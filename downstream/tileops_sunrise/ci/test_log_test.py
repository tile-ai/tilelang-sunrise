"""CPU-only integration checks for console streaming and plain log retention."""

import json
import os
import shlex
import signal
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path

CI_DIR = Path(__file__).resolve().parent


class TestLogging(unittest.TestCase):
    def setUp(self):
        state = CI_DIR.parent / ".ci-state"
        state.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="log-test-", dir=state)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.console = self.root / "console"
        self.raw = self.root / "tileops-test.log"

    def start(self, body, exit_code=0, trace_limit=None, setup=""):
        record = json.dumps(
            dict(
                record_kind="case_result",
                case="fixture",
                status="FAIL" if exit_code else "PASS",
                exit_code=exit_code,
                elapsed_seconds=1,
            )
        )
        script = f"""
set -e
SCRIPT_DIR={shlex.quote(str(CI_DIR))}
REPORT={shlex.quote(str(self.root))}
source "$SCRIPT_DIR/test_logging.sh"
ci_failure_report_dir () {{ echo "$REPORT"; }}
ci_prepare_failure_reports () {{ :; }}
ci_cleanup_state () {{ trap - EXIT TERM INT; exit "$1"; }}
trap 'exit 143' TERM INT
tileops_prepare_test_log
echo "$TILEOPS_PROGRESS_PID" > "$REPORT/helper.pid"
printf '%s\\n' {shlex.quote(record)} > "$REPORT/tileops.jsonl"
{setup}
{{
    {body}
    exit {exit_code}
}} > "$TILEOPS_TEST_LOG" 2>&1
"""
        self.proc = subprocess.Popen(
            ["bash", "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "CASE_REPEAT": "1"},
        )
        self.addCleanup(self.stop)
        self.received = 0

        def consume():
            # GitLab keeps consuming output after it stops retaining the trace.
            with self.console.open("wb", buffering=0) as output:
                while chunk := os.read(self.proc.stdout.fileno(), 65536):
                    retained = chunk
                    if trace_limit is not None:
                        retained = chunk[: max(0, trace_limit - self.received)]
                    output.write(retained)
                    self.received += len(chunk)

        self.reader = threading.Thread(target=consume)
        self.reader.start()

    def stop(self):
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait()
        self.reader.join(timeout=5)
        self.proc.stdout.close()

    def await_output(self, expected):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if self.console.exists() and expected in self.console.read_bytes():
                self.assertIsNone(self.proc.poll(), "output arrived only after exit")
                return
            time.sleep(0.02)
        self.fail(f"Missing live output {expected!r}")

    def finish(self, expected_exit):
        self.assertEqual(self.proc.wait(timeout=10), expected_exit)
        self.reader.join(timeout=5)
        self.assertFalse(self.reader.is_alive())
        self.assertTrue(self.raw.is_file())
        self.assertFalse(self.raw.with_suffix(".log.gz").exists())
        self.assertFalse(self.raw.with_suffix(".log.done").exists())
        helper = int((self.root / "helper.pid").read_text())
        self.assertFalse(Path(f"/proc/{helper}").exists(), "stream helper leaked")
        return self.raw.read_bytes()

    def test_detailed_stdout_stderr_and_partial_line_are_live(self):
        self.start(
            "printf 'test_example PASSED\\n'; printf 'compiler detail\\n' >&2; "
            "printf 'partial'; "
            'while [ ! -f "$REPORT/release" ]; do sleep .02; done; '
            "printf -- '-tail'"
        )
        self.await_output(b"test_example PASSED\ncompiler detail\npartial")
        (self.root / "release").touch()
        self.assertEqual(self.finish(0), b"test_example PASSED\ncompiler detail\npartial-tail")
        self.assertIn(b"partial-tail", self.console.read_bytes())

    def test_trace_limit_preserves_full_plain_log(self):
        payload = b"x" * (5 * 1024 * 1024) + b"\nAFTER_TRACE_LIMIT\n"
        source = self.root / "payload"
        source.write_bytes(payload)
        self.start('cat "$REPORT/payload"', trace_limit=4096)
        self.assertEqual(self.finish(0), payload)
        self.assertEqual(self.console.stat().st_size, 4096)
        self.assertGreater(self.received, len(payload))
        self.assertNotIn(b"AFTER_TRACE_LIMIT", self.console.read_bytes())

    def test_failure_preserves_exit_and_details(self):
        self.start("printf 'failure detail\\n' >&2", exit_code=7)
        self.assertEqual(self.finish(7), b"failure detail\n")
        self.assertIn(b"failure detail", self.console.read_bytes())

    def test_timeout_exit_is_preserved(self):
        self.start("printf 'timeout evidence\\n'; timeout .2 sleep 5")
        self.assertEqual(self.finish(124), b"timeout evidence\n")

    def test_cancel_retains_partial_log_and_cleans_helper(self):
        self.start("printf 'in-flight detail\\n'; sleep 30 & wait")
        self.await_output(b"in-flight detail")
        os.killpg(self.proc.pid, signal.SIGTERM)
        self.assertIn(b"in-flight detail", self.finish(143))

    def test_helper_failure_does_not_interrupt_file_recording(self):
        self.start("printf 'after helper failure\\n'", setup='kill -KILL "$TILEOPS_PROGRESS_PID"')
        self.assertEqual(self.finish(2), b"after helper failure\n")

    def test_summary_failure_keeps_plain_log(self):
        self.start("printf 'retained detail\\n'", setup='echo invalid > "$REPORT/tileops.jsonl"')
        self.assertEqual(self.finish(2), b"retained detail\n")


if __name__ == "__main__":
    unittest.main()
