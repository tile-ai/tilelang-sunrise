"""Stream the complete test log and retain it as a plain-text artifact."""

import collections
import json
import signal
import sys
import time
from pathlib import Path


def follow(raw_log: Path) -> None:
    done = raw_log.with_suffix(".log.done")
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    # The runner writes this file directly, independently of GitLab's trace limit.
    with raw_log.open("rb") as source:
        while True:
            chunk = source.read(65536)
            if chunk:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            elif stopping or done.exists():
                return
            else:
                time.sleep(0.2)


def finish(report_dir: Path) -> None:
    raw_log = report_dir / "tileops-test.log"
    if not raw_log.is_file():
        raise RuntimeError("full test log missing")
    print(f"Full test log (plain-text artifact): {raw_log}")

    counts = collections.Counter()
    report = report_dir / "tileops.jsonl"
    if not report.exists():
        raise RuntimeError("case report missing; inspect the full test log")
    with report.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record.get("record_kind") != "case_result":
                continue
            status = record["status"]
            counts[status] += 1
            # JSON escaping keeps multiline/untrusted labels on a single line.
            case = json.dumps(record["case"], ensure_ascii=True)[:400]
            print(f"{status}: {case} (exit {record['exit_code']}, {record['elapsed_seconds']}s)")
    if not counts:
        raise RuntimeError("no completed case results; inspect the full test log")
    print("TileOPs results: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(
        f"Attempt exit codes/durations: {report}; "
        f"node results: {report_dir / 'tileops-pytest.jsonl'}"
    )


if __name__ == "__main__":
    try:
        if sys.argv[1] == "--follow":
            follow(Path(sys.argv[2]))
        else:
            finish(Path(sys.argv[1]))
    except Exception as error:
        print(f"ERROR: test log processing failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
