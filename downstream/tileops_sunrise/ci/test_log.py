"""Show live case progress and archive the complete shared-runner output."""

import collections
import gzip
import json
import shutil
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
    prefixes = ("Total test cases:", ">>>>>>> running case ", "Case timeout:", "  -> ", "ERROR:")
    pending = ""
    with raw_log.open(encoding="utf-8", errors="replace") as source:
        while True:
            chunk = source.readline()
            if chunk:
                pending += chunk
                if pending.endswith("\n"):
                    if pending.startswith(prefixes):
                        print(pending, end="", flush=True)
                    pending = ""
            elif stopping or done.exists():
                if pending.startswith(prefixes):
                    print(pending, flush=True)
                return
            else:
                time.sleep(0.2)


def finish(report_dir: Path) -> None:
    raw_log = report_dir / "tileops-test.log"
    archive = raw_log.with_suffix(".log.gz")
    # Retain the original on any archive failure, including a full filesystem.
    with raw_log.open("rb") as source, gzip.open(archive, "wb") as target:
        shutil.copyfileobj(source, target)
    raw_log.unlink()
    print(f"Full test log (case/attempt/command boundaries included): {archive}")

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
