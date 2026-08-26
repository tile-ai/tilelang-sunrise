"""Record pytest collection markers and per-phase outcomes as JSONL.

The CI runner executes one test file per pytest process to keep PTPU work
strictly serial.  Every process appends to the same job-local artifact, making
the exact parameterized nodeids and skip reasons available for migration
classification without changing test semantics.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _output_path() -> Path | None:
    value = os.getenv("PYTEST_NODE_AUDIT_PATH")
    return Path(value) if value else None


def _bounded_text(value, env_name: str, default: int) -> str:
    text = str(value)
    limit = int(os.getenv(env_name, str(default)))
    if len(text) <= limit:
        return text
    return f"...<truncated {len(text) - limit} chars>..." + text[-limit:]


def _bounded_repr(value) -> str:
    return _bounded_text(repr(value), "PYTEST_NODE_AUDIT_MARKER_CHARS", 1000)


def _write(event: dict) -> None:
    output = _output_path()
    if output is None:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 2,
        "record_kind": "pytest_event",
        "suite": os.getenv("PYTEST_NODE_AUDIT_SUITE", "tilelang"),
        "pipeline_id": os.getenv("CI_PIPELINE_ID", ""),
        "job_id": os.getenv("CI_JOB_ID", ""),
        "commit_sha": os.getenv("CI_COMMIT_SHA", ""),
        **event,
    }
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _marker_record(marker) -> dict:
    return {
        "name": marker.name,
        "args": [_bounded_repr(value) for value in marker.args],
        "kwargs": {key: _bounded_repr(value) for key, value in marker.kwargs.items()},
    }


def pytest_sessionstart(session) -> None:
    """Restore test cwd and sibling imports without exposing the checkout root."""
    source_root = Path(os.getenv("PYTEST_NODE_AUDIT_SOURCE_ROOT", "/")).resolve()
    for argument in session.config.args:
        test_path = Path(str(argument).split("::", maxsplit=1)[0])
        if test_path.suffix == ".py":
            test_dir = test_path.resolve().parent
            if test_dir != source_root and str(test_dir) not in sys.path:
                sys.path.insert(0, str(test_dir))
    test_cwd = os.getenv("PYTEST_NODE_AUDIT_TEST_CWD")
    if test_cwd:
        os.chdir(test_cwd)


def pytest_collection_modifyitems(items) -> None:
    for item in items:
        _write(
            {
                "event": "collected",
                "nodeid": item.nodeid,
                "markers": [_marker_record(marker) for marker in item.iter_markers()],
            }
        )


def pytest_collectreport(report) -> None:
    if report.failed:
        _write(
            {
                "event": "collection_error",
                "nodeid": report.nodeid,
                "outcome": report.outcome,
                "reason": _bounded_text(report.longrepr, "PYTEST_NODE_AUDIT_REASON_CHARS", 12000),
            }
        )


def pytest_runtest_logreport(report) -> None:
    reason = ""
    if report.skipped:
        if isinstance(report.longrepr, tuple) and len(report.longrepr) == 3:
            reason = _bounded_text(report.longrepr[2], "PYTEST_NODE_AUDIT_REASON_CHARS", 12000)
        else:
            reason = _bounded_text(report.longrepr, "PYTEST_NODE_AUDIT_REASON_CHARS", 12000)
    elif report.failed:
        reason = _bounded_text(report.longrepr, "PYTEST_NODE_AUDIT_REASON_CHARS", 12000)
    _write(
        {
            "event": "outcome",
            "nodeid": report.nodeid,
            "phase": report.when,
            "outcome": report.outcome,
            "reason": reason,
            "duration_seconds": report.duration,
        }
    )
