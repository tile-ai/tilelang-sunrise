#!/usr/bin/env python3
"""Synthesize a validated JUnit report from a suite's per-case JSONL.

The final ``case_result`` decides the testcase verdict. Every ``case_attempt``
is retained in JUnit properties/system-out, including failures before a retry
passes, and testcase time is the sum of all attempts.

Pure stdlib; no external dependencies.
"""

import argparse
import contextlib
import json
import os
import sys
import tempfile
from xml.sax.saxutils import escape, quoteattr


FAILURE_STATUSES = {"FAIL", "TIMEOUT", "NOT_RUN"}
VALID_STATUSES = FAILURE_STATUSES | {"PASS", "SKIPPED"}


class ReportError(ValueError):
    """The JSONL cannot support a complete, trustworthy JUnit report."""


def _clean(text):
    """Remove characters forbidden by XML 1.0, including lone surrogates."""
    return "".join(
        char
        for char in text
        if char in "\t\n\r" or 0x20 <= ord(char) <= 0xD7FF or 0xE000 <= ord(char) <= 0xFFFD or 0x10000 <= ord(char) <= 0x10FFFF
    )


def _require_text(record, key, line_number, *, nonempty=False):
    value = record.get(key)
    if not isinstance(value, str) or (nonempty and not value):
        qualifier = "non-empty " if nonempty else ""
        raise ReportError(f"line {line_number}: {key!r} must be a {qualifier}string")
    return value


def _require_integer(record, key, line_number, *, minimum=None):
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReportError(f"line {line_number}: {key!r} must be an integer")
    if minimum is not None and value < minimum:
        raise ReportError(f"line {line_number}: {key!r} must be >= {minimum}")
    return value


def _validate_record(record, line_number, suite):
    if not isinstance(record, dict):
        raise ReportError(f"line {line_number}: record must be a JSON object")
    if record.get("schema_version") != 2:
        raise ReportError(f"line {line_number}: unsupported or missing schema_version")

    kind = _require_text(record, "record_kind", line_number, nonempty=True)
    if kind not in {"case_attempt", "case_result", "device_recovery"}:
        raise ReportError(f"line {line_number}: unsupported record_kind {kind!r}")
    record_suite = _require_text(record, "suite", line_number, nonempty=True)
    if record_suite != suite:
        raise ReportError(f"line {line_number}: suite {record_suite!r} does not match {suite!r}")

    _require_text(record, "case", line_number, nonempty=True)
    if kind == "device_recovery":
        _require_integer(record, "attempt", line_number, minimum=1)
        _require_text(record, "device", line_number, nonempty=True)
        if record.get("action") != "reset":
            raise ReportError(f"line {line_number}: unsupported recovery action")
        status = _require_text(record, "status", line_number, nonempty=True)
        if status not in {"PASS", "FAIL", "SKIPPED"}:
            raise ReportError(f"line {line_number}: unsupported recovery status")
        exit_code = _require_integer(record, "exit_code", line_number)
        reason = _require_text(record, "reason", line_number)
        if (status == "FAIL") != (exit_code != 0) or (status != "PASS" and not reason):
            raise ReportError(f"line {line_number}: inconsistent recovery outcome")
        return record

    _require_text(record, "command", line_number, nonempty=True)
    if "timeout_seconds" in record:
        _require_integer(record, "timeout_seconds", line_number, minimum=1)
    status = _require_text(record, "status", line_number, nonempty=True)
    if status not in VALID_STATUSES:
        raise ReportError(f"line {line_number}: unsupported status {status!r}")
    _require_integer(record, "exit_code", line_number)
    _require_integer(record, "elapsed_seconds", line_number, minimum=0)
    reason = _require_text(record, "failure_reason", line_number)
    _require_text(record, "log_tail", line_number)
    if status in FAILURE_STATUSES and not reason:
        raise ReportError(f"line {line_number}: {status} record has no failure_reason")

    if status == "NOT_RUN" and (kind != "case_result" or record["elapsed_seconds"] != 0 or record["exit_code"] == 0):
        raise ReportError(f"line {line_number}: invalid NOT_RUN result")

    if kind == "case_attempt":
        attempt = _require_integer(record, "attempt", line_number, minimum=1)
        total_attempts = _require_integer(record, "total_attempts", line_number, minimum=1)
        if attempt > total_attempts:
            raise ReportError(f"line {line_number}: attempt exceeds total_attempts")
    return record


def load_cases(jsonl_path, suite, expected_cases):
    if expected_cases < 1:
        raise ReportError("expected case count must be positive")
    if not os.path.isfile(jsonl_path):
        raise ReportError(f"{jsonl_path} not found")

    states = {}
    order = []
    record_count = 0
    try:
        with open(jsonl_path, encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, 1):
                if not raw_line.strip():
                    raise ReportError(f"line {line_number}: empty JSONL record")
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise ReportError(f"line {line_number}: malformed JSON: {error.msg}") from error
                record = _validate_record(record, line_number, suite)
                record_count += 1

                case_id = record["case"]
                if case_id not in states:
                    states[case_id] = {"attempts": [], "recoveries": [], "result": None}
                    order.append(case_id)
                state = states[case_id]

                if record["record_kind"] == "device_recovery":
                    attempts = state["attempts"]
                    if state["result"] is not None or not attempts:
                        raise ReportError(f"line {line_number}: recovery outside an active attempt for {case_id!r}")
                    if record["attempt"] != attempts[-1]["attempt"] or attempts[-1]["status"] != "TIMEOUT":
                        raise ReportError(f"line {line_number}: recovery does not match a timeout attempt for {case_id!r}")
                    if any(item["attempt"] == record["attempt"] for item in state["recoveries"]):
                        raise ReportError(f"line {line_number}: duplicate recovery for {case_id!r}")
                    state["recoveries"].append(record)
                    continue

                if record["record_kind"] == "case_attempt":
                    if state["result"] is not None:
                        raise ReportError(f"line {line_number}: attempt appears after result for {case_id!r}")
                    attempts = state["attempts"]
                    expected_attempt = len(attempts) + 1
                    if record["attempt"] != expected_attempt:
                        raise ReportError(
                            f"line {line_number}: {case_id!r} attempt sequence expected {expected_attempt}, got {record['attempt']}"
                        )
                    if attempts:
                        if attempts[-1]["status"] in {"PASS", "SKIPPED"}:
                            raise ReportError(f"line {line_number}: attempt follows terminal status for {case_id!r}")
                        first = attempts[0]
                        if record["total_attempts"] != first["total_attempts"]:
                            raise ReportError(f"line {line_number}: inconsistent attempt limit for {case_id!r}")
                        if record["command"] != first["command"]:
                            raise ReportError(f"line {line_number}: inconsistent command for {case_id!r}")
                        if record.get("timeout_seconds") != first.get("timeout_seconds"):
                            raise ReportError(f"line {line_number}: inconsistent timeout for {case_id!r}")
                    attempts.append(record)
                    continue

                if state["result"] is not None:
                    raise ReportError(f"line {line_number}: duplicate case_result for {case_id!r}")
                attempts = state["attempts"]
                if not attempts:
                    aborted = any(
                        prior["result"] is not None
                        and prior["result"]["status"] == "TIMEOUT"
                        and prior["recoveries"]
                        and prior["recoveries"][-1]["attempt"] == len(prior["attempts"])
                        and prior["recoveries"][-1]["status"] != "PASS"
                        for prior in states.values()
                    )
                    if record["status"] != "NOT_RUN" or not aborted:
                        raise ReportError(f"line {line_number}: case_result for {case_id!r} has no attempts")
                    state["result"] = record
                    continue
                final_attempt = attempts[-1]
                if record.get("timeout_seconds") != final_attempt.get("timeout_seconds"):
                    raise ReportError(f"line {line_number}: case_result timeout disagrees with final attempt")
                for key in (
                    "command",
                    "status",
                    "exit_code",
                    "elapsed_seconds",
                    "failure_reason",
                    "log_tail",
                ):
                    if record[key] != final_attempt[key]:
                        raise ReportError(f"line {line_number}: case_result {key!r} disagrees with final attempt")
                aborted = bool(
                    state["recoveries"]
                    and state["recoveries"][-1]["attempt"] == final_attempt["attempt"]
                    and state["recoveries"][-1]["status"] != "PASS"
                )
                if record["status"] in FAILURE_STATUSES and len(attempts) != final_attempt["total_attempts"] and not aborted:
                    raise ReportError(f"line {line_number}: failed case {case_id!r} did not record every retry")
                for attempt in attempts:
                    if (
                        attempt["status"] == "TIMEOUT"
                        and "timeout_seconds" in attempt
                        and not any(item["attempt"] == attempt["attempt"] for item in state["recoveries"])
                    ):
                        raise ReportError(f"line {line_number}: missing device recovery for {case_id!r} attempt {attempt['attempt']}")
                state["result"] = record
    except UnicodeError as error:
        raise ReportError(f"{jsonl_path} is not valid UTF-8: {error}") from error

    if record_count == 0:
        raise ReportError(f"{jsonl_path} is empty")
    incomplete = [case_id for case_id in order if states[case_id]["result"] is None]
    if incomplete:
        raise ReportError(f"missing case_result for: {', '.join(incomplete)}")
    if len(order) != expected_cases:
        raise ReportError(f"expected {expected_cases} case_result records, found {len(order)}")
    return [states[case_id] for case_id in order]


def _attempt_history(attempts, recoveries):
    if not attempts:
        return "Not run: device recovery failed."
    lines = [f"Attempts used: {len(attempts)}/{attempts[0]['total_attempts']}"]
    for attempt in attempts:
        lines.append(
            f"Attempt {attempt['attempt']}/{attempt['total_attempts']}: {attempt['status']} "
            f"({attempt['elapsed_seconds']}s, exit_code={attempt['exit_code']})"
        )
        if attempt["failure_reason"]:
            lines.append(f"Failure reason: {attempt['failure_reason']}")
        if "timeout_seconds" in attempt:
            lines.append(f"Timeout: {attempt['timeout_seconds']}s")
        if attempt["log_tail"]:
            lines.append("Log tail:")
            lines.append(attempt["log_tail"])
        for recovery in recoveries:
            if recovery["attempt"] == attempt["attempt"]:
                lines.append(
                    f"Device recovery: {recovery['action']} device={recovery['device']} "
                    f"status={recovery['status']} exit_code={recovery['exit_code']} reason={recovery['reason']}"
                )
    return "\n".join(lines)


def render(suite, cases):
    failures = sum(1 for case in cases if case["result"]["status"] in FAILURE_STATUSES)
    skipped = sum(1 for case in cases if case["result"]["status"] == "SKIPPED")
    elapsed_by_case = [sum(attempt["elapsed_seconds"] for attempt in case["attempts"]) for case in cases]
    total_time = sum(elapsed_by_case)

    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<testsuites>"]
    lines.append(
        f'  <testsuite name={quoteattr(_clean(suite))} tests="{len(cases)}" failures="{failures}" skipped="{skipped}" time="{total_time}">'
    )
    for case, elapsed in zip(cases, elapsed_by_case):
        result = case["result"]
        attempts = case["attempts"]
        status = result["status"]
        statuses = ",".join(attempt["status"] for attempt in attempts)
        durations = ",".join(str(attempt["elapsed_seconds"]) for attempt in attempts)
        lines.append(f'    <testcase classname={quoteattr(_clean(suite))} name={quoteattr(_clean(result["case"]))} time="{elapsed}">')
        lines.append("      <properties>")
        lines.append(f'        <property name="attempts_used" value="{len(attempts)}"/>')
        lines.append(f'        <property name="attempt_limit" value="{attempts[0]["total_attempts"] if attempts else 0}"/>')
        lines.append(f'        <property name="attempt_statuses" value={quoteattr(_clean(statuses))}/>')
        lines.append(f'        <property name="attempt_durations_seconds" value={quoteattr(durations)}/>')
        if "timeout_seconds" in result:
            lines.append(f'        <property name="timeout_seconds" value="{result["timeout_seconds"]}"/>')
        if case["recoveries"]:
            lines.append(f'        <property name="device_recovery_count" value="{len(case["recoveries"])}"/>')
        lines.append("      </properties>")
        if status in FAILURE_STATUSES:
            message = result["failure_reason"] or status
            lines.append(
                f"      <failure message={quoteattr(_clean(message))} "
                f"type={quoteattr(status)}>{escape(_clean(result['log_tail']))}</failure>"
            )
        elif status == "SKIPPED":
            lines.append(f"      <skipped message={quoteattr(_clean(result['failure_reason']))}/>")
        lines.append(f"      <system-out>{escape(_clean(_attempt_history(attempts, case['recoveries'])))}</system-out>")
        lines.append("    </testcase>")
    lines.extend(("  </testsuite>", "</testsuites>"))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, help="suite name (JSONL basename)")
    parser.add_argument("--report-dir", required=True, help="ci_failure_reports directory")
    parser.add_argument("--expected-cases", required=True, type=int, help="number of cases in the resolved run plan")
    args = parser.parse_args()

    jsonl_path = os.path.join(args.report_dir, f"{args.suite}.jsonl")
    xml_path = os.path.join(args.report_dir, f"junit-{args.suite}.xml")
    temporary_path = None
    try:
        if os.path.exists(xml_path):
            os.remove(xml_path)
        cases = load_cases(jsonl_path, args.suite, args.expected_cases)
        document = render(args.suite, cases)
        os.makedirs(args.report_dir, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(prefix=f".junit-{args.suite}.", suffix=".tmp", dir=args.report_dir, text=True)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(document)
        os.replace(temporary_path, xml_path)
        temporary_path = None
    except (OSError, ReportError) as error:
        if temporary_path:
            with contextlib.suppress(OSError):
                os.remove(temporary_path)
        print(f"junit_from_jsonl: ERROR: {error}", file=sys.stderr)
        return 1

    print(f"junit_from_jsonl: wrote {xml_path} ({len(cases)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
