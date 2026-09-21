import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

import pytest


ROOT = Path(__file__).resolve().parents[2]
CI_LIB = ROOT / "ci" / "lib.sh"
CI_RUN = ROOT / "ci" / "run.sh"
JUNIT_GENERATOR = ROOT / "ci" / "junit_from_jsonl.py"


def _attempt(
    case,
    status,
    elapsed,
    attempt=1,
    total_attempts=1,
    *,
    exit_code=0,
    failure_reason="",
    log_tail="",
):
    return {
        "schema_version": 2,
        "record_kind": "case_attempt",
        "suite": "tilelang",
        "case": case,
        "command": f"python {case}",
        "status": status,
        "exit_code": exit_code,
        "elapsed_seconds": elapsed,
        "attempt": attempt,
        "total_attempts": total_attempts,
        "failure_reason": failure_reason,
        "log_tail": log_tail,
    }


def _result(final_attempt):
    record = dict(final_attempt)
    record["record_kind"] = "case_result"
    del record["attempt"]
    del record["total_attempts"]
    return record


def _write_jsonl(report_dir, records):
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(record, ensure_ascii=True) + "\n" for record in records)
    (report_dir / "tilelang.jsonl").write_text(payload, encoding="utf-8")


def _run_generator(report_dir, expected_cases):
    return subprocess.run(
        [
            sys.executable,
            str(JUNIT_GENERATOR),
            "--suite",
            "tilelang",
            "--report-dir",
            str(report_dir),
            "--expected-cases",
            str(expected_cases),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def _properties(testcase):
    return {item.attrib["name"]: item.attrib["value"] for item in testcase.findall("./properties/property")}


def test_retry_before_pass_keeps_all_attempts_and_total_time(tmp_path):
    first = _attempt(
        "examples/blocksparse_attention/test_example_blocksparse_attention.py",
        "TIMEOUT",
        900,
        attempt=1,
        total_attempts=3,
        exit_code=124,
        failure_reason="timed out after 900s",
        log_tail="first timeout <& tail",
    )
    second = _attempt(
        first["case"],
        "TIMEOUT",
        901,
        attempt=2,
        total_attempts=3,
        exit_code=124,
        failure_reason="timed out after 900s",
        log_tail="second timeout tail",
    )
    third = _attempt(first["case"], "PASS", 16, attempt=3, total_attempts=3)
    report_dir = tmp_path / "reports"
    _write_jsonl(report_dir, [first, second, third, _result(third)])

    completed = _run_generator(report_dir, expected_cases=1)

    assert completed.returncode == 0, completed.stderr
    suite = ElementTree.parse(report_dir / "junit-tilelang.xml").getroot().find("testsuite")
    assert suite is not None
    assert suite.attrib == {
        "name": "tilelang",
        "tests": "1",
        "failures": "0",
        "skipped": "0",
        "time": "1817",
    }
    testcase = suite.find("testcase")
    assert testcase is not None
    assert testcase.attrib["time"] == "1817"
    assert testcase.find("failure") is None
    assert _properties(testcase) == {
        "attempts_used": "3",
        "attempt_limit": "3",
        "attempt_statuses": "TIMEOUT,TIMEOUT,PASS",
        "attempt_durations_seconds": "900,901,16",
    }
    history = testcase.findtext("system-out", default="")
    assert "Attempts used: 3/3" in history
    assert "Attempt 1/3: TIMEOUT (900s, exit_code=124)" in history
    assert "Failure reason: timed out after 900s" in history
    assert "first timeout <& tail" in history
    assert "Attempt 3/3: PASS (16s, exit_code=0)" in history


def test_normal_verdicts_and_xml_illegal_characters_are_rendered_safely(tmp_path):
    passed = _attempt("pass.py", "PASS", 1)
    failed = _attempt(
        "fail<&\x00\ud800.py",
        "FAIL",
        2,
        exit_code=1,
        failure_reason="broken <&\x00\ud800 reason",
        log_tail="trace <&\x00\ud800 tail",
    )
    skipped = _attempt(
        "skip.py",
        "SKIPPED",
        3,
        exit_code=5,
        failure_reason="all skipped",
    )
    report_dir = tmp_path / "reports"
    _write_jsonl(
        report_dir,
        [passed, _result(passed), failed, _result(failed), skipped, _result(skipped)],
    )

    completed = _run_generator(report_dir, expected_cases=3)

    assert completed.returncode == 0, completed.stderr
    xml_path = report_dir / "junit-tilelang.xml"
    raw_xml = xml_path.read_text(encoding="utf-8")
    assert "\x00" not in raw_xml
    assert "\ud800" not in raw_xml
    suite = ElementTree.parse(xml_path).getroot().find("testsuite")
    assert suite is not None
    assert suite.attrib["tests"] == "3"
    assert suite.attrib["failures"] == "1"
    assert suite.attrib["skipped"] == "1"
    assert suite.attrib["time"] == "6"
    testcases = {item.attrib["name"]: item for item in suite.findall("testcase")}
    assert testcases["pass.py"].find("failure") is None
    failure = testcases["fail<&.py"].find("failure")
    assert failure is not None
    assert failure.attrib == {"message": "broken <& reason", "type": "FAIL"}
    assert failure.text == "trace <& tail"
    assert testcases["skip.py"].find("skipped") is not None
    assert all(testcase.find("system-out") is not None for testcase in testcases.values())


@pytest.mark.parametrize(
    "scenario",
    ["missing", "empty", "malformed", "missing_field", "unfinished_case", "count_mismatch"],
)
def test_invalid_or_incomplete_jsonl_fails_without_leaving_junit(tmp_path, scenario):
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    xml_path = report_dir / "junit-tilelang.xml"
    xml_path.write_text("stale", encoding="utf-8")
    jsonl_path = report_dir / "tilelang.jsonl"
    expected_cases = 1

    attempt = _attempt("pass.py", "PASS", 1)
    if scenario == "empty":
        jsonl_path.write_text("", encoding="utf-8")
    elif scenario == "malformed":
        valid = "".join(json.dumps(record) + "\n" for record in (attempt, _result(attempt)))
        jsonl_path.write_text(valid + "{not-json}\n", encoding="utf-8")
    elif scenario == "missing_field":
        del attempt["status"]
        _write_jsonl(report_dir, [attempt])
    elif scenario == "unfinished_case":
        _write_jsonl(report_dir, [attempt])
    elif scenario == "count_mismatch":
        _write_jsonl(report_dir, [attempt, _result(attempt)])
        expected_cases = 2

    completed = _run_generator(report_dir, expected_cases)

    assert completed.returncode != 0
    assert "junit_from_jsonl: ERROR:" in completed.stderr
    assert not xml_path.exists()


@pytest.mark.parametrize(
    ("run_action", "run_returncode", "report_returncode", "expected_returncode"),
    [
        ("return", 0, 0, 0),
        ("return", 0, 9, 9),
        ("return", 7, 0, 7),
        ("return", 7, 9, 7),
        ("exit", 7, 0, 7),
        ("exit", 7, 9, 7),
    ],
)
def test_run_sh_propagates_report_failure_without_overwriting_test_failure(
    tmp_path, run_action, run_returncode, report_returncode, expected_returncode
):
    project = tmp_path / "project"
    ci_dir = project / "ci"
    ci_dir.mkdir(parents=True)
    shutil.copy2(CI_RUN, ci_dir / "run.sh")
    (project / "case.py").write_text("pass\n", encoding="utf-8")
    (ci_dir / "lib.sh").write_text(
        """\
ci_parse_list_into () {
    while IFS= read -r line; do
        CI_CMDS+=("true")
        CI_LABELS+=("$line")
        CI_CASE_MODES+=("command")
    done < "$1"
}
ci_failure_report_dir () { echo "$ROOT/reports"; }
ci_init_state () {
    CI_TMP_DIR="$ROOT/tmp"
    mkdir -p "$CI_TMP_DIR"
    trap ci_cleanup_state EXIT
}
ci_create_conda_env () { :; }
ci_configure_ptcc () { :; }
ci_export_tang_env () { :; }
ci_install_tilelang_whl () { :; }
ci_prepare_installed_wheel_env () { :; }
ci_assert_installed_tilelang_wheel () { :; }
ci_assert_runtime_stack () { :; }
ci_cleanup_state () {
    local trapped_ret=$?
    local cleanup_ret="${1:-$trapped_ret}"
    trap - EXIT
    printf '%s\n' "$cleanup_ret" > "$ROOT/cleanup-status"
    exit "$cleanup_ret"
}
ci_run_prepared_cases () {
    if [[ "${STUB_RUN_ACTION}" == "exit" ]]; then
        exit "${STUB_RUN_RET}"
    fi
    return "${STUB_RUN_RET}"
}
""",
        encoding="utf-8",
    )
    (ci_dir / "junit_from_jsonl.py").write_text(
        """\
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--suite", required=True)
parser.add_argument("--report-dir", required=True)
parser.add_argument("--expected-cases", required=True, type=int)
args = parser.parse_args()
if args.expected_cases != 1:
    sys.exit(19)
sys.exit(int(os.environ["STUB_REPORT_RET"]))
""",
        encoding="utf-8",
    )
    home = tmp_path / "home"
    temp_dir = tmp_path / "tmp"
    home.mkdir()
    temp_dir.mkdir()
    (home / ".bashrc").write_text("exit 97\n", encoding="utf-8")
    environment = os.environ.copy()
    environment.update(
        HOME=str(home),
        TMPDIR=str(temp_dir),
        STUB_RUN_ACTION=run_action,
        STUB_RUN_RET=str(run_returncode),
        STUB_REPORT_RET=str(report_returncode),
    )

    completed = subprocess.run(
        ["bash", str(ci_dir / "run.sh"), "--ci", "--case", "case.py"],
        cwd=project,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == expected_returncode, completed.stdout + completed.stderr
    assert (project / "cleanup-status").read_text(encoding="utf-8").strip() == str(expected_returncode)
    if report_returncode:
        assert f"JUnit synthesis failed (exit {report_returncode})" in completed.stderr
    if run_returncode and report_returncode:
        assert f"Preserving test/recording exit code {run_returncode}" in completed.stderr


def test_case_recording_failure_makes_successful_test_run_fail(tmp_path):
    shell = f"""
set -e
source {shlex.quote(str(CI_LIB))}
ci_prepare_failure_reports () {{ return 0; }}
ci_check_card_state () {{ return 0; }}
ci_run_timed () {{ : > "$2"; return 0; }}
ci_record_case_attempt () {{ return 17; }}
ci_record_case_result () {{ return 18; }}
CI_CMDS=("true")
CI_LABELS=("case.py")
CI_CASE_MODES=("command")
CASE_REPEAT=1
TILELANG_CACHE_DIR={shlex.quote(str(tmp_path / "cache"))}
ci_run_prepared_cases tilelang
"""
    environment = os.environ.copy()
    environment["TMPDIR"] = str(tmp_path)

    completed = subprocess.run(
        ["bash", "-c", shell],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2, completed.stdout + completed.stderr
    assert "failed to record attempt 1 for case.py" in completed.stderr
    assert "failed to record final result for case.py" in completed.stderr


def test_sourced_ci_library_is_non_executable_and_has_no_shebang():
    assert CI_LIB.stat().st_mode & 0o111 == 0
    assert not CI_LIB.read_bytes().startswith(b"#!")


def _recovery(attempt, status="SKIPPED"):
    return {
        "schema_version": 2,
        "record_kind": "device_recovery",
        "suite": "tilelang",
        "case": attempt["case"],
        "attempt": attempt["attempt"],
        "action": "reset",
        "device": "6",
        "status": status,
        "exit_code": 17 if status == "FAIL" else 0,
        "reason": "" if status == "PASS" else "reset unavailable <&>",
    }


@pytest.mark.parametrize("recovery_status", ["PASS", "FAIL", "SKIPPED"])
def test_timeout_and_device_recovery_preserve_retry_verdict(tmp_path, recovery_status):
    first = _attempt("case.py", "TIMEOUT", 1, total_attempts=2, exit_code=124, failure_reason="deadline")
    second = _attempt("case.py", "PASS", 2, attempt=2, total_attempts=2)
    first["timeout_seconds"] = second["timeout_seconds"] = 1
    recovery = _recovery(first, recovery_status)
    report_dir = tmp_path / "reports"
    _write_jsonl(report_dir, [first, recovery, second, _result(second)])
    completed = _run_generator(report_dir, 1)
    assert completed.returncode == 0, completed.stderr
    suite = ElementTree.parse(report_dir / "junit-tilelang.xml").getroot().find("testsuite")
    assert (suite.attrib["tests"], suite.attrib["failures"], suite.attrib["time"]) == ("1", "0", "3")
    testcase = suite.find("testcase")
    assert _properties(testcase)["timeout_seconds"] == "1"
    assert _properties(testcase)["device_recovery_count"] == "1"
    output = testcase.findtext("system-out")
    assert f"Device recovery: reset device=6 status={recovery_status}" in output
    assert "Attempt 2/2: PASS" in output


@pytest.mark.parametrize(
    "fault",
    [
        "missing_recovery",
        "duplicate_recovery",
        "wrong_attempt",
        "wrong_case",
        "after_result",
        "wrong_action",
        "wrong_status",
        "wrong_exit",
        "timeout_zero",
        "timeout_bool",
        "timeout_mismatch",
    ],
)
def test_invalid_timeout_or_recovery_evidence_fails_closed(tmp_path, fault):
    attempt = _attempt("case.py", "TIMEOUT", 1, exit_code=124, failure_reason="deadline")
    attempt["timeout_seconds"] = 1
    recovery = _recovery(attempt)
    result = _result(attempt)
    records = [attempt, recovery, result]
    if fault == "missing_recovery":
        records.remove(recovery)
    elif fault == "duplicate_recovery":
        records.insert(2, dict(recovery))
    elif fault == "wrong_attempt":
        recovery["attempt"] = 2
    elif fault == "wrong_case":
        recovery["case"] = "another.py"
    elif fault == "after_result":
        records = [attempt, result, recovery]
    elif fault == "wrong_action":
        recovery["action"] = "unknown"
    elif fault == "wrong_status":
        recovery["status"] = "UNKNOWN"
    elif fault == "wrong_exit":
        recovery["exit_code"] = 17
    elif fault == "timeout_zero":
        attempt["timeout_seconds"] = 0
    elif fault == "timeout_bool":
        attempt["timeout_seconds"] = True
    elif fault == "timeout_mismatch":
        result["timeout_seconds"] = 2
    report_dir = tmp_path / "reports"
    _write_jsonl(report_dir, records)
    completed = _run_generator(report_dir, 1)
    assert completed.returncode != 0
    assert "junit_from_jsonl: ERROR:" in completed.stderr
    assert not (report_dir / "junit-tilelang.xml").exists()


@pytest.mark.parametrize("recording_fails", [False, True])
def test_run_sh_timeout_retry_and_recovery_are_end_to_end(tmp_path, recording_fails):
    project = tmp_path / "project"
    ci_dir = project / "ci"
    ci_dir.mkdir(parents=True)
    shutil.copy2(CI_RUN, ci_dir / "run.sh")
    shutil.copy2(JUNIT_GENERATOR, ci_dir / "junit_from_jsonl.py")
    library = CI_LIB.read_text() + "\nci_check_card_state () { :; }\nci_configure_ptcc () { :; }\n"
    library += 'ci_reset_gpu_on_timeout () { ci_record_device_recovery "$1" "$2" "$3" 6 PASS 0 ""; }\n'
    if recording_fails:
        library += "ci_record_device_recovery () { return 19; }\n"
    (ci_dir / "lib.sh").write_text(library, encoding="utf-8")
    case_path = project / "case.py"
    case_path.write_text(
        "from pathlib import Path\nimport time\n"
        "state = Path(__file__).with_suffix('.state')\n"
        "if not state.exists():\n    state.write_text('first attempt')\n    time.sleep(5)\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    report_dir = tmp_path / "reports"
    environment.update(
        TMPDIR=str(tmp_path),
        CI_FAILURE_REPORT_DIR=str(report_dir),
        TILELANG_CACHE_DIR=str(tmp_path / "cache"),
        CASE_TIMEOUT="1",
        CASE_REPEAT="2",
        TILELANG_CI_RESET_MODE="disabled",
        TANG_VISIBLE_DEVICES="6",
    )
    completed = subprocess.run(
        ["bash", str(ci_dir / "run.sh"), "--case", f"python {case_path}"],
        cwd=project,
        env=environment,
        text=True,
        capture_output=True,
        timeout=20,
    )
    records = [json.loads(line) for line in (report_dir / "tilelang.jsonl").read_text().splitlines()]
    assert records[0]["status"] == "TIMEOUT"
    assert records[-1]["status"] == ("TIMEOUT" if recording_fails else "PASS")
    assert records[-1]["timeout_seconds"] == 1
    xml_path = report_dir / "junit-tilelang.xml"
    if recording_fails:
        assert completed.returncode == 1, completed.stdout + completed.stderr
        assert not xml_path.exists()
        assert "device recovery failed" in completed.stderr
    else:
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert [record["record_kind"] for record in records] == ["case_attempt", "device_recovery", "case_attempt", "case_result"]
        assert records[1]["status"] == "PASS"
        testcase = ElementTree.parse(xml_path).getroot().find("testsuite/testcase")
        assert _properties(testcase)["attempt_statuses"] == "TIMEOUT,PASS"
        assert _properties(testcase)["device_recovery_count"] == "1"


@pytest.mark.parametrize("recovery_mode", ["disabled", "failed"])
def test_failed_recovery_stops_execution_and_reports_unrun_cases(tmp_path, recovery_mode):
    plan = tmp_path / "cases.txt"
    plan.write_text("python first.py\npython second.py\n")
    report_dir = tmp_path / "reports"
    launches = tmp_path / "launches"
    script = f"""
source {shlex.quote(str(CI_LIB))}
ci_check_card_state () {{ :; }}
ci_run_timed () {{
    echo invoked >> {shlex.quote(str(launches))}
    echo timeout > "$2"
    CI_LAST_RUN_TIMED_OUT=1
    return 124
}}
"""
    if recovery_mode == "failed":
        script += 'ci_reset_gpu_on_timeout () { ci_record_device_recovery "$1" "$2" "$3" 0 FAIL 1 "reset failed"; return 1; }\n'
    script += f"ci_run_test_list {shlex.quote(str(plan))} command tilelang\n"
    env = {
        **os.environ,
        "TMPDIR": str(tmp_path),
        "CI_FAILURE_REPORT_DIR": str(report_dir),
        "TILELANG_CACHE_DIR": str(tmp_path / "cache"),
        "CASE_REPEAT": "3",
        "CASE_TIMEOUT": "1",
        "TILELANG_CI_RESET_MODE": "disabled",
        "TANG_VISIBLE_DEVICES": "0",
        "TEST_MARKER": "",
    }
    completed = subprocess.run(["bash", "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert launches.read_text().splitlines() == ["invoked"]
    records = [json.loads(line) for line in (report_dir / "tilelang.jsonl").read_text().splitlines()]
    assert [r["record_kind"] for r in records] == ["case_attempt", "device_recovery", "case_result", "case_result"]
    assert records[2]["status"] == "TIMEOUT"
    assert records[3]["status"] == "NOT_RUN"
    generated = _run_generator(report_dir, 2)
    assert generated.returncode == 0, generated.stderr
    suite = ElementTree.parse(report_dir / "junit-tilelang.xml").getroot().find("testsuite")
    assert suite.attrib["failures"] == "2"
    assert suite.attrib["skipped"] == "0"
    unrun = suite.findall("testcase")[1]
    assert unrun.find("failure").attrib["type"] == "NOT_RUN"
    assert _properties(unrun)["attempts_used"] == "0"


def test_not_run_requires_failed_recovery_evidence(tmp_path):
    passed = _attempt("first.py", "PASS", 1)
    unrun = _result(_attempt("second.py", "NOT_RUN", 0, exit_code=1, failure_reason="device unavailable"))
    report_dir = tmp_path / "reports"
    _write_jsonl(report_dir, [passed, _result(passed), unrun])
    completed = _run_generator(report_dir, 2)
    assert completed.returncode != 0
    assert not (report_dir / "junit-tilelang.xml").exists()
