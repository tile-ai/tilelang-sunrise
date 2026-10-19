import json
import os
import re
import shlex
import sys
import subprocess
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
CI_LIST = ROOT / "ci_test_case_list_tilelang.txt"
S3_LIST = ROOT / "ci_test_case_list_tilelang_s3.txt"


def test_ci_cache_directory_is_exported_to_child_processes(tmp_path):
    shell = f"""source {shlex.quote(str(ROOT / "ci/lib.sh"))}
unset TILELANG_CACHE_DIR
ci_init_state {shlex.quote(str(tmp_path))} test
trap - EXIT TERM INT
{shlex.quote(sys.executable)} -c 'import os; print("CHILD_CACHE=" + os.environ["TILELANG_CACHE_DIR"])'
"""
    result = subprocess.run(["bash", "-c", shell], check=True, text=True, capture_output=True)
    child_cache = next(line.removeprefix("CHILD_CACHE=") for line in result.stdout.splitlines() if line.startswith("CHILD_CACHE="))
    assert Path(child_cache).parent.parent == tmp_path / ".ci-state"
    assert Path(child_cache).name == "tilelang-cache"


@pytest.fixture
def ptcc_resolver(monkeypatch):
    spec = importlib.util.spec_from_file_location("ptcc_resolver", ROOT / "tilelang" / "_ptcc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("PTCC_PATH", "PTCC_JIT_PROFILE", "TANG_HOME", "TANG_PATH", "TANGRT_PATH"):
        monkeypatch.delenv(name, raising=False)
    return module


@pytest.mark.parametrize("profile", ["llvm20", "llvm22"])
def test_ptcc_jit_profile_defaults(ptcc_resolver, monkeypatch, profile):
    monkeypatch.setenv("PTCC_JIT_PROFILE", profile)
    options = ptcc_resolver.default_jit_options()
    assert "--tang-gpu-arch=stcu" in options
    assert "--tang-gpu-arch=stcuv2" not in options
    assert ("-fstpu-warp-alu" in options) == (profile == "llvm20")
    assert ("-fno-stpu-warp-alu" in options) == (profile == "llvm22")
    monkeypatch.setitem(ptcc_resolver._JIT_OPTIMIZATION_FLAGS, profile, ("-O1",))
    changed = ptcc_resolver.default_jit_options()
    assert "-O1" in changed and "-O3" not in changed
    monkeypatch.setenv("PTCC_JIT_PROFILE", "invalid")
    with pytest.raises(ValueError, match="PTCC_JIT_PROFILE"):
        ptcc_resolver.default_jit_options()


@pytest.mark.parametrize("layout", ["bin/ptcc", "toolchains/llvm/prebuilt/linux-x86_64/bin/ptcc"])
def test_ptcc_precedence(ptcc_resolver, monkeypatch, tmp_path, layout):
    toolkit = tmp_path / "tool kit"
    fallback = toolkit / layout
    fallback.parent.mkdir(parents=True)
    fallback.write_text("#!/bin/sh\necho ptcc-test\n")
    fallback.chmod(0o755)
    monkeypatch.setenv("TANG_HOME", str(toolkit))
    monkeypatch.setattr(ptcc_resolver.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(ptcc_resolver.shutil, "which", lambda _: None)
    assert ptcc_resolver.resolve_ptcc(str(toolkit)) == str(fallback)
    explicit = tmp_path / "custom ptcc"
    explicit.write_text("#!/bin/sh\necho custom-ptcc\n")
    explicit.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PTCC_PATH", "custom ptcc")
    monkeypatch.setattr(ptcc_resolver.shutil, "which", lambda _: str(fallback))
    assert ptcc_resolver.resolve_ptcc(str(toolkit)) == str(explicit)
    monkeypatch.setenv("PTCC_PATH", "")
    assert ptcc_resolver.resolve_ptcc(str(toolkit)) == str(fallback)
    monkeypatch.setattr(ptcc_resolver.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError):
        ptcc_resolver.resolve_ptcc("")


def test_selected_ptcc_is_shared_with_build(ptcc_resolver, monkeypatch, tmp_path):
    compiler = tmp_path / "custom ptcc"
    compiler.write_text("#!/bin/sh\necho test-version\n")
    compiler.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PTCC_PATH", "custom ptcc")
    monkeypatch.setenv("SKBUILD_CMAKE_ARGS", "-DEXISTING_OPTION=ON")
    assert ptcc_resolver.resolve_ptcc("") == str(compiler)
    shell = f"""source {shlex.quote(str(ROOT / "ci/lib.sh"))}
ci_configure_ptcc || exit
ci_set_tilelang_build_env "$PWD"
cd /
printf 'SELECTED=%s\\n' "$PTCC_PATH"
printf 'WHEEL_ARGS=%s\\n' "$SKBUILD_CMAKE_ARGS"
"""
    result = subprocess.run(["bash", "-c", shell], check=True, text=True, capture_output=True)
    assert f"SELECTED={compiler}" in result.stdout
    wheel_args = next(line.removeprefix("WHEEL_ARGS=") for line in result.stdout.splitlines() if line.startswith("WHEEL_ARGS="))
    assert {
        "-DEXISTING_OPTION=ON",
        f"-DCMAKE_TANG_COMPILER={compiler}",
        "-DCMAKE_TANG_FLAGS=--tang-gpu-arch=stcu",
    } <= set(wheel_args.split(";"))


@pytest.mark.parametrize("invalid", ["missing", "directory", "non-executable"])
def test_ptcc_invalid_override_never_falls_back(ptcc_resolver, monkeypatch, tmp_path, invalid):
    candidate = tmp_path / invalid
    if invalid == "directory":
        candidate.mkdir()
    elif invalid == "non-executable":
        candidate.write_text("compiler")
    monkeypatch.setenv("PTCC_PATH", str(candidate))
    monkeypatch.setattr(ptcc_resolver.shutil, "which", lambda _: "/valid/system/ptcc")
    with pytest.raises(RuntimeError, match="PTCC_PATH"):
        ptcc_resolver.resolve_ptcc("")
    if invalid == "missing":
        result = subprocess.run(["bash", "-c", f"source {shlex.quote(str(ROOT / 'ci/lib.sh'))}; ci_configure_ptcc"], capture_output=True)
        assert result.returncode != 0


def test_ptcc_identity_changes_in_new_process(ptcc_resolver, tmp_path):
    compiler = tmp_path / "ptcc"
    compiler.write_bytes(b"version-one")
    first = ptcc_resolver.compiler_identity(str(compiler))
    assert ptcc_resolver.compiler_identity(str(compiler)) == first
    alternate = tmp_path / "other-ptcc"
    alternate.write_bytes(b"version-one")
    assert ptcc_resolver.compiler_identity(str(alternate)) != first
    compiler.write_bytes(b"version-two")
    script = "import runpy,json,sys; m=runpy.run_path(sys.argv[1]); print(json.dumps(m['compiler_identity'](sys.argv[2])))"
    result = subprocess.run(
        [sys.executable, "-c", script, str(ROOT / "tilelang/_ptcc.py"), str(compiler)],
        check=True,
        text=True,
        capture_output=True,
    )
    assert tuple(json.loads(result.stdout)) != first


def test_ptcc_dry_run_does_not_require_compiler(monkeypatch, tmp_path):
    monkeypatch.setenv("PTCC_PATH", str(tmp_path / "missing"))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    result = subprocess.run(
        ["bash", str(ROOT / "ci/run.sh"), "--dry-run", "--case", "examples/elementwise/test_example_elementwise.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Dry-run plan" in result.stdout


def _enabled_lines(path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]


def _pytest_paths():
    return [line for line in _enabled_lines(CI_LIST) if not line.startswith("python ")]


def _direct_commands():
    return [line for line in _enabled_lines(CI_LIST) if line.startswith("python ")]


def test_ci_shell_scripts_have_valid_syntax():
    scripts = [
        ROOT / "ci" / "build.sh",
        ROOT / "ci" / "github_actions.sh",
        ROOT / "ci" / "github_runner" / "manage.sh",
        ROOT / "ci" / "github_runner" / "preflight.sh",
        ROOT / "ci" / "lib.sh",
        ROOT / "ci" / "lint.sh",
        ROOT / "ci" / "run.sh",
        ROOT / "ci" / "test.sh",
        ROOT / "ci" / "s3" / "test_s3.sh",
    ]
    subprocess.run(["bash", "-n", *map(str, scripts)], check=True)


def test_every_enabled_ci_command_references_an_existing_file():
    missing = [case_path for case_path in _pytest_paths() if not (ROOT / case_path).is_file()]
    for line in _direct_commands():
        argv = shlex.split(line)
        case_path = argv[1]
        if not (ROOT / case_path).is_file():
            missing.append(case_path)
    missing += [case_path for case_path in _s3_list_paths() if not (ROOT / case_path).is_file()]
    assert not missing


_S3_ENTRY_RE = re.compile(r"^(?P<path>[^\s:]+\.py)(?:::(?P<nodeid>\S+))?(?:\s+-m\s+(?P<marker>[A-Za-z_][A-Za-z0-9_]*))?$")


def _s3_list_entries():
    """(path, nodeid_or_None, marker_or_None) for every enabled S3 list entry.

    Grammar: ``<path>[::<nodeid>][ -m <marker>]``. The trailing ``-m`` modifier
    narrows the entry to the tests carrying that marker, which is how a file
    whose arch-agnostic cases are not triaged for S3 yet is enlisted while only
    its S3-only cases run (docs/s3_list_s3_only_selection_design.md).
    """
    entries = []
    for line in _enabled_lines(S3_LIST):
        match = _S3_ENTRY_RE.match(line)
        assert match is not None, f"unparsable S3 list entry: {line!r}"
        entries.append((match["path"], match["nodeid"], match["marker"]))
    return entries


def _s3_list_paths():
    """Path part of the enabled S3 list entries (bare paths, like the main list)."""
    return [path for path, _, _ in _s3_list_entries()]


def test_conda_shell_bootstraps_from_home_without_user_startup(tmp_path):
    conda_base = tmp_path / "home" / "miniconda3"
    conda_exe = conda_base / "bin" / "conda"
    conda_sh = conda_base / "etc" / "profile.d" / "conda.sh"
    conda_exe.parent.mkdir(parents=True)
    conda_sh.parent.mkdir(parents=True)
    conda_exe.write_text(
        f'#!/bin/sh\nif [ "$1" = info ] && [ "$2" = --base ]; then\n  printf \'%s\\n\' {shlex.quote(str(conda_base))}\n  exit 0\nfi\nexit 2\n',
        encoding="utf-8",
    )
    conda_sh.write_text("conda() { :; }\n", encoding="utf-8")
    conda_exe.chmod(0o755)
    (tmp_path / "home" / ".bashrc").write_text("exit 97\n", encoding="utf-8")

    shell = f"""
set -e
source {shlex.quote(str(ROOT / "ci" / "lib.sh"))}
ci_ensure_conda_shell
type -t conda
"""
    environment = os.environ.copy()
    environment.update(HOME=str(tmp_path / "home"), PATH="/usr/bin:/bin")
    environment.pop("CONDA_EXE", None)
    environment.pop("CONDA_PREFIX", None)

    completed = subprocess.run(
        ["/bin/bash", "-c", shell],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    assert completed.stdout.strip() == "function"


def test_wheel_runtime_environment_drops_source_checkout_overrides(tmp_path):
    shell = f"""
set -e
source {shlex.quote(str(ROOT / "ci" / "lib.sh"))}
TILELANG_HOME=/current/checkout
ci_prepare_installed_wheel_env /current/checkout/ci
env
"""
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH="/other/checkout",
        TILELANG_HOME="/other/checkout",
        TVM_HOME="/other/tvm",
        TVM_PREBUILD_PATH="/other/tvm/build",
        TVM_SOURCE_DIR="/other/tvm",
        TVM_LIBRARY_PATH="/other/tvm/lib",
        TL_TEMPLATE_PATH="/other/checkout/src/tl_templates",
        TILELANG_TEST_INSTALLED_WHEEL="1",
        TILELANG_WHEEL_PREFIX="/other/prefix",
    )

    completed = subprocess.run(
        ["bash", "-c", shell],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    child_env = dict(line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line)
    assert child_env["PYTHONPATH"] == "/current/checkout/ci"
    assert child_env["PYTHONNOUSERSITE"] == "1"
    for name in (
        "TILELANG_HOME",
        "TVM_HOME",
        "TVM_PREBUILD_PATH",
        "TVM_SOURCE_DIR",
        "TVM_LIBRARY_PATH",
        "TL_TEMPLATE_PATH",
        "TILELANG_TEST_INSTALLED_WHEEL",
        "TILELANG_WHEEL_PREFIX",
    ):
        assert name not in child_env


def test_cmake_resolver_skips_a_broken_configured_wrapper(tmp_path):
    broken = tmp_path / "cmake"
    fallback = tmp_path / "cmake3"
    broken.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    fallback.write_text("#!/bin/sh\nprintf 'cmake version test\\n'\n", encoding="utf-8")
    broken.chmod(0o755)
    fallback.chmod(0o755)

    shell = f"""
set -e
source {shlex.quote(str(ROOT / "ci" / "lib.sh"))}
CMAKE_ROOT={shlex.quote(str(broken))}
ci_resolve_cmake
"""
    environment = os.environ.copy()
    environment["PATH"] = str(tmp_path)

    completed = subprocess.run(
        ["/bin/bash", "-c", shell],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    assert completed.stdout.strip() == str(fallback)


def test_ci_records_every_retry_attempt(tmp_path):
    case_list = tmp_path / "cases.txt"
    case_list.write_text("python fake.py\n", encoding="utf-8")
    report_dir = tmp_path / "reports"
    script = f"""
source {shlex.quote(str(ROOT / "ci" / "lib.sh"))}
ci_assert_runtime_stack () {{ :; }}
ci_check_card_state () {{ :; }}
stub_attempt=0
ci_run_timed () {{
    stub_attempt=$((stub_attempt + 1))
    printf 'stub attempt %s\n' "$stub_attempt" > "$2"
    [[ $stub_attempt -gt 1 ]]
}}
ci_run_test_list {shlex.quote(str(case_list))} command retry_suite
"""
    env = {
        **os.environ,
        "CASE_REPEAT": "2",
        "CASE_TIMEOUT": "900",
        "CI_FAILURE_REPORT_DIR": str(report_dir),
        "TILELANG_CACHE_DIR": str(tmp_path / "cache"),
        "TMPDIR": str(tmp_path),
        "TEST_MARKER": "",
    }
    subprocess.run(["bash", "-c", script], cwd=ROOT, env=env, check=True)

    records = [json.loads(line) for line in (report_dir / "retry_suite.jsonl").read_text().splitlines()]
    assert [record["record_kind"] for record in records] == ["case_attempt", "case_attempt", "case_result"]
    assert records[0]["status"] == "FAIL"
    assert records[0]["attempt"] == 1
    assert records[0]["total_attempts"] == 2
    assert "stub attempt 1" in records[0]["log_tail"]
    assert records[1]["status"] == "PASS"
    assert records[1]["attempt"] == 2
    assert records[1]["total_attempts"] == 2
    assert records[2]["status"] == "PASS"


def test_ci_honors_audited_per_case_timeout_mapping(tmp_path):
    case_list = tmp_path / "cases.txt"
    case_list.write_text("python fake.py\n", encoding="utf-8")
    timeout_file = tmp_path / "cases_timeouts.tsv"
    timeout_file.write_text("fake.py\t3600\n", encoding="utf-8")
    report_dir = tmp_path / "reports"
    captured_timeout = tmp_path / "captured-timeout.txt"
    script = f"""
source {shlex.quote(str(ROOT / "ci" / "lib.sh"))}
ci_assert_runtime_stack () {{ :; }}
ci_check_card_state () {{ :; }}
ci_run_timed () {{
    printf '%s\n' "$1" > "$CAPTURED_TIMEOUT"
    : > "$2"
}}
ci_run_test_list {shlex.quote(str(case_list))} command timeout_override_suite
"""
    env = {
        **os.environ,
        "CASE_REPEAT": "1",
        "CASE_TIMEOUT": "900",
        "CAPTURED_TIMEOUT": str(captured_timeout),
        "CI_FAILURE_REPORT_DIR": str(report_dir),
        "TILELANG_CACHE_DIR": str(tmp_path / "cache"),
        "TMPDIR": str(tmp_path),
        "TEST_MARKER": "",
    }
    subprocess.run(["bash", "-c", script], cwd=ROOT, env=env, check=True)

    assert captured_timeout.read_text(encoding="utf-8").strip() == "3600"
    records = [json.loads(line) for line in (report_dir / "timeout_override_suite.jsonl").read_text().splitlines()]
    assert [record["timeout_seconds"] for record in records] == [3600, 3600]

    timeout_file.write_text("stale.py\t3600\n", encoding="utf-8")
    rejected = subprocess.run(["bash", "-c", script], cwd=ROOT, env=env, text=True, capture_output=True, check=False)
    assert rejected.returncode == 1
    assert "Timeout mapping does not match a case" in rejected.stderr


def test_ci_treats_nonzero_exit_at_wall_clock_limit_as_timeout(tmp_path):
    case_list = tmp_path / "cases.txt"
    case_list.write_text("python fake.py\n", encoding="utf-8")
    report_dir = tmp_path / "reports"
    script = f"""
source {shlex.quote(str(ROOT / "ci" / "lib.sh"))}
ci_result_is_timeout 1 900 900
! ci_result_is_timeout 1 899 900
ci_assert_runtime_stack () {{ :; }}
ci_check_card_state () {{ :; }}
ci_run_timed () {{
    printf 'terminated at deadline\n' > "$2"
    CI_LAST_RUN_TIMED_OUT=1
    return 143
}}
if ci_run_test_list {shlex.quote(str(case_list))} command timeout_suite; then
    exit 99
fi
"""
    env = {
        **os.environ,
        "CASE_REPEAT": "1",
        "CASE_TIMEOUT": "900",
        "CI_FAILURE_REPORT_DIR": str(report_dir),
        "TILELANG_CACHE_DIR": str(tmp_path / "cache"),
        "TILELANG_CI_RESET_MODE": "disabled",
        "TANG_VISIBLE_DEVICES": "0",
        "TMPDIR": str(tmp_path),
        "TEST_MARKER": "",
    }
    subprocess.run(["bash", "-c", script], cwd=ROOT, env=env, check=True)

    records = [json.loads(line) for line in (report_dir / "timeout_suite.jsonl").read_text().splitlines()]
    assert [record["record_kind"] for record in records] == ["case_attempt", "device_recovery", "case_result"]
    assert records[0]["status"] == "TIMEOUT"
    assert records[0]["exit_code"] == 143
    assert records[0]["failure_reason"] == "timed out after 900s (exit 143)"
    assert records[1]["action"] == "reset"
    assert records[1]["status"] == "SKIPPED"
    assert records[1]["reason"] == "reset mode is disabled"
    assert records[2]["status"] == "TIMEOUT"


def test_github_adapter_accepts_only_same_repository_push_or_dispatch(tmp_path):
    adapter = ROOT / "ci" / "github_actions.sh"
    event_path = tmp_path / "event.json"
    github_env = tmp_path / "github-env"
    event_path.write_text(
        json.dumps(
            {
                "before": "1" * 40,
                "repository": {"full_name": "tile-ai/tilelang-sunrise", "default_branch": "main"},
            }
        ),
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "GITHUB_EVENT_PATH": str(event_path),
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REPOSITORY": "tile-ai/tilelang-sunrise",
        "GITHUB_REPOSITORY_ID": "12345",
        "GITHUB_RUN_ID": "67890",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_WORKSPACE": str(ROOT),
        "GITHUB_SHA": "2" * 40,
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_ENV": str(github_env),
    }
    subprocess.run(["bash", str(adapter), "trust"], cwd=ROOT, env=env, check=True)
    subprocess.run(["bash", str(adapter), "export", "tilelang_build"], cwd=ROOT, env=env, check=True)
    exported = dict(line.split("=", 1) for line in github_env.read_text(encoding="utf-8").splitlines())
    assert exported["CI_COMMIT_BEFORE_SHA"] == "1" * 40
    assert exported["CI_COMMIT_SHA"] == "2" * 40
    assert exported["CI_JOB_NAME"] == "tilelang_build"
    assert exported["TANG_VISIBLE_DEVICES"] == "0"
    assert exported["TILELANG_CI_RESET_MODE"] == "sudo-n"
    assert exported["TILELANG_CI_PUBLIC_LOGS"] == "1"

    env["GITHUB_EVENT_NAME"] = "pull_request"
    rejected = subprocess.run(["bash", str(adapter), "trust"], cwd=ROOT, env=env, text=True, capture_output=True)
    assert rejected.returncode != 0
    assert "refusing untrusted GitHub event" in rejected.stderr
