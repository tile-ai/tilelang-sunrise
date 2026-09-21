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


def _pytest_inventory_paths():
    paths = []
    for raw_line in CI_LIST.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("# "):
            line = line[2:].strip()
        if line.endswith(".py") and not line.startswith("python "):
            paths.append(line)
    return paths


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
        ROOT / "ci" / "s3" / "lib_s3.sh",
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
    assert not missing


def test_pytest_inventory_exactly_matches_current_test_universe():
    expected = sorted(
        str(path.relative_to(ROOT))
        for base in (ROOT / "testing" / "python", ROOT / "examples")
        for path in base.rglob("test_*.py")
        if ROOT / "testing" / "python" / "s3" not in path.parents
    )
    assert _pytest_inventory_paths() == expected


def test_s3_ci_list_exactly_matches_preserved_s3_tests():
    expected = sorted(str(path.relative_to(ROOT)) for path in (ROOT / "testing" / "python" / "s3").glob("test_*.py"))
    commands = _enabled_lines(S3_LIST)
    assert all(command.startswith("python ") for command in commands)
    assert sorted(shlex.split(command)[1] for command in commands) == expected


def test_direct_scripts_use_the_supported_python_command_mode():
    commands = _direct_commands()
    assert commands
    assert all(command.startswith("python ") for command in commands)
    assert not (ROOT / "ci_test_case_list_tilelang_scripts.txt").exists()


def test_ci_records_exact_pytest_nodeids_and_preserves_all_suite_reports():
    # The CI test path is now ci/run.sh --ci; ci/test.sh is a thin shim that
    # delegates to it, so the audit-plugin env that used to live in ci/test.sh
    # moved into ci/run.sh verbatim.
    run_script = (ROOT / "ci" / "run.sh").read_text(encoding="utf-8")
    test_shim = (ROOT / "ci" / "test.sh").read_text(encoding="utf-8")
    library = (ROOT / "ci" / "lib.sh").read_text(encoding="utf-8")
    preflight = (ROOT / "ci" / "github_runner" / "preflight.sh").read_text(encoding="utf-8")
    assert (ROOT / "ci" / "pytest_node_audit.py").is_file()
    # ci/test.sh still routes to the CI run path (run.sh --ci on the tilelang list)
    # so its existing callers keep the same behavior.
    assert 'run.sh" --ci --list' in test_shim
    assert "ci_test_case_list_tilelang.txt" in test_shim
    # Exact pytest nodeids: the node-audit plugin + its output path are wired on
    # the CI test path (moved from ci/test.sh to ci/run.sh --ci).
    assert 'PYTEST_AUDIT_PLUGIN_ARGS="-p pytest_node_audit"' in run_script
    assert 'PYTEST_NODE_AUDIT_PATH="$(ci_failure_report_dir)/tilelang_nodes.jsonl"' in run_script
    assert 'PYTEST_LAUNCH_CWD="$CI_TMP_DIR"' in run_script
    assert 'PYTEST_NODE_AUDIT_TEST_CWD="$TILELANG_HOME"' in run_script
    assert 'PYTEST_NODE_AUDIT_SOURCE_ROOT="$TILELANG_HOME"' in run_script
    assert "TILELANG_TEST_INSTALLED_WHEEL=1" in run_script
    assert 'TILELANG_DEFAULT_TARGET="${TILELANG_DEFAULT_TARGET:-tang}"' in run_script
    assert "--import-mode=importlib" in library
    assert "case_modes" in library
    assert 'current_mode="${case_modes[$i]}"' in library
    assert "ci_prepare_failure_reports ()" in library
    assert "CI_FAILURE_REPORTS_INITIALIZED=1" in library
    assert run_script.count("ci_run_test_list") == 1
    assert 'CI_FAILURE_REPORT_ROOT="$evidence_dir/ci_failure_reports"' in preflight
    assert 'CI_FAILURE_REPORT_DIR="$CI_FAILURE_REPORT_ROOT/$name"' in preflight


def test_wheel_ci_disables_testing_conftest_source_shadowing():
    conftest = (ROOT / "testing" / "conftest.py").read_text(encoding="utf-8")
    assert 'os.environ.get("TILELANG_TEST_INSTALLED_WHEEL") == "1"' in conftest
    assert "sys.path[:]" in conftest
    assert 'SOURCE_PACKAGE_ROOT = os.path.join(REPO_ROOT, "tilelang")' in conftest
    assert "resolved TileLang from the checkout" in conftest
    assert 'os.environ.get("TILELANG_WHEEL_PREFIX")' in conftest
    assert "outside the isolated environment" in conftest


def test_ci_entry_points_do_not_source_user_shell_startup():
    for relative_path in ("ci/build.sh", "ci/install.sh", "ci/run.sh", "ci/test.sh"):
        script = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "source ~/.bashrc" not in script


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


def _require_snippets(path: Path, *snippets: str) -> None:
    """Assert snippets exist without putting the whole file into the assertion (avoids huge diffs)."""
    text = path.read_text(encoding="utf-8")
    missing = [s for s in snippets if s not in text]
    assert not missing, f"{path} missing snippets:\n" + "\n".join(f"  - {s!r}" for s in missing)


def _forbid_snippets(path: Path, *snippets: str) -> None:
    text = path.read_text(encoding="utf-8")
    present = [s for s in snippets if s in text]
    assert not present, f"{path} must not contain:\n" + "\n".join(f"  - {s!r}" for s in present)


def test_ci_uses_isolated_temp_state_and_platform_scoped_timeout_reset():
    lib_path = ROOT / "ci" / "lib.sh"
    _forbid_snippets(lib_path, "torch-c-dlpack-ext", "~/.cache/pip", "passwd=")
    _require_snippets(
        lib_path,
        'TILELANG_CI_RESET_MODE="${TILELANG_CI_RESET_MODE:-password}"',
        'TILELANG_CI_PUBLIC_LOGS="${TILELANG_CI_PUBLIC_LOGS:-0}"',
        '[[ -z "${SUDO_MAGICWORD:-}" ]]',
        "printf '%s\\n' \"$SUDO_MAGICWORD\" | sudo -S -p ''",
        "/usr/bin/sudo -n /usr/bin/pt_smi -r -i 0",
        "details redacted for public CI",
        'TILELANG_CI_PUBLIC_LOGS:-0}" == "1"',
        "sunrise_device_summary.json",
        'CI_STATE_ROOT="${base}/.ci-state/',
        'export TMPDIR="$CI_TMP_DIR"',
        "ci_on_job_cancel ()",
        "ci_check_card_state ()",
        "ci_save_device_logs ()",
        "ci_run_timed ()",
        'case_repeat="${CASE_REPEAT:-3}"',
    )


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


def test_preflight_cleans_generated_device_summary_before_git_cleanliness_gate():
    preflight = (ROOT / "ci" / "github_runner" / "preflight.sh").read_text(encoding="utf-8")
    assert 'rm -f "$SOURCE_DIR/sunrise_device_summary.json"' in preflight
    assert preflight.index('cleanup_source_device_summary "$validation_failed"') < preflight.index(
        'git status --short > "$evidence_dir/final_git_status.txt"'
    )


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


def test_tang_ci_treats_llvm_home_as_optional_and_verifies_the_wheel():
    library = (ROOT / "ci" / "lib.sh").read_text(encoding="utf-8")
    build = (ROOT / "ci" / "install.sh").read_text(encoding="utf-8")
    assert "${LLVM_HOME:?" not in library
    assert 'if [[ -n "$LLVM_HOME" ]]' in library
    assert "ci_assert_tilelang_tang_registration ()" in library
    assert 'tvm.get_global_func("target.build.tilelang_tang", allow_missing=True)' in library
    assert "ci_assert_installed_tilelang_wheel" in build


def test_ci_builds_only_from_vendored_dependencies():
    lib = (ROOT / "ci" / "lib.sh").read_text(encoding="utf-8")
    assert "ci_update_pinned_submodule" not in lib
    assert "submodule update" not in lib
    assert "gitlab." + "sunrise-ai.com" not in lib
    for relative_path in (
        "3rdparty/tvm_sunrise/LICENSE",
        "3rdparty/tvm_sunrise/NOTICE",
        "3rdparty/tvm_sunrise/3rdparty/tvm-ffi/LICENSE",
        "3rdparty/tvm_sunrise/3rdparty/tvm-ffi/NOTICE",
        "3rdparty/tvm_sunrise/3rdparty/tvm-ffi/3rdparty/dlpack/LICENSE",
        "3rdparty/tvm_sunrise/3rdparty/tvm-ffi/3rdparty/libbacktrace/LICENSE",
    ):
        assert (ROOT / relative_path).is_file()
    assert "-DTANG_DIR=${TANG_CMAKE_PACKAGE_DIR}" in lib
    assert "-DTANGRT_DIR=${TANGRT_CMAKE_PACKAGE_DIR}" in lib
    assert "-DUSE_CUDA=OFF" in lib
    assert "-DUSE_OPENCL=OFF" in lib
    assert "-DUSE_CUTLASS=OFF" in lib
    assert 'tang_cmake_prefix="${TANGRT_PATH%/}/targets/linux-x86_64"' in lib
    assert 'export CMAKE_PREFIX_PATH="${tang_cmake_prefix}:${conda_cmake_prefix}' in lib


def test_pipeline_preserves_project_jobs_and_manual_s3_entry():
    config = (ROOT / ".gitlab-ci.yml").read_text(encoding="utf-8")
    assert "ai_" + "mr_review" not in config
    assert "ai_" + "mr_failed_case_analysis" not in config
    assert "ai_" + "analysis" not in config
    assert "[ai-" + "review]" not in config
    assert "\ntilelang_build:" in config
    assert "\ntilelang_test:" in config
    assert "\ntileops_pipeline:" not in config
    assert "\ntilekernels_pipeline:" not in config
    assert "\nvalidate:" in config
    assert "ci/validate_operator.sh" in config
    assert "- OP: tileops_sunrise" in config
    assert "- OP: tilekernels_sunrise" in config
    assert "tilelang-puzzles" not in config
    assert "\ntilelang_test_s3:" in config
    s3_job = config.split("\ntilelang_test_s3:", 1)[1].split("\nvalidate:", 1)[0]
    assert "allow_failure: true" in s3_job
    assert "when: manual" in s3_job
    assert "TANG_S3_PTCC_PATH" in s3_job
    assert "ci_save_device_logs" in config
    assert "- dmesg.log" in config
    assert "- pt.log" in config
    for lint_job in ("lint_changed", "lint_all"):
        lint_config = config.split(f"\n{lint_job}:", 1)[1].split("\n\n", 1)[0]
        assert "tags: [runtime, pt200]" in lint_config


def test_tilelang_ci_owns_lint_and_downstream_validation_boundaries():
    root_precommit = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "exclude: '^(build|3rdparty|downstream)/.*$'" in root_precommit

    root_ci = (ROOT / ".gitlab-ci.yml").read_text(encoding="utf-8")
    tilelang_change_scope = root_ci.split(".tilelang_code_changes:", 1)[1].split(".ci_rules:", 1)[0]
    assert "3rdparty/" not in tilelang_change_scope
    assert "downstream/" not in tilelang_change_scope
    validate_script = ROOT / "ci" / "validate_operator.sh"
    assert validate_script.is_file()
    validate_text = validate_script.read_text(encoding="utf-8")
    assert "downstream/tileops_sunrise" in validate_text
    assert "downstream/tilekernels_sunrise" in validate_text
    assert "git clone" not in validate_text
    assert "gitlab." + "sunrise-ai.com" not in validate_text

    test_list = (ROOT / "ci_test_case_list_tilelang.txt").read_text(encoding="utf-8")
    assert "# testing/python/cache/test_tilelang_cuda_binary_cache.py" in test_list
    enabled_cases = {line.strip() for line in test_list.splitlines() if not line.lstrip().startswith("#")}
    assert "testing/python/cache/test_tilelang_cuda_binary_cache.py" not in enabled_cases


def test_root_lint_supports_public_python39_runner_without_shell_profile():
    root_precommit = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    lint_script = (ROOT / "ci" / "lint.sh").read_text(encoding="utf-8")
    assert "rev: v0.9.29" in root_precommit
    assert "ensure_py39" in lint_script
    assert "sys.version_info[:2]>=(3,9)" in lint_script
    assert "source ~/.bashrc" not in lint_script


def test_ci_conda_activation_is_initialized_for_nested_shells():
    ci_lib = (ROOT / "ci" / "lib.sh").read_text(encoding="utf-8")
    assert "if command -v conda" in ci_lib
    assert 'source "$HOME/.bashrc"' in ci_lib
    assert 'conda_base="$("$conda_exe" info --base)"' in ci_lib
    assert 'source "$conda_init"' in ci_lib

    for relative_path in ("ci/build.sh", "ci/test.sh", "ci/validate_operator.sh"):
        script = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "source ~/.bashrc" not in script


def test_cython_version_matches_the_cp39_limited_api_contract():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    ci_lib = (ROOT / "ci" / "lib.sh").read_text(encoding="utf-8")
    assert 'wheel.py-api = "cp39"' in pyproject
    assert '"cython>=3.1.0,<3.3"' in pyproject
    assert 'conda install numpy psutil "cython>=3.1.0,<3.3" pytest -y' in ci_lib


def test_release_tree_has_no_submodule_metadata_or_internal_repositories():
    assert not list(ROOT.rglob(".gitmodules"))
    assert not (ROOT / "downstream" / "tilelang-puzzles").exists()

    scanned_files = [ROOT / ".gitlab-ci.yml", *sorted((ROOT / "ci").rglob("*.sh"))]
    for path in scanned_files:
        text = path.read_text(encoding="utf-8")
        assert "gitlab." + "sunrise-ai.com" not in text
        assert "packaging." + "sunrise-ai.com" not in text


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


def test_github_workflows_are_pinned_public_safe_thin_adapters():
    workflow_dir = ROOT / ".github" / "workflows"
    workflows = sorted(workflow_dir.glob("*.yml"))
    assert [path.name for path in workflows] == ["sunrise-lint.yml", "sunrise-s2.yml"]
    assert sorted(path.name for path in (ROOT / ".github" / "workflows-archive").glob("*.yml")) == [
        "ci.yml",
        "dist.yml",
        "pr-regression-test-bot.yml",
        "pr-reminder-bot.yml",
        "publish-docs.yml",
    ]

    text = "\n".join(path.read_text(encoding="utf-8") for path in workflows)
    _forbid_snippets(
        workflow_dir / "sunrise-s2.yml",
        "pull_request",
        "pull_request_target",
        "issue_comment",
        "SUDO_MAGICWORD",
        "dmesg.log",
        "pt.log",
        "gitlab." + "sunrise-ai.com",
        "packaging." + "sunrise-ai.com",
    )
    assert "pull_request" not in (workflow_dir / "sunrise-lint.yml").read_text(encoding="utf-8")
    assert "permissions:\n  contents: read" in text
    assert '"!dependabot/**"' in text
    assert "cancel-in-progress: false" in (workflow_dir / "sunrise-s2.yml").read_text(encoding="utf-8")
    assert "[self-hosted, linux, x64, sunrise-s2, tilelang-sunrise]" in text

    action_uses = re.findall(r"^\s*uses:\s*(\S+)", text, flags=re.MULTILINE)
    assert action_uses
    assert all(re.fullmatch(r"actions/(checkout|upload-artifact|download-artifact)@[0-9a-f]{40}", use) for use in action_uses)

    for job_name, entrypoint in (
        ("lint_changed", "bash ci/lint.sh changed"),
        ("lint_all", "bash ci/lint.sh all"),
        ("tilelang_build", "bash ci/build.sh"),
        ("tilelang_test", "bash ci/test.sh"),
        ("tilekernels_sunrise", "bash ci/validate_operator.sh"),
        ("tileops_sunrise", "bash ci/validate_operator.sh"),
    ):
        assert f"  {job_name}:" in text
        assert entrypoint in text

    tilekernels_job = text.split("\n  tilekernels_sunrise:", 1)[1].split("\n  tileops_sunrise:", 1)[0]
    tileops_job = text.split("\n  tileops_sunrise:", 1)[1]
    assert "needs: [tilelang_build, tilelang_test]" in tilekernels_job
    assert "always() && needs.tilelang_build.result == 'success'" in tilekernels_job
    assert "needs.tilelang_test.result == 'failure'" in tilekernels_job
    assert "needs: [tilelang_build, tilelang_test, tilekernels_sunrise]" in tileops_job
    assert "always() && needs.tilelang_build.result == 'success'" in tileops_job
    assert "needs.tilelang_test.result == 'failure'" in tileops_job
    assert "needs.tilekernels_sunrise.result == 'failure'" in tileops_job

    gitlab = (ROOT / ".gitlab-ci.yml").read_text(encoding="utf-8")
    for entrypoint in ("ci/lint.sh changed", "ci/lint.sh all", "ci/build.sh", "ci/test.sh", "ci/validate_operator.sh"):
        assert entrypoint in gitlab
    assert "- OP: tileops_sunrise" in gitlab
    assert "- OP: tilekernels_sunrise" in gitlab
    assert (ROOT / "downstream" / "tileops_sunrise" / "ci_test_case_list_tileops_timeouts.tsv").read_text(encoding="utf-8").splitlines()[
        -1
    ] == "tests/ops/test_grouped_gemm.py\t2700"


def test_github_runner_provisioning_is_pinned_and_fail_closed():
    manager_path = ROOT / "ci" / "github_runner" / "manage.sh"
    preflight_path = ROOT / "ci" / "github_runner" / "preflight.sh"
    manager = manager_path.read_text(encoding="utf-8")
    preflight = preflight_path.read_text(encoding="utf-8")
    combined = manager + preflight
    _forbid_snippets(
        manager_path,
        "SUDO_MAGICWORD",
        "rm -rf",
        "gitlab." + "sunrise-ai.com",
        "packaging." + "sunrise-ai.com",
        "/tmp/",
        "/mnt/github-actions/",
    )
    _require_snippets(
        manager_path,
        'readonly BASE_DIR="/home/github-actions/tilelang-sunrise"',
        'download_checked "$RUNNER_URL" "$RUNNER_SHA256"',
        'download_checked "$MINIFORGE_URL" "$MINIFORGE_SHA256"',
        'echo "$TORCH_WHEEL_SHA256  $WHEEL_DIR/$TORCH_WHEEL_NAME" | sha256sum -c -',
        'echo "$TRITON_WHEEL_SHA256  $WHEEL_DIR/$TRITON_WHEEL_NAME" | sha256sum -c -',
        'echo "$PINNED_PTCC_SHA256  $path" | sha256sum -c -',
        "NOPASSWD: /usr/bin/pt_smi -r -i 0",
        "meta skuid ${runner_uid} ip daddr 127.0.0.1 tcp dport 3128 accept",
        "meta skuid ${runner_uid} reject",
        "IPAddressDeny=any",
        "IPAddressAllow=127.0.0.1",
        "TemporaryFileSystem=/mnt:ro",
        "BindPaths=$BASE_DIR",
        "BindReadOnlyPaths=$PINNED_PTCC_PATH:$SYSTEM_PTCC_PATH",
        "InaccessiblePaths=-/run/avahi-daemon -/run/cups -/run/dbus",
        "DeviceAllow=/dev/ptpu0 rw",
        'git -c safe.directory="$destination" -C "$destination"',
        "require_preflight_pass",
        "refresh-units",
        "rollback-list",
    )
    for pin_name in ("RUNNER_SHA256", "MINIFORGE_SHA256", "TORCH_WHEEL_SHA256", "TRITON_WHEEL_SHA256", "PINNED_PTCC_SHA256"):
        assert re.search(rf'readonly {pin_name}="[0-9a-f]{{64}}"', manager)

    register_runner = manager.split("register_runner ()", 1)[1].split("unregister_runner ()", 1)[0]
    start_runner = manager.split("start_runner ()", 1)[1].split("stop_runner ()", 1)[0]
    for activation_action in (register_runner, start_runner):
        assert activation_action.index("require_public_disclosure_approval") < activation_action.index("require_preflight_pass")
    assert register_runner.index("require_preflight_pass") < register_runner.index("registration token")

    disclosure_check = manager.split("public_disclosure_is_approved ()", 1)[1].split("require_public_disclosure_approval ()", 1)[0]
    assert '[[ -f "$PUBLIC_DISCLOSURE_APPROVAL_FILE" && ! -L "$PUBLIC_DISCLOSURE_APPROVAL_FILE" ]]' in disclosure_check
    assert '"0:0:600"' in disclosure_check
    assert '"$PUBLIC_DISCLOSURE_APPROVAL_VALUE"' in disclosure_check

    assert "ACTIONS_RUNNER_INPUT_TOKEN" in manager
    assert '/usr/bin/env -i "${RUNNER_ENVIRONMENT[@]}"' in manager
    assert "--preserve-environment" not in manager
    assert '--token "$registration_token"' not in manager
    assert '--token "$removal_token"' not in manager

    runner_unit = manager.split('write_file "/etc/systemd/system/$RUNNER_SERVICE"', 1)[1].split(
        'write_file "/etc/systemd/system/$PREFLIGHT_SERVICE"', 1
    )[0]
    preflight_unit = manager.split('write_file "/etc/systemd/system/$PREFLIGHT_SERVICE"', 1)[1].split("EOF\n}", 1)[0]
    assert "InaccessiblePaths=-/run/avahi-daemon -/run/cups -/run/dbus" in runner_unit
    assert "/run/systemd/userdb /media /srv /root /var/log/pt200 $EVIDENCE_DIR $SOURCE_ROOT" in runner_unit
    runner_writable = next(line for line in runner_unit.splitlines() if line.startswith("ReadWritePaths="))
    assert "$EVIDENCE_DIR" not in runner_writable
    assert "$SOURCE_ROOT" not in runner_writable
    assert "$RUNNER_DIR/run-helper.sh" in runner_writable
    assert 'local helper="$RUNNER_DIR/run-helper.sh"' in manager
    assert "for state_file in .env .path" in manager
    nnp_implying_settings = (
        "DynamicUser=",
        "LockPersonality=",
        "MemoryDenyWriteExecute=",
        "NoNewPrivileges=true",
        "PrivateDevices=",
        "ProtectClock=",
        "ProtectHostname=",
        "ProtectKernelLogs=",
        "ProtectKernelModules=",
        "ProtectKernelTunables=",
        "RestrictAddressFamilies=",
        "RestrictNamespaces=",
        "RestrictRealtime=",
        "RestrictSUIDSGID=",
        "SystemCallArchitectures=",
        "SystemCallFilter=",
        "SystemCallLog=",
    )
    for hardware_unit in (runner_unit, preflight_unit):
        assert "ConditionPathExists=$PINNED_PTCC_PATH" in hardware_unit
        assert "BindReadOnlyPaths=$PINNED_PTCC_PATH:$SYSTEM_PTCC_PATH" in hardware_unit
        assert "$TOOLCHAIN_DIR" in next(line for line in hardware_unit.splitlines() if line.startswith("ReadOnlyPaths="))
        for setting in nnp_implying_settings:
            assert setting not in hardware_unit
        for retained_setting in (
            "PrivateTmp=true",
            "ProtectSystem=strict",
            "ProtectHome=tmpfs",
            "ProtectControlGroups=true",
            "IPAddressDeny=any",
            "IPAddressAllow=127.0.0.1",
            "DevicePolicy=closed",
        ):
            assert retained_setting in hardware_unit

    _require_snippets(
        preflight_path,
        '(set -Eeuo pipefail; "$@")',
        'os.walk("/run")',
        "candidate.connect(path)",
        "host contract found reachable host service sockets",
        "--write-out '%{http_connect}'",
        '[[ "$no_new_privs" != "0" ]]',
        "expect_tcp_blocked 127.0.0.1 22",
        "expect_tcp_blocked 127.0.0.1 111",
        "expect_tcp_blocked 127.0.0.1 139",
        "expect_tcp_blocked 127.0.0.1 445",
        'success_path="$evidence_dir/PREFLIGHT_SUCCESS"',
        'printf \'SOURCE_SHA=%s\\n\' "$SOURCE_SHA" > "$success_path"',
    )
    for job in ("lint_changed", "lint_all", "tilelang_build", "tilelang_test", "tilekernels_sunrise", "tileops_sunrise"):
        assert f"run_job {job}" in preflight
    blocking_jobs = (
        "run_job tilelang_test bash ci/test.sh || validation_failed=1",
        "run_job tilekernels_sunrise run_tilekernels || validation_failed=1",
        "run_job tileops_sunrise run_tileops || validation_failed=1",
    )
    assert [preflight.index(command) for command in blocking_jobs] == sorted(preflight.index(command) for command in blocking_jobs)
    assert "if (( validation_failed != 0 )); then" in preflight
    assert "TANG_VISIBLE_DEVICES=0" in combined
    assert "TILELANG_CI_PUBLIC_LOGS=1" in combined
