import json
import os
import re
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CI_LIST = ROOT / "ci_test_case_list_tilelang.txt"
S3_LIST = ROOT / "ci_test_case_list_tilelang_s3.txt"


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
    test_script = (ROOT / "ci" / "test.sh").read_text(encoding="utf-8")
    library = (ROOT / "ci" / "lib.sh").read_text(encoding="utf-8")
    preflight = (ROOT / "ci" / "github_runner" / "preflight.sh").read_text(encoding="utf-8")
    assert (ROOT / "ci" / "pytest_node_audit.py").is_file()
    assert 'PYTEST_AUDIT_PLUGIN_ARGS="-p pytest_node_audit"' in test_script
    assert 'PYTEST_NODE_AUDIT_PATH="$(ci_failure_report_dir)/tilelang_nodes.jsonl"' in test_script
    assert 'PYTEST_LAUNCH_CWD="$CI_TMP_DIR"' in test_script
    assert 'PYTEST_NODE_AUDIT_TEST_CWD="$TILELANG_HOME"' in test_script
    assert 'PYTEST_NODE_AUDIT_SOURCE_ROOT="$TILELANG_HOME"' in test_script
    assert "TILELANG_TEST_INSTALLED_WHEEL=1" in test_script
    assert 'TILELANG_DEFAULT_TARGET="${TILELANG_DEFAULT_TARGET:-tang}"' in test_script
    assert "--import-mode=importlib" in library
    assert "case_modes" in library
    assert 'current_mode="${case_modes[$i]}"' in library
    assert "ci_prepare_failure_reports ()" in library
    assert "CI_FAILURE_REPORTS_INITIALIZED=1" in library
    assert test_script.count("ci_run_test_list") == 1
    assert 'CI_FAILURE_REPORT_ROOT="$evidence_dir/ci_failure_reports"' in preflight
    assert 'CI_FAILURE_REPORT_DIR="$CI_FAILURE_REPORT_ROOT/$name"' in preflight


def test_wheel_ci_disables_testing_conftest_source_shadowing():
    conftest = (ROOT / "testing" / "conftest.py").read_text(encoding="utf-8")
    assert 'os.environ.get("TILELANG_TEST_INSTALLED_WHEEL") == "1"' in conftest
    assert "sys.path[:]" in conftest
    assert 'SOURCE_PACKAGE_ROOT = os.path.join(REPO_ROOT, "tilelang")' in conftest
    assert "resolved TileLang from the checkout" in conftest


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
    assert records[0]["will_retry"] is True
    assert "stub attempt 1" in records[0]["log_tail"]
    assert records[1]["status"] == "PASS"
    assert records[1]["attempt"] == 2
    assert records[1]["will_retry"] is False
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
    assert "Timeout mapping does not match a case" in rejected.stdout


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
    return 143
}}
if ci_run_test_list {shlex.quote(str(case_list))} command timeout_suite; then
    exit 99
fi
"""
    env = {
        **os.environ,
        "CASE_REPEAT": "1",
        "CASE_TIMEOUT": "0",
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
    assert records[0]["failure_reason"] == "timed out after 0s (exit 143)"
    assert records[1]["action"] == "reset"
    assert records[1]["status"] == "SKIPPED"
    assert records[1]["reason"] == "reset mode is disabled"
    assert records[2]["status"] == "TIMEOUT"


def test_tang_ci_treats_llvm_home_as_optional_and_verifies_the_wheel():
    library = (ROOT / "ci" / "lib.sh").read_text(encoding="utf-8")
    build = (ROOT / "ci" / "build.sh").read_text(encoding="utf-8")
    assert "${LLVM_HOME:?" not in library
    assert 'if [[ -n "$LLVM_HOME" ]]' in library
    assert "ci_assert_tilelang_tang_registration ()" in library
    assert 'tvm.get_global_func("target.build.tilelang_tang", allow_missing=True)' in library
    assert "ci_assert_tilelang_tang_registration" in build


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
    assert "start tilelang S3 ISS simulator test (preserved, unverified)" in config
    assert "when: manual" in config
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


def test_cython_version_matches_the_cp38_limited_api_contract():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    ci_lib = (ROOT / "ci" / "lib.sh").read_text(encoding="utf-8")
    assert 'wheel.py-api = "cp38"' in pyproject
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
