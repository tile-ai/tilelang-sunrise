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


def test_ci_uses_isolated_temp_state_and_masked_timeout_reset_secret():
    lib_path = ROOT / "ci" / "lib.sh"
    _forbid_snippets(lib_path, "torch-c-dlpack-ext", "~/.cache/pip", "passwd=", "sudo -n pt_smi")
    _require_snippets(
        lib_path,
        '[[ -z "${SUDO_MAGICWORD:-}" ]]',
        "printf '%s\\n' \"$SUDO_MAGICWORD\" | sudo -S -p ''",
        'timeout --foreground --kill-after=10s 60 pt_smi -r -i "$dev"',
        'CI_STATE_ROOT="${base}/.ci-state/',
        'export TMPDIR="$CI_TMP_DIR"',
        "ci_on_job_cancel ()",
        "ci_check_card_state ()",
        "ci_save_device_logs ()",
        "ci_run_timed ()",
        'case_repeat="${CASE_REPEAT:-3}"',
    )


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
    assert "if ! command -v conda" in ci_lib
    assert 'source "$HOME/.bashrc"' in ci_lib
    assert 'conda_base="$(conda info --base)"' in ci_lib
    assert 'source "$conda_init"' in ci_lib

    for relative_path in ("ci/build.sh", "ci/test.sh", "ci/validate_operator.sh"):
        script = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "source ~/.bashrc" not in script


def test_release_tree_has_no_submodule_metadata_or_internal_repositories():
    assert not list(ROOT.rglob(".gitmodules"))
    assert not (ROOT / "downstream" / "tilelang-puzzles").exists()

    scanned_files = [ROOT / ".gitlab-ci.yml", *sorted((ROOT / "ci").rglob("*.sh"))]
    for path in scanned_files:
        text = path.read_text(encoding="utf-8")
        assert "gitlab." + "sunrise-ai.com" not in text
        assert "packaging." + "sunrise-ai.com" not in text
