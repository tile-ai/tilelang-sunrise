#!/bin/bash
# Install tilelang + tvm-ffi from dist/ whls, run S3 ISS simulator tests.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The shared CI library already carries the S3 ISS simulator toolchain paths, so
# this job only differs from the S2 job by the test list it runs.
source "$(dirname "$SCRIPT_DIR")/lib.sh"

TILELANG_HOME="$(cd "$SCRIPT_DIR/../.." && pwd)"
DIST_DIR="$TILELANG_HOME/dist"

# S3 mode: tang/stcuv2 through the ISS simulator (sets TILELANG_DEFAULT_TARGET,
# TANG_ARCH and TANG_SIMULATOR; the ISS toolchain paths come from ci/lib.sh).
ci_apply_test_mode s3 || exit 1

time {
ci_init_state "$TILELANG_HOME"
# Same installed-wheel setup as the CI flavor of ci/run.sh: the pytest-mode cases
# have to import the wheel, not the checkout. testing/conftest.py points sys.path
# at the checkout unless TILELANG_TEST_INSTALLED_WHEEL=1, and the checkout has no
# build/ tree in CI -- the s3 files only survived because they run as direct
# commands, where sys.path[0] is the script's own directory.
ci_prepare_installed_wheel_env "$SCRIPT_DIR"
ci_create_conda_env
ci_export_tang_env
# triton is required by some tilelang test cases.
check_exec pip3 install "$TARGET_TRITON_PKG"
ci_install_tilelang_whl "$DIST_DIR"
# Also exports TILELANG_WHEEL_PREFIX for the conftest guard below.
check_exec ci_assert_installed_tilelang_wheel
}

export TILELANG_TEST_INSTALLED_WHEEL=1
# Run pytest cases from outside the checkout as well: `python -m pytest` puts the
# cwd on sys.path.
export PYTEST_LAUNCH_CWD="$CI_TMP_DIR"

time {
# Bare paths, exactly like the main list: every entry runs under pytest. (The
# runner still accepts a "python x.py" line for a direct command if one is ever
# needed.)
ci_run_test_list "$TILELANG_HOME/ci_test_case_list_tilelang_s3.txt"
}
exit $?
