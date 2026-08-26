#!/bin/bash
# Install tilelang + tvm-ffi from dist/ whls, run S3 ISS simulator tests.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Source the shared CI library first (S2 base), then overlay S3 overrides.
source "$(dirname "$SCRIPT_DIR")/lib.sh"
source "$SCRIPT_DIR/lib_s3.sh"

TILELANG_HOME="$(cd "$SCRIPT_DIR/../.." && pwd)"
DIST_DIR="$TILELANG_HOME/dist"

time {
ci_init_state "$TILELANG_HOME"
ci_create_conda_env
ci_export_tang_env
# triton is required by some tilelang test cases.
check_exec pip3 install "$TARGET_TRITON_PKG"
ci_install_tilelang_whl "$DIST_DIR"
}

time {
ci_run_test_list "$TILELANG_HOME/ci_test_case_list_tilelang_s3.txt" command
}
exit $?
