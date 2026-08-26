#!/bin/bash
# Install tilelang + tvm-ffi from dist/ whls, run TileLang's own test list.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

TILELANG_HOME="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST_DIR="$TILELANG_HOME/dist"

time {
ci_init_state "$TILELANG_HOME"
ci_create_conda_env
ci_export_tang_env
ci_install_tilelang_whl "$DIST_DIR"
}

time {
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTEST_AUDIT_PLUGIN_ARGS="-p pytest_node_audit"
export PYTEST_NODE_AUDIT_PATH="$(ci_failure_report_dir)/tilelang_nodes.jsonl"
export PYTEST_NODE_AUDIT_SUITE="tilelang"
export PYTEST_LAUNCH_CWD="$CI_TMP_DIR"
export PYTEST_NODE_AUDIT_TEST_CWD="$TILELANG_HOME"
export PYTEST_NODE_AUDIT_SOURCE_ROOT="$TILELANG_HOME"
export TILELANG_TEST_INSTALLED_WHEEL=1
# The S2 runner contains CUDA toolchain metadata as well as PTPU.  Auto target
# detection therefore prefers CUDA even when no CUDA device is visible.
export TILELANG_DEFAULT_TARGET="${TILELANG_DEFAULT_TARGET:-tang}"

test_ret=0
ci_run_test_list "$TILELANG_HOME/ci_test_case_list_tilelang.txt" pytest tilelang || test_ret=1
}
exit "$test_ret"
