#!/bin/bash
# Set up tilelang, install TileOPs, run its test suite.
# In the TileLang-Sunrise monorepo this script runs from the TileOPs child
# pipeline. It builds the vendored TileLang source unless an explicit wheel
# directory is supplied by the caller.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TILEOPS_HOME="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- 1. Resolve the vendored TileLang source ----
TILELANG_SOURCE_DIR="${TILELANG_SOURCE_DIR:-$(cd "$TILEOPS_HOME/../.." && pwd)}"
TILELANG_CI_DIR="${TILELANG_CI_DIR:-$TILELANG_SOURCE_DIR/ci}"
if [[ ! -f "$TILELANG_CI_DIR/lib.sh" ]]; then
    echo "TileLang CI library not found under $TILELANG_SOURCE_DIR" >&2
    exit 1
fi

if [ -n "${TILELANG_WHL_DIR:-}" ]; then
    : "${TILELANG_CI_DIR:?TILELANG_CI_DIR must be set when TILELANG_WHL_DIR is set}"
fi

# ---- 2. Source the single shared lib.sh ----
source "$TILELANG_CI_DIR/lib.sh"

# ---- 3. Obtain the tilelang whl (standalone builds it in an isolated BUILD env) ----
if [ -z "${TILELANG_WHL_DIR:-}" ]; then
    echo "================================ Building tilelang whl (isolated build env) ================================"
    bash "$TILELANG_CI_DIR/build.sh" || { echo "tilelang build.sh failed"; exit 1; }
    TILELANG_WHL_DIR="$TILELANG_SOURCE_DIR/dist"
fi

# ---- 4. Clean TEST env: install whl + TileOPs, run tests ----
time {
ci_init_state "$TILEOPS_HOME"
ci_create_conda_env
ci_export_tang_env
ci_install_tilelang_whl "$TILELANG_WHL_DIR"

echo "================================ Installing TileOPs ================================"
cd "$TILEOPS_HOME"
check_exec pip install -e . -v
pip install pytest-xdist pyyaml
}

time {
ci_run_test_list "$TILEOPS_HOME/ci_test_case_list_tileops.txt" pytest
}
exit $?
