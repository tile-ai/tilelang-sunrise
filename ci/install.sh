#!/bin/bash
# Canonical install/build entry for TileLang.
# Builds tvm-ffi + tvm + tilelang whls into dist/, installs them into the
# isolated conda env, and smoke-tests the import from outside the source tree.
#
# Usage: ci/install.sh [--fresh]
#   --fresh  clear script-managed cache before building. ci_init_state already
#            wipes its own per-run state dir every time; --fresh additionally
#            clears the tilelang kernel cache. It never deletes anything outside
#            script-managed state; without it the script is otherwise idempotent.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

FRESH=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fresh) FRESH=1; shift ;;
        -h|--help) echo "Usage: ci/install.sh [--fresh]"; exit 0 ;;
        *) echo "ERROR: unknown argument: $1"; echo "Usage: ci/install.sh [--fresh]"; exit 2 ;;
    esac
done

ci_configure_ptcc || exit 1

TILELANG_HOME="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST_DIR="$TILELANG_HOME/dist"

time {
ci_init_state "$TILELANG_HOME" build
ci_prepare_installed_wheel_env
ci_create_conda_env
ci_export_tang_env
export TILELANG_HOME

check_exec python "$SCRIPT_DIR/check_gemm_header.py" --include-dir "$TILELANG_HOME/src" \
    --output "$CI_STATE_ROOT/gemm-header-source"

if [[ $FRESH -eq 1 ]]; then
    echo "--fresh: clearing tilelang cache ${TILELANG_CACHE_DIR}"
    rm -rf "${TILELANG_CACHE_DIR}"
fi

rm -f "$DIST_DIR"/*.whl
mkdir -p "$DIST_DIR"

echo "================================ build tvm-ffi + tvm ================================"
ci_build_tvm "$TILELANG_HOME" "$DIST_DIR"

echo "================================ build tilelang whl ================================"
ci_set_tilelang_build_env "$TILELANG_HOME"
pushd "$TILELANG_HOME"
    export USE_TANG=ON
    export USE_CUDA=OFF
    pip3 uninstall tilelang-sunrise -y 2>/dev/null || true
    check_exec pip install "scikit-build-core" "z3-solver>=4.13.0,<4.15.5" "patchelf>=0.17.2"
    check_exec pip wheel --no-build-isolation --no-deps . -w "$DIST_DIR" -v
popd

echo "================================ install tilelang + tvm-ffi whls ================================"
ci_install_tilelang_whl "$DIST_DIR"
}

echo "================================ Build artifacts ================================"
ls -la "$DIST_DIR"/

# Install smoke test: import from OUTSIDE the source tree so the installed wheel is
# exercised, not the ./tilelang checkout that would otherwise shadow it on sys.path.
echo "================================ import smoke test ================================"
ci_prepare_installed_wheel_env
check_exec ci_assert_installed_tilelang_wheel
installed_templates_dir="$(python -c 'from importlib.metadata import distribution; print(distribution("tilelang-sunrise").locate_file("tilelang/src"))')"
check_exec python "$SCRIPT_DIR/check_gemm_header.py" --include-dir "$installed_templates_dir" \
    --output "$CI_STATE_ROOT/gemm-header-installed"
echo "Install phase complete. tvm-ffi + tilelang whls are in $DIST_DIR/ and installed."

# Best-effort: on a LOCAL developer setup, install the pre-commit git hooks so
# commits get lint (matching TileOPs/TileKernels one_click_install.sh). Never in
# CI, never fatal. Runs last so it cannot affect the build/smoke-test exit status.
if [[ -n "${GITLAB_CI:-}" || -n "${CI:-}" || -n "${CI_JOB_ID:-}" ]]; then
    echo "[install] CI detected; skipping local pre-commit hook install"
elif ! git -C "$TILELANG_HOME" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "[install] not a git work tree; skipping local pre-commit hook install"
else
    pc_cmd=""
    if python -m pre_commit --version >/dev/null 2>&1; then
        pc_cmd="python -m pre_commit"
    elif command -v pre-commit >/dev/null 2>&1; then
        pc_cmd="pre-commit"
    fi
    if [[ -n "$pc_cmd" ]]; then
        if (cd "$TILELANG_HOME" && $pc_cmd install --hook-type pre-commit --hook-type pre-push); then
            echo "[install] pre-commit git hooks installed (pre-commit, pre-push)"
        else
            echo "[install] pre-commit hook install failed (non-fatal); run '$pc_cmd install --hook-type pre-commit --hook-type pre-push' from $TILELANG_HOME to enable it"
        fi
    else
        echo "[install] pre-commit not found; skipping local git hook. To enable commit-time lint: pip install pre-commit && pre-commit install"
    fi
fi
