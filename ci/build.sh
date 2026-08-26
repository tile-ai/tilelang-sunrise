#!/bin/bash
# Build tvm-ffi + tvm + tilelang from source, produce whls into dist/.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

TILELANG_HOME="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST_DIR="$TILELANG_HOME/dist"

time {
ci_init_state "$TILELANG_HOME" build
ci_create_conda_env
ci_export_tang_env
export TILELANG_HOME

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
}

echo "================================ Build artifacts ================================"
ls -la "$DIST_DIR"/
echo "Build phase complete. tvm-ffi + tilelang whls are in $DIST_DIR/"
