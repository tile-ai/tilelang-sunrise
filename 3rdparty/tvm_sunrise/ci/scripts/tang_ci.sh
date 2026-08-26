#!/bin/bash
# Modified by TileLang-Sunrise contributors in 2026: use vendored dependencies
# and require vendor runtime packages to be provided explicitly.

set -euo pipefail

unset PYTHONHOME
unset PYTHONPATH
export PYTHONNOUSERSITE=1

mode="${1:-}"
if [[ "$mode" != "on" && "$mode" != "off" ]]; then
  echo "Usage: $0 <on|off>" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
run_key="${CI_JOB_ID:-local-$$}"
state_root="$repo_root/.ci-state/${run_key}-tang-${mode}"
build_dir="$state_root/build"
env_prefix="$state_root/conda"
dist_dir="$state_root/dist"
result_dir="$repo_root/.ci-results"
cmake_bin="${CMAKE_BIN:-/usr/bin/cmake}"
test -x "$cmake_bin"

cleanup() {
  local exit_code=$?
  trap - EXIT
  set +e
  if [[ -d "$env_prefix" ]]; then
    conda env remove --prefix "$env_prefix" -y
  fi
  rm -rf "$state_root"
  exit "$exit_code"
}
trap cleanup EXIT

rm -rf "$state_root"
mkdir -p "$build_dir" "$dist_dir" "$result_dir" "$state_root/tmp"
export TMPDIR="$state_root/tmp"

test -f "$repo_root/3rdparty/tvm-ffi/LICENSE"
test -f "$repo_root/3rdparty/tvm-ffi/3rdparty/dlpack/LICENSE"
test -f "$repo_root/3rdparty/tvm-ffi/3rdparty/libbacktrace/LICENSE"

conda create --prefix "$env_prefix" python=3.10 gtest -y
python_bin="$env_prefix/bin/python"
test -x "$python_bin"
"$python_bin" - "$env_prefix" <<'PY'
import sys

expected_prefix = sys.argv[1]
assert sys.prefix == expected_prefix, (sys.prefix, expected_prefix)
assert not any("/envs/" in path and not path.startswith(expected_prefix) for path in sys.path), sys.path
PY

"$python_bin" -m pip install --upgrade pip
"$python_bin" -m pip install \
  cloudpickle decorator ml-dtypes numpy packaging psutil pytest scipy tornado typing_extensions

SETUPTOOLS_SCM_PRETEND_VERSION="${TARGET_TVM_FFI_VERSION:-0.1.11+sunrise.1}" \
  "$python_bin" -m pip wheel --no-deps "$repo_root/3rdparty/tvm-ffi" -w "$dist_dir"
ffi_wheel="$(find "$dist_dir" -maxdepth 1 -type f -name 'apache_tvm_ffi-*.whl' -print -quit)"
test -n "$ffi_wheel"
"$python_bin" -m pip install "$ffi_wheel"

cmake_args=(
  -S "$repo_root"
  -B "$build_dir"
  -DUSE_HEXAGON=OFF
  -DUSE_LLVM=OFF
  -DUSE_CUDA=OFF
  -DUSE_OPENCL=OFF
  -DUSE_CUTLASS=OFF
)

if [[ "$mode" == "on" ]]; then
  tangrt_path="${TANGRT_PATH:-/usr/local/tangrt}"
  tang_prefix="${tangrt_path%/}/targets/linux-x86_64"
  export LD_LIBRARY_PATH="${tangrt_path%/}/lib/linux-x86_64:/usr/lib64:$env_prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  cmake_args+=(
    -DUSE_TANG=ON
    -DCMAKE_TANG_COMPILER="${PTCC_PATH:-${tangrt_path%/}/toolchains/llvm/prebuilt/linux-x86_64/bin/ptcc}"
    -DTANG_TOOLKIT_ROOT_DIR="$tangrt_path"
    -DTANG_DIR="$tang_prefix/lib/cmake/TANG"
    -DTANGRT_DIR="$tang_prefix/lib/cmake/TANGRT"
    -DCMAKE_MODULE_PATH="${CMAKE_PATH:-${tangrt_path%/}/cmake}"
    "-DCMAKE_PREFIX_PATH=$tang_prefix;$env_prefix"
  )
else
  cmake_args+=(
    -DUSE_TANG=OFF
    "-DCMAKE_PREFIX_PATH=$env_prefix"
  )
fi

"$cmake_bin" "${cmake_args[@]}"
"$cmake_bin" --build "$build_dir" --parallel "$(nproc)"

export PYTHONPATH="$repo_root/python${PYTHONPATH:+:$PYTHONPATH}"
export TVM_LIBRARY_PATH="$build_dir/lib"

if [[ "$mode" == "off" ]]; then
  "$python_bin" -c "import tvm; assert not tvm.runtime.enabled('tang')"
  exit 0
fi

test -n "${TANG_VISIBLE_DEVICES:-}"
"$python_bin" -m pip install \
  "torch==${TARGET_TORCH_VERSION:-2.6.0}" \
  --index-url "${TARGET_TORCH_PKG_URL:-https://download.pytorch.org/whl/cpu}"
: "${TARGET_TORCH_PTPU_PKG:?Set TARGET_TORCH_PTPU_PKG to an accessible torch_ptpu wheel}"
"$python_bin" -m pip install "$TARGET_TORCH_PTPU_PKG"

"$python_bin" - <<'PY'
import torch
import torch_ptpu

assert torch.ptpu.is_available(), "PTPU is unavailable"
lhs = torch.randn((64, 64), device="ptpu", dtype=torch.float32)
rhs = torch.randn((64, 64), device="ptpu", dtype=torch.float32)
out = lhs @ rhs
torch.ptpu.synchronize()
assert torch.isfinite(out.cpu()).all()
PY

"$python_bin" -m pytest -q \
  --confcutdir="$repo_root/tests" \
  --junitxml="$result_dir/tvm-tang-on.xml" \
  "$repo_root/tests/python/contrib/test_tang.py" \
  "$repo_root/tests/python/target/test_target_tang.py" \
  "$repo_root/tests/python/target/test_target_tang_intrin.py"

"$cmake_bin" --build "$build_dir" --target cpptest --parallel "$(nproc)"
"$build_dir/cpptest" --gtest_filter='TANGLaunchUtils.*'
