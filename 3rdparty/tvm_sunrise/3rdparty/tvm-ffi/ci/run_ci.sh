#!/bin/bash
# Modified by TileLang-Sunrise contributors in 2026: use the vendored DLPack
# and libbacktrace sources and require vendor packages explicitly.

set -euo pipefail

unset PYTHONHOME
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-0.1.11+sunrise.1}"

mode="${1:-}"
if [[ "$mode" != "cpu" && "$mode" != "ptpu" ]]; then
  echo "Usage: $0 <cpu|ptpu>" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_key="${CI_JOB_ID:-local-$$}"
state_root="$repo_root/.ci-state/${run_key}-tvm-ffi-${mode}"
env_prefix="$state_root/conda"
python_bin="$env_prefix/bin/python"

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
mkdir -p "$state_root/tmp"
export TMPDIR="$state_root/tmp"

conda create --prefix "$env_prefix" python=3.10 -y
test -x "$python_bin"
"$python_bin" - "$env_prefix" <<'PY'
import sys

expected_prefix = sys.argv[1]
assert sys.prefix == expected_prefix, (sys.prefix, expected_prefix)
assert not any("/envs/" in path and not path.startswith(expected_prefix) for path in sys.path), sys.path
PY

"$python_bin" -m pip install --upgrade pip

if [[ "$mode" == "cpu" ]]; then
  test -f "$repo_root/3rdparty/dlpack/LICENSE"
  test -f "$repo_root/3rdparty/libbacktrace/LICENSE"
  "$python_bin" -m pip install build pytest numpy

  rm -rf "$repo_root/dist"
  "$python_bin" -m build --wheel --outdir "$repo_root/dist" "$repo_root"
  wheel="$(find "$repo_root/dist" -maxdepth 1 -type f -name 'apache_tvm_ffi-0.1.11+sunrise.1-*.whl' -print -quit)"
  test -n "$wheel"
  test "$(find "$repo_root/dist" -maxdepth 1 -type f -name '*.whl' | wc -l)" -eq 1
  sha256sum "$wheel" > "$repo_root/dist/SHA256SUMS"
  "$python_bin" -m pip install "$wheel"
  "$python_bin" -c "import tvm_ffi; assert tvm_ffi.__version__ == '0.1.11+sunrise.1', tvm_ffi.__version__"
  "$python_bin" -m pytest -q \
    "$repo_root/tests/python/test_device.py" \
    "$repo_root/tests/python/test_error.py" \
    "$repo_root/tests/python/test_tang_torch_detection.py"
  exit 0
fi

test -n "${TANG_VISIBLE_DEVICES:-}"
sha256sum --check "$repo_root/dist/SHA256SUMS"
wheel="$(find "$repo_root/dist" -maxdepth 1 -type f -name 'apache_tvm_ffi-0.1.11+sunrise.1-*.whl' -print -quit)"
test -n "$wheel"
"$python_bin" -m pip install "$wheel" pytest numpy
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

"$python_bin" -m pytest -q "$repo_root/tests/python/test_tang_ptpu_smoke.py"
