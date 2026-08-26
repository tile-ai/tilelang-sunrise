"""Tests for the per-call ``nan_propagate`` kwarg on T.reduce_max / reduce_min /
reduce_absmax for float16 and bfloat16 buffers (CUDA and TANG)."""

import math

import torch

import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang.utils.device import get_current_device

_DTYPES = [("float16", T.float16, torch.float16), ("bfloat16", T.bfloat16, torch.bfloat16)]


def _compile(prim_func):
    target = "tang" if torch.ptpu.is_available() else "cuda"
    return tilelang.compile(prim_func, out_idx=-1, target=target)


def _to_float(x):
    """Move to CPU and convert to float32 for NaN checks (ptpu has limited op support)."""
    return x.to(device="cpu", dtype=torch.float32)


def _make_reduce_kernel(reduce_fn, length, dtype, *, nan_propagate):

    @T.prim_func
    def kernel(a: T.Tensor((length,), dtype), out: T.Tensor((1,), dtype)):
        with T.Kernel(1, threads=32):
            frag = T.alloc_fragment((length,), dtype)
            out_frag = T.alloc_fragment((1,), dtype)
            T.copy(a, frag)
            reduce_fn(frag, out_frag, nan_propagate=nan_propagate)
            T.copy(out_frag, out)

    return kernel


# ---------------------------------------------------------------------------
# Source-level checks: confirm the right reducer / intrinsic is emitted.
# ---------------------------------------------------------------------------


def test_reduce_max_default_uses_plain_op():
    k = _compile(_make_reduce_kernel(T.reduce_max, 64, T.float16, nan_propagate=False))
    src = k.get_kernel_source()
    assert "tl::MaxOp" in src and "MaxOpNan" not in src


def test_reduce_max_nan_propagate_uses_nan_op():
    k = _compile(_make_reduce_kernel(T.reduce_max, 64, T.float16, nan_propagate=True))
    src = k.get_kernel_source()
    assert "tl::MaxOpNan" in src


def test_reduce_min_nan_propagate_uses_nan_op():
    k = _compile(_make_reduce_kernel(T.reduce_min, 64, T.bfloat16, nan_propagate=True))
    src = k.get_kernel_source()
    assert "tl::MinOpNan" in src


def test_reduce_absmax_nan_propagate_uses_nan_op():
    k = _compile(_make_reduce_kernel(T.reduce_absmax, 64, T.float16, nan_propagate=True))
    src = k.get_kernel_source()
    assert "tl::MaxOpNan" in src


# ---------------------------------------------------------------------------
# Runtime behavioral checks: NaN actually propagates only when requested.
# ---------------------------------------------------------------------------


def test_reduce_max_runtime_nan_behavior():
    for _, tl_dtype, torch_dtype in _DTYPES:
        length = 64
        a = torch.arange(length, dtype=torch.float32).to(torch_dtype).to(get_current_device())
        a[7] = float("nan")

        k_default = _compile(_make_reduce_kernel(T.reduce_max, length, tl_dtype, nan_propagate=False))
        k_nan = _compile(_make_reduce_kernel(T.reduce_max, length, tl_dtype, nan_propagate=True))

        out_default = _to_float(k_default(a))
        out_nan = _to_float(k_nan(a))

        assert not math.isnan(out_default.item()), f"{tl_dtype}: default reduce_max should ignore NaN, got {out_default}"
        assert math.isnan(out_nan.item()), f"{tl_dtype}: nan_propagate reduce_max should return NaN, got {out_nan}"


def test_reduce_min_runtime_nan_behavior():
    for _, tl_dtype, torch_dtype in _DTYPES:
        length = 64
        a = torch.arange(length, dtype=torch.float32).to(torch_dtype).to(get_current_device())
        a[13] = float("nan")

        k_default = _compile(_make_reduce_kernel(T.reduce_min, length, tl_dtype, nan_propagate=False))
        k_nan = _compile(_make_reduce_kernel(T.reduce_min, length, tl_dtype, nan_propagate=True))

        assert not math.isnan(_to_float(k_default(a)).item())
        assert math.isnan(_to_float(k_nan(a)).item())


# ---------------------------------------------------------------------------
# clear=False path: the write-back merge (MakeUpdate) must also honor
# nan_propagate.  See GH-2697.
# ---------------------------------------------------------------------------


def _make_reduce_clear_false_kernel(reduce_fn, length, dtype, *, nan_propagate):

    @T.prim_func
    def kernel(a: T.Tensor((length,), dtype), out: T.Tensor((1,), dtype)):
        with T.Kernel(1, threads=32):
            frag = T.alloc_fragment((length,), dtype)
            out_frag = T.alloc_fragment((1,), dtype)
            T.copy(a, frag)
            # Seed the output fragment with -1 so the NaN test is meaningful.
            out_frag[0] = -1.0
            reduce_fn(frag, out_frag, clear=False, nan_propagate=nan_propagate)
            T.copy(out_frag, out)

    return kernel


def test_reduce_max_clear_false_nan_propagate():
    """clear=False + nan_propagate=True must yield NaN (GH-2697)."""
    for _, tl_dtype, torch_dtype in _DTYPES:
        length = 64
        a = torch.arange(length, dtype=torch.float32).to(torch_dtype).to(get_current_device())
        a[7] = float("nan")

        k_nan = _compile(_make_reduce_clear_false_kernel(T.reduce_max, length, tl_dtype, nan_propagate=True))
        out = _to_float(k_nan(a))
        assert math.isnan(out.item()), f"{tl_dtype}: reduce_max clear=False nan_propagate=True should return NaN, got {out.item()}"

        k_default = _compile(_make_reduce_clear_false_kernel(T.reduce_max, length, tl_dtype, nan_propagate=False))
        out = _to_float(k_default(a))
        assert not math.isnan(out.item()), f"{tl_dtype}: reduce_max clear=False nan_propagate=False should not return NaN"


def test_reduce_min_clear_false_nan_propagate():
    for _, tl_dtype, torch_dtype in _DTYPES:
        length = 64
        a = torch.arange(length, dtype=torch.float32).to(torch_dtype).to(get_current_device())
        a[13] = float("nan")

        k_nan = _compile(_make_reduce_clear_false_kernel(T.reduce_min, length, tl_dtype, nan_propagate=True))
        out = _to_float(k_nan(a))
        assert math.isnan(out.item()), f"{tl_dtype}: reduce_min clear=False nan_propagate=True should return NaN, got {out.item()}"

        k_default = _compile(_make_reduce_clear_false_kernel(T.reduce_min, length, tl_dtype, nan_propagate=False))
        out = _to_float(k_default(a))
        assert not math.isnan(out.item())


def test_reduce_absmax_clear_false_nan_propagate():
    for _, tl_dtype, torch_dtype in _DTYPES:
        length = 64
        a = torch.arange(length, dtype=torch.float32).to(torch_dtype).to(get_current_device())
        a[13] = float("nan")

        k_nan = _compile(_make_reduce_clear_false_kernel(T.reduce_absmax, length, tl_dtype, nan_propagate=True))
        out = _to_float(k_nan(a))
        assert math.isnan(out.item()), f"{tl_dtype}: reduce_absmax clear=False nan_propagate=True should return NaN, got {out.item()}"

        k_default = _compile(_make_reduce_clear_false_kernel(T.reduce_absmax, length, tl_dtype, nan_propagate=False))
        out = _to_float(k_default(a))
        assert not math.isnan(out.item())


if __name__ == "__main__":
    tilelang.testing.main()
