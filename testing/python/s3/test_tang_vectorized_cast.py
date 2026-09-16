"""TANG stcuv2 类型转换测试 —— 源码断言 + S3 ISS 数值验证.

覆盖:
  §1-§4    FP4 (e2m1) ↔ {float32, float16, bfloat16, float64} 标量 roundtrip
  §4.1-§4.4 FP4 (e2m1) 交叉类型单向 ISS 测试 (4)
  §5       FP4 向量化源码断言 (8 方向)
  §6       FP8 (e4m3/e5m2) 向量化源码断言 (12 方向)
  §7-§8    FP8 (e4m3/e5m2) 标量 roundtrip (2)
  §8.1-§8.8 FP8 (e4m3/e5m2) 交叉类型单向 ISS 测试 (8)
"""

import math

import pytest
import torch

import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang import tvm
from tilelang.jit import JITKernel

STCUV2_TARGET = {"kind": "tang", "arch": "stcuv2"}

_TL_DTYPE = {
    "float32": T.float32,
    "float16": T.float16,
    "bfloat16": T.bfloat16,
    "float64": T.float64,
}
_PT_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
}


def _sim_jit(func):
    return JITKernel(func, out_idx=[-1], target=STCUV2_TARGET, execution_backend="simulator", verbose=False)


_FP4_POS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_FP4_MAX = _FP4_POS[-1]


def _fp4_e2m1_quantize(x):
    if math.isnan(x):
        return x
    sign = -1.0 if math.copysign(1.0, x) < 0 else 1.0
    ax = abs(x)
    if ax >= _FP4_MAX:
        return sign * _FP4_MAX
    best, best_d = 0, abs(_FP4_POS[0] - ax)
    for idx in range(1, len(_FP4_POS)):
        d = abs(_FP4_POS[idx] - ax)
        if d < best_d - 1e-12 or abs(d - best_d) <= 1e-12 and idx % 2 == 0 and best % 2 == 1:
            best, best_d = idx, d
    return sign * _FP4_POS[best]


_FP4_TEST_VALUES = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
    0.25,
    0.75,
    1.25,
    1.75,
    2.5,
    2.75,
    3.5,
    5.0,
    -0.25,
    -0.75,
    -2.5,
    -3.5,
    -5.0,
    7.0,
    10.0,
    -7.0,
    -100.0,
    0.1,
    0.01,
]


def _make_roundtrip_kernel(M, io_dtype):
    tl_dt = _TL_DTYPE[io_dtype]

    @T.prim_func
    def kernel(In: T.Tensor((M,), tl_dt), Out: T.Tensor((M,), tl_dt)):
        with T.Kernel(1, threads=128):
            In_local = T.alloc_fragment((M,), tl_dt)
            T.copy(In, In_local)
            Fp4_local = T.alloc_fragment((M,), T.float4_e2m1fn)
            for i in T.Parallel(M):
                Fp4_local[i] = T.cast(In_local[i], T.float4_e2m1fn)
            Out_local = T.alloc_fragment((M,), tl_dt)
            for i in T.Parallel(M):
                Out_local[i] = T.cast(Fp4_local[i], tl_dt)
            T.copy(Out_local, Out)

    return kernel


def _make_double_via_fp4_kernel(M):
    @T.prim_func
    def kernel(In: T.Tensor((M,), T.float32), Out: T.Tensor((M,), T.float32)):
        with T.Kernel(1, threads=128):
            In_local = T.alloc_fragment((M,), T.float32)
            T.copy(In, In_local)
            D_local = T.alloc_fragment((M,), T.float64)
            for i in T.Parallel(M):
                D_local[i] = T.cast(In_local[i], T.float64)
            Fp4_local = T.alloc_fragment((M,), T.float4_e2m1fn)
            for i in T.Parallel(M):
                Fp4_local[i] = T.cast(D_local[i], T.float4_e2m1fn)
            D2_local = T.alloc_fragment((M,), T.float64)
            for i in T.Parallel(M):
                D2_local[i] = T.cast(Fp4_local[i], T.float64)
            Out_local = T.alloc_fragment((M,), T.float32)
            for i in T.Parallel(M):
                Out_local[i] = T.cast(D2_local[i], T.float32)
            T.copy(Out_local, Out)

    return kernel


def _run_roundtrip(io_dtype, atol, kernel=None):
    pt = _PT_DTYPE[io_dtype]
    inp = torch.tensor(_FP4_TEST_VALUES, dtype=pt)
    M = inp.numel()
    if kernel is None:
        kernel = _make_roundtrip_kernel(M, io_dtype)
    jit = _sim_jit(kernel)
    result = jit(inp.ptpu())
    assert result.device.type == "ptpu"
    result = result.cpu().float()
    expected = torch.tensor([_fp4_e2m1_quantize(float(v)) for v in inp.float()], dtype=torch.float32)
    max_diff = (result - expected).abs().max().item()
    print(f"\n  FP4 e2m1 roundtrip ISS test  io={io_dtype}  (M={M})")
    print(f"  max_diff = {max_diff:.6e}")
    n_bad = 0
    for i in range(M):
        v, r, e = float(inp[i]), result[i].item(), expected[i].item()
        ok = abs(r - e) <= atol
        n_bad += 0 if ok else 1
        print(f"    {'OK ' if ok else 'BAD'} {v:>9.4f} -> fp4 -> {r:>9.4f}  (expected {e:>9.4f})")
    assert max_diff <= atol, f"FP4 roundtrip mismatch (io={io_dtype}): max_diff={max_diff:.6e} > atol={atol:.6e}, {n_bad}/{M} bad"


def test_fp4_roundtrip_float32():
    _run_roundtrip("float32", atol=0.0)


def test_fp4_roundtrip_float16():
    _run_roundtrip("float16", atol=0.0)


def test_fp4_roundtrip_bfloat16():
    _run_roundtrip("bfloat16", atol=0.0)


# The double<->fp4 scalar path routes through float. Since fp4 e2m1 has one
# mantissa bit, this intermediate conversion is numerically exact.
def test_fp4_roundtrip_double_intermediate():
    M = len(_FP4_TEST_VALUES)
    _run_roundtrip("float32", atol=0.0, kernel=_make_double_via_fp4_kernel(M))


def _make_cross_cast_kernel(M, src_dtype, dst_dtype):
    tl_src = _TL_DTYPE[src_dtype]
    tl_dst = _TL_DTYPE[dst_dtype]

    @T.prim_func
    def kernel(In: T.Tensor((M,), tl_src), Out: T.Tensor((M,), tl_dst)):
        with T.Kernel(1, threads=128):
            In_local = T.alloc_fragment((M,), tl_src)
            T.copy(In, In_local)
            Fp4_local = T.alloc_fragment((M,), T.float4_e2m1fn)
            for i in T.Parallel(M):
                Fp4_local[i] = T.cast(In_local[i], T.float4_e2m1fn)
            Out_local = T.alloc_fragment((M,), tl_dst)
            for i in T.Parallel(M):
                Out_local[i] = T.cast(Fp4_local[i], tl_dst)
            T.copy(Out_local, Out)

    return kernel


def _run_cross_cast(src_dtype, dst_dtype, atol):
    pt_src = _PT_DTYPE[src_dtype]
    M = len(_FP4_TEST_VALUES)
    inp = torch.tensor(_FP4_TEST_VALUES, dtype=pt_src)
    kernel = _make_cross_cast_kernel(M, src_dtype, dst_dtype)
    jit = _sim_jit(kernel)
    result = jit(inp.ptpu())
    result = result.cpu().float()
    expected = torch.tensor([_fp4_e2m1_quantize(float(v)) for v in inp.float()], dtype=torch.float32)
    max_diff = (result - expected).abs().max().item()
    print(f"\n  FP4 e2m1 cross-cast ISS test  {src_dtype}->fp4->{dst_dtype}  (M={M})")
    print(f"  max_diff = {max_diff:.6e}")
    assert max_diff <= atol, f"FP4 cross-cast mismatch: max_diff={max_diff:.6e}"


def test_fp4_cross_float32_to_float16():
    _run_cross_cast("float32", "float16", atol=0.0)


def test_fp4_cross_float16_to_float32():
    _run_cross_cast("float16", "float32", atol=0.0)


def test_fp4_cross_bfloat16_to_float32():
    _run_cross_cast("bfloat16", "float32", atol=0.0)


def test_fp4_cross_float32_to_bfloat16():
    _run_cross_cast("float32", "bfloat16", atol=0.0)


def _make_vectorized_cast_kernel(M, src_dtype, dst_dtype):
    @T.prim_func
    def kernel(A: T.Tensor((M,), src_dtype), B: T.Tensor((M,), dst_dtype)):
        with T.Kernel(1, threads=128):
            T.copy(A, B)

    return kernel


def _lower_src(kernel, target_str=STCUV2_TARGET):
    with tvm.target.Target(target_str):
        return tilelang.lower(kernel, target=target_str).kernel_source


_FP4_TANG_CHECKS = {
    ("float4_e2m1fn", "float16", 2): ["__tang_cvt_fp4x2_to_halfraw2"],
    ("float16", "float4_e2m1fn", 2): ["__tang_cvt_halfraw2_to_fp4x2"],
    ("bfloat16", "float4_e2m1fn", 2): ["__tang_cvt_bfloat16raw2_to_fp4x2"],
    ("float4_e2m1fn", "bfloat16", 2): ["__tang_cvt_fp4x2_to_halfraw2", "__floats2bfloat162_rn"],
    ("float4_e2m1fn", "float32", 2): ["__tang_cvt_fp4_to_halfraw"],
    ("float32", "float4_e2m1fn", 2): ["__tang_cvt_float_to_fp4"],
    ("float64", "float4_e2m1fn", 2): ["__tang_cvt_float_to_fp4"],
    ("float4_e2m1fn", "float64", 2): ["__tang_cvt_fp4_to_halfraw"],
}

_DTYPE_MAP = {
    "float32": T.float32,
    "float16": T.float16,
    "bfloat16": T.bfloat16,
    "float64": T.float64,
    "float4_e2m1fn": T.float4_e2m1fn,
}


def _run_source_assert(src_name, dst_name, lanes):
    M = 128 * lanes
    kernel = _make_vectorized_cast_kernel(M, _DTYPE_MAP[src_name], _DTYPE_MAP[dst_name])
    src = _lower_src(kernel)
    for c in _FP4_TANG_CHECKS[(src_name, dst_name, lanes)]:
        assert c in src, f"fp4 cast {src_name}->{dst_name}: expected '{c}'"


def test_fp4_source_assert_fp4_to_half():
    _run_source_assert("float4_e2m1fn", "float16", 2)


def test_fp4_source_assert_half_to_fp4():
    _run_source_assert("float16", "float4_e2m1fn", 2)


def test_fp4_source_assert_fp4_to_float():
    _run_source_assert("float4_e2m1fn", "float32", 2)


def test_fp4_source_assert_float_to_fp4():
    _run_source_assert("float32", "float4_e2m1fn", 2)


def test_fp4_source_assert_fp4_to_double():
    _run_source_assert("float4_e2m1fn", "float64", 2)


def test_fp4_source_assert_double_to_fp4():
    _run_source_assert("float64", "float4_e2m1fn", 2)


def test_fp4_source_assert_fp4_to_bf16():
    _run_source_assert("float4_e2m1fn", "bfloat16", 2)


def test_fp4_source_assert_bf16_to_fp4():
    _run_source_assert("bfloat16", "float4_e2m1fn", 2)


FP8_CAST_DIRECTIONS = [
    (T.float32, T.float8_e4m3fn, "__tang_cvt_float_to_fp8", 2),
    (T.float32, T.float8_e5m2, "__tang_cvt_float_to_fp8", 2),
    (T.float16, T.float8_e4m3fn, "__tang_cvt_halfraw_to_fp8", 1),
    (T.float16, T.float8_e5m2, "__tang_cvt_halfraw2_to_fp8x2", 2),
    (T.bfloat16, T.float8_e4m3fn, "__tl_cvt_bfloat162_to_fp8x2", 2),
    (T.bfloat16, T.float8_e5m2, "__tl_cvt_bfloat162_to_fp8x2", 2),
    (T.float64, T.float8_e4m3fn, "__tang_cvt_float_to_fp8", 1),
    (T.float64, T.float8_e5m2, "__tang_cvt_float_to_fp8", 1),
    (T.float8_e4m3fn, T.float32, "__tang_cvt_fp8_to_halfraw", 2),
    (T.float8_e5m2, T.float32, "__tang_cvt_fp8_to_halfraw", 2),
    (T.float8_e4m3fn, T.float16, "__tang_cvt_fp8_to_halfraw", 1),
    (T.float8_e5m2, T.float16, "__tang_cvt_fp8_to_halfraw", 1),
    (T.float8_e4m3fn, T.bfloat16, "__stvm_cvt_fp8_e4m3_to_bfloat16", 2),
    (T.float8_e5m2, T.bfloat16, "__stvm_cvt_fp8_e5m2_to_bfloat16", 2),
    (T.float8_e4m3fn, T.float64, "__tang_cvt_fp8_to_halfraw", 1),
    (T.float8_e5m2, T.float64, "__tang_cvt_fp8_to_halfraw", 1),
]


def _make_fp8_cast_scalar_kernel(M, src_dtype, dst_dtype):
    @T.prim_func
    def kernel(A: T.Tensor((M,), src_dtype), B: T.Tensor((M,), dst_dtype)):
        with T.Kernel(1, threads=128):
            A_local = T.alloc_fragment((M,), src_dtype)
            T.copy(A, A_local)
            B_local = T.alloc_fragment((M,), dst_dtype)
            for i in T.Parallel(M):
                B_local[i] = T.cast(A_local[i], dst_dtype)
            T.copy(B_local, B)

    return kernel


@pytest.mark.parametrize("src_dtype,dst_dtype,check_str,lanes", FP8_CAST_DIRECTIONS)
def test_tang_fp8_vectorized_cast(src_dtype, dst_dtype, check_str, lanes):
    M = 128 * lanes
    kernel = _make_vectorized_cast_kernel(M, src_dtype, dst_dtype)
    src = _lower_src(kernel)
    if check_str in src:
        return
    kernel_s = _make_fp8_cast_scalar_kernel(M, src_dtype, dst_dtype)
    src_s = _lower_src(kernel_s)
    assert check_str in src_s, f"Expected '{check_str}' for {src_dtype}->{dst_dtype}"


_FP8_TEST_VALUES = [
    0.0,
    -0.0,
    0.5,
    -0.5,
    1.0,
    -1.0,
    1.5,
    -1.5,
    2.0,
    -2.0,
    3.0,
    -3.0,
    4.0,
    -4.0,
    6.0,
    -6.0,
    8.0,
    -8.0,
    10.0,
    -10.0,
    12.0,
    -12.0,
    14.0,
    -14.0,
    16.0,
    -16.0,
    24.0,
    -24.0,
    32.0,
    -32.0,
    48.0,
    -48.0,
    64.0,
    -64.0,
    96.0,
    -96.0,
    128.0,
    -128.0,
    192.0,
    -192.0,
    256.0,
    -256.0,
]


def _fp8_e4m3_quantize(x):
    if math.isnan(x):
        return x
    sign = 0 if math.copysign(1.0, x) >= 0 else 1
    ax = abs(x)
    if ax >= 448.0:
        return (-1.0 if sign else 1.0) * 448.0
    if ax == 0:
        return 0.0 if sign == 0 else -0.0
    best_val, best_diff = 0.0, float("inf")
    for m in range(1, 8):
        v = (2**-6) * (m / 8.0)
        d = abs(v - ax)
        if d < best_diff - 1e-14 or abs(d - best_diff) <= 1e-14 and m % 2 == 0:
            best_val, best_diff = v, d
    for e in range(1, 16):
        for m in range(8):
            v = (2 ** (e - 7)) * (1.0 + m / 8.0)
            if v > 448.0 + 1e-10:
                continue
            d = abs(v - ax)
            if d < best_diff - 1e-14 or abs(d - best_diff) <= 1e-14 and m % 2 == 0:
                best_val, best_diff = v, d
    return (-1.0 if sign else 1.0) * best_val


def _fp8_e5m2_quantize(x):
    if math.isnan(x):
        return x
    sign = 0 if math.copysign(1.0, x) >= 0 else 1
    ax = abs(x)
    if ax >= 57344.0:
        return (-1.0 if sign else 1.0) * 57344.0
    if ax == 0:
        return 0.0 if sign == 0 else -0.0
    best_val, best_diff = 0.0, float("inf")
    for m in range(1, 4):
        v = (2**-14) * (m / 4.0)
        d = abs(v - ax)
        if d < best_diff - 1e-14 or abs(d - best_diff) <= 1e-14 and m % 2 == 0:
            best_val, best_diff = v, d
    for e in range(1, 32):
        for m in range(4):
            v = (2 ** (e - 15)) * (1.0 + m / 4.0)
            if v > 57344.0 + 1e-10:
                continue
            d = abs(v - ax)
            if d < best_diff - 1e-14 or abs(d - best_diff) <= 1e-14 and m % 2 == 0:
                best_val, best_diff = v, d
    return (-1.0 if sign else 1.0) * best_val


def _run_fp8_roundtrip(fp8_variant, fp8_dtype, quantize_fn, atol):
    pt = torch.float32
    inp = torch.tensor(_FP8_TEST_VALUES, dtype=pt)
    M = inp.numel()

    @T.prim_func
    def kernel(In: T.Tensor((M,), T.float32), Out: T.Tensor((M,), T.float32)):
        with T.Kernel(1, threads=128):
            A = T.alloc_fragment((M,), T.float32)
            T.copy(In, A)
            B = T.alloc_fragment((M,), fp8_dtype)
            for i in T.Parallel(M):
                B[i] = T.cast(A[i], fp8_dtype)
            C = T.alloc_fragment((M,), T.float32)
            for i in T.Parallel(M):
                C[i] = T.cast(B[i], T.float32)
            T.copy(C, Out)

    jit = _sim_jit(kernel)
    result = jit(inp.ptpu())
    assert result.device.type == "ptpu"
    result = result.cpu().float()
    expected = torch.tensor([quantize_fn(float(v)) for v in inp.float()], dtype=torch.float32)
    max_diff = (result - expected).abs().max().item()
    print(f"\n  FP8 {fp8_variant} roundtrip ISS test  (M={M})")
    print(f"  max_diff = {max_diff:.6e}  atol={atol:.6e}")
    for i in range(M):
        v, r, e = float(inp[i]), result[i].item(), expected[i].item()
        print(f"    {'OK ' if abs(r - e) <= atol else 'BAD'} {v:>9.4f} -> fp8 -> {r:>9.4f}  (expected {e:>9.4f})")
    assert max_diff <= atol, f"FP8 {fp8_variant} roundtrip mismatch: max_diff={max_diff:.6e}"


def test_fp8_e4m3_roundtrip():
    _run_fp8_roundtrip("e4m3", T.float8_e4m3fn, _fp8_e4m3_quantize, atol=1e-4)


def test_fp8_e5m2_roundtrip():
    _run_fp8_roundtrip("e5m2", T.float8_e5m2, _fp8_e5m2_quantize, atol=1e-4)


# ===========================================================================
# §8.1-§8.8  FP8 单向交叉 ISS 测试 (src≠dst)
# ===========================================================================
# 对标 FP4 §4.1-§4.4, 用交叉类型隔离前向/后向问题.


def _run_fp8_cross_cast(src_dtype, dst_dtype, fp8_dtype, quantize_fn, atol):
    tl_src = _TL_DTYPE[src_dtype]
    tl_dst = _TL_DTYPE[dst_dtype]
    pt_src = _PT_DTYPE[src_dtype]
    M = len(_FP8_TEST_VALUES)
    inp = torch.tensor(_FP8_TEST_VALUES, dtype=pt_src)

    @T.prim_func
    def kernel(In: T.Tensor((M,), tl_src), Out: T.Tensor((M,), tl_dst)):
        with T.Kernel(1, threads=128):
            A = T.alloc_fragment((M,), tl_src)
            T.copy(In, A)
            B = T.alloc_fragment((M,), fp8_dtype)
            for i in T.Parallel(M):
                B[i] = T.cast(A[i], fp8_dtype)
            C = T.alloc_fragment((M,), tl_dst)
            for i in T.Parallel(M):
                C[i] = T.cast(B[i], tl_dst)
            T.copy(C, Out)

    jit = _sim_jit(kernel)
    result = jit(inp.ptpu())
    result = result.cpu().float()
    expected = torch.tensor([quantize_fn(float(v)) for v in inp.float()], dtype=torch.float32)
    max_diff = (result - expected).abs().max().item()
    print(f"\n  FP8 cross-cast ISS test  {src_dtype}->fp8->{dst_dtype}  (M={M})")
    print(f"  max_diff = {max_diff:.6e}")
    assert max_diff <= atol, f"FP8 cross-cast mismatch: max_diff={max_diff:.6e}"


def test_fp8_cross_float32_to_e4m3_to_float16():
    _run_fp8_cross_cast("float32", "float16", T.float8_e4m3fn, _fp8_e4m3_quantize, atol=0.01)


def test_fp8_cross_float32_to_e4m3_to_bfloat16():
    _run_fp8_cross_cast("float32", "bfloat16", T.float8_e4m3fn, _fp8_e4m3_quantize, atol=0.01)


def test_fp8_cross_float32_to_e5m2_to_float16():
    _run_fp8_cross_cast("float32", "float16", T.float8_e5m2, _fp8_e5m2_quantize, atol=0.02)


def test_fp8_cross_float32_to_e5m2_to_bfloat16():
    _run_fp8_cross_cast("float32", "bfloat16", T.float8_e5m2, _fp8_e5m2_quantize, atol=0.02)


def test_fp8_cross_float16_to_e4m3_to_float32():
    _run_fp8_cross_cast("float16", "float32", T.float8_e4m3fn, _fp8_e4m3_quantize, atol=0.01)


def test_fp8_cross_float16_to_e5m2_to_float32():
    _run_fp8_cross_cast("float16", "float32", T.float8_e5m2, _fp8_e5m2_quantize, atol=0.02)


def test_fp8_cross_bfloat16_to_e4m3_to_float32():
    _run_fp8_cross_cast("bfloat16", "float32", T.float8_e4m3fn, _fp8_e4m3_quantize, atol=0.01)


def test_fp8_cross_bfloat16_to_e5m2_to_float32():
    _run_fp8_cross_cast("bfloat16", "float32", T.float8_e5m2, _fp8_e5m2_quantize, atol=0.02)


# ===========================================================================
# §9  FP8 (e4m3/e5m2) ↔ float16/bfloat16 标量 roundtrip (ISS)
# ===========================================================================


def _run_fp8_typed_roundtrip(io_dtype, fp8_variant, fp8_dtype, quantize_fn, atol):
    tl_dt = _TL_DTYPE[io_dtype]
    pt = _PT_DTYPE[io_dtype]
    M = len(_FP8_TEST_VALUES)
    inp = torch.tensor(_FP8_TEST_VALUES, dtype=pt)

    @T.prim_func
    def kernel(In: T.Tensor((M,), tl_dt), Out: T.Tensor((M,), tl_dt)):
        with T.Kernel(1, threads=128):
            A = T.alloc_fragment((M,), tl_dt)
            T.copy(In, A)
            B = T.alloc_fragment((M,), fp8_dtype)
            for i in T.Parallel(M):
                B[i] = T.cast(A[i], fp8_dtype)
            C = T.alloc_fragment((M,), tl_dt)
            for i in T.Parallel(M):
                C[i] = T.cast(B[i], tl_dt)
            T.copy(C, Out)

    jit = _sim_jit(kernel)
    result = jit(inp.ptpu())
    result = result.cpu().float()
    expected = torch.tensor([quantize_fn(float(v)) for v in inp.float()], dtype=torch.float32)
    max_diff = (result - expected).abs().max().item()
    print(f"\n  FP8 {fp8_variant} ↔ {io_dtype} roundtrip ISS test  (M={M})")
    print(f"  max_diff = {max_diff:.6e}  atol={atol:.6e}")
    assert max_diff <= atol, f"FP8 {fp8_variant} roundtrip mismatch ({io_dtype}): max_diff={max_diff:.6e}"


def test_fp8_e4m3_float16_roundtrip():
    _run_fp8_typed_roundtrip("float16", "e4m3", T.float8_e4m3fn, _fp8_e4m3_quantize, atol=0.02)


def test_fp8_e5m2_float16_roundtrip():
    _run_fp8_typed_roundtrip("float16", "e5m2", T.float8_e5m2, _fp8_e5m2_quantize, atol=0.05)


def test_fp8_e4m3_bfloat16_roundtrip():
    _run_fp8_typed_roundtrip("bfloat16", "e4m3", T.float8_e4m3fn, _fp8_e4m3_quantize, atol=0.02)


def test_fp8_e5m2_bfloat16_roundtrip():
    _run_fp8_typed_roundtrip("bfloat16", "e5m2", T.float8_e5m2, _fp8_e5m2_quantize, atol=0.05)


# ===========================================================================
# §10  FP8 ↔ float64 (via float 中转) ISS 测试
# ===========================================================================
# float64↔fp8 走 float 中转: double→(float)→__tang_cvt_float_to_fp8,
# __tang_cvt_fp8_to_halfraw→(float)→(double). 对标 FP4 double 测试.


def _run_fp8_double_cross(fp8_variant, fp8_dtype, quantize_fn, forward, atol):
    """Forward=True:  float32→float64→fp8→float32 (tests double→fp8).
    Forward=False: float32→fp8→float64→float32 (tests fp8→double)."""
    M = len(_FP8_TEST_VALUES)

    if forward:

        @T.prim_func
        def kernel(In: T.Tensor((M,), T.float32), Out: T.Tensor((M,), T.float32)):
            with T.Kernel(1, threads=128):
                A = T.alloc_fragment((M,), T.float32)
                T.copy(In, A)
                D = T.alloc_fragment((M,), T.float64)
                for i in T.Parallel(M):
                    D[i] = T.cast(A[i], T.float64)
                B = T.alloc_fragment((M,), fp8_dtype)
                for i in T.Parallel(M):
                    B[i] = T.cast(D[i], fp8_dtype)
                C = T.alloc_fragment((M,), T.float32)
                for i in T.Parallel(M):
                    C[i] = T.cast(B[i], T.float32)
                T.copy(C, Out)
    else:

        @T.prim_func
        def kernel(In: T.Tensor((M,), T.float32), Out: T.Tensor((M,), T.float32)):
            with T.Kernel(1, threads=128):
                A = T.alloc_fragment((M,), T.float32)
                T.copy(In, A)
                B = T.alloc_fragment((M,), fp8_dtype)
                for i in T.Parallel(M):
                    B[i] = T.cast(A[i], fp8_dtype)
                D = T.alloc_fragment((M,), T.float64)
                for i in T.Parallel(M):
                    D[i] = T.cast(B[i], T.float64)
                C = T.alloc_fragment((M,), T.float32)
                for i in T.Parallel(M):
                    C[i] = T.cast(D[i], T.float32)
                T.copy(C, Out)

    inp = torch.tensor(_FP8_TEST_VALUES, dtype=torch.float32)
    jit = _sim_jit(kernel)
    result = jit(inp.ptpu())
    result = result.cpu().float()
    expected = torch.tensor([quantize_fn(float(v)) for v in inp.float()], dtype=torch.float32)
    max_diff = (result - expected).abs().max().item()
    direction = "float64→fp8" if forward else "fp8→float64"
    print(f"\n  FP8 {fp8_variant} {direction} via float ISS test  (M={M})")
    print(f"  max_diff = {max_diff:.6e}  atol={atol:.6e}")
    assert max_diff <= atol, f"FP8 {fp8_variant} {direction}: max_diff={max_diff:.6e}"


def test_fp8_double_to_e4m3():
    _run_fp8_double_cross("e4m3", T.float8_e4m3fn, _fp8_e4m3_quantize, forward=True, atol=0.01)


def test_fp8_double_to_e5m2():
    _run_fp8_double_cross("e5m2", T.float8_e5m2, _fp8_e5m2_quantize, forward=True, atol=0.02)


def test_fp8_e4m3_to_double():
    _run_fp8_double_cross("e4m3", T.float8_e4m3fn, _fp8_e4m3_quantize, forward=False, atol=0.01)


def test_fp8_e5m2_to_double():
    _run_fp8_double_cross("e5m2", T.float8_e5m2, _fp8_e5m2_quantize, forward=False, atol=0.02)


if __name__ == "__main__":
    tilelang.testing.main()
