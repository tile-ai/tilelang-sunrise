"""TANG stcuv2 类型转换测试 —— 源码断言.

覆盖:
  FP4 (e2m1) 向量化源码断言 (8 方向)
  FP8 (e4m3/e5m2) 向量化源码断言
"""

import pytest

import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang import tvm

STCUV2_TARGET = {"kind": "tang", "arch": "stcuv2"}


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


if __name__ == "__main__":
    tilelang.testing.main()
