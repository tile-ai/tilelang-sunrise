"""stcuv2 tcgen05 LDT/STT (T.tcgen05_ld / T.tcgen05_st) 测试。

覆盖 tensor memory <-> fragment register 的拷贝原语(S3 专属,`.16x256b` shape,
单 warp 循环实现,详见 docs/s3_tmem_fragment_ldt_stt_layout.md)。warp 专门化
(指定非 0 warp)见 test_tang_tcgen05_warp_specialize.py。

§1 Lowering 断言    test_ldst_lowering_*   (对生成源码做正则)

这些原语仅在 stcuv2 上支持;在 stcu 上不应产生对应指令。
"""

import re

import pytest

import tilelang
import tilelang.testing
import tilelang.tang.language as T
from tilelang import tvm as tvm

STCUV2_TARGET = {"kind": "tang", "arch": "stcuv2"}

_TL_DTYPE = {"float16": T.float16, "bfloat16": T.bfloat16}


# ===========================================================================
# Shared kernels / helpers
# ===========================================================================


def _make_ld_kernel(M, N, K, threads, dtype):
    """gemm -> tcgen05_ld(TMEM->fragment) -> copy fragment to global."""
    dt = _TL_DTYPE[dtype]

    @T.prim_func
    def kernel(A: T.Tensor((M, K), dt), B: T.Tensor((N, K), dt), C: T.Tensor((M, N), T.float32)):
        with T.Kernel(1, 1, threads=threads):
            A_s = T.alloc_shared((M, K), dt)
            B_s = T.alloc_shared((N, K), dt)
            C_t = T.alloc_tmem((M, N), T.float32)
            C_f = T.alloc_fragment((M, N), T.float32)
            T.copy(A, A_s)
            T.copy(B, B_s)
            T.gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=True)
            T.tcgen05_ld(C_f, C_t)
            T.copy(C_f, C)

    return kernel


def _make_st_roundtrip_kernel(M, N, K, threads, dtype):
    """gemm -> ld -> st(fragment->TMEM2) -> ld(TMEM2->fragment) -> global.

    Exercises tcgen05_st into a fresh (non-MMA) TMEM buffer, then reads it back;
    a correct round-trip must reproduce the GEMM result.
    """
    dt = _TL_DTYPE[dtype]

    @T.prim_func
    def kernel(A: T.Tensor((M, K), dt), B: T.Tensor((N, K), dt), C: T.Tensor((M, N), T.float32)):
        with T.Kernel(1, 1, threads=threads):
            A_s = T.alloc_shared((M, K), dt)
            B_s = T.alloc_shared((N, K), dt)
            C_t = T.alloc_tmem((M, N), T.float32)
            C_t2 = T.alloc_tmem((M, N), T.float32)
            C_f = T.alloc_fragment((M, N), T.float32)
            C_f2 = T.alloc_fragment((M, N), T.float32)
            T.copy(A, A_s)
            T.copy(B, B_s)
            T.gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=True)
            T.tcgen05_ld(C_f, C_t)
            T.tcgen05_st(C_t2, C_f)
            T.tcgen05_ld(C_f2, C_t2)
            T.copy(C_f2, C)

    return kernel


def _lower_src(func, target=STCUV2_TARGET):
    with tvm.target.Target(target):
        return tilelang.lower(func, target=target).kernel_source


# ===========================================================================
# §1  Lowering assertions
# ===========================================================================


def test_ld_lowering_emits_ldt_16x256b():
    """tcgen05_ld lowers to a single-warp-guarded ldt_16x256b + fence_ldt.

    N=64 -> num_chunks = 64/8 = 8 -> ldt_16x256b_x8. M=128 -> 8 sub-blocks,
    all driven by the first warp of thread_bounds (threadIdx.x < 32 here).
    """
    src = _lower_src(_make_ld_kernel(128, 64, 64, 128, "float16"))
    assert re.search(r"ldt_16x256b_x8\(", src), "expected ldt_16x256b_x8"
    assert re.search(r"fence_ldt\(\)", src), "expected fence_ldt"
    # warp-0-only guard
    assert re.search(r"threadIdx\.x\)? *< *32", src) or re.search(r"threadIdx\.x[^;]*32", src), "expected warp-0 guard"
    # must not fall back to the CUDA tcgen05.ld extern
    assert "tcgen05.ld" not in src


def test_st_lowering_emits_stt_16x256b():
    """tcgen05_st lowers to stt_16x256b + fence_stt."""
    src = _lower_src(_make_st_roundtrip_kernel(128, 64, 64, 128, "float16"))
    assert re.search(r"stt_16x256b_x8\(", src), "expected stt_16x256b_x8"
    assert re.search(r"fence_stt\(\)", src), "expected fence_stt"
    assert re.search(r"ldt_16x256b_x8\(", src), "roundtrip also needs ldt"


def test_ld_rows_must_be_multiple_of_16():
    """A tile whose row count is not a multiple of 16 is rejected at lowering."""
    with pytest.raises(Exception, match=r"16x256b tmem fragment rows must be a multiple of 32"):
        _lower_src(_make_ld_kernel(24, 64, 64, 128, "float16"))


if __name__ == "__main__":
    tilelang.testing.main()
