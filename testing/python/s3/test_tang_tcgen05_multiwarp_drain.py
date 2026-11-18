"""stcuv2 多-warp TMEM 写回 (drain) 测试。

需求:GEMM 的累加器写回 (TMEM -> global) 不再只由 warp0 单独完成,而是把
BM/16 个 16-行 strip 轮转分摊给 block 的所有 warp,每个 warp 用 warp-collective
的 ``ldt_16x256b`` 拷回自己负责的 strip。

关键点:tcgen05 的 MMA 是**异步**且**只由 warp0 发起并 fence** 的,所以在多 warp
写回前必须插入一次 block 级 barrier(``__syncthreads``),让所有参与写回的 warp
都能看到已完成的累加器 —— 否则 warp>0 会读到尚未落地的 TMEM(garbage)。这条
barrier 由 `tl.tang.transform.ExpandS3TmemDrain` pass 在 IR 里生成
(``nwarps>1`` 时,位于 ``warp < nwarps`` guard 之外,因此 block-wide)。

一个标准的 ``T.copy(C_tmem, C_global)`` 在 128 线程 block 上会自动展开成 4-warp
写回,无需用户手写切分。

§1 Lowering 断言   test_drain_lowering_is_multiwarp

这些特性仅在 stcuv2 上支持。
"""

import re

import pytest

import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang import tvm as tvm

STCUV2_TARGET = {"kind": "tang", "arch": "stcuv2"}

_TL_DTYPE = {"float16": T.float16, "bfloat16": T.bfloat16}


def _make_gemm(M, N, K, BK, dtype, drain_warps=None):
    """Standard TN GEMM whose epilogue is a plain T.copy(C_tmem, C_global).

    The drain spreads across every warp of the 128-thread block by default. When
    ``drain_warps`` is given, only that many warps run the ldt writeback (the
    others still hit the MMA-publish block barrier) -- decoupling the writeback
    warp count from the copy/MMA warps.
    """
    dt = _TL_DTYPE[dtype]

    @T.prim_func
    def kernel(A: T.Tensor((M, K), dt), B: T.Tensor((N, K), dt), C: T.Tensor((M, N), T.float32)):
        with T.Kernel(1, 1, threads=128):
            A_s = T.alloc_shared((M, BK), dt)
            B_s = T.alloc_shared((N, BK), dt)
            C_t = T.alloc_tmem((M, N), T.float32)
            for k in T.Pipelined(K // BK, num_stages=3):
                T.copy(A[0, k * BK], A_s)
                T.copy(B[0, k * BK], B_s)
                T.gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=(k == 0))
            if drain_warps is None:
                T.copy(C_t, C)
            else:
                T.copy(C_t, C, drain_warps=drain_warps)

    return kernel


def _lower_src(func, target=STCUV2_TARGET):
    with tvm.target.Target(target):
        return tilelang.lower(func, target=target).kernel_source


# ===========================================================================
# §1  Lowering assertions
# ===========================================================================


def test_drain_lowering_is_multiwarp():
    """A plain drain lowers to a per-warp strip loop + a block barrier.

    The drain's geometry is emitted by the ExpandS3TmemDrain pass (in IR), so
    the shape pinned here is the round-robin strip split across all four warps of
    the 128-thread block: BM/(nwarps*16) = 2 strips per warp, with the
    MMA-publish barrier in front of the warp guard (block-wide, never guarded).
    """
    src = _lower_src(_make_gemm(128, 128, 64, 64, "float16"))
    assert re.search(r"ldt_16x256b_x16\(", src), "expected 16x256b ldt drain"
    assert re.search(r"for \(int k = 0; k < 2; \+\+k\)", src), "expected BM/(nwarps*16) = 2 round-robin strips per warp"
    # a block barrier must precede the multi-warp ldt so warps>0 see the MMA out
    assert re.search(r"__syncthreads\(\);\s*uint _out\[64\];", src), (
        "expected the MMA-publish barrier before the drain (and outside the guard)"
    )


@pytest.mark.parametrize("drain_warps,strips", [(1, 8), (2, 4), (4, 2)])
def test_drain_warps_knob_lowering(drain_warps, strips):
    """`drain_warps=N` caps the writeback to N warps; the barrier stays block-wide.

    N warps split the 8 row strips of the 128-row tile, so each takes
    BM/(N*16) strips. The __syncthreads (MMA publish) is emitted before the
    `warp < N` guard -- it sits OUTSIDE it -- so reducing N never deadlocks.
    """
    src = _lower_src(_make_gemm(128, 128, 64, 64, "float16", drain_warps=drain_warps))
    assert re.search(rf"for \(int k = 0; k < {strips}; \+\+k\)", src), f"expected {strips} strips per warp (BM/({drain_warps}*16))"
    if drain_warps > 1:
        assert re.search(r"__syncthreads\(\);\s*uint _out\[64\];", src), "multi-warp drain needs the block barrier before the warp guard"
    if drain_warps < 4:
        # Restricting the writeback keeps a warp guard; for all four warps of a
        # 128-thread block the guard is provably true and folds away.
        assert re.search(rf"if \(\(\(int\)threadIdx\.x\) < {drain_warps * 32}\)", src), (
            f"expected drain restricted to {drain_warps} warp(s)"
        )


if __name__ == "__main__":
    tilelang.testing.main()
