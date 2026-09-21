"""stcuv2 多-warp TMEM 写回 (drain) 测试。

需求:GEMM 的累加器写回 (TMEM -> global) 不再只由 warp0 单独完成,而是把
BM/16 个 16-行 strip 轮转分摊给 block 的所有 warp,每个 warp 用 warp-collective
的 ``ldt_16x256b`` 拷回自己负责的 strip。

关键点:tcgen05 的 MMA 是**异步**且**只由 warp0 发起并 fence** 的,所以在多 warp
写回前必须插入一次 block 级 barrier(``__syncthreads``),让所有参与写回的 warp
都能看到已完成的累加器 —— 否则 warp>0 会读到尚未落地的 TMEM(garbage)。这条
barrier 由 codegen 在 drain 内建里自动生成 (nwarps>1 时)。

一个标准的 ``T.copy(C_tmem, C_global)`` 在 128 线程 block 上会自动展开成 4-warp
写回,无需用户手写切分。

§1 Lowering 断言   test_drain_lowering_is_multiwarp
§2 数值 (via ISS)  test_multiwarp_drain_numeric / test_multiwarp_drain_per_strip

这些特性仅在 stcuv2 上支持。
"""

import re

import pytest
import torch

import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.jit import JITKernel

STCUV2_TARGET = {"kind": "tang", "arch": "stcuv2"}

_TL_DTYPE = {"float16": T.float16, "bfloat16": T.bfloat16}
_PT_DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16}


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


def _run(kernel, M, N, K, dtype):
    pt = _PT_DTYPE[dtype]
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=pt)
    B = torch.randn(N, K, dtype=pt)
    ref = A.float() @ B.float().T
    jit = JITKernel(kernel, out_idx=[-1], target=STCUV2_TARGET, execution_backend="simulator", verbose=False)
    C = jit(A.ptpu(), B.ptpu()).cpu().float()
    return C, ref


# ===========================================================================
# §1  Lowering assertions
# ===========================================================================


def test_drain_lowering_is_multiwarp():
    """A plain drain lowers to a per-warp strided strip loop + a block barrier."""
    src = _lower_src(_make_gemm(128, 128, 64, 64, "float16"))
    # per-warp id and warp-guarded strip loop
    assert re.search(r"_w\s*=\s*\(\(int\)threadIdx\.x\)\s*/\s*32", src), "expected per-warp id (_w = threadIdx.x / 32)"
    # The counter's name comes from the codegen's name supply (it must not
    # collide with an enclosing loop variable), so match it back-referentially
    # rather than pinning a literal.
    assert re.search(r"for\s*\(int (\w+) = _w\*16;.*\1 \+= 64\)", src), "expected round-robin strip loop with nwarps*16 (=64) stride"
    # a block barrier must precede the multi-warp ldt so warps>0 see the MMA out
    assert "__syncthreads();" in src, "expected a block barrier before drain"
    assert re.search(r"ldt_16x256b_x16\(", src), "expected 16x256b ldt drain"


@pytest.mark.parametrize("drain_warps,stride", [(1, 16), (2, 32), (4, 64)])
def test_drain_warps_knob_lowering(drain_warps, stride):
    """`drain_warps=N` caps the writeback to N warps; the barrier stays block-wide.

    The __syncthreads (MMA publish) is emitted for all threads regardless -- it
    sits OUTSIDE the `if (_w < N)` guard -- so reducing N never deadlocks.
    """
    src = _lower_src(_make_gemm(128, 128, 64, 64, "float16", drain_warps=drain_warps))
    assert re.search(rf"if \(_w < {drain_warps}\)", src), f"expected drain restricted to {drain_warps} warp(s)"
    assert re.search(rf"for\s*\(int (\w+) = _w\*16;.*\1 \+= {stride}\)", src), f"expected strip stride nwarps*16={stride}"
    if drain_warps > 1:
        assert "__syncthreads();" in src, "multi-warp drain needs the block barrier"


# ===========================================================================
# §2  Numeric (via ISS)
# ===========================================================================


# N is fixed to 128 (the standard TMEM accumulator tile width the drain
# supports); BM (=M) is varied to exercise 8/4/2 row-strips across 4 warps.
@pytest.mark.parametrize(
    "M,N,K,BK,dtype",
    [
        (128, 128, 64, 64, "float16"),
        (64, 128, 64, 64, "float16"),
        (32, 128, 64, 64, "float16"),
        (128, 128, 64, 64, "bfloat16"),
    ],
)
def test_multiwarp_drain_numeric(M, N, K, BK, dtype):
    """Multi-warp drain must reproduce the GEMM result."""
    C, ref = _run(_make_gemm(M, N, K, BK, dtype), M, N, K, dtype)
    rel = (C - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
    assert rel < 1e-2, f"rel_err too large: {rel:.3e}"


@pytest.mark.parametrize("drain_warps", [1, 2, 4])
def test_drain_warps_knob_numeric(drain_warps):
    """A capped-warp drain (decoupled writeback) still reproduces the GEMM.

    drain_warps=1 => a single warp writes back while the other 3 only cross the
    MMA-publish barrier; drain_warps=2/4 => 2/4 warps share the writeback.
    """
    M, N, K, BK = 128, 128, 64, 64
    C, ref = _run(_make_gemm(M, N, K, BK, "float16", drain_warps=drain_warps), M, N, K, "float16")
    rel = (C - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
    assert rel < 1e-2, f"drain_warps={drain_warps}: rel_err too large: {rel:.3e}"


def test_multiwarp_drain_per_strip():
    """Every 16-row strip (each owned by a different warp) is correct.

    A per-strip check pins down that ALL warps -- not just warp 0 -- wrote their
    slice, i.e. that the block barrier really published the accumulator to the
    non-issuing warps.
    """
    M, N, K, BK = 128, 128, 64, 64
    C, ref = _run(_make_gemm(M, N, K, BK, "float16"), M, N, K, "float16")
    for s in range(0, M, 16):
        band = (C[s : s + 16] - ref[s : s + 16]).abs().max().item() / (ref[s : s + 16].abs().max().item() + 1e-6)
        assert band < 1e-2, f"strip rows[{s}:{s + 16}] wrong: rel={band:.3e}"


if __name__ == "__main__":
    tilelang.testing.main()
