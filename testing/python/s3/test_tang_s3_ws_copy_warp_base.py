"""Regression tests for warp-specialized (``T.ws``) bulk copies.

Both bugs pinned here were silent, which is why they need a test rather than a
comment:

* the tcgen5 MMA emitter derived its warp index from the *absolute*
  ``threadIdx``, so a ``T.ws(w, 32)`` region with ``w >= 1`` made the
  ``warp < warp_m * warp_n`` guard statically false and Simplify folded the MMA
  (and its ``fence_tc``) away while the TMEM drain kept running (commit
  b82aa03c);
* the bulk-copy helper gated its work -- *and its barrier arrival* -- on the
  absolute ``__warpid() == 0``. Inside ``T.ws(w, 32)`` the copy reports
  ``nwarps == 1`` and takes that path, so a copy outside warp 0 moved no bytes
  and posted no arrival, and every consumer of its ready barrier deadlocked on
  the ISS. Nothing in the protocol had a wrong count; the copy simply never ran.

The structural tests pin the fix: every emitted leaf runs inside the warp range
its ``T.ws`` asked for.
"""

import re

import pytest

import tilelang
import tilelang.tang.language as T
from tilelang import tvm

STCUV2_TARGET = {"kind": "tang", "arch": "stcuv2"}

M = N = 256
K = 512
BM = BN = 128
BK = 64
STAGES = 2
THREADS = 128


def _build(a_warp, b_warp, mma_warp):
    """Producers on ``a_warp``/``b_warp``, consumer on ``mma_warp`` (or all)."""

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((N, K), T.float16),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=THREADS) as (bx, by):
            A_s = T.alloc_shared((BM, BK), T.float16)
            B_s = T.alloc_shared((BN, BK), T.float16)
            C_t = T.alloc_tmem((BM, BN), T.float32)
            for k in T.Pipelined(K // BK, num_stages=STAGES):
                with T.ws(a_warp, warp_group_size=32):
                    T.copy(A[by * BM, k * BK], A_s)
                with T.ws(b_warp, warp_group_size=32):
                    T.copy(B[bx * BN, k * BK], B_s)
                if mma_warp is None:
                    T.gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=(k == 0))
                else:
                    with T.ws(mma_warp, warp_group_size=32):
                        T.gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=(k == 0))
            T.copy(C_t, C[by * BM, bx * BN])

    return kernel


def _lower(a_warp, b_warp, mma_warp):
    with tvm.target.Target(STCUV2_TARGET):
        return tilelang.lower(_build(a_warp, b_warp, mma_warp), target=STCUV2_TARGET).kernel_source


def _leaf_warp_bases(src):
    """The region each emitted g2s leaf runs in, as the warp of its range start.

    The helper's trailing (warp_base) argument is gone: the lowering folds the
    base into the leaf's own addresses and gate. That gate is a *provable* fact
    now (a 1-warp region's ``warp_id == 0`` folds away exactly because the region
    guard already says which warp this is), so every leaf must sit in the region
    its ``T.ws`` requested.
    """
    # Three spellings of the same region guard: ``lo <= tx && tx < hi`` as the
    # lowering emits it, ``tx < hi`` for a region that starts at warp 0 (what
    # Simplify leaves of it), and no guard at all when the whole block is one.
    ranged = re.compile(r"if \(\((\d+) <= \(\(int\)threadIdx\.x\)\) && \(\(\(int\)threadIdx\.x\) < (\d+)\)\)")
    upper = re.compile(r"if \(\(\(int\)threadIdx\.x\) < (\d+)\)")
    leaf = re.compile(r"fcpg2s(?:<[^>]*>)?\(")
    bases, region = [], None
    for line in src.splitlines():
        m = ranged.search(line)
        if m:
            region = int(m.group(1)) // 32
            continue
        if leaf.search(line):
            bases.append(region)
            continue
        m = upper.search(line)
        if m:
            region = 0
    return bases


@pytest.mark.parametrize("a_warp,b_warp,expected", [(0, 1, {0, 1}), (1, 2, {1, 2})])
def test_ws_copy_passes_its_region_warp_base(a_warp, b_warp, expected):
    """A copy inside ``T.ws(w, 32)`` must run in *that* region.

    The bug this pins: a copy whose warp base is read from the block takes the
    single-warp fast path inside ``T.ws(w, 32)``, so for ``w >= 1`` it moves no
    bytes and posts no arrival and the consumer waits forever. The fix used to be
    an explicit warp-base argument to the helper; the lowering emits the copy as
    IR now, so the base is folded into the leaf's addresses and the copy's gate,
    and what remains observable is that each leaf sits inside the warp range its
    ``T.ws`` asked for.
    """
    src = _lower(a_warp, b_warp, None)
    bases = _leaf_warp_bases(src)
    assert bases, "no g2s leaf emitted:\n" + src
    assert None not in bases, f"a leaf outside any region guard: {bases}\n{src}"
    assert set(bases) == expected, f"regions {bases}, expected {expected}:\n{src}"
    # The region runs the transfer and publishes completion to the consumer.
    assert "arrive_on_g2s_fence" in src or "fence_g2s_arrive(" in src, src
    # The blocking fallback is not region-relative yet, so it must not be used
    # here -- if that changes, this assertion is the reminder to fix it too.
    assert "tang_bulk_g2s<" not in src, src


def _blocking_copy_kernel(warp):
    """A loop the async pass leaves alone, so the blocking helper is emitted."""

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((N, K), T.float16),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=THREADS) as (bx, by):
            A_s = T.alloc_shared((BM, BK), T.float16)
            B_s = T.alloc_shared((BN, BK), T.float16)
            C_t = T.alloc_tmem((BM, BN), T.float32)
            # T.serial, not T.Pipelined: with nothing rotated the async rewrite
            # refuses and the copies stay blocking.
            for k in T.serial(K // BK):
                with T.ws(warp, warp_group_size=32):
                    T.copy(A[by * BM, k * BK], A_s)
                with T.ws(1, warp_group_size=32):
                    T.copy(B[bx * BN, k * BK], B_s)
                T.gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=(k == 0))
            T.copy(C_t, C[by * BM, bx * BN])

    return kernel


@pytest.mark.parametrize("warp", [0, 2])
def test_ws_blocking_copy_also_passes_its_region_warp_base(warp):
    """The same region warp base must reach the *blocking* copy as well.

    Whenever the async pass refuses (here: nothing is rotated), the copy keeps
    the blocking form. Its base is folded into the leaf and region guard; a copy
    confined to another warp must stream that region's stripes.
    """
    with tvm.target.Target(STCUV2_TARGET):
        src = tilelang.lower(_blocking_copy_kernel(warp), target=STCUV2_TARGET).kernel_source
    bases = _leaf_warp_bases(src)
    assert bases, "no g2s leaf emitted:\n" + src
    assert None not in bases, f"a leaf outside any region guard: {bases}\n{src}"
    # The B copy is always on warp 1; the A copy carries the parameter.
    assert warp in bases, f"regions {bases} lack warp {warp}:\n{src}"
    assert 1 in bases, f"regions {bases} lack warp 1:\n{src}"
    # Blocking really means blocking: the copy keeps its own drain and nothing
    # arrives on a barrier for it.
    assert "fence_mem(" in src, src
    assert "fence_g2s_arrive(" not in src, src
    assert "arrive_on_g2s_fence" not in src, src
    assert "sync_wait(" not in src, src
