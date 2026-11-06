"""Warp-specialised pipeline: the role split, and where its synchronisation sits.

The kernel below is the shape the role-split pipeline is about: warp 0 produces
A, warp 1 produces B, warp 2 consumes both with a tcgen5 MMA, and warp 3 does
nothing inside the loop but still has to reach every block-wide construct. Every
assertion reads the generated device source, because that is where the failure
modes of this rewrite live:

* a ``__syncthreads()`` inside the loop re-serialises the roles -- the whole
  point of the async form is that the producers, the consumer and the idle warp
  never rendezvous once the pipeline is running;
* a barrier array declared *after* the shared-memory buffer lands outside the
  window the CTA maps, and then its state stops updating: the phase never flips
  and every wait on it spins forever. That is exactly what
  ``tang(s3): declare the mbarrier array before the shared-memory buffer`` fixed,
  and it is asserted here so it cannot come back.
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
BK = 64  # <= the 128 B MMA operand row, so the M1b segmentation does not kick in
STAGES = 2
THREADS = 128
WARP = 32


def ws_gemm(stages: int = STAGES, mma_warp: int = 2):
    """Role-split GEMM: two producer warps, one consumer warp, one idle warp."""

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
            for k in T.Pipelined(K // BK, num_stages=stages):
                with T.ws(0, warp_group_size=WARP):
                    T.copy(A[by * BM, k * BK], A_s)
                with T.ws(1, warp_group_size=WARP):
                    T.copy(B[bx * BN, k * BK], B_s)
                with T.ws(mma_warp, warp_group_size=WARP):
                    T.gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=(k == 0))
            T.copy(C_t, C[by * BM, bx * BN])

    return kernel


def source(stages: int = STAGES) -> str:
    """Lower the WS kernel through the production mbarrier path."""
    with tvm.target.Target(STCUV2_TARGET):
        return tilelang.lower(ws_gemm(stages), target=STCUV2_TARGET).kernel_source


# ===========================================================================
# Source helpers
# ===========================================================================


def _loop_body(src: str) -> str:
    """The body of the pipelined ``for (int k = 0; ...)`` loop, braces matched."""
    start = src.index("for (int k = 0;")
    open_brace = src.index("{", start)
    depth = 0
    for i in range(open_brace, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[open_brace + 1 : i]
    raise AssertionError("unterminated pipeline loop")


def _line_of(src: str, needle: str) -> int:
    for i, line in enumerate(src.splitlines(), 1):
        if needle in line:
            return i
    raise AssertionError(f"{needle!r} not in source")


# ===========================================================================
# §1  The roles never rendezvous inside the loop
# ===========================================================================


def test_role_split_loop_has_no_block_wide_sync():
    """A block-wide sync in the loop is the role split thrown away."""
    body = _loop_body(source())
    assert "__syncthreads" not in body, body
    # The roles must still be there, otherwise this asserts nothing: the copy
    # guards and the MMA guard are the warp ranges of the three roles.
    assert "((int)threadIdx.x) < 32" in body, body
    assert "64 <= ((int)threadIdx.x)" in body, body
    assert "tang_tcgen05_mma_ss" in body, body


# ===========================================================================
# §2  Both pipeline edges use mbarriers
# ===========================================================================


def test_the_default_puts_both_edges_on_mbarriers():
    """Ready (copy -> MMA) and free (MMA -> copy) both use mbarrier slots."""
    body = _loop_body(source())

    # Ready waits use the same versioned slots across prologue, loop and
    # epilogue.
    assert re.search(r"mbarrier\[[^\]]+\]\.wait\(", body), body
    # The arrival is published by the copy's own warp (an arrive on the g2s
    # group), and it is the *only* publication: there is no named-barrier
    # arrival for the copy to double-count. The op form spelled that as the
    # kNoBarrierArrive sentinel inside the helper's argument list; the IR form
    # has no helper to tell.
    assert "arrive_on_g2s_fence()" in body, body
    assert "((uint)4294967295)" not in body, body
    assert "fence_g2s_arrive(" not in body, body
    # The free edge is on a slot too: the MMA's release arrives on the TC fence
    # and the producer waits the phase parity of the generation it reuses.
    assert "arrive_on_tc_fence" in body, body
    assert "fence_tc_arrive(" not in body, body
    assert "sync_wait(" not in body, body


# ===========================================================================
# §3  The barrier array lives inside the shared window
# ===========================================================================


@pytest.mark.parametrize(("stages", "slot_count"), [(2, 8), (3, 12)])
def test_prologue_and_epilogue_reuse_versioned_slots(stages, slot_count):
    """The array contains two operands x two edges x ``stages`` slots."""
    src = source(stages)
    assert f"mbarrier_mem[{slot_count}]" in src, src
    literal_ids = {int(slot) for slot in re.findall(r"(?<!_)mbarrier\[(\d+)\]", src)}
    assert literal_ids
    assert max(literal_ids) == slot_count - 1, literal_ids


def test_barrier_array_is_declared_before_the_shared_buffer():
    """Declared after the dyn buffer it lands outside the CTA's shared window.

    The dyn buffer is the whole window (the tiles fill it exactly), so anything
    declared after it is not mapped: the barrier state stops updating and every
    wait spins forever. Order is the whole assertion -- this is the regression
    guard for that hang.
    """
    src = source()
    barrier = _line_of(src, "mbarrier_mem[")
    window = _line_of(src, "buf_dyn_shmem[")
    assert barrier < window, f"mbarrier_mem at {barrier}, buf_dyn_shmem at {window}\n{src}"
    # ... and the pointer that everything uses is derived from it right there.
    assert _line_of(src, "reinterpret_cast<Barrier*>") == barrier + 1, src
