"""stcuv2 warp-specialized tcgen05 TMEM drain 测试。

验证 warp specialization:让 warp0 之外的某个 warp 执行 TMEM->fragment 的
ldt drain。正确做法是把 drain(sync_wait + tcgen05_ld + 消费拷贝)包在一个
32 线程的 warp-specialized 区域里 —— ``T.ws(warp, warp_group_size=32)`` ——
这样 ``thread_bounds`` 对 ldt 与其 fragment 消费者一并收窄到该 warp;再配合
生产者→消费者的 ``tcgen05_sync_arrive`` / ``tcgen05_sync_wait`` 握手。

§1 Lowering 断言   test_ws_drain_lowering_* / test_standalone_warp_param_fails

这些原语仅在 stcuv2 上支持。
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


def _make_ws_drain_kernel(M, N, K, drain_warp, dtype, barrier_id=1):
    """gemm (warp0) -> sync handshake -> warpN ldt drain -> global.

    Producer (warp 0, whose MMA fences internally) signals ``barrier_id``; the
    drain warp waits, then runs ``tcgen05_ld`` + the fragment->global copy inside
    a 32-thread warp-specialized region.
    """
    dt = _TL_DTYPE[dtype]

    @T.prim_func
    def kernel(A: T.Tensor((M, K), dt), B: T.Tensor((N, K), dt), C: T.Tensor((M, N), T.float32)):
        with T.Kernel(1, 1, threads=128):
            A_s = T.alloc_shared((M, K), dt)
            B_s = T.alloc_shared((N, K), dt)
            C_t = T.alloc_tmem((M, N), T.float32)
            C_f = T.alloc_fragment((M, N), T.float32)
            T.copy(A, A_s)
            T.copy(B, B_s)
            with T.ws(0, warp_group_size=32):
                T.gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=True)
                T.tcgen05_sync_arrive(barrier_id)
            with T.ws(drain_warp, warp_group_size=32):
                T.tcgen05_sync_wait(barrier_id, 1, 1)
                T.tcgen05_ld(C_f, C_t)
                T.copy(C_f, C)

    return kernel


def _lower_src(func, target=STCUV2_TARGET):
    with tvm.target.Target(target):
        return tilelang.lower(func, target=target).kernel_source


# ===========================================================================
# §1  Lowering assertions
# ===========================================================================


def test_ws_drain_lowering_emits_sync_and_warp_guard():
    """warp-2 drain emits ldt + sync_arrive/sync_wait + a [64,96) warp guard."""
    src = _lower_src(_make_ws_drain_kernel(128, 64, 64, 2, "float16"))
    assert re.search(r"ldt_16x256b_x8\(", src), "expected ldt_16x256b_x8"
    assert "sync_arrive(1)" in src, "expected producer sync_arrive"
    assert re.search(r"sync_wait\(1,", src), "expected consumer sync_wait"
    # drain restricted to warp 2 (threads [64, 96))
    assert re.search(r"64 <= \(\(int\)threadIdx\.x\)", src) or re.search(r"\(\(int\)threadIdx\.x\) < 96", src), "expected [64,96) guard"


@pytest.mark.parametrize("drain_warp,lo,hi", [(1, 32, 64), (3, 96, 128)])
def test_ws_drain_lowering_warp_range(drain_warp, lo, hi):
    """Each drain warp restricts the ldt/consumer to its own 32-lane range."""
    src = _lower_src(_make_ws_drain_kernel(128, 64, 64, drain_warp, "float16"))
    assert re.search(r"ldt_16x256b_x8\(", src)
    assert (str(lo) in src) and (str(hi) in src)


def test_warp0_drain_no_ws_lowering():
    """A plain warp-0 drain (no ws region) still lowers to a [0,32) ldt."""
    src = _lower_src(_make_ws_drain_kernel(128, 64, 64, 0, "float16"))
    assert re.search(r"ldt_16x256b_x8\(", src)


def _make_fused_drain_in_ws_kernel(M, N, K, drain_warp, dtype):
    """MISUSE: the *fused* TMEM->global drain (`T.copy(C_tmem, C)`) placed inside
    a warp-specialized region. Under ``T.ws(...,32)`` the epilogue copy is laid
    out over a single 32-thread warp, so the loop collapses (128x128 -> extent
    512) and the fused ldt drain -- whose row-strip / lane geometry assumes a
    128-thread copy -- cannot be reconstructed. Lowering must reject this loudly
    and steer the user to the tcgen05_ld warp-specialization idiom.
    """
    dt = _TL_DTYPE[dtype]

    @T.prim_func
    def kernel(A: T.Tensor((M, K), dt), B: T.Tensor((N, K), dt), C: T.Tensor((M, N), T.float32)):
        with T.Kernel(1, 1, threads=128):
            A_s = T.alloc_shared((M, K), dt)
            B_s = T.alloc_shared((N, K), dt)
            C_t = T.alloc_tmem((M, N), T.float32)
            T.copy(A, A_s)
            T.copy(B, B_s)
            T.gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=True)
            with T.ws(drain_warp, warp_group_size=32):
                T.copy(C_t, C)

    return kernel


def test_fused_drain_in_ws_fails_loudly():
    """Guard: `T.copy(C_tmem, C)` inside `T.ws(...)` is rejected with a clear msg.

    Silently lowering it would emit a garbage drain (the 32-thread copy collapses
    the loop geometry the fused ldt drain depends on). The guard must raise and
    steer the user to the explicit single-warp writeback idiom.
    """
    with pytest.raises(Exception) as excinfo:
        _lower_src(_make_fused_drain_in_ws_kernel(128, 128, 64, 2, "float16"))
    msg = str(excinfo.value)
    assert "warp-specialized" in msg or "T.ws" in msg, f"guard message should mention the T.ws misuse, got: {msg[:400]}"


def _make_ws_region_kernel(group, warp_group_size, threads=128):
    """The smallest kernel that has a ``T.ws`` region, for the range check.

    Deliberately not the drain kernel: the check is argument validation in
    ``T.ws`` itself and raises while the function is still being traced, so
    nothing about TMEM is involved.
    """

    @T.prim_func
    def kernel(A: T.Tensor((threads,), T.float16), C: T.Tensor((threads,), T.float16)):
        with T.Kernel(1, threads=threads):
            A_s = T.alloc_shared((threads,), T.float16)
            with T.ws(group, warp_group_size=warp_group_size):
                T.copy(A, A_s)
            T.copy(A_s, C)

    return kernel


@pytest.mark.parametrize("group,size", [(4, 32), (2, 64), (1, 128)])
def test_ws_group_outside_the_block_is_refused(group, size):
    """A region the kernel never launches must be named at the T.ws, not later.

    The threads simply never enter it, so whatever it was supposed to produce is
    missing and the first pass to care reports something unrelated: an MMA in
    ``T.ws(2, warp_group_size=64)`` of a 128-thread kernel used to surface as
    ``The layout of tmem buffer C_t is not defined in the layout map`` from
    lower_s3_shared_tmem.cc, which sends the reader to the wrong file.
    """
    with pytest.raises(ValueError) as excinfo:
        _lower_src(_make_ws_region_kernel(group, size))
    msg = str(excinfo.value)
    assert "covers threads" in msg and "128" in msg, msg


@pytest.mark.parametrize("group,size", [(3, 32), (1, 64), (0, 128)])
def test_ws_group_at_the_last_slot_is_allowed(group, size):
    """The boundary the check must not eat: the block's final group."""
    assert _lower_src(_make_ws_region_kernel(group, size))


if __name__ == "__main__":
    tilelang.testing.main()
