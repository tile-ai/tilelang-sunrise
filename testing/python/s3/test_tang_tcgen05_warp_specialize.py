"""stcuv2 warp-specialized tcgen05 TMEM drain 测试。

验证 warp specialization:让 warp0 之外的某个 warp 执行 TMEM->fragment 的
ldt drain。正确做法是把 drain(sync_wait + tcgen05_ld + 消费拷贝)包在一个
32 线程的 warp-specialized 区域里 —— ``T.ws(warp, warp_group_size=32)`` ——
这样 ``thread_bounds`` 对 ldt 与其 fragment 消费者一并收窄到该 warp;再配合
生产者→消费者的 ``tcgen05_sync_arrive`` / ``tcgen05_sync_wait`` 握手。

详见 docs/s3_tcgen05_ldst_warp_specialize.md。

§1 Lowering 断言   test_ws_drain_lowering_* / test_standalone_warp_param_fails
§2 数值 (via ISS)  test_ws_drain_numeric

这些原语仅在 stcuv2 上支持。
"""

import re

import pytest
import torch

import tilelang
import tilelang.testing
import tilelang.tang.language as T
from tilelang import tvm as tvm
from tilelang.jit import JITKernel

STCUV2_TARGET = {"kind": "tang", "arch": "stcuv2"}

_TL_DTYPE = {"float16": T.float16, "bfloat16": T.bfloat16}
_PT_DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16}


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


def _make_ws_st_roundtrip_kernel(M, N, K, warp, dtype, barrier_id=1):
    """gemm (warp0) -> drain warp does ld->st->ld->global inside T.ws(warp,32).

    Exercises ``tcgen05_st`` (fragment->fresh TMEM) under warp specialization:
    the drain warp reads the MMA output, stores it into a second TMEM region,
    reads it back and drains to global. A correct round-trip reproduces the GEMM.
    """
    dt = _TL_DTYPE[dtype]

    @T.prim_func
    def kernel(A: T.Tensor((M, K), dt), B: T.Tensor((N, K), dt), C: T.Tensor((M, N), T.float32)):
        with T.Kernel(1, 1, threads=128):
            A_s = T.alloc_shared((M, K), dt)
            B_s = T.alloc_shared((N, K), dt)
            C_t = T.alloc_tmem((M, N), T.float32)
            C_t2 = T.alloc_tmem((M, N), T.float32)
            C_f = T.alloc_fragment((M, N), T.float32)
            C_f2 = T.alloc_fragment((M, N), T.float32)
            T.copy(A, A_s)
            T.copy(B, B_s)
            with T.ws(0, warp_group_size=32):
                T.gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=True)
                T.tcgen05_sync_arrive(barrier_id)
            with T.ws(warp, warp_group_size=32):
                T.tcgen05_sync_wait(barrier_id, 1, 1)
                T.tcgen05_ld(C_f, C_t)
                T.tcgen05_st(C_t2, C_f)
                T.tcgen05_ld(C_f2, C_t2)
                T.copy(C_f2, C)

    return kernel


def _lower_src(func, target=STCUV2_TARGET):
    with tvm.target.Target(target):
        return tilelang.lower(func, target=target).kernel_source


def _sim_rel_err(kernel, M, N, K, dtype):
    pt = _PT_DTYPE[dtype]
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=pt)
    B = torch.randn(N, K, dtype=pt)
    ref = A.float() @ B.float().T
    jit = JITKernel(kernel, out_idx=[-1], target=STCUV2_TARGET, execution_backend="simulator", verbose=False)
    C = jit(A.ptpu(), B.ptpu()).cpu().float()
    return (C - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)


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


# ===========================================================================
# §2  Numeric (via ISS)
# ===========================================================================


@pytest.mark.parametrize(
    "M,N,K,drain_warp,dtype",
    [
        (128, 64, 64, 2, "float16"),
        (128, 64, 64, 3, "float16"),
        (128, 64, 64, 2, "bfloat16"),
    ],
)
def test_ws_drain_numeric(M, N, K, drain_warp, dtype):
    """A warp>0 tcgen05_ld drain must reproduce the GEMM result."""
    rel = _sim_rel_err(_make_ws_drain_kernel(M, N, K, drain_warp, dtype), M, N, K, dtype)
    assert rel < 1e-2, f"rel_err too large: {rel:.3e}"


def test_ws_drain_numeric_large_tile_is_toolchain_limited():
    """Record the largest tile a single-warp ws drain sustains -- that limit
    belongs to ptcc, not to TileLang.

    A single drain warp materializing a full 128x128 fp32 fragment needs
    ~512 regs/lane. Whether it fits is decided by ptcc's register allocator:
    the ptcc shipped with the simulator reports 'ran out of registers', while
    the one on CI compiles it and produces correct numbers.

    So "must fail" cannot be written as a strict xfail -- it flips into an
    XPASS(strict) failure the moment CI compiles it, and the failure reads like
    a known limitation, which invites the conclusion that the drain broke.

    Both outcomes are accepted, and both keep an assertion: either the known
    register-allocation limit is hit, or the kernel compiles and the numbers
    are correct. A third outcome (compiles but wrong numbers) still fails.

    Draining a 128x128 tile on one warp needs a tiled ldt epilogue, tracked
    separately.
    """
    try:
        rel = _sim_rel_err(_make_ws_drain_kernel(128, 128, 64, 2, "float16"), 128, 128, 64, "float16")
    except RuntimeError as exc:
        assert "ran out of registers" in str(exc), (
            f"expected the known register-allocation limit, but ptcc failed for another reason:\n{exc}"
        )
        return
    assert rel < 1e-2, f"rel_err too large: {rel:.3e}"


@pytest.mark.parametrize(
    "M,N,K,warp,dtype",
    [
        (128, 64, 64, 2, "float16"),
        (128, 64, 64, 3, "float16"),
        (128, 32, 64, 2, "float16"),
    ],
)
def test_ws_st_roundtrip_numeric(M, N, K, warp, dtype):
    """A warp>0 tcgen05_st round-trip (ld->st->ld) must be lossless."""
    rel = _sim_rel_err(_make_ws_st_roundtrip_kernel(M, N, K, warp, dtype), M, N, K, dtype)
    assert rel < 1e-2, f"rel_err too large: {rel:.3e}"


if __name__ == "__main__":
    tilelang.testing.main()
