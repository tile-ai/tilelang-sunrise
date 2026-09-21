"""stcuv2 tcgen05 thread-sync fence 测试。

T.tcgen05_before_thread_sync / T.tcgen05_after_thread_sync 是 S3 专属的
tensor-core fence 原语,lower 为 tang::ptx::fence_tc<fg>()。这里只做 lowering
断言(对生成源码正则匹配),不跑 ISS。

两个原语在 TANG 硬件上映射到同一条 ``tang::ptx::fence_tc<fg>()`` 指令,
区别仅在于 IR 中的**位置**(before/after sync),详见
docs/tang_tcgen05_fence_semantics.md。
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


# ===========================================================================
# Shared helpers
# ===========================================================================


def _lower_src(func, target=STCUV2_TARGET):
    with tvm.target.Target(target):
        return tilelang.lower(func, target=target).kernel_source


def _assert_in_ir(txt: str, *patterns: str):
    """Assert each regex pattern appears in the IR text."""
    for pat in patterns:
        assert re.search(pat, txt), f"Expected '{pat}' NOT found in IR:\n{txt[:1024]}"


def _assert_not_in_ir(txt: str, *patterns: str):
    """Assert each regex pattern does NOT appear in the IR text."""
    for pat in patterns:
        assert not re.search(pat, txt), f"Unexpected '{pat}' found in IR:\n{txt[:1024]}"


# ===========================================================================
# §0  API existence
# ===========================================================================


def test_fence_apis_exist():
    """tcgen05 fence API 在 T 命名空间中可用。"""
    assert callable(T.tcgen05_before_thread_sync)
    assert callable(T.tcgen05_after_thread_sync)


# ===========================================================================
# §1  Lowering — both primitives together
# ===========================================================================


def _make_fence_kernel(fence_group=0):
    """Kernel with both before-MMA and after-MMA fences (the canonical pair)."""
    # K=64: fp16 K=128 needs a 256-byte operand row, over the 128-byte MMA cap.
    M = N = 128
    K = 64

    @T.prim_func
    def kernel(A: T.Tensor((M, K), T.float16), B: T.Tensor((N, K), T.float16), C: T.Tensor((M, N), T.float32)):
        with T.Kernel(1, 1, threads=128):
            A_s = T.alloc_shared((M, K), T.float16)
            B_s = T.alloc_shared((N, K), T.float16)
            C_t = T.alloc_tmem((M, N), T.float32)
            T.copy(A, A_s)
            T.copy(B, B_s)
            T.tcgen05_before_thread_sync(fence_group)
            T.gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=True)
            T.tcgen05_after_thread_sync(fence_group)
            T.copy(C_t, C)

    return kernel


def test_fence_group_0_emits_fence_tc0():
    """fence_group=0 lowers to fence_tc<0>()."""
    src = _lower_src(_make_fence_kernel(0))
    _assert_in_ir(src, r"fence_tc<0>\(\)")
    _assert_not_in_ir(src, r"fence_tc<1>\(\)")


def test_fence_group_1_emits_fence_tc1():
    """fence_group=1 lowers to fence_tc<1>() for the user's two fences.

    The GEMM itself also emits an implicit group-0 completion fence after the
    MMA (see docs/tang_tcgen05_fence_semantics.md), so fence_tc<0> legitimately
    appears once; only the user's fences must be group 1.
    """
    src = _lower_src(_make_fence_kernel(1))
    count1 = len(re.findall(r"fence_tc<1>\(\)", src))
    assert count1 == 2, f"expected 2 fence_tc<1> calls, got {count1}"


def test_fence_count_is_exactly_2():
    """When both before+after are used, exactly two user fence_tc calls appear.

    Uses fence_group=1 so the user's two fences are countable independently of
    the implicit group-0 completion fence the GEMM emits after the MMA.
    """
    src = _lower_src(_make_fence_kernel(1))
    count = len(re.findall(r"fence_tc<1>\(\)", src))
    assert count == 2, f"expected 2 fence_tc<1> calls, got {count}"


def test_fence_ordering_before_mma_after():
    """fence_tc appears before and after the MMA in the correct order:

    fence_tc → ... → gemm/mma → ... → fence_tc
    """
    src = _lower_src(_make_fence_kernel(0))
    assert re.search(r"fence_tc<\d>\(\)[\s\S]*?(?:mma|gemm)[\s\S]*?fence_tc<\d>\(\)", src), "expected fence_tc → mma → fence_tc ordering"


def test_default_fence_group_is_0():
    """Calling before_thread_sync / after_thread_sync without arguments
    defaults to fence_group=0."""

    @T.prim_func
    def kernel(A: T.Tensor((128, 64), T.float16), B: T.Tensor((128, 64), T.float16), C: T.Tensor((128, 128), T.float32)):
        with T.Kernel(1, 1, threads=128):
            A_s = T.alloc_shared((128, 64), T.float16)
            B_s = T.alloc_shared((128, 64), T.float16)
            C_t = T.alloc_tmem((128, 128), T.float32)
            T.copy(A, A_s)
            T.copy(B, B_s)
            T.tcgen05_before_thread_sync()  # no arg → defaults to 0
            T.gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=True)
            T.tcgen05_after_thread_sync()  # no arg → defaults to 0
            T.copy(C_t, C)

    src = _lower_src(kernel)
    _assert_in_ir(src, r"fence_tc<0>\(\)")
    _assert_not_in_ir(src, r"fence_tc<1>\(\)")


# ===========================================================================
# §2  Each primitive in isolation
# ===========================================================================


def test_before_only_emits_single_fence():
    """tcgen05_before_thread_sync alone emits exactly one fence_tc of its group.

    Uses fence_group=1 so the user's fence is countable independently of the
    implicit group-0 completion fence the GEMM emits after the MMA.
    """

    @T.prim_func
    def kernel(A: T.Tensor((128, 64), T.float16), B: T.Tensor((128, 64), T.float16), C: T.Tensor((128, 128), T.float32)):
        with T.Kernel(1, 1, threads=128):
            A_s = T.alloc_shared((128, 64), T.float16)
            B_s = T.alloc_shared((128, 64), T.float16)
            C_t = T.alloc_tmem((128, 128), T.float32)
            T.copy(A, A_s)
            T.copy(B, B_s)
            T.tcgen05_before_thread_sync(1)
            T.gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=True)
            T.copy(C_t, C)

    src = _lower_src(kernel)
    _assert_in_ir(src, r"fence_tc<1>\(\)")
    count = len(re.findall(r"fence_tc<1>\(\)", src))
    assert count == 1, f"expected exactly 1 fence_tc<1>, got {count}"


def test_after_only_emits_single_fence():
    """tcgen05_after_thread_sync alone emits exactly one fence_tc of its group.

    Uses fence_group=1 so the user's fence is countable independently of the
    implicit group-0 completion fence the GEMM emits after the MMA.
    """

    @T.prim_func
    def kernel(A: T.Tensor((128, 64), T.float16), B: T.Tensor((128, 64), T.float16), C: T.Tensor((128, 128), T.float32)):
        with T.Kernel(1, 1, threads=128):
            A_s = T.alloc_shared((128, 64), T.float16)
            B_s = T.alloc_shared((128, 64), T.float16)
            C_t = T.alloc_tmem((128, 128), T.float32)
            T.copy(A, A_s)
            T.copy(B, B_s)
            T.gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=True)
            T.tcgen05_after_thread_sync(1)
            T.copy(C_t, C)

    src = _lower_src(kernel)
    _assert_in_ir(src, r"fence_tc<1>\(\)")
    count = len(re.findall(r"fence_tc<1>\(\)", src))
    assert count == 1, f"expected exactly 1 fence_tc<1>, got {count}"


# ===========================================================================
# §3  Negative tests — invalid fence_group
# ===========================================================================


def test_fence_group_out_of_range_rejected():
    """fence_group=2 raises AssertionError.

    NOTE: ``@T.prim_func`` 在装饰阶段 eager trace, 异常在函数定义时抛出,
    故 ``pytest.raises`` 须包裹装饰语句本身。
    """
    with pytest.raises(AssertionError, match="fence_group must be 0 or 1"):

        @T.prim_func
        def kernel(A: T.Tensor((128, 64), T.float16), B: T.Tensor((128, 64), T.float16), C: T.Tensor((128, 128), T.float32)):
            with T.Kernel(1, 1, threads=128):
                A_s = T.alloc_shared((128, 64), T.float16)
                B_s = T.alloc_shared((128, 64), T.float16)
                C_t = T.alloc_tmem((128, 128), T.float32)
                T.copy(A, A_s)
                T.copy(B, B_s)
                T.tcgen05_before_thread_sync(2)  # invalid
                T.gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=True)
                T.copy(C_t, C)


# ===========================================================================
# §4  S3(stcuv2) exclusive: fence_tc 只应在 stcuv2 上出现
# ===========================================================================


def test_fence_only_on_stcuv2():
    """fence_tc is an S3 (stcuv2) exclusive feature — lowering on stcuv2
    must emit it in the generated kernel source."""

    @T.prim_func
    def kernel(A: T.Tensor((128, 64), T.float16), B: T.Tensor((128, 64), T.float16), C: T.Tensor((128, 128), T.float32)):
        with T.Kernel(1, 1, threads=128):
            A_s = T.alloc_shared((128, 64), T.float16)
            B_s = T.alloc_shared((128, 64), T.float16)
            C_t = T.alloc_tmem((128, 128), T.float32)
            T.copy(A, A_s)
            T.copy(B, B_s)
            T.tcgen05_before_thread_sync(0)
            T.gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=True)
            T.tcgen05_after_thread_sync(0)
            T.copy(C_t, C)

    src = _lower_src(kernel, target=STCUV2_TARGET)
    _assert_in_ir(src, r"fence_tc<0>\(\)")


# ===========================================================================
# §5  ISS numeric — cross-warp producer→consumer copy with fence_tc
# ===========================================================================
#
# 验证跨 warp 场景: warp 0 将 fragment 写入 TMEM(stt)→ fence_tc→
# sync_arrive; warp 2 sync_wait→ 从 TMEM 读回(ldt)→ 写 global。
# fence_tc 保证 producer warp 的 stt 对 consumer warp 的 ldt 可见。


def _make_cross_warp_copy_kernel(M: int, N: int, producer_warp: int = 0, consumer_warp: int = 2, barrier_id: int = 1):
    """Cross-warp fragment→TMEM→fragment copy through tcgen05 primitives.

    Producer warp:  global → fragment → stt(TMEM) → fence_tc → sync_arrive
    Consumer warp:  sync_wait → ldt(TMEM→fragment) → global
    """

    @T.prim_func
    def kernel(A: T.Tensor((M, N), T.float32), B: T.Tensor((M, N), T.float32)):
        with T.Kernel(1, 1, threads=128):
            C_t = T.alloc_tmem((M, N), T.float32)
            C_f_prod = T.alloc_fragment((M, N), T.float32)
            C_f_cons = T.alloc_fragment((M, N), T.float32)

            # Producer: warp 0 writes data into TMEM
            with T.ws(producer_warp, warp_group_size=32):
                T.copy(A, C_f_prod)
                T.tcgen05_st(C_t, C_f_prod)
                T.tcgen05_before_thread_sync(0)
                T.tcgen05_sync_arrive(barrier_id)

            # Consumer: warp 2 reads data from TMEM
            with T.ws(consumer_warp, warp_group_size=32):
                T.tcgen05_sync_wait(barrier_id, 1, 1)
                T.tcgen05_after_thread_sync(0)
                T.tcgen05_ld(C_f_cons, C_t)
                T.copy(C_f_cons, B)

    return kernel


@pytest.mark.parametrize("M,N", [(128, 64), (64, 64), (128, 128)])
def test_cross_warp_copy_numeric(M, N):
    """Cross-warp st→fence_tc→ld copy must be lossless on ISS."""
    kernel = _make_cross_warp_copy_kernel(M, N)
    torch.manual_seed(42)
    A = torch.randn(M, N, dtype=torch.float32)
    jit = JITKernel(kernel, out_idx=[-1], target=STCUV2_TARGET, execution_backend="simulator", verbose=False)
    B = jit(A.ptpu()).cpu().float()
    rel = (B - A).abs().max().item() / (A.abs().max().item() + 1e-6)
    assert rel < 1e-2, f"cross-warp copy rel_err too large: {rel:.3e}"


# ===========================================================================
# §6  Negative: cross-warp copy WITHOUT fence_tc — xfail until HW/cmodel
# ===========================================================================
#
# 去掉 fence_tc 后, producer warp 的 tcgen05_st 可能尚未完成时 consumer
# warp 的 tcgen05_ld 就已执行, 读到未初始化的 TMEM 数据。
#
# 当前 ISS simulator 对 tcgen05 是同步建模, 无法暴露这个 race, 故标记为
# xfail(reason="ISS synchronous, HW/cmodel expected to fail").
# 等 cycle-accurate cmodel 或 S3 硬件就绪后, 去掉 xfail 标记即可验证
# fence_tc 的必要性。


def _make_cross_warp_copy_no_fence_kernel(M: int, N: int):
    """Same as _make_cross_warp_copy_kernel but WITHOUT fence_tc calls."""

    @T.prim_func
    def kernel(A: T.Tensor((M, N), T.float32), B: T.Tensor((M, N), T.float32)):
        with T.Kernel(1, 1, threads=128):
            C_t = T.alloc_tmem((M, N), T.float32)
            C_f_prod = T.alloc_fragment((M, N), T.float32)
            C_f_cons = T.alloc_fragment((M, N), T.float32)

            with T.ws(0, warp_group_size=32):
                T.copy(A, C_f_prod)
                T.tcgen05_st(C_t, C_f_prod)
                # NOTE: NO fence_tc here — deliberately omitted
                T.tcgen05_sync_arrive(1)

            with T.ws(2, warp_group_size=32):
                T.tcgen05_sync_wait(1, 1, 1)
                # NOTE: NO fence_tc here — deliberately omitted
                T.tcgen05_ld(C_f_cons, C_t)
                T.copy(C_f_cons, B)

    return kernel


@pytest.mark.xfail(reason="ISS synchronous; HW/cmodel expected to fail (fence_tc absent)")
@pytest.mark.parametrize("M,N", [(128, 64), (64, 64)])
def test_cross_warp_copy_no_fence(M, N):
    """Cross-warp st→ld WITHOUT fence_tc — expected to fail on real HW.

    On ISS the stt completes synchronously before the other warp runs, so
    the test passes.  On cycle-accurate cmodel or S3 silicon, the consumer
    warp is expected to read stale TMEM, causing a numerical mismatch.
    Remove the xfail marker when running on those targets.
    """
    kernel = _make_cross_warp_copy_no_fence_kernel(M, N)
    torch.manual_seed(42)
    A = torch.randn(M, N, dtype=torch.float32)
    jit = JITKernel(kernel, out_idx=[-1], target=STCUV2_TARGET, execution_backend="simulator", verbose=False)
    B = jit(A.ptpu()).cpu().float()
    rel = (B - A).abs().max().item() / (A.abs().max().item() + 1e-6)
    assert rel < 1e-2, f"cross-warp no-fence copy rel_err too large: {rel:.3e}"


if __name__ == "__main__":
    tilelang.testing.main()
