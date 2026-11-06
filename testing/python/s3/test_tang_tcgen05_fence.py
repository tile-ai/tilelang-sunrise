"""stcuv2 tcgen05 thread-sync fence 测试。

T.tcgen05_before_thread_sync / T.tcgen05_after_thread_sync 是 S3 专属的
tensor-core fence 原语,lower 为 tang::ptx::fence_tc<fg>()。这里只做 lowering
断言(对生成源码正则匹配)。

两个原语在 TANG 硬件上映射到同一条 ``tang::ptx::fence_tc<fg>()`` 指令,
区别仅在于 IR 中的**位置**(before/after sync),详见
docs/tang_tcgen05_fence_semantics.md。
"""

import re

import pytest

import tilelang
import tilelang.testing
import tilelang.tang.language as T
from tilelang import tvm as tvm

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
# §4  S3(stcuv2) exclusive: fence_tc should appear on stcuv2 only
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


if __name__ == "__main__":
    tilelang.testing.main()
