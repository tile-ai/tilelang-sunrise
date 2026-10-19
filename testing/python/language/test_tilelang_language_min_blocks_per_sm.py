"""Tests for T.annotate_min_blocks_per_sm → __launch_bounds__(maxThreads, minBlocks)."""

import pytest

import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


@tl.jit(out_idx=[2])
def _kernel_min_blocks_per_sm():
    @T.prim_func
    def main(
        A: T.Tensor((128, 128), "float32"),
        B: T.Tensor((128, 128), "float32"),
        C: T.Tensor((128, 128), "float32"),
    ):
        with T.Kernel(128, threads=128) as bx:
            T.annotate_min_blocks_per_sm(2)
            for i in T.serial(128):
                C[bx, i] = A[bx, i] + B[bx, i]

    return main


@tl.jit(out_idx=[2])
def _kernel_default_min_blocks_per_sm():
    @T.prim_func
    def main(
        A: T.Tensor((128, 128), "float32"),
        B: T.Tensor((128, 128), "float32"),
        C: T.Tensor((128, 128), "float32"),
    ):
        with T.Kernel(128, threads=128) as bx:
            for i in T.serial(128):
                C[bx, i] = A[bx, i] + B[bx, i]

    return main


def test_annotate_min_blocks_per_sm_launch_bounds():
    """Codegen should emit the second __launch_bounds__ argument from the annotation."""
    src = _kernel_min_blocks_per_sm.get_kernel_source()
    assert "__launch_bounds__(128, 2)" in src


def test_default_min_blocks_per_sm_launch_bounds():
    """Codegen should retain one resident block as the default."""
    src = _kernel_default_min_blocks_per_sm.get_kernel_source()
    assert "__launch_bounds__(128, 1)" in src


@pytest.mark.parametrize("min_blocks", [None, 2])
def test_tang_min_blocks_per_sm_source(min_blocks):
    """Explicit TANG source avoids default-target or compiled JIT cache ambiguity."""
    if tvm.get_global_func("target.build.tilelang_tang_without_compile", allow_missing=True) is None:
        pytest.skip("TileLang was built without the TANG code generator")

    @T.prim_func
    def main(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=128):
            if min_blocks is not None:
                T.annotate_min_blocks_per_sm(min_blocks)
            tid = T.get_thread_binding()
            B[tid] = A[tid]

    source = tl.lower(main, target={"kind": "tang", "arch": "stcu"}).kernel_source
    assert "#include <tang.h>" in source
    expected = 1 if min_blocks is None else min_blocks
    assert f"__launch_bounds__(128, {expected})" in source


if __name__ == "__main__":
    tilelang.testing.main()
