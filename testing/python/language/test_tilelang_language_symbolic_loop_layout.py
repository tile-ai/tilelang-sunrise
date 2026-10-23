"""Injectivity proofs for loop layouts over symbolic iteration spaces.

A T.Parallel loop whose extent is symbolic is partitioned with a padded tail,
so its loop layout is injective but not bijective and the domain cannot be
enumerated. #2719 replaced the unchecked NoCheck fallback with a real proof,
which could not handle a symbolic stride; the left-inverse proof restores it.
"""

import pytest
import torch
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.layout import Layout
from tilelang.utils.device import get_current_device


@pytest.fixture(
    params=[
        pytest.param("cuda", marks=tilelang.testing.requires_cuda.marks()),
        pytest.param("tang", marks=pytest.mark.skipif(not tilelang.testing.is_tang_target(), reason="Requires TANG target")),
    ]
)
def execution_target(request):
    return request.param


@pytest.mark.parametrize("dynamic_axis", [0, 1])
def test_symbolic_parallel_loop_layout(dynamic_axis, execution_target):
    n = T.dynamic("n")
    shape = (n, 16) if dynamic_axis == 0 else (16, n)

    @T.prim_func
    def kernel(A: T.Tensor(shape, "int32"), B: T.Tensor(shape, "int32")):
        with T.Kernel(1, threads=128):
            for i, j in T.Parallel(*shape):
                B[i, j] = A[i, j]

    target = "cuda" if execution_target == "cuda" else tilelang.env.get_default_target()
    device = "cuda" if execution_target == "cuda" else get_current_device()
    compiled = tilelang.compile(kernel, out_idx=[1], target=target)
    for length in (7, 129):
        a = torch.randint(-10, 11, (length, 16) if dynamic_axis == 0 else (16, length), device=device, dtype=torch.int32)
        torch.testing.assert_close(compiled(a).cpu(), a.cpu(), rtol=0, atol=0)


def test_symbolic_noninjective_shared_layout_rejected(execution_target):
    # Layout inference checks injectivity before the symbolic-extent rejection
    # in makeBufferWithLayout, so a non-injective symbolic map must still fail
    # on injectivity: the left-inverse proof must not accept it.
    n = T.dynamic("n")

    @T.prim_func
    def kernel(A: T.Tensor((16, n), "int32")):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((16, n), "int32")
            # Both coordinates occur, but adjacent rows overlap.
            T.annotate_layout({shared: Layout((16, n), lambda i, j: i * (n - 1) + j)})
            for i, j in T.Parallel(16, n):
                shared[i, j] = A[i, j]
            for i, j in T.Parallel(16, n):
                A[i, j] = shared[i, j]

    target = "cuda" if execution_target == "cuda" else tilelang.env.get_default_target()
    with pytest.raises(ValueError, match="must be injective"):
        tilelang.compile(kernel, target=target)


if __name__ == "__main__":
    tilelang.testing.main()
