"""Regression test: consecutive kernel calls must return independent output tensors.

Covers the _output_cache aliasing bug where caching and reusing the same
torch.Tensor across calls caused in-place overwrite of previously-returned
results.
"""

import torch
import tilelang
import tilelang.language as T
import tilelang.testing


def _get_device():
    if torch.ptpu.is_available():
        return "ptpu"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def test_output_tensor_independence():
    """Consecutive calls to the same kernel must return distinct output tensors."""
    N = 64
    device = _get_device()

    @tilelang.jit(out_idx=-1)
    def add_kernel(N, dtype=T.float32):
        @T.prim_func
        def main(
            A: T.Tensor((N,), dtype),
            B: T.Tensor((N,), dtype),
            C: T.Tensor((N,), dtype),
        ):
            with T.Kernel(T.ceildiv(N, 256), threads=256) as (bx,):
                for i in T.Parallel(256):
                    idx = bx * 256 + i
                    if idx < N:
                        C[idx] = A[idx] + B[idx]

        return main

    a = torch.randn(N, device=device)
    b = torch.randn(N, device=device)
    x = torch.full((N,), 100.0, device=device)
    y = torch.full((N,), 200.0, device=device)

    # Compile: non-tensor params first, tensor params second
    kernel = add_kernel(N)

    # Warmup
    _ = kernel(a, b)

    r1 = kernel(a, b)
    r2 = kernel(x, y)

    # r1 and r2 must be independent tensors (different storage)
    assert r1.data_ptr() != r2.data_ptr(), "output tensors alias across calls: kernel() reuses the same tensor"

    # r1 must still hold a + b (not corrupted by the second call)
    tilelang.testing.torch_assert_close(r1, a + b, rtol=1e-5, atol=1e-5)

    # r2 must hold x + y
    tilelang.testing.torch_assert_close(r2, x + y, rtol=1e-5, atol=1e-5)


def test_output_independence_dynamic_shape():
    """Same check with T.dynamic() — exercises the tirx.Var resolution path in func()."""
    N = 64
    device = _get_device()

    @tilelang.jit(out_idx=-1)
    def add_kernel(N, block_size=256, dtype=T.float32):
        @T.prim_func
        def main(
            A: T.Tensor((N,), dtype),
            B: T.Tensor((N,), dtype),
            C: T.Tensor((N,), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as (bx,):
                for i in T.Parallel(block_size):
                    idx = bx * block_size + i
                    if idx < N:
                        C[idx] = A[idx] + B[idx]

        return main

    # T.dynamic("n") makes N a tirx.Var → triggers the dynamic shape path
    kernel = add_kernel(T.dynamic("n"))

    a = torch.randn(N, device=device)
    b = torch.randn(N, device=device)
    x = torch.full((N,), 100.0, device=device)
    y = torch.full((N,), 200.0, device=device)

    _ = kernel(a, b)

    r1 = kernel(a, b)
    r2 = kernel(x, y)

    assert r1.data_ptr() != r2.data_ptr(), "dynamic shape: output tensors alias across calls"
    tilelang.testing.torch_assert_close(r1, a + b, rtol=1e-5, atol=1e-5)
    tilelang.testing.torch_assert_close(r2, x + y, rtol=1e-5, atol=1e-5)


def test_non_contiguous_inputs_rejected():
    """Non-contiguous inputs are rejected with explicit error, not silently corrupted.

    Covers the require_contiguous=False from_dlpack path — DLPack correctly
    preserves strides of sliced tensors.  The runtime's packed ABI validation
    rejects strides that don't match the compile-time signature, preventing
    silent data corruption.
    """
    N = 64
    device = _get_device()

    @tilelang.jit(out_idx=-1)
    def add_kernel(N, block_size=256, dtype=T.float32):
        @T.prim_func
        def main(
            A: T.Tensor((N,), dtype),
            B: T.Tensor((N,), dtype),
            C: T.Tensor((N,), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as (bx,):
                for i in T.Parallel(block_size):
                    idx = bx * block_size + i
                    if idx < N:
                        C[idx] = A[idx] + B[idx]

        return main

    kernel = add_kernel(N)

    # Sliced inputs: shape (N,) but stride=2 (non-contiguous)
    a_long = torch.randn(N * 2, device=device)
    b_long = torch.randn(N * 2, device=device)
    a_sliced = a_long[::2]
    b_sliced = b_long[::2]

    assert not a_sliced.is_contiguous(), "sliced tensor must be non-contiguous"

    # Stride metadata is correctly passed via DLPack, but the runtime
    # rejects it with a clear error (no silent corruption).
    import pytest

    with pytest.raises(RuntimeError, match="packed ABI constraint"):
        _ = kernel(a_sliced, b_sliced)


if __name__ == "__main__":
    tilelang.testing.main()
