"""Non-contiguous tensor correctness tests for the tvm_ffi adapter.

These tests verify that the adapter correctly handles non-contiguous input tensors
(transpose and slice views). Since the generated device kernel flattens all buffer
accesses to 1-D linear offsets using compile-time strides (codegen_c.cc:806:
"Load from non-flat memory not supported"), the kernel cannot read DLPack strides
at runtime. Non-contiguous inputs would produce silently wrong results, so these
tests document the current behaviour and guard against regressions.

Test scenarios:
  - Non-contiguous transpose input (T().contiguous() path is correct,
    raw T (non-contiguous) triggers a from_dlpack RuntimeError on the contiguous check)
  - Non-contiguous slice input
  - verify that explicit .contiguous() on the input makes it pass
"""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.utils.device import get_current_device


# ---------------------------------------------------------------------------
# Simple elementwise-add kernel — minimal surface for non-contiguous testing
# ---------------------------------------------------------------------------
def elementwise_add(M, N, dtype=T.float32):
    """Return an (M,N) elementwise-add prim_func."""

    @T.prim_func
    def kernel(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(M * N // 256, threads=256) as bx:
            for i in T.Parallel(256):
                idx = bx * 256 + i
                if idx < M * N:
                    row = idx // N
                    col = idx % N
                    C[row, col] = A[row, col] + B[row, col]

    return kernel


def _allclose(actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-5, rtol: float = 1e-5) -> bool:
    return torch.allclose(actual.cpu(), expected.cpu().to(actual.dtype), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Non-contiguous input: transpose
# ---------------------------------------------------------------------------
def test_non_contiguous_transpose_input_errors():
    """Transposed (non-contiguous) input should be rejected by from_dlpack.

    The tvm_ffi adapter uses require_contiguous=True in from_dlpack because the
    generated kernel flattens accesses assuming row-major contiguous layout.
    Passing a transposed view of a tensor would silently produce wrong results
    if allowed, so from_dlpack enforces contiguity.
    """
    M, N = 256, 128
    kernel = tilelang.compile(elementwise_add(M, N), out_idx=[-1], execution_backend="tvm_ffi")
    device = get_current_device()

    A = torch.randn(M, N, dtype=torch.float32, device=device)
    B = torch.randn(M, N, dtype=torch.float32, device=device)

    # Transposed view — strides are swapped, tensor is non-contiguous
    A_t = A.T  # shape (N, M), non-contiguous
    assert not A_t.is_contiguous(), "Expected A.T to be non-contiguous"

    # Passing the transposed tensor with the original shape mismatches dimensions:
    # the kernel declares T.Tensor((M,N), ...) but receives shape (N,M).
    # This is a separate dimension mismatch, not a contiguous check.
    # What we actually care about: a valid-shape non-contiguous tensor.
    # Let's construct one by transposing twice; same logical layout but
    # non-contiguous.
    A_noncontig = A.T.contiguous().T  # shape (M,N), logical layout matches, non-contiguous
    assert A_noncontig.shape == (M, N)
    assert not A_noncontig.is_contiguous()

    # This should raise RuntimeError because require_contiguous=True rejects
    # non-contiguous memory.
    with pytest.raises(RuntimeError, match="contiguous|from_dlpack|strides|packed ABI"):
        kernel(A_noncontig, B)


def test_non_contiguous_transpose_contiguous_works():
    """Explicit .contiguous() on a transposed tensor makes it work."""
    M, N = 256, 128
    kernel = tilelang.compile(elementwise_add(M, N), out_idx=[-1], execution_backend="tvm_ffi")
    device = get_current_device()

    A = torch.randn(N, M, dtype=torch.float32, device=device)  # stored transposed
    B = torch.randn(M, N, dtype=torch.float32, device=device)

    # A has shape (N, M), .T gives (M, N) but non-contiguous
    # .contiguous() makes it a proper (M, N) tensor the kernel can consume
    A_contig = A.T.contiguous()
    assert A_contig.shape == (M, N)
    assert A_contig.is_contiguous()

    C = kernel(A_contig, B)
    expected = A_contig + B
    assert _allclose(C, expected), "Contiguous-transpose path produced wrong result"


# ---------------------------------------------------------------------------
# Non-contiguous input: slice
# ---------------------------------------------------------------------------
def test_non_contiguous_slice_input_errors():
    """Sliced (non-contiguous) input should be rejected.

    A slice of a larger tensor along a non-unit stride dimension is
    non-contiguous because the row stride is the larger tensor's stride.
    """
    M_small, N = 128, 128
    _kernel = tilelang.compile(elementwise_add(M_small, N), out_idx=[-1], execution_backend="tvm_ffi")
    device = get_current_device()

    # Create a larger tensor and slice a sub-region — non-contiguous
    M_large = 512
    A_large = torch.randn(M_large, N, dtype=torch.float32, device=device)
    A_slice = A_large[::2, :]  # every other row, shape (256, N)
    # This slice is larger than the kernel expects, so it's a dimension mismatch
    # instead of a contiguity issue. Let's create the right-sized slice.
    M_small = 64
    kernel2 = tilelang.compile(elementwise_add(M_small, N), out_idx=[-1], execution_backend="tvm_ffi")
    A_slice = A_large[:M_small:2, :]  # wrong shape — let me think again

    # Simplest: slice a tensor of the same shape with stride
    A = torch.randn(M_small * 2, N, dtype=torch.float32, device=device)
    B = torch.randn(M_small, N, dtype=torch.float32, device=device)
    A_slice = A[:M_small, :]  # contiguous slice (stride 1 in rows)
    assert A_slice.is_contiguous(), "contiguous slice should pass"

    A_stride = A[::2, :]  # non-contiguous slice
    assert not A_stride.is_contiguous(), "strided slice should be non-contiguous"
    assert A_stride.shape == (M_small, N)

    with pytest.raises(RuntimeError, match="contiguous|from_dlpack|strides|packed ABI"):
        kernel2(A_stride, B)


def test_non_contiguous_slice_contiguous_works():
    """Explicit .contiguous() on a sliced tensor makes it work."""
    M_small, N = 64, 128

    kernel = tilelang.compile(elementwise_add(M_small, N), out_idx=[-1], execution_backend="tvm_ffi")
    device = get_current_device()

    # Create a larger tensor and slice with stride
    A_large = torch.randn(M_small * 2, N, dtype=torch.float32, device=device)
    B = torch.randn(M_small, N, dtype=torch.float32, device=device)

    A_slice = A_large[::2, :].contiguous()
    assert A_slice.is_contiguous()
    assert A_slice.shape == (M_small, N)

    C = kernel(A_slice, B)
    expected = A_slice + B
    assert _allclose(C, expected), "Contiguous-slice path produced wrong result"


# ---------------------------------------------------------------------------
# Non-contiguous input: torch.as_strided
# ---------------------------------------------------------------------------
def test_non_contiguous_as_strided_errors():
    """as_strided creates a non-contiguous view — should be rejected."""
    M, N = 128, 128
    kernel = tilelang.compile(elementwise_add(M, N), out_idx=[-1], execution_backend="tvm_ffi")
    device = get_current_device()

    A_base = torch.randn(M * 2, N, dtype=torch.float32, device=device)
    B = torch.randn(M, N, dtype=torch.float32, device=device)

    # Create a view with stride=2 in the first dim — non-contiguous
    A_strided = torch.as_strided(A_base, (M, N), (N * 2, 1))
    assert A_strided.shape == (M, N)
    assert not A_strided.is_contiguous()

    with pytest.raises(RuntimeError, match="contiguous|from_dlpack|strides|packed ABI"):
        kernel(A_strided, B)


# ---------------------------------------------------------------------------
# Symmetric transpose: input already in correct layout
# ---------------------------------------------------------------------------
def test_input_correctly_pre_transposed():
    """Pass a tensor whose logical layout matches kernel expectation.

    When the kernel declares A_shape = (N, M) (transposed in the kernel) and the
    user provides a tensor already in (N, M) layout (contiguous), it should work.
    This is the normal transpose-GEMM path.
    """
    M, N = 128, 64

    # Kernel that expects A in (N, M) transposed layout
    @T.prim_func
    def kernel(
        A: T.Tensor((N, M), T.float32),
        B: T.Tensor((M, N), T.float32),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(M * N // 256, threads=256) as bx:
            for i in T.Parallel(256):
                idx = bx * 256 + i
                if idx < M * N:
                    row = idx // N
                    col = idx % N
                    # A is (N,M), B is (M,N) — use transposed access for A
                    C[row, col] = A[col, row] + B[row, col]

    kernel_fn = tilelang.compile(kernel, out_idx=[-1], execution_backend="tvm_ffi")
    device = get_current_device()

    # A stored as (M, N) but passed as T → (N, M) non-contiguous
    A_orig = torch.randn(M, N, dtype=torch.float32, device=device)
    B = torch.randn(M, N, dtype=torch.float32, device=device)

    # A.T has shape (N, M) but is non-contiguous — should be rejected
    A_transposed = A_orig.T
    assert A_transposed.shape == (N, M)
    assert not A_transposed.is_contiguous()

    with pytest.raises(RuntimeError, match="contiguous|from_dlpack|strides|packed ABI"):
        kernel_fn(A_transposed, B)

    # A_contig is (N, M) contiguous — should work
    A_contig = A_orig.T.contiguous()
    assert A_contig.shape == (N, M)
    assert A_contig.is_contiguous()

    result = kernel_fn(A_contig, B)
    expected = A_orig + B  # A_contig[col, row] == A_orig[row, col]
    assert _allclose(result, expected), "Pre-transposed contiguous input produced wrong result"


if __name__ == "__main__":
    tilelang.testing.main()
