import tilelang
import tilelang.language as T
import tilelang.testing
import torch
from tilelang.utils.device import get_current_device


def test_tensor_annot_mul():
    device = get_current_device()

    # There is a known issue where the cython execution backend fails to build with T.symbolic.
    # Forcing the TVM FFI execution backend to avoid the issue on HIP.
    @tilelang.jit(execution_backend="tvm_ffi")
    def example_tensor_annot():
        n = T.symbolic("n")

        @T.prim_func
        def kernel(
            A: T.Tensor((n * 4,), T.int32),
        ):
            with T.Kernel(1) as _:
                for i in range(n * 4):
                    A[i] = 0

        return kernel

    ker = example_tensor_annot()
    A = torch.arange(16, dtype=torch.int32, device=device)
    ker(A)
    expected = torch.zeros(16, dtype=torch.int32)
    assert torch.equal(A.cpu(), expected)


def test_tensor_annot_add():
    device = get_current_device()

    # There is a known issue where the cython execution backend fails to build with T.symbolic.
    # Forcing the TVM FFI execution backend to avoid the issue on HIP.
    @tilelang.jit(execution_backend="tvm_ffi")
    def example_tensor_annot():
        n = T.symbolic("n")

        @T.prim_func
        def kernel(
            A: T.Tensor((n + 1,), T.int32),
        ):
            with T.Kernel(1) as _:
                for i in range(n + 1):
                    A[i] = 0

        return kernel

    ker = example_tensor_annot()
    A = torch.arange(16, dtype=torch.int32, device=device)
    ker(A)
    expected = torch.zeros(16, dtype=torch.int32)
    assert torch.equal(A.cpu(), expected)


def test_tensor_annot_mul_add():
    device = get_current_device()

    # There is a known issue where the cython execution backend fails to build with T.symbolic.
    # Forcing the TVM FFI execution backend to avoid the issue on HIP.
    @tilelang.jit(execution_backend="tvm_ffi")
    def example_tensor_annot():
        n = T.symbolic("n")

        @T.prim_func
        def kernel(
            A: T.Tensor((n * 3 + 1,), T.int32),
        ):
            with T.Kernel(1) as _:
                for i in range(n * 3 + 1):
                    A[i] = 0

        return kernel

    ker = example_tensor_annot()
    A = torch.arange(16, dtype=torch.int32, device=device)
    ker(A)
    expected = torch.zeros(16, dtype=torch.int32)
    assert torch.equal(A.cpu(), expected)


if __name__ == "__main__":
    tilelang.testing.main()
