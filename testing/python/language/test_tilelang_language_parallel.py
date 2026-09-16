import tilelang
import tilelang.language as T
import torch
import tilelang.testing
import pytest
from tilelang.utils.device import get_current_device

tilelang.testing.set_random_seed()


@tilelang.jit(out_idx=[1])
def parallel_elementwise_static(length=256, dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((length,), dtype),
        B: T.Tensor((length,), dtype),
    ):
        with T.Kernel(1, threads=length) as _:
            for i in T.Parallel(length):
                B[i] = A[i] + 1.0

    return main


@tilelang.jit(out_idx=[1])
def parallel_elementwise_dynamic(max_len=512, threads=256, dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((max_len,), dtype),
        B: T.Tensor((max_len,), dtype),
        valid_len: T.int32,
    ):
        with T.Kernel(1, threads=threads) as _:
            for i in T.Parallel(max_len):
                B[i] = 0.0
            span = T.min(valid_len, max_len)
            for i in T.Parallel(span):
                B[i] = A[i] - 1.0

    return main


def _require_accelerator_tensor(shape, dtype=torch.float32):
    device = get_current_device()
    if device.type not in ("cuda", "ptpu"):
        pytest.skip("CUDA or PTPU not available")
    try:
        return torch.randn(*shape, device=device, dtype=dtype)
    except RuntimeError as err:
        pytest.skip(f"Accelerator runtime unavailable: {err}")


def _assert_close_on_host(actual, expected):
    if actual.device.type == "ptpu":
        torch.ptpu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=1e-5, rtol=1e-5)


PARALLEL_DYNAMIC_VALID_LENGTHS = [0, 13, 200, 600]


def test_parallel_static_extent():
    kernel = parallel_elementwise_static(length=256)
    data = _require_accelerator_tensor((256,), torch.float32)
    result = kernel(data)
    _assert_close_on_host(result, data + 1.0)


@pytest.mark.parametrize(
    "valid_len",
    PARALLEL_DYNAMIC_VALID_LENGTHS,
    ids=[f"valid_len={value}" for value in PARALLEL_DYNAMIC_VALID_LENGTHS],
)
def test_parallel_dynamic_extent(valid_len):
    kernel = parallel_elementwise_dynamic(max_len=512, threads=256)
    data = _require_accelerator_tensor((512,), torch.float32)
    out = kernel(data, valid_len)
    reference = torch.zeros_like(data, device="cpu")
    clip = min(valid_len, data.shape[0])
    if data.device.type == "ptpu":
        torch.ptpu.synchronize()
    reference[:clip] = data[:clip].cpu() - 1.0
    _assert_close_on_host(out, reference)


def test_parallel_vectorize_var():
    @tilelang.jit(out_idx=[1])
    def kernel():
        @T.prim_func
        def main(A: T.Tensor((256,), T.float32), B: T.Tensor((256,), T.float32)):
            with T.Kernel(1, threads=128):
                x = T.alloc_fragment((256,), T.float32)
                y = T.alloc_fragment((256,), T.float32)
                scale = T.alloc_var(T.float32, init=3.0)
                T.copy(A, x)
                for i in T.Parallel(256):
                    y[i] = x[i] * scale
                T.copy(y, B)

        return main

    compiled = kernel.compile()
    # Register-register loops vectorize while the scalar operand broadcasts.
    assert "float2" in compiled.get_kernel_source()
    data = _require_accelerator_tensor((256,), torch.float32)
    output = compiled(data)
    if data.device.type == "ptpu":
        torch.ptpu.synchronize()
    _assert_close_on_host(output, data.cpu() * 3.0)


if __name__ == "__main__":
    tilelang.testing.main()
