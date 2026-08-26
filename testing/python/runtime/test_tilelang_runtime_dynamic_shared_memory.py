import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.utils.device import get_current_device


@tilelang.jit
def dynamic_smem_kernel():
    # Symbolic length to drive dynamic shared memory allocation
    length = T.symbolic("len", dtype=T.int32)  # noqa: F821

    @T.prim_func
    def main(global_tensor: T.Tensor[(length,), T.int32]):  # noqa: F821
        # Launch a simple kernel that copies from global memory into shared memory
        # using a dynamically-sized allocation. No writes back to global_tensor.
        with T.Kernel(1, threads=32) as _:
            buffer_shared = T.alloc_shared((length,), dtype=T.int32)  # noqa: F821
            T.copy(buffer_shared, global_tensor)

    return main


def _require_accelerator_tensor(shape, dtype):
    device = get_current_device()
    if device.type not in ("cuda", "ptpu"):
        pytest.skip("CUDA or PTPU not available")
    try:
        return torch.randint(0, 100, shape, dtype=dtype, device=device)
    except RuntimeError as err:
        pytest.skip(f"Accelerator runtime unavailable: {err}")


def _run_and_check(kernel, n):
    a = _require_accelerator_tensor((n,), torch.int32)
    kernel(a)
    if a.device.type == "ptpu":
        torch.ptpu.synchronize()
    else:
        torch.cuda.synchronize()


def test_dynamic_shared_memory_varies_across_calls():
    kernel = dynamic_smem_kernel()

    # Run with different dynamic shared memory sizes across invocations
    _run_and_check(kernel, 100)
    _run_and_check(kernel, 200)
    # Repeat sizes to exercise attribute caching path
    _run_and_check(kernel, 200)
    _run_and_check(kernel, 100)


if __name__ == "__main__":
    tilelang.testing.main()
