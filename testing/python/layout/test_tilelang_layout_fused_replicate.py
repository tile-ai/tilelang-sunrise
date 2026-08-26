import pytest
import torch

import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang.utils.device import get_current_device

tilelang.testing.set_random_seed()

VEC_SIZE = 32


@tilelang.jit
def fused_index_kernel(B: int, M: int, N: int, BLOCK_MN: int, BLOCK_K: int):
    @T.prim_func
    def main(
        a: T.Buffer((B, M, N), T.bfloat16),
        a_out: T.Buffer((B, M, N), T.float32),
    ):
        with T.Kernel(
            T.ceildiv(M, BLOCK_MN),
            T.ceildiv(N, BLOCK_K),
            B,
            threads=128,
        ) as (pid_m, pid_n, pid_b):
            a_fp32_local = T.alloc_fragment((BLOCK_MN * BLOCK_K // VEC_SIZE, VEC_SIZE), T.float32)
            offs_m = pid_m * BLOCK_MN
            offs_n = pid_n * BLOCK_K

            for i, j in T.Parallel(BLOCK_MN, BLOCK_K):
                idx = i * BLOCK_K + j
                a_out[pid_b, offs_m + i, offs_n + j] = a_fp32_local[idx // VEC_SIZE, idx % VEC_SIZE]

    return main


def _require_accelerator_tensor(shape, dtype):
    device = get_current_device()
    if device.type not in ("cuda", "ptpu"):
        pytest.skip("CUDA or PTPU not available")
    try:
        return torch.randn(*shape, device=device, dtype=dtype)
    except RuntimeError as err:
        pytest.skip(f"Accelerator runtime unavailable: {err}")


def test_layout_infer_compiles_and_runs():
    B, M, N = 1, 32, 64
    BLOCK_MN, BLOCK_K = 32, 64
    kernel = fused_index_kernel(B, M, N, BLOCK_MN, BLOCK_K)

    a = _require_accelerator_tensor((B, M, N), torch.bfloat16)
    a_out = torch.empty((B, M, N), dtype=torch.float32, device=a.device)

    # Ensure kernel compiles and executes without layout inversion failure
    kernel(a, a_out)
    if a.device.type == "ptpu":
        torch.ptpu.synchronize()

    assert a_out.shape == a.shape
    assert a_out.dtype == torch.float32


if __name__ == "__main__":
    tilelang.testing.main()
