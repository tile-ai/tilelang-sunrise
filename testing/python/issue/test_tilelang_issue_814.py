import tilelang
import tilelang.testing
import tilelang.language as T
import torch
from tilelang.utils.device import get_current_device


@tilelang.jit
def _tmp_var_kernel(N, block_N, dtype=T.float32):
    @T.prim_func
    def kernel(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=128) as bx:
            for i in T.Parallel(block_N):
                idx = bx * block_N + i
                tmp = T.max(A[idx], 1)
                B[idx] = tmp / 2
                A[idx] = tmp * 2

    return kernel


def run_tmp_var_test(N=1024, block_N=128):
    kernel = _tmp_var_kernel(N, block_N)
    device = get_current_device()

    a = torch.randn(N, device=device, dtype=torch.float)
    b = torch.empty(N, device=device, dtype=torch.float)

    a_ref = a.clone()
    kernel(a, b)

    # Reference computation on CPU (ptpu has limited op support)
    a_cpu = a_ref.cpu()
    tmp_ref = torch.maximum(a_cpu, torch.tensor(1.0, dtype=torch.float))
    b_ref = tmp_ref / 2
    a_ref = tmp_ref * 2

    # Compare on CPU
    tilelang.testing.torch_assert_close(a.cpu(), a_ref, rtol=1e-2, atol=1e-2)
    tilelang.testing.torch_assert_close(b.cpu(), b_ref, rtol=1e-2, atol=1e-2)


def test_issue_814():
    """Test that temporary variables are correctly handled and not over-inlined"""
    run_tmp_var_test(N=1024, block_N=128)


if __name__ == "__main__":
    tilelang.testing.main()
