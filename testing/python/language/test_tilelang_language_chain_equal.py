import tilelang
import tilelang.testing
import tilelang.language as T
import torch
from tilelang.utils.device import get_current_device


@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
)
def chain_equal(N, block_size, dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as bx:
            for lane in T.Parallel(block_size):
                idx = bx * block_size + lane
                A[idx] = B[idx] = C[idx] = 1

    return main


def run_chain_equal(N=128, block_size=64, dtype=T.float32):
    kernel = chain_equal(N, block_size, dtype)
    device = get_current_device()
    A = torch.zeros((N,), dtype=torch.float32, device=device)
    B = torch.zeros((N,), dtype=torch.float32, device=device)
    C = torch.zeros((N,), dtype=torch.float32, device=device)
    kernel(A, B, C)
    ref = torch.ones_like(A)
    torch.testing.assert_close(A.cpu(), ref.cpu())
    torch.testing.assert_close(B.cpu(), ref.cpu())
    torch.testing.assert_close(C.cpu(), ref.cpu())


def test_chain_equal():
    run_chain_equal()


if __name__ == "__main__":
    tilelang.testing.main()
