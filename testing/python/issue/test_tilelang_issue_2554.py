import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.utils.device import get_current_device


def test_runtime_unknown_sign_vector_negative_index_load():
    device = get_current_device()

    @T.prim_func
    def main(A: T.Tensor((1024,), T.float32), B: T.Tensor((4, 4), T.float32)):
        with T.Kernel(1, threads=1) as _:
            for t in T.serial(4):
                B[t, T.Ramp(0, 1, 4)] = A[T.Ramp(t - 2, 1, 4)]

    kernel = tilelang.compile(main, out_idx=[1])

    a = torch.arange(1024, device=device, dtype=torch.float32)
    b = kernel(a)
    if device.type == "ptpu":
        torch.ptpu.synchronize(device)
    expected = torch.tensor(
        [
            [1022.0, 1023.0, 0.0, 1.0],
            [1023.0, 0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(b.cpu(), expected)


def test_runtime_unknown_sign_vector_negative_index_store():
    device = get_current_device()

    @T.prim_func
    def main(B: T.Tensor((4, 4), T.float32), A: T.Tensor((1024,), T.float32)):
        with T.Kernel(1, threads=1) as _:
            for i in T.serial(1024):
                A[i] = T.float32(-1)
            for t in T.serial(4):
                A[T.Ramp(t - 2, 1, 4)] = B[t, T.Ramp(0, 1, 4)]

    kernel = tilelang.compile(main, out_idx=[1])

    b = torch.arange(16, device=device, dtype=torch.float32).reshape(4, 4)
    a = kernel(b)
    if device.type == "ptpu":
        torch.ptpu.synchronize(device)

    b_cpu = b.cpu()
    expected = torch.full((1024,), -1, dtype=torch.float32)
    expected[0] = b_cpu[2, 0]
    expected[1] = b_cpu[3, 0]
    expected[2] = b_cpu[3, 1]
    expected[3] = b_cpu[3, 2]
    expected[4] = b_cpu[3, 3]
    expected[1022] = b_cpu[0, 0]
    expected[1023] = b_cpu[1, 0]

    torch.testing.assert_close(a.cpu(), expected)


if __name__ == "__main__":
    tilelang.testing.main()
