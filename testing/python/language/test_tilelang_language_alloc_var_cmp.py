import warnings

import tilelang
import tilelang.language as T
import tilelang.testing
import torch
from tilelang.utils.device import get_current_device


@tilelang.jit
def kernel_alloc_var_eq(data):
    N = T.dynamic("N")
    data: T.Tensor[[N], T.int32]
    out = T.empty([N], T.int32)

    with T.Kernel(1) as _:
        for i in T.serial(N):
            cond = T.alloc_var(T.bool)
            cond = data[i] == 2
            if cond:
                out[i] = 1
            else:
                out[i] = 0
    return out


@tilelang.jit
def kernel_alloc_var_ne(data):
    N = T.dynamic("N")
    data: T.Tensor[[N], T.int32]
    out = T.empty([N], T.int32)

    with T.Kernel(1) as _:
        for i in T.serial(N):
            cond = T.alloc_var(T.bool)
            cond = data[i] != 0
            if cond:
                out[i] = 1
            else:
                out[i] = 0
    return out


def test_alloc_var_eq():
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Immutable value.*is re-bound")
        device = get_current_device()
        data = torch.tensor([0, 1, 2, 3, 2], dtype=torch.int32, device=device)
        result = kernel_alloc_var_eq(data)
        expected = torch.tensor([0, 0, 1, 0, 1], dtype=torch.int32, device=device)
        if device.type == "ptpu":
            torch.ptpu.synchronize()
        torch.testing.assert_close(result.cpu(), expected.cpu())


def test_alloc_var_ne():
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Immutable value.*is re-bound")
        device = get_current_device()
        data = torch.tensor([0, 1, 2, 3, 0], dtype=torch.int32, device=device)
        result = kernel_alloc_var_ne(data)
        expected = torch.tensor([0, 1, 1, 1, 0], dtype=torch.int32, device=device)
        if device.type == "ptpu":
            torch.ptpu.synchronize()
        torch.testing.assert_close(result.cpu(), expected.cpu())


if __name__ == "__main__":
    tilelang.testing.main()
