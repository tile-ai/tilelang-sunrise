import torch
import pytest
from tilelang.utils.device import get_current_device

import tilelang
import tilelang.testing
import tilelang.language as T


@tilelang.jit
def get_kernel(reduce_op: str, dtype: str):
    assert reduce_op in ["sum", "max", "min", "bitand", "bitor"]

    @T.prim_func
    def main(x: T.Tensor((32), dtype)):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding(0)
            local_val = T.alloc_local([1], dtype)
            local_val[0] = x[tx]
            reduced_val = T.alloc_local([1], dtype)
            if reduce_op == "sum":
                reduced_val[0] = T.warp_reduce_sum(local_val[0])
            elif reduce_op == "max":
                reduced_val[0] = T.warp_reduce_max(local_val[0])
            elif reduce_op == "min":
                reduced_val[0] = T.warp_reduce_min(local_val[0])
            elif reduce_op == "bitand":
                reduced_val[0] = T.warp_reduce_bitand(local_val[0])
            elif reduce_op == "bitor":
                reduced_val[0] = T.warp_reduce_bitor(local_val[0])
            x[tx] = reduced_val[0]

    return main


def test_warp_reduce_sum():
    device = get_current_device()
    a = torch.randn((32,), dtype=torch.float32, device=device)
    kernel = get_kernel("sum", T.float32)
    if device.type == "ptpu":
        torch.ptpu.synchronize(device)
    a_cpu = a.cpu()
    ref = torch.full_like(a_cpu, a_cpu.sum())
    kernel(a)
    if device.type == "ptpu":
        torch.ptpu.synchronize(device)
    torch.testing.assert_close(a.cpu(), ref)


def test_warp_reduce_max():
    device = get_current_device()
    a = torch.randn((32,), dtype=torch.float32, device=device)
    kernel = get_kernel("max", T.float32)
    print(kernel.get_kernel_source())
    if device.type == "ptpu":
        torch.ptpu.synchronize(device)
    a_cpu = a.cpu()
    ref = torch.full_like(a_cpu, a_cpu.max())
    kernel(a)
    if device.type == "ptpu":
        torch.ptpu.synchronize(device)
    torch.testing.assert_close(a.cpu(), ref)


def test_warp_reduce_min():
    device = get_current_device()
    a = torch.randn((32,), dtype=torch.float32, device=device)
    kernel = get_kernel("min", T.float32)
    if device.type == "ptpu":
        torch.ptpu.synchronize(device)
    a_cpu = a.cpu()
    ref = torch.full_like(a_cpu, a_cpu.min())
    kernel(a)
    if device.type == "ptpu":
        torch.ptpu.synchronize(device)
    torch.testing.assert_close(a.cpu(), ref)


def test_warp_reduce_bitand():
    device = get_current_device()
    a = torch.randint(0, 100, size=(32,), dtype=torch.int32).to(device)
    kernel = get_kernel("bitand", T.int32)
    if device.type == "ptpu":
        torch.ptpu.synchronize(device)
    a_cpu = a.cpu()
    ref_val = int(a_cpu[0].item())
    for i in range(1, a_cpu.shape[0]):
        ref_val &= int(a_cpu[i].item())
    ref = torch.full((32,), ref_val, dtype=torch.int32)
    kernel(a)
    if device.type == "ptpu":
        torch.ptpu.synchronize(device)
    torch.testing.assert_close(a.cpu(), ref)


def test_warp_reduce_bitor():
    device = get_current_device()
    a = torch.randint(0, 100, size=(32,), dtype=torch.int32).to(device)
    kernel = get_kernel("bitor", T.int32)
    if device.type == "ptpu":
        torch.ptpu.synchronize(device)
    a_cpu = a.cpu()
    ref_val = int(a_cpu[0].item())
    for i in range(1, a_cpu.shape[0]):
        ref_val |= int(a_cpu[i].item())
    ref = torch.full((32,), ref_val, dtype=torch.int32)
    kernel(a)
    if device.type == "ptpu":
        torch.ptpu.synchronize(device)
    torch.testing.assert_close(a.cpu(), ref)


WARP_REDUCE_CASES_64 = [
    # (op, dtype, N)
    ("sum", "int64", 32),
    ("max", "int64", 32),
    ("min", "int64", 32),
    ("bitand", "int64", 32),
    ("bitor", "int64", 32),
]


@pytest.mark.parametrize(
    ("op", "dtype", "N"),
    WARP_REDUCE_CASES_64,
)
def test_warp_reduce_64(op, dtype, N):
    def warp_reduce_ref(a):
        if op == "sum":
            return torch.full_like(a, a.sum())
        elif op == "max":
            return torch.full_like(a, a.max())
        elif op == "min":
            return torch.full_like(a, a.min())
        elif op == "bitand":
            ref_val = a[0]
            for i in range(1, a.shape[0]):
                ref_val = ref_val & a[i]
            return torch.full_like(a, ref_val)
        elif op == "bitor":
            ref_val = a[0]
            for i in range(1, a.shape[0]):
                ref_val = ref_val | a[i]
            return torch.full_like(a, ref_val)
        raise AssertionError(f"Unknown op: {op}")

    torch_dtype = getattr(torch, dtype)
    tl_dtype = getattr(T, dtype)

    device = get_current_device()
    a = torch.randint(1 << 32, (1 << 63) - 1, (N,), dtype=torch_dtype).to(device)
    if device.type == "ptpu":
        torch.ptpu.synchronize(device)
    ref = warp_reduce_ref(a.cpu())

    kernel = get_kernel(op, tl_dtype)
    kernel(a)
    if device.type == "ptpu":
        torch.ptpu.synchronize(device)

    torch.testing.assert_close(a.cpu(), ref)


if __name__ == "__main__":
    tilelang.testing.main()
