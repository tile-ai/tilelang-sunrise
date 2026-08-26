import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.utils.device import get_current_device


LANES = 32


@pytest.mark.parametrize(
    "dtype,torch_dtype,value",
    [("int8", torch.int8, -37), ("uint8", torch.uint8, 213)],
)
def test_runtime_int8_broadcast_to_32_lanes(dtype, torch_dtype, value):
    device = get_current_device()

    @T.prim_func
    def main(src: T.Tensor((1,), dtype), out: T.Tensor((LANES,), dtype)):
        with T.Kernel(1, threads=1) as _:
            # Scalar global stores keep this test runnable on pre-SM100 GPUs.
            packed = T.alloc_local((LANES,), dtype)
            packed[0:LANES] = T.Broadcast(src[0], LANES)
            for i in T.serial(LANES):
                out[i] = packed[i]

    target = tilelang.env.get_default_target()
    kernel = tilelang.compile(main, target=target)
    if str(target).startswith("cuda"):
        constructor = "make_ulonglong4" if dtype == "uint8" else "make_longlong4"
        assert constructor in kernel.get_kernel_source()
    src = torch.tensor([value], dtype=torch_dtype, device=device)
    out = torch.empty(LANES, dtype=torch_dtype, device=device)
    kernel(src, out)
    if device.type == "ptpu":
        torch.ptpu.synchronize(device)

    torch.testing.assert_close(out, torch.full_like(out, value), rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype,torch_dtype,value",
    [("int8", torch.int8, -37), ("uint8", torch.uint8, 213)],
)
def test_constant_int8_broadcast_to_32_lanes(dtype, torch_dtype, value):
    device = get_current_device()

    @T.prim_func
    def main(out: T.Tensor((LANES,), dtype)):
        with T.Kernel(1, threads=1) as _:
            packed = T.alloc_local((LANES,), dtype)
            packed[0:LANES] = T.Broadcast(T.cast(value, dtype), LANES)
            for i in T.serial(LANES):
                out[i] = packed[i]

    target = tilelang.env.get_default_target()
    kernel = tilelang.compile(main, target=target)
    if str(target).startswith("cuda"):
        constructor = "make_ulonglong4" if dtype == "uint8" else "make_longlong4"
        assert constructor in kernel.get_kernel_source()
    out = torch.empty(LANES, dtype=torch_dtype, device=device)
    kernel(out)
    if device.type == "ptpu":
        torch.ptpu.synchronize(device)

    torch.testing.assert_close(out, torch.full_like(out, value), rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
