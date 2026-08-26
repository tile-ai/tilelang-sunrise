# ruff: noqa

import torch
import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang.utils.device import get_current_device


def _accelerator():
    device = get_current_device()
    return device, torch.ptpu if device.type == "ptpu" else torch.cuda


@tilelang.jit
def _empty_kernel():
    @T.prim_func
    def empty_kernel():
        with T.Kernel(1, threads=32) as thread_idx:
            pass

    return empty_kernel


def test_empty_kernel_lowering():
    # Empty kernels need an initialized accelerator context.
    _, accelerator = _accelerator()
    accelerator.set_device(0)
    kernel = _empty_kernel()
    kernel()


@tilelang.jit
def _empty_with_dead_code_kernel():
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def buggy_kernel(x: T.Tensor[(num_tokens,), T.float32]):
        with T.Kernel(num_tokens, threads=32) as pid:
            y = x[pid]

    return buggy_kernel


def test_empty_with_dead_code_kernel():
    kernel = _empty_with_dead_code_kernel()
    device, _ = _accelerator()
    x = torch.randn((128,), dtype=torch.float32, device=device)
    kernel(x)


@tilelang.jit
def _empty_kernel_with_binding_variants(use_tuple_binding: bool = False):
    @T.prim_func
    def kernel_with_tuple_kernel_binding():
        with T.Kernel(1, threads=32) as (pid,):
            print(pid)
            pass

    @T.prim_func
    def kernel_with_scalar_kernel_binding():
        with T.Kernel(1, threads=32) as pid:
            print(pid)
            pass

    return kernel_with_tuple_kernel_binding if use_tuple_binding else kernel_with_scalar_kernel_binding


def test_empty_kernel_with_binding_variants():
    _, accelerator = _accelerator()
    accelerator.set_device(0)
    kernel = _empty_kernel_with_binding_variants()
    kernel()

    tuple_kernel = _empty_kernel_with_binding_variants(use_tuple_binding=True)
    tuple_kernel()


if __name__ == "__main__":
    tilelang.testing.main()
