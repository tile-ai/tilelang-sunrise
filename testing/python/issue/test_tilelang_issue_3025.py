import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.utils.device import get_current_device
from tilelang.utils.language import retrieve_stride


M = 4
N = 3
PITCH = 8


def _make_store_kernel():
    @T.prim_func
    def main(dst: T.StridedTensor((M, N), (PITCH, 1), "int32")):
        with T.Kernel(1, threads=1):
            for row in T.serial(M):
                T.stg32(dst[row, 0], T.Cast("uint32", row + 1))

    return main


def _make_load_kernel():
    @T.prim_func
    def main(
        src: T.StridedTensor((M, N), (PITCH, 1), "int32"),
        out: T.Tensor((M,), "int32"),
    ):
        with T.Kernel(1, threads=1):
            for row in T.serial(M):
                out[row] = T.reinterpret(T.ldg32(src[row, 0]), "int32")

    return main


def test_retrieve_stride_preserves_scalar_rank():
    scalar = T.Tensor((), "float32")
    assert retrieve_stride(scalar) == []


def test_strided_stg32_uses_declared_stride():
    kernel = tilelang.compile(_make_store_kernel())
    physical = torch.zeros(M * PITCH, dtype=torch.int32, device=get_current_device())
    view = physical.as_strided((M, N), (PITCH, 1))

    kernel(view)

    expected = torch.arange(1, M + 1, dtype=torch.int32, device=get_current_device())
    if view.device.type == "ptpu":
        torch.ptpu.synchronize(view.device)
        view, expected = view.cpu(), expected.cpu()
    torch.testing.assert_close(view[:, 0], expected, rtol=0, atol=0)
    source = kernel.get_kernel_source()
    assert "dst[(row * 8)]" in source
    assert "dst[(row * 3)]" not in source


def test_strided_ldg32_uses_declared_stride():
    physical = torch.arange(M * PITCH, dtype=torch.int32, device=get_current_device())
    view = physical.as_strided((M, N), (PITCH, 1))
    out = torch.empty(M, dtype=torch.int32, device=get_current_device())
    kernel = tilelang.compile(_make_load_kernel())

    kernel(view, out)

    if out.device.type == "ptpu":
        torch.ptpu.synchronize(out.device)
        out, view = out.cpu(), view.cpu()
    torch.testing.assert_close(out, view[:, 0], rtol=0, atol=0)
    source = kernel.get_kernel_source()
    assert "src[(row * 8)]" in source
    assert "src[(row * 3)]" not in source


def test_predicated_ldg32_stg32():
    @T.prim_func
    def main(src: T.Tensor((4,), "int32"), dst: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=1):
            for row in T.serial(4):
                value = T.ldg32(src[row], row % 2 == 0)
                T.stg32(dst[row], value, row < 3)

    device = get_current_device()
    src = torch.tensor([11, 22, 33, 44], dtype=torch.int32, device=device)
    dst = torch.full((4,), -1, dtype=torch.int32, device=device)
    kernel = tilelang.compile(main)
    kernel(src, dst)
    if dst.device.type == "ptpu":
        torch.ptpu.synchronize(dst.device)
    torch.testing.assert_close(dst.cpu(), torch.tensor([11, 0, 33, -1], dtype=torch.int32), rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
