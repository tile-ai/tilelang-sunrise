import pytest
import tilelang
import tilelang.language as T
import torch
from tvm import tirx
import tilelang.testing
from tilelang.backend.target import determine_target
from tilelang.utils.device import get_current_device


@tilelang.jit
def kernel_with_warp_sync():
    target_kind = determine_target(tilelang.env.get_default_target(), return_object=True).kind.name

    @T.prim_func
    def main(
        A: T.Tensor((1,), "int32"),
        B: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            if tx == 0:
                if target_kind != "tang":
                    tirx.call_extern("void", "__nanosleep", 100)
                A[0] = -1
            T.sync_warp()
            if tx == 1:
                B[0] = A[0]

    return main


def test_warp_sync():
    device = get_current_device()
    a = torch.empty((1), device=device, dtype=torch.int32)
    b = torch.empty((1), device=device, dtype=torch.int32)
    kernel = kernel_with_warp_sync()
    source = kernel.get_kernel_source()
    assert "__syncwarp" in source
    kernel(a, b)
    torch.testing.assert_close(b.cpu(), torch.tensor([-1], dtype=torch.int32))


@tilelang.jit
def kernel_with_masked_warp_sync():
    @T.prim_func
    def main(A: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            T.sync_warp(0x0000FFFF)
            A[tx] = tx

    return main


def test_masked_warp_sync_codegen():
    source = kernel_with_masked_warp_sync().get_kernel_source()
    assert "__syncwarp(65535)" in source


@tilelang.jit(out_idx=[-1])
def kernel_with_shfl_sync():
    @T.prim_func
    def main(
        A: T.Tensor((32,), "int32"),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            val = tx * 10
            broadcast = T.shfl_sync(val, 31)
            A[tx] = broadcast

    return main


def test_shfl_sync():
    # Guards the shfl_sync codegen branch (CUDA and TANG both emit __shfl_sync).
    # Runs on whichever backend TileLang targets in this environment.
    kernel = kernel_with_shfl_sync()
    assert "__shfl_sync" in kernel.get_kernel_source()
    a = kernel()
    # Every lane receives lane 31's value: 31 * 10 = 310.
    expected = torch.full((32,), 310, dtype=torch.int32)
    torch.testing.assert_close(a.cpu(), expected)


# (width, delta) pairs exercise the segmented out-of-range fallback:
#   width=32 is the full warp; width=16 splits the warp into two independent
#     segments, so the shuffle must wrap within a segment, not across it.
#   delta=0    -> identity (every lane reads itself)
#   delta<width-> a block of lanes shifts, the edge lanes keep their own value
#   delta>=width -> every lane is out of range (identity)
# Using value == lane_id (which differs per lane) while delta/width are small
# constants and the mask is 0xFFFFFFFF also makes any mask/value/delta/width
# argument-order slip in codegen produce a wrong result, so it is caught here.
_SHFL_CONFIGS = [(32, 0), (32, 1), (32, 8), (32, 32), (16, 1), (16, 8)]


def _expected_shfl(direction: str, width: int, delta: int) -> torch.Tensor:
    # CUDA __shfl_{down,up}_sync semantics: lanes are grouped into width-sized
    # segments; lane i reads segment-local position (pos +/- delta), or keeps
    # its own value when that position leaves the segment.
    out = torch.arange(32, dtype=torch.int32)
    for i in range(32):
        seg, pos = i // width, i % width
        src = pos + delta if direction == "down" else pos - delta
        if 0 <= src < width:
            out[i] = seg * width + src
    return out


@tilelang.jit(out_idx=[-1])
def kernel_with_shfl_down(delta: int = 1, width: int = 32):
    @T.prim_func
    def main(A: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            A[tx] = T.shfl_down(tx, delta, width=width)

    return main


@pytest.mark.parametrize(("width", "delta"), _SHFL_CONFIGS)
def test_shfl_down(width: int, delta: int):
    # Runs on the TANG backend and asserts the CUDA __shfl_down_sync result, so
    # any TANG/CUDA divergence (out-of-range fallback, segment width, or an
    # argument-order slip) fails rather than silently passing.
    kernel = kernel_with_shfl_down(delta, width)
    assert "__shfl_down_sync" in kernel.get_kernel_source()
    A = kernel()
    torch.testing.assert_close(A.cpu(), _expected_shfl("down", width, delta))


@tilelang.jit(out_idx=[-1])
def kernel_with_shfl_up(delta: int = 1, width: int = 32):
    @T.prim_func
    def main(A: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            A[tx] = T.shfl_up(tx, delta, width=width)

    return main


@pytest.mark.parametrize(("width", "delta"), _SHFL_CONFIGS)
def test_shfl_up(width: int, delta: int):
    # Runs on the TANG backend and asserts the CUDA __shfl_up_sync result, so a
    # TANG/CUDA mismatch (out-of-range fallback, segment width, or an
    # argument-order slip) fails rather than silently passing.
    kernel = kernel_with_shfl_up(delta, width)
    assert "__shfl_up_sync" in kernel.get_kernel_source()
    A = kernel()
    torch.testing.assert_close(A.cpu(), _expected_shfl("up", width, delta))


if __name__ == "__main__":
    tilelang.testing.main()
