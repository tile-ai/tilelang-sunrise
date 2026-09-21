import tilelang
import tilelang.testing
import tilelang.language as T
import pytest
import torch
from tilelang.utils.device import get_current_device


def _make_round_kernel(dtype):
    @T.prim_func
    def main(A: T.Tensor((16,), dtype), B: T.Tensor((16,), dtype)):
        with T.Kernel(1, threads=16):
            for i in T.Parallel(16):
                B[i] = T.round(A[i], "ties-away-from-zero")

    return main


def test_round_ties_away_from_zero_compiles_for_bfloat16():
    tilelang.compile(_make_round_kernel("bfloat16"))


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_round_ties_away_from_zero_values(dtype):
    values = [-2.5, -1.5, -0.5, -0.25, 0.0, 0.25, 0.5, 1.5, 2.5, -3.0, 3.0, -4.25, 4.25, -4.75, 4.75, 8.0]
    expected = [-3, -2, -1, 0, 0, 0, 1, 2, 3, -3, 3, -4, 4, -5, 5, 8]
    torch_dtype = getattr(torch, dtype)
    src = torch.tensor(values, dtype=torch_dtype, device=get_current_device())
    dst = torch.empty_like(src)
    tilelang.compile(_make_round_kernel(dtype))(src, dst)
    if dst.device.type == "ptpu":
        torch.ptpu.synchronize(dst.device)
    torch.testing.assert_close(dst.cpu(), torch.tensor(expected, dtype=torch_dtype), rtol=0, atol=0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 9)
def test_round_ties_away_from_zero_compiles_for_float8():
    for dtype in ("float8_e4m3", "float8_e5m2"):
        tilelang.compile(_make_round_kernel(dtype), target="cuda")


if __name__ == "__main__":
    tilelang.testing.main()
