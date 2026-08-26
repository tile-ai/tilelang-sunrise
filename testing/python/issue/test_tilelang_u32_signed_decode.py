import math

import pytest
import torch

import tilelang
import tilelang.testing
import tilelang.language as T
from examples.dequantize_gemm.quantize.quantization import _tir_u32_to_int_to_float
from tilelang.utils.device import get_current_device


def _pack_signed_values(values: list[int], nbit: int, device: torch.device) -> torch.Tensor:
    lanes_per_word = 32 // nbit
    words = []
    mask = (1 << nbit) - 1
    for word_start in range(0, len(values), lanes_per_word):
        packed = 0
        for pos, value in enumerate(values[word_start : word_start + lanes_per_word]):
            packed |= (value & mask) << (pos * nbit)
        words.append(packed)
    return torch.tensor(words, dtype=torch.uint32, device=device)


@pytest.mark.parametrize(
    "nbit,values",
    [
        (2, [-2, -1, 0, 1]),
        (4, list(range(-8, 8))),
        (8, [-128, -17, -1, 0, 1, 42, 127]),
    ],
)
def test_u32_to_int_to_float_sign_extends_subword_values(nbit, values):
    """Signed sub-word decode should preserve negative values from uint32 storage."""

    lanes_per_word = 32 // nbit
    num_words = math.ceil(len(values) / lanes_per_word)
    num_values = len(values)

    @T.prim_func
    def main(packed_values: T.Tensor((num_words,), "uint32"), decoded_values: T.Tensor((num_values,), "float32")):
        with T.Kernel(1, threads=1) as _:
            for i in T.serial(num_values):
                decoded_values[i] = _tir_u32_to_int_to_float(nbit, packed_values[i // lanes_per_word], i % lanes_per_word, "float32")

    device = get_current_device()
    kernel = tilelang.compile(main)
    packed = _pack_signed_values(values, nbit, device)
    out = torch.empty(num_values, dtype=torch.float32, device=device)
    kernel(packed, out)
    if device.type == "ptpu":
        torch.ptpu.synchronize()
    torch.testing.assert_close(out.cpu(), torch.tensor(values, dtype=torch.float32), rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
