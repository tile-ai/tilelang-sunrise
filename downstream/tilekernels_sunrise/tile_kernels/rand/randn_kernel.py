# Temporary replacement for the extremely slow torch_npu.randn.
import math

import torch

from tile_kernels.config import get_device, get_max_ub_per_vector_core, get_num_vec_cores, is_ascend
from tile_kernels.rand.seed import reserve_seed
from tile_kernels.utils import ceil_div, next_power_of_two


def get_randn_config(dtype: torch.dtype, num_elements: int) -> tuple[int, int, int, int]:
    tile_size, unroll_factor = 1024, 4
    alignment = tile_size * unroll_factor
    num_cores = get_num_vec_cores()

    # Double buffering overlaps SIMT generation with UB -> GM stores.
    max_block_size = get_max_ub_per_vector_core() // (2 * dtype.itemsize * alignment) * alignment
    num_elements_per_core = max(alignment, ceil_div(num_elements, num_cores))
    block_size = min(max_block_size, next_power_of_two(num_elements_per_core))
    num_stages = 1 if num_elements <= num_cores * block_size else 2
    return block_size, num_stages, tile_size, unroll_factor


def randn(
    *size: int | tuple[int, ...],
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    shape = size[0] if len(size) == 1 and isinstance(size[0], tuple) else size
    device = device or get_device()
    if not is_ascend():
        return torch.randn(shape, dtype=dtype, device=device)

    assert torch.device(device).type == 'npu', 'Ascend randn only supports NPU tensors'
    assert dtype in (torch.bfloat16, torch.float32), 'Ascend randn only supports bfloat16 and float32'
    # reserve_seed runs on the host, so graph replay would reuse the captured seed.
    assert not torch.npu.is_current_stream_capturing(), (
        'Ascend randn does not support NPU graph capture now: graph replay would reuse the same random seed'
    )

    from tile_kernels.rand.randn_asc import get_randn_kernel_asc

    num_elements = math.prod(shape)
    output = torch.empty(shape, dtype=dtype, device=device)
    if num_elements == 0:
        return output
    block_size, num_stages, tile_size, unroll_factor = get_randn_config(dtype, num_elements)
    kernel = get_randn_kernel_asc(dtype, get_num_vec_cores(), block_size, num_stages, tile_size, unroll_factor)
    num_blocks = ceil_div(num_elements, block_size)
    kernel(output.view(-1), reserve_seed(num_blocks))
    return output


def randn_like(input: torch.Tensor) -> torch.Tensor:
    return randn(input.shape, dtype=input.dtype, device=input.device)
