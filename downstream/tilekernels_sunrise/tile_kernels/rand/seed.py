import torch

from tile_kernels.config import get_is_tang


_seed = torch.initial_seed()


def set_seed(seed: int) -> None:
    global _seed
    torch.manual_seed(seed)
    if get_is_tang():
        device = torch.device('ptpu', torch.ptpu.current_device())
        generator = torch.Generator(device=device).manual_seed(seed)
        torch.ptpu.set_rng_state(generator.get_state(), device=device)
    _seed = seed


def reserve_seed(num_blocks: int) -> int:
    global _seed
    seed = _seed
    _seed += num_blocks
    seed = (seed + 0x9E3779B9) & 0xFFFFFFFF
    seed = ((seed ^ (seed >> 16)) * 0x21F0AAAD) & 0xFFFFFFFF
    seed = ((seed ^ (seed >> 15)) * 0x735A2D97) & 0xFFFFFFFF
    return (seed ^ (seed >> 15)) & 0xFFFFFFFF
