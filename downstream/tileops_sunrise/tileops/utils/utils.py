import functools

import torch

str2dtype = {
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'float32': torch.float32,
    "int32": torch.int32
}


def is_hopper() -> bool:
    return False  # PTPU is not Hopper


@functools.lru_cache(maxsize=1)
def is_h200():
    return False  # PTPU is not H200


def get_sm_version() -> None:
    return None  # SM version does not apply to PTPU


def get_device() -> str:
    return "ptpu"
