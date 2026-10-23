"""Default TileLang language facade.

The default language surface is the CUDA dialect. Backend-neutral definitions
live in :mod:`tilelang.language.common`, while CUDA-only extensions are
provided by :mod:`tilelang.cuda.language`.
"""

from __future__ import annotations

from tilelang.cuda.language import *  # noqa: F401,F403
from tilelang.cuda.language import __all__ as __all__  # noqa: F401

# Imported by name so static type checkers resolve the CUDA-typed signatures
# through this facade (they cannot evaluate the dynamic __all__).
from tilelang.cuda.language import (  # noqa: F401
    Kernel,
    Parallel,
    Unroll,
    atomic_add,
    copy,
    gemm,
    gemm_blockscaled,
    gemm_sp,
    im2col,
    reduce_absmax,
    reduce_max,
    reduce_min,
    unroll,
)

__tilelang_dialect__ = "cuda"
