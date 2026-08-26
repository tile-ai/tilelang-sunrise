"""Backward-compatible sparse utility imports.

The v0.1.13 upstream tree moved the implementation to the shipped
``examples.gemm_sp`` module. Keep the historical ``tilelang.utils.sparse``
entry point available for installed users and downstream code while sharing
the single implementation.
"""

from examples.gemm_sp.sparse_utils import (
    GROUP_CONFIG,
    arange_semi_sparse,
    compress,
    get_e_factor,
    get_e_replicate_factor,
    randint_semi_sparse,
    randn_semi_sparse,
    torch_compress,
)

__all__ = [
    "GROUP_CONFIG",
    "arange_semi_sparse",
    "compress",
    "get_e_factor",
    "get_e_replicate_factor",
    "randint_semi_sparse",
    "randn_semi_sparse",
    "torch_compress",
]
