# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under the MIT License.
# Adapted and modified for TileOps GatedDeltaNet prefill integration.

from typing import TYPE_CHECKING

from .gemm_lowering import install_gemm_v1

install_gemm_v1()

from .cp_fwd import correct_initial_states, get_warmup_chunks  # noqa: E402

if TYPE_CHECKING:
    from .fused_fwd import fused_gdr_fwd
    from .prepare_h import fused_gdr_h

__all__ = [
    "correct_initial_states",
    "fused_gdr_fwd",
    "fused_gdr_h",
    "get_warmup_chunks",
]


def __getattr__(name):
    # Importing the CP helpers must not initialize CUDA-only fused kernels.
    if name == "fused_gdr_fwd":
        from .fused_fwd import fused_gdr_fwd

        return fused_gdr_fwd
    if name == "fused_gdr_h":
        from .prepare_h import fused_gdr_h

        return fused_gdr_h
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
