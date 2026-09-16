"""TANG language dialect: CUDA dialect plus TANG (stcuv2) extensions.

The TANG backend is built on top of the CUDA code-generation infrastructure
(TMEM allocation, warp specialization, tcgen05 gemm, ...), so the TANG dialect
starts from the full CUDA facade and layers the stcuv2-only primitives on top:

* ``tcgen05_ld`` / ``tcgen05_st`` — explicit TMEM <-> register-fragment movement.
* ``tcgen05_before_thread_sync`` / ``tcgen05_after_thread_sync`` — tensor-core
  ordering fences (shadow the CUDA no-arg intrinsics of the same name).
* ``tcgen05_sync_arrive`` / ``tcgen05_sync_wait`` — cross-warp TMEM handshake.
* ``tang_stmatrix`` / ``tang_ldmatrix`` — S3-explicit shared <-> fragment matrix
  store / load.

These are only meaningful on ``tang -arch=stcuv2``; using them on another
target fails during code generation.
"""

from __future__ import annotations

from tilelang.cuda.language import *  # noqa: F401,F403
from tilelang.cuda.language import __all__ as _CUDA_ALL

# Single-warp (warp_group_size=32) WarpSpecialize is a TANG/stcuv2-only need
# (warp-specialized TMEM ldt/stt drains), so it shadows the CUDA dialect's `ws`
# (which is fixed at 128) here on the TANG dialect only.
from tilelang.tang.language.warpgroup import (  # noqa: F401
    WarpSpecialize,
    ws,
)

from tilelang.language.tang_tcgen05 import (  # noqa: F401
    tang_cp_tmem_to_shared,
    tcgen05_after_thread_sync,
    tcgen05_before_thread_sync,
    tcgen05_cp,
    tcgen05_ld,
    tcgen05_st,
    tcgen05_sync_arrive,
    tcgen05_sync_wait,
)
from tilelang.language.builtin import (  # noqa: F401
    tang_ldmatrix,
    tang_stmatrix,
)

_TANG_API_ALL = (
    "tang_cp_tmem_to_shared",
    "tcgen05_after_thread_sync",
    "tcgen05_before_thread_sync",
    "tcgen05_cp",
    "tcgen05_ld",
    "tcgen05_st",
    "tcgen05_sync_arrive",
    "tcgen05_sync_wait",
    "tang_ldmatrix",
    "tang_stmatrix",
)

__tilelang_dialect__ = "tang"
__all__ = tuple(dict.fromkeys((*_CUDA_ALL, *_TANG_API_ALL)))

del _CUDA_ALL, _TANG_API_ALL
