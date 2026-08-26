"""Default TileLang language facade.

The default language surface is the CUDA dialect. Backend-neutral definitions
live in :mod:`tilelang.language.common`, while CUDA-only extensions are
provided by :mod:`tilelang.cuda.language`.
"""

from __future__ import annotations

from tilelang.cuda.language import *  # noqa: F401,F403
from tilelang.cuda.language import __all__ as __all__  # noqa: F401

__tilelang_dialect__ = "cuda"
