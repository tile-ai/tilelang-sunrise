"""Backward-compatible re-export of the shared device-assert intrinsic.

``device_assert`` is backend-neutral (lowered by both the CUDA and TANG
codegens); its canonical home is :mod:`tilelang.language.debug`.  This module
only forwards the shared implementation so existing
``from tilelang.cuda.debug import device_assert`` imports keep working.
"""

from tilelang.language.debug import device_assert, get_stack_str  # noqa: F401
