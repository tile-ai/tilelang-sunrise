"""TANG-specific transformation frontends."""

from .. import _ffi_api


def LowerSharedBarrier():
    """Turn shared.barrier allocations into an mbarrier init prologue.

    TANG has its own rewrite rather than the CUDA one: that file is only
    compiled when USE_CUDA is on, and its cluster-barrier branch has no TANG
    counterpart.
    """
    return _ffi_api.LowerSharedBarrier()  # type: ignore


def LowerSharedTmem():
    """Lower TANG shared.tmem buffers to address holders."""
    return _ffi_api.LowerSharedTmem()  # type: ignore


def LowerTangTmemDrain():
    """Rewrite an stcuv2 TMEM-to-global drain into its collective intrinsic."""
    return _ffi_api.LowerTangTmemDrain()  # type: ignore


def InjectPTSAsyncCopy():
    """Rewrite eligible stcu global-to-shared copies to PTS async copies."""
    return _ffi_api.InjectPTSAsyncCopy()  # type: ignore


def LowerLDGSTG():
    """Lower TANG global-memory loads and stores to explicit intrinsics."""
    return _ffi_api.LowerLDGSTG()  # type: ignore


__all__ = ["InjectPTSAsyncCopy", "LowerLDGSTG", "LowerSharedBarrier", "LowerSharedTmem", "LowerTangTmemDrain"]
