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


def ExpandTangTmemDrain():
    """Expand the fused TMEM->global drain intrinsic into its loop nest.

    The drain's geometry (warp split, 16-row strip loop, warp-collective
    ``ldt_16x256b_x16`` and the per-lane store pattern) becomes ordinary TIR, so
    the code generator only prints statements. Must run after
    ``RemoveRedundantSyncs`` and before ``MakePackedAPI``: the loops, the
    MMA-publish barrier and the scratch array it emits are final.
    """
    return _ffi_api.ExpandTangTmemDrain()  # type: ignore


def InjectTcgen05Fence():
    """Insert ``fence_tc<fg>`` at conservative TCGEN05/TMEM handoff points.

    stcuv2 counterpart of ``tl.cuda.transform.InjectTcgen05Fence``: wraps
    shared-memory syncs with a TC fence and fences the linear wait/arrive
    handoffs that bracket a TCGEN05/TMEM use. Fences written by hand are never
    duplicated, and the pass is a no-op on non-stcuv2 targets or functions
    without TMEM operations. The injected fence uses fence group 0.
    """
    return _ffi_api.InjectTcgen05Fence()  # type: ignore


def PromoteTmemAccumulator():
    """Turn a local.fragment T.gemm accumulator into a shared.tmem one.

    stcuv2 has no register-destination MMA, so a fragment accumulator has no
    lowering there at all. This promotes it to tensor memory and folds the
    preceding T.clear into the MMA's own clear_accum.
    """
    return _ffi_api.PromoteTmemAccumulator()  # type: ignore


def InjectPTSAsyncCopy():
    """Rewrite eligible stcu global-to-shared copies to PTS async copies."""
    return _ffi_api.InjectPTSAsyncCopy()  # type: ignore


def LowerLDGSTG():
    """Lower TANG global-memory loads and stores to explicit intrinsics."""
    return _ffi_api.LowerLDGSTG()  # type: ignore


__all__ = [
    "ExpandTangTmemDrain",
    "InjectPTSAsyncCopy",
    "InjectTcgen05Fence",
    "LowerLDGSTG",
    "LowerSharedBarrier",
    "LowerSharedTmem",
    "LowerTangTmemDrain",
    "PromoteTmemAccumulator",
]
