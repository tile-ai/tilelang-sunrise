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


def ExpandS3TmemDrain():
    """Expand the fused TMEM->global drain intrinsic into its loop nest.

    The drain's geometry (warp split, 16-row strip loop, warp-collective
    ``ldt_16x256b_x16`` and the per-lane store pattern) becomes ordinary TIR, so
    the code generator only prints statements. Must run after
    ``RemoveRedundantSyncs`` and before ``MakePackedAPI``: the loops, the
    MMA-publish barrier and the scratch array it emits are final.
    """
    return _ffi_api.ExpandS3TmemDrain()  # type: ignore


def MaterializeTangMbarriers():
    """Materialise late TMEM-publication and partial-sync mbarriers (stcuv2).

    Tensor memory has no ThreadSync equivalent: the edge is "the copy engine
    wrote TMEM" against "the MMA reads it", and the MMA is issued by a different
    warp. Lowering states the publish as a marker op, positioned by the staging
    it follows. The same pass also converts surviving out-of-loop partial shared
    syncs wider than one warp. Both protocols retain separate rewrite logic but
    share one slot plan and one mbarrier publication prologue.
    """
    return _ffi_api.MaterializeTangMbarriers()  # type: ignore


def FinalizeS3ThreadSyncs():
    """Finalize stcuv2 TCGEN05/TMEM and ThreadSync handoffs.

    stcuv2 counterpart of ``tl.cuda.transform.InjectTcgen05Fence``: wraps
    shared-memory syncs with a TC fence and fences the linear wait/arrive
    handoffs that bracket a TCGEN05/TMEM use. It then removes ThreadSync
    barriers covered by complete S3 ready/free certificates. Fence injection
    and certificate finalization honor their debug configs independently.
    The pass is a no-op on non-stcuv2 targets.
    """
    return _ffi_api.FinalizeS3ThreadSyncs()  # type: ignore


def AnnotateS3PipelinePlan():
    """Attach S3 dependency plans before generic software pipelining."""
    return _ffi_api.AnnotateS3PipelinePlan()  # type: ignore


def FinalizeS3PipelineInjection():
    """Finalize S3 barrier ownership after generic software pipelining."""
    return _ffi_api.FinalizeS3PipelineInjection()  # type: ignore


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


def VerifyMmaDescriptors():
    """Check the tcgen05 MMA descriptor contract (stcuv2 only, no rewriting).

    The STCU backend's TCProcess pass reads each A/B/instruction descriptor
    straight off the MMA intrinsic's operands and rejects anything in between
    with an unattributable "Missing description information" error. This pass
    checks the same contract while the offending statement still exists in the
    IR. Adjacency and sharing are deliberately *not* checked: TCProcess matches
    by def-use, so a descriptor may live in another block or feed several MMAs.
    """
    return _ffi_api.VerifyMmaDescriptors()  # type: ignore


def VerifySharedSwizzle():
    """Check that the bulk copies of one shared buffer agree on its swizzle.

    ``tang_swizzle_atom_bytes`` hangs off a copy statement while the swizzle is
    a property of the buffer, so annotating one leg of a global->shared->global
    round trip and not the other would write swizzled and read unswizzled: it
    compiles, runs, and silently scrambles the data. This
    pass reports the disagreement with both use sites. Copies on the other paths
    (tmem<->shared) are ignored, so the cpt2s drain's one-sided annotation stays
    legal. No rewriting.
    """
    return _ffi_api.VerifySharedSwizzle()  # type: ignore


def InjectS3AsyncCopy():
    """Materialize the S3 pipeline free edge and conservative fallback (stcuv2).

    ``InjectSoftwarePipeline`` owns the unconditional engine-copy ready edge:
    barrier allocation/counting, copy declaration and consumer wait. This pass
    consumes the lowered copy records and adds the independent free/overwrite
    edge (MMA retirement -> next copy). It also retains the conservative
    fallback for shapes whose ready edge is not owned by the generic pipeline.
    Anything it cannot classify exactly is left blocking (correct, just slower).

    Must run after ``LowerTileOp`` (which creates the bulk-copy calls) and
    before ``LowerAccessPtr`` (while the buffer version is still an
    ``access_ptr`` offset argument).
    """
    return _ffi_api.InjectS3AsyncCopy()  # type: ignore


__all__ = [
    "ExpandS3TmemDrain",
    "InjectPTSAsyncCopy",
    "InjectS3AsyncCopy",
    "FinalizeS3ThreadSyncs",
    "LowerLDGSTG",
    "LowerSharedBarrier",
    "LowerSharedTmem",
    "LowerTangTmemDrain",
    "MaterializeTangMbarriers",
    "PromoteTmemAccumulator",
    "VerifyMmaDescriptors",
    "VerifySharedSwizzle",
]
