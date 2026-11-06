"""TANG lowering pipeline for STCU and STCUV2."""

from __future__ import annotations

from tvm import IRModule, s_tir, tirx
from tvm.target import Target
from tvm.tirx import PrimFunc, SBlock
from tvm.tirx.stmt_functor import post_order_visit

import tilelang
from tilelang.backend.pass_pipeline import PassPipeline
from tilelang.backend.pass_pipeline.pipeline_utils import (
    LayoutVisual,
    allow_global_thread_synchronization,
    allow_vectorize,
    should_disable_gemm_pad,
    should_disable_loop_peeling,
    should_disable_merge_loop,
    should_disable_remove_redundant_syncs,
    should_disable_shared_memory_reuse,
    should_enable_aggressive_merge,
    should_enable_hoist_copy_addresses,
    should_enable_race_check,
    should_force_let_inline,
)

from .subtarget import TangSubtarget as S
from .subtarget import pass_filter, subtarget_matches


def _module_has_shared_barrier(
    mod: IRModule,
    scopes: tuple[str, ...] = ("shared.barrier", "shared.cluster_barrier"),
) -> bool:
    """Whether any function allocates a barrier buffer in one of ``scopes``
    (i.e. uses ``T.alloc_barrier`` / ``T.alloc_cluster_barrier``).

    TANG rejects a barrier allocation on an arch that has no mbarrier hardware,
    so the failure names the arch instead of dying inside ptcc. Both barrier
    scopes count: a cluster barrier is a shared mbarrier too, and the whole
    ``__mbarrier_*`` family is gated on ``__Tang_ARCH__ >= 200``.
    """
    found = False

    def visit(node):
        nonlocal found
        if isinstance(node, SBlock):
            for buffer in node.alloc_buffers:
                if buffer.scope() in scopes:
                    found = True

    for _, func in mod.functions.items():
        if isinstance(func, PrimFunc):
            post_order_visit(func.body, visit)
    return found


def TANGPassPipelineBodyPrologue(mod: IRModule, target: Target) -> IRModule:
    """Lower Tile IR for stcu/stcuv2 without entering CUDA-only passes."""
    mod = tirx.transform.BindTarget(target)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    pass_ctx = tilelang.transform.get_pass_context()

    if should_force_let_inline():
        mod = tilelang.transform.LetInline()(mod)
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    if should_enable_race_check():
        mod = tilelang.transform.VerifyParallelLoop()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.CanonicalizeLegacyReducer()(mod)
    mod = tilelang.transform.VerifyReducerEpoch()(mod)
    mod = tilelang.transform.VerifyBufferInit()(mod)

    # stcuv2 cannot accumulate a GEMM into registers, so a `T.alloc_fragment`
    # accumulator is promoted to tensor memory here. It has to run before
    # PipelinePlanning (the folded `clear_accum` predicate is built from the
    # reduction loop var, which software pipelining is about to peel and
    # reorder) and therefore also before LayoutInference, which is where the
    # fragment layout would otherwise be fixed.
    mod = pass_filter(tilelang.tang.transform.PromoteTmemAccumulator, S.STCUV2)()(mod)

    mod = tilelang.transform.IfStmtBinding()(mod)
    mod = tilelang.transform.PipelinePlanning()(mod)
    mod = pass_filter(tilelang.tang.transform.AnnotateS3PipelinePlan, S.STCUV2)()(mod)
    mod = tilelang.transform.InjectSoftwarePipeline()(mod)
    mod = pass_filter(tilelang.tang.transform.FinalizeS3PipelineInjection, S.STCUV2)()(mod)
    mod = tilelang.transform.Simplify()(mod)

    # Shared tile lowering and vector planning consult Target::Current while
    # selecting target-dispatched operators and legal vector widths.  Keep the
    # same TANG target bound through LegalizeVectorizedLoop so vectorized tile
    # operators cannot accidentally observe no target.
    with target:
        mod = tilelang.transform.LayoutInference()(mod)
        mod = tilelang.transform.ReducerPlanAndMaterialize()(mod)
        if subtarget_matches(target, S.STCU):
            if not should_disable_loop_peeling(pass_ctx=pass_ctx):
                mod = tilelang.transform.LoopPeeling()(mod)
            if not should_disable_gemm_pad(pass_ctx=pass_ctx):
                mod = tilelang.transform.PadGemmTail()(mod)
        LayoutVisual(mod)
        # One buffer, one swizzle: every bulk copy of a shared buffer must agree
        # on `tang_swizzle_atom_bytes`, otherwise the swizzled write is read back
        # unswizzled and the data is silently scrambled.
        # Checked before LowerTileOp, while the copies and their annotations
        # still exist: src/tang/transform/verify_s3_shared_swizzle.cc.
        mod = pass_filter(tilelang.tang.transform.VerifySharedSwizzle, S.STCUV2)()(mod)
        mod = tilelang.transform.LowerTileOp()(mod)
        # Pipelined bulk copies commit into a named sync barrier (fence.arrive)
        # instead of stalling the warp in fence_mem + __syncthreads, and the
        # consuming MMA waits on that barrier's parity: no block-wide sync and
        # the DMA of the next tile stays in flight. Runs after LowerTileOp
        # (which creates the copies) and before LowerAccessPtr below (while the
        # buffer version is still an access_ptr offset argument).
        # src/tang/transform/inject_s3_async_copy.cc
        #
        # ISP also preserves the loop region's warp-role declaration, so this
        # late pass only materializes the declared protocol.
        mod = pass_filter(tilelang.tang.transform.InjectS3AsyncCopy, S.STCUV2)()(mod)
        # Consume tl.tang_bulk_async_commit records before LowerAccessPtr
        # rewrites the access_ptr offset that identifies each buffer version.
        mod = tilelang.transform.VerifyReducerConsumed()(mod)
        mod = tilelang.transform.DecoupleTypeCast()(mod)
        mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    mod = tilelang.transform.LowerAccessPtr()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.HoistNonRestrictParams()(mod)

    return mod


def TANGPassPipelineBody(mod: IRModule, target: Target) -> IRModule:
    mod = TANGPassPipelineBodyPrologue(mod, target)
    pass_ctx = tilelang.transform.get_pass_context()

    mod = tilelang.tang.transform.LowerSharedTmem()(mod)
    mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    # T.alloc_cluster_barrier lowers to a shared.cluster_barrier allocation.
    # LowerSharedBarrier publishes its init with a cluster-scoped storage sync
    # (tl::cluster_sync), and T.mbarrier_arrive(mbar, cta_id) with a non-zero
    # rank -- as emitted by T.copy_cluster(remote_barrier=...) -- lowers to a
    # DSHM remote mbarrier arrive (tl::cluster_arrive_barrier). That path needs
    # the DSHM builtins/intrinsics/ISel, which are on LLVM develop.
    # TANG mbarriers exist only on stcuv2: the whole __mbarrier_* family is
    # gated on __Tang_ARCH__ >= 200. Reject T.alloc_barrier on stcu here rather
    # than letting ptcc fail on the gated builtins with no mention of the arch.
    if _module_has_shared_barrier(mod) and not subtarget_matches(target, S.STCUV2):
        raise ValueError(
            f"T.alloc_barrier() requires arch=stcuv2, but the current target is "
            f"{target}. TANG mbarrier operations (the Barrier type, mbarrier_init, "
            f"mbarrier_arrive, the parity waits) are only compiled for "
            f"__Tang_ARCH__ >= 200. Use T.sync_threads() on stcu instead."
        )
    mod = pass_filter(tilelang.tang.transform.LowerSharedBarrier, S.STCUV2)()(mod)
    mod = pass_filter(tilelang.tang.transform.LowerTangTmemDrain, S.STCUV2)()(mod)
    mod = tilelang.transform.HoistGlobalBufferAllocations()(mod)
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tirx.transform.NarrowDataType(32)(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    mod = tirx.transform.Simplify()(mod)
    # Runs after Simplify so pointer indexing is already flattened before
    # merging; this avoids interference with VectorizeLoop and StorageRewrite
    # below and reduces loop launch overhead for independent copy operations.
    if not should_disable_merge_loop(pass_ctx=pass_ctx):
        mod = tilelang.transform.MergeLoop()(mod)
    # The final vectorizer also performs target-specific atomic planning.
    with target:
        mod = tilelang.transform.VectorizeLoop(enable_vectorize=allow_vectorize(pass_ctx=pass_ctx))(mod)
    mod = tilelang.transform.StorageRewrite()(mod)
    mod = tilelang.transform.LoopUnswitching()(mod)
    # Runs before UnrollLoop so that loops UnrollLoop would expand are still
    # available as serial loops with a live loop_var; after VectorizeLoop, since
    # replacing an affine index with a runtime scalar defeats vectorization, and
    # after StorageRewrite, so the address scalars are not drawn into storage
    # reuse planning.
    if should_enable_hoist_copy_addresses(pass_ctx=pass_ctx):
        mod = tilelang.transform.HoistCopyAddresses()(mod)
    mod = tilelang.transform.UnrollLoop()(mod)
    mod = s_tir.transform.RenormalizeSplitPattern()(mod)
    mod = tirx.transform.Simplify()(mod)
    mod = tirx.transform.RemoveNoOp()(mod)
    mod = s_tir.transform.HoistIfThenElse()(mod)

    mod = tirx.transform.VerifyMemory()(mod)
    mod = tirx.transform.AnnotateEntryFunc()(mod)
    mod = s_tir.transform.InferFragment()(mod)
    mod = tilelang.transform.LowerThreadAllreduce()(mod)
    if allow_global_thread_synchronization(pass_ctx=pass_ctx):
        mod = tilelang.transform.ThreadSync("global")(mod)
    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)
    mod = tilelang.transform.AnnotateReadOnlyParams()(mod)

    aggressive_merge = should_enable_aggressive_merge(pass_ctx=pass_ctx, target=target)
    disable_reuse = should_disable_shared_memory_reuse(pass_ctx=pass_ctx)
    mod = tilelang.transform.MergeSharedMemoryAllocations(
        enable_aggressive_merge=aggressive_merge,
        disable_reuse=disable_reuse,
    )(mod)
    mod = tilelang.transform.ThreadSync("shared")(mod)
    mod = tilelang.transform.ThreadSync("shared.dyn")(mod)
    # Finalize S3 synchronization after ThreadSync has emitted its conservative
    # rendezvous: first attach TC/TMEM domain fences, then remove only barriers
    # covered by complete S3 ready/free certificates. The pass itself honors
    # the fence-injection and redundant-sync debug configs independently.
    mod = pass_filter(tilelang.tang.transform.FinalizeS3ThreadSyncs, S.STCUV2)()(mod)
    mod = tilelang.transform.MergeIfStmt()(mod)
    mod = pass_filter(tilelang.tang.transform.InjectPTSAsyncCopy, S.STCU)()(mod)
    # STCU has no S3 handshake and still needs the ordinary cleanup after its
    # PTS async-copy rewrite.
    if subtarget_matches(target, S.STCU) and not should_disable_remove_redundant_syncs(pass_ctx=pass_ctx):
        mod = tilelang.transform.RemoveRedundantSyncs()(mod)
    # Expand the fused TMEM->global drain into its loop nest in IR, so the code
    # generator only prints statements (src/tang/transform/lower_s3_shared_tmem.cc).
    # Deliberately late: after RemoveRedundantSyncs, so the MMA-publish barrier
    # the drain needs (nwarps > 1) cannot be elided, and before MakePackedAPI,
    # which is where the global buffer's flat store indices are still the ones
    # the fused intrinsic carried.
    mod = pass_filter(tilelang.tang.transform.ExpandS3TmemDrain, S.STCUV2)()(mod)
    # Materialise the two late mbarrier protocols with one slot plan/prologue:
    # shared->TMEM engine publication, which ThreadSync cannot model, and
    # surviving partial shared syncs wider than one warp. Late on purpose: only
    # syncs that survived every elision pass are rewritten.
    mod = pass_filter(tilelang.tang.transform.MaterializeTangMbarriers, S.STCUV2)()(mod)
    # Last IR-level check before codegen: every tcgen05 MMA descriptor operand
    # must still be the descriptor value op the STCU backend's TCProcess pass
    # reads (src/tang/transform/verify_s3_mma_descriptors.cc). It is a no-op on
    # well-formed IR; it turns "a pass bound/cast a descriptor" into a TIR
    # diagnostic instead of an unattributable backend failure.
    mod = pass_filter(tilelang.tang.transform.VerifyMmaDescriptors, S.STCUV2)()(mod)
    mod = tilelang.transform.MakePackedAPI()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)
    return mod


tang_pipeline = PassPipeline("tang", TANGPassPipelineBody)
