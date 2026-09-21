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

    TANG rejects barrier allocations an arch has no hardware for here so the
    generated code fails fast with a clear message instead of deep inside ptcc.
    Called twice with different ``scopes``: once narrowed to cluster barriers,
    which TANG has no level for at all, and once for any barrier, which needs
    stcuv2.
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

    mod = tilelang.transform.IfStmtBinding()(mod)
    mod = tilelang.transform.PipelinePlanning()(mod)
    mod = tilelang.transform.InjectSoftwarePipeline()(mod)
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
        mod = tilelang.transform.LowerTileOp()(mod)
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
    # TANG has no cluster level, so a cluster-scoped barrier has no counterpart
    # at any arch: mbarrier.arrive takes no cta_id operand and there is no
    # cluster-wide storage sync to publish the init with.
    if _module_has_shared_barrier(mod, ("shared.cluster_barrier",)):
        raise ValueError(
            "T.alloc_cluster_barrier() has no TANG equivalent: TANG has no cluster "
            "level, and mbarrier.arrive takes no cta_id operand. Use "
            "T.alloc_barrier() for block-scoped synchronisation."
        )
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
    mod = tilelang.transform.MergeIfStmt()(mod)
    mod = pass_filter(tilelang.tang.transform.InjectPTSAsyncCopy, S.STCU)()(mod)
    if not should_disable_remove_redundant_syncs(pass_ctx=pass_ctx):
        mod = tilelang.transform.RemoveRedundantSyncs()(mod)
    mod = tilelang.transform.MakePackedAPI()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)
    return mod


tang_pipeline = PassPipeline("tang", TANGPassPipelineBody)
