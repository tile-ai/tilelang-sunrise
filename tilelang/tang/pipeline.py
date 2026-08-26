"""TANG lowering pipeline built on the v0.1.12 backend registry."""

from __future__ import annotations

from tvm import IRModule, s_tir, tirx
from tvm.target import Target

import tilelang
from tilelang.backend.pass_pipeline import PassPipeline, register_pipeline
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
from .subtarget import pass_filter


def TANGPassPipelineBody(mod: IRModule, target: Target) -> IRModule:
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
    mod = tilelang.transform.LayoutReducer()(mod)

    mod = tilelang.transform.IfStmtBinding()(mod)
    mod = tilelang.transform.PipelinePlanning()(mod)
    mod = tilelang.transform.InjectSoftwarePipeline()(mod)
    mod = tilelang.transform.Simplify()(mod)

    mod = tilelang.transform.LayoutInference()(mod)
    if not should_disable_loop_peeling(pass_ctx=pass_ctx):
        mod = tilelang.transform.LoopPeeling()(mod)
    if not should_disable_gemm_pad(pass_ctx=pass_ctx):
        mod = tilelang.transform.PadGemmTail()(mod)
    LayoutVisual(mod)
    # Shared tile lowering and vector planning consult Target::Current while
    # selecting target-dispatched operators and legal vector widths.  Keep the
    # same TANG target bound through LegalizeVectorizedLoop so vectorized tile
    # operators cannot accidentally observe no target.
    with target:
        mod = tilelang.transform.LowerTileOp()(mod)
        mod = tilelang.transform.DecoupleTypeCast()(mod)
        mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    mod = tilelang.transform.LowerAccessPtr()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.HoistNonRestrictParams()(mod)

    mod = tilelang.tang.transform.LowerSharedTmem()(mod)
    mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
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

register_pipeline(tang_pipeline)
