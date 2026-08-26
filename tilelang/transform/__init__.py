"""Wrapping transformations."""
# pylint: disable=invalid-name, unsupported-binary-operation

from . import _ffi_api
from .simplify import Simplify, simplify_prim_func, LetInline  # noqa: F401
from .pass_config import PassConfigKey  # noqa: F401
from tilelang import tvm as tvm  # noqa: F401
from tvm.ir.transform import PassContext  # noqa: F401
from .add_bufstore_wrapper import AddWrapperForSingleBufStore  # noqa: F401
from .hoist_broadcast_values import HoistBroadcastValues  # noqa: F401
from .decouple_type_cast import DecoupleTypeCast  # noqa: F401


def get_pass_context():
    """Get the current pass context"""
    return PassContext.current()


def ClusterPlanning():
    """ClusterPlanning

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.ClusterPlanning()  # type: ignore


def PipelinePlanning():
    """infer the fragment/shared memory layout

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.PipelinePlanning()  # type: ignore


def LayoutInference():
    """LayoutInference

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LayoutInference()  # type: ignore


def LowerTileOp():
    """LowerTileOp

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LowerTileOp()  # type: ignore


def InjectSoftwarePipeline():
    """InjectSoftwarePipeline

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InjectSoftwarePipeline()  # type: ignore


def LegalizeNegativeIndex():
    """Legalize negative indices in buffer loads.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LegalizeNegativeIndex()  # type: ignore


def InjectAssumes():
    """Inject Assumes for natural shape boundary conditions. And convert Assumes in Evaluate(Call(...)) form
    (tvm builtin assume call) to AttrNode form.

    Returns:
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InjectAssumes()


def VerifyParallelLoop():
    """VerifyParallelLoop

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.VerifyParallelLoop()  # type: ignore


def ThreadSync(storage_scope: str):
    """Insert sync between parallel read/write of shared buffers.

    Parameters
    ----------
    storage_scope: str
        The target storage scope.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.ThreadSync(storage_scope)  # type: ignore


def IfStmtBinding():
    """IfStmtBinding

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.IfStmtBinding()  # type: ignore


def MergeIfStmt():
    """MergeIfStmt

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.MergeIfStmt()  # type: ignore


def MergeLoop():
    """MergeLoop

    Merge adjacent For loops that have the same iteration space and
    compatible annotations. Adjacent parallel loops from independent
    copy operations are merged into a single loop with a SeqStmt body,
    reducing loop overhead and improving memory-level parallelism.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.MergeLoop()  # type: ignore


def LoopPeeling():
    """LoopPeeling

    Peel the M/N tail of GEMM block copies when a matrix dimension is not
    divisible by the block size. Shrinks the copy extent along blockIdx-indexed
    dimensions to ``min(block, shape - min)`` so the last grid block copies only
    the remainder instead of reading/writing out of bounds.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LoopPeeling()  # type: ignore


def PadGemmTail():
    """PadGemmTail

    Pad the M/N/K tail of GEMM block copies when a matrix dimension is not
    divisible by the block size. Wraps the copy in an if/else over the padded
    boundary so the main branch proves its loads in-bounds (no per-element
    predicate) while the tail branch keeps the full block extent and reads 0
    for out-of-bounds loads.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.PadGemmTail()  # type: ignore


def LoopUnswitching():
    """LoopUnswitching: Hoist loop-invariant if statements out of loops.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LoopUnswitching()  # type: ignore


def LegalizeVectorizedLoop():
    """LegalizeLoopVectorize

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LegalizeVectorizedLoop()  # type: ignore


def LegalizeSafeMemoryAccess():
    """LegalizeLoopVectorize

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LegalizeSafeMemoryAccess()  # type: ignore


def LowerAccessPtr():
    """Lower TileLang frontend `tl.access_ptr` to `tir.builtin.tvm_access_ptr`.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.LowerAccessPtr()  # type: ignore


def MakePackedAPI():
    """MakePackedAPI

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.MakePackedAPI()  # type: ignore


def MaterializeKernelLaunch(lower_thread_binding: bool = True):
    """Materialize the target-neutral kernel launch nest (thread_binding
    For loops emitted by T.Kernel) into a backend-specific form. Each
    backend pipeline decides the mode for itself:

    Parameters
    ----------
    lower_thread_binding : bool
        If True (SIMT backends, e.g. CUDA/ROCm/Metal), lower the
        blockIdx.*/threadIdx.* loops into thread_extent AttrStmts.
        If False (backends without SIMT, e.g. CPU), lower blockIdx.*
        loops into plain serial For loops and ignore threadIdx.* loops
        (their extents are dropped; the loop vars are pinned to 0).

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.MaterializeKernelLaunch(lower_thread_binding)  # type: ignore


def AnnotateDeviceRegions():
    """AnnotateDeviceRegions

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.AnnotateDeviceRegions()  # type: ignore


def SplitHostDevice():
    """Split host/device functions even for empty kernels.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.SplitHostDevice()  # type: ignore


def AnnotateReadOnlyParams():
    """Annotate read-only handle parameters for PrimFuncs.

    Adds attribute `tl.readonly_param_indices` listing param indices that are
    never written, enabling CUDA codegen to emit `const` qualifiers to unlock
    read-only cache loads.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.AnnotateReadOnlyParams()  # type: ignore


def VectorizeLoop(enable_vectorize: bool = True):
    """VectorizeLoop

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.VectorizeLoop(enable_vectorize)  # type: ignore


def InjectPTSAsyncCopy():
    """Rewrite global↔shared memory copy on TANG with async DMA intrinsics.

    Replaces BufferStore nodes inside ``async_scope`` blocks with
    ``pts_load_async`` / ``pts_store_async`` calls, which later codegen to
    ``async_load`` / ``async_store``. The codegen emits inline asm for
    cop4 (128-bit) DMA when widening is requested via pass config.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.InjectPTSAsyncCopy()  # type: ignore


def HoistCopyAddresses():
    """Hoist copy address computations out of inner loops.

    Converts loop-var-based copy addresses to incremental induction variables,
    reducing integer ALU pressure from ``k * stride`` patterns.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.HoistCopyAddresses()  # type: ignore


def ConfigIndexBitwidth():
    """Config index bitwidth.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    ----
    """
    return _ffi_api.ConfigIndexBitwidth()  # type: ignore


def FlattenBuffer():
    """FlattenBuffer

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.FlattenBuffer()  # type: ignore


def RemoveRedundantSyncs():
    """RemoveRedundantSyncs

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.RemoveRedundantSyncs()  # type: ignore


def MergeSharedMemoryAllocations(enable_aggressive_merge: bool = False, align_bytes: int = 16, disable_reuse: bool = False):
    """MergeSharedMemoryAllocations

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.MergeSharedMemoryAllocations(enable_aggressive_merge, align_bytes, disable_reuse)  # type: ignore


def PlanAndUpdateBufferAllocationLocation():
    """Plan and update buffer allocation locations within PrimFuncs.

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.PlanAndUpdateBufferAllocationLocation()  # type: ignore


def HoistGlobalBufferAllocations():
    """Hoist global buffer allocations to the top of the block (host side).

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.HoistGlobalBufferAllocations()  # type: ignore


def HoistNonRestrictParams():
    return _ffi_api.HoistNonRestrictParams()  # type: ignore


def StorageRewrite():
    """StorageRewrite

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.StorageRewrite()  # type: ignore


def LowerOpaqueBlock():
    """LowerOpaqueBlock"""
    return _ffi_api.LowerOpaqueBlock()  # type: ignore


def LowerThreadAllreduce():
    """LowerThreadAllreduce"""
    return _ffi_api.LowerThreadAllreduce()  # type: ignore


def LowerIntrin():
    """LowerIntrin"""
    return _ffi_api.LowerIntrin()  # type: ignore


def LowerDeviceKernelLaunch():
    """
    Create and return a transform pass that lowers device kernel launch constructs to target-specific IR.

    This pass transforms high-level device kernel launch and related intrinsics into lower-level
    IR suitable for backend code generation and device-side lowering.

    Returns:
        tvm.transform.Pass: The transform pass that performs device kernel launch lowering.
    """
    return _ffi_api.LowerDeviceKernelLaunch()  # type: ignore


def LayoutReducer():
    """
    Return a TVM transform pass that performs layout reduction/normalization.

    This wrapper delegates to the underlying FFI implementation and returns a pass object suitable for use in a PassContext or pass pipeline. The pass is intended to simplify or reduce tensor/layout-related representations during relay/tile transformations.

    Returns:
        The transform pass object produced by the FFI backend.
    """
    return _ffi_api.LayoutReducer()  # type: ignore


def UnrollLoop():
    """Unroll loops as in Halide pipeline.

    This pass unrolls loops based on configuration options including:
    - auto_max_step: Threshold of number of steps to be automatically unrolled
    - auto_max_depth: Maximum nested level of loops that can be automatically unrolled
    - auto_max_extent: Maximum extent of loop that will be unrolled
    - explicit_unroll: Whether to explicitly unroll instead of setting a pragma
    - unroll_local_access: Whether to always unroll local access

    Returns
    -------
    fpass : tvm.transform.Pass
        The result pass
    """
    return _ffi_api.UnrollLoop()  # type: ignore
