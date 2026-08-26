from __future__ import annotations

# TODO: Add more documentation for each pass config

import logging
import warnings
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class PassConfigKey(str, Enum):
    """Pass configuration keys for TileLang compiler."""

    # TileLang specific configs: TL_XX

    TL_SIMPLIFY = "tl.Simplify"
    """Configuration for TileLang simplification passes.

    This is a dict-based config with the following options:
    - transitively_prove_inequalities: bool, default False
    - convert_boolean_to_and_of_ors: bool, default False
    - apply_constraints_to_boolean_branches: bool, default False
    - propagate_knowns_to_prove_conditional: bool, default False
    - propagate_knowns_to_simplify_expressions: bool, default False
    - enable_simplify_let_inline: bool, default True

    Usage:
        with tvm.transform.PassContext(config={
            "tl.Simplify": {"enable_simplify_let_inline": False}
        }):
            mod = tl.transform.Simplify()(mod)
    """

    # TL_SIMPLIFY sub-config keys
    TL_SIMPLIFY_TRANSITIVELY_PROVE_INEQUALITIES = "transitively_prove_inequalities"
    """Enable transitive inequality proving in simplification. Default: False"""

    TL_SIMPLIFY_CONVERT_BOOLEAN_TO_AND_OF_ORS = "convert_boolean_to_and_of_ors"
    """Convert boolean expressions to AND of ORs form. Default: False"""

    TL_SIMPLIFY_APPLY_CONSTRAINTS_TO_BOOLEAN_BRANCHES = "apply_constraints_to_boolean_branches"
    """Apply constraints to simplify boolean branches. Default: False"""

    TL_SIMPLIFY_PROPAGATE_KNOWNS_TO_PROVE_CONDITIONAL = "propagate_knowns_to_prove_conditional"
    """Propagate known values to prove conditionals. Default: False"""

    TL_SIMPLIFY_PROPAGATE_KNOWNS_TO_SIMPLIFY_EXPRESSIONS = "propagate_knowns_to_simplify_expressions"
    """Propagate known values to simplify expressions. Default: False"""

    TL_SIMPLIFY_ENABLE_LET_INLINE = "enable_simplify_let_inline"
    """Enable inlining of let statements during simplification. Default: True"""

    TL_DISABLE_DATA_RACE_CHECK = "tl.disable_data_race_check"
    """Disable data race check in TileLang. Default: False"""

    TL_DISABLE_PRELOWER_SEMANTIC_CHECK = "tl.disable_prelower_semantic_check"
    """Disable Python-side pre-lower semantic checks. Default: False"""

    TL_DISABLE_WARP_SPECIALIZED = "tl.disable_warp_specialized"
    """Disable warp specialization optimization. Default: False"""

    TL_ENABLE_FAST_MATH = "tl.enable_fast_math"
    """
        Enable fast math optimization. Default: False
        if enabled, --use_fast_math will be passed to nvcc
    """

    TL_PTXAS_REGISTER_USAGE_LEVEL = "tl.ptxas_register_usage_level"
    """The PTXAS register usage level in [0, 10], which controls the
    aggressiveness of optimizations that affect register usage. Default: None"""

    TL_DEVICE_COMPILE_FLAGS = "tl.device_compile_flags"
    """Additional device compiler flags passed to nvcc/NVRTC.

    Accepts either a string (parsed with shell-like splitting) or a list of
    strings. Typical usage is to provide extra include paths, defines or
    ptxas options, e.g.:

    - "-I/opt/include -DMY_SWITCH=1 --ptxas-options=--verbose"
    - ["-I/opt/include", "-DMY_SWITCH=1", "--ptxas-options=--verbose"]

    These flags are appended to the compiler options used in the tvm_ffi
    CUDA compile callback. Default: None
    """

    TL_TANG_DISABLE_WARP_ALU = "tl.tang_disable_warp_alu"
    """Disable the TANG compiler warp-ALU optimization for affected kernels."""

    TL_CONFIG_INDEX_BITWIDTH = "tl.config_index_bitwidth"
    """Bitwidth for configuration indices. Default: 32"""

    TL_DISABLE_TMA_LOWER = "tl.disable_tma_lower"
    """Deprecated flag — prevents plain T.copy() from auto-lowering to TMA store.

    Temporarily re-enabled for backward compatibility. Will be removed in
    v0.1.10.
    """

    TL_DISABLE_SAFE_MEMORY_ACCESS = "tl.disable_safe_memory_legalize"
    """Disable safe memory access optimization. Default: False"""

    TL_DISABLE_VECTORIZE_256 = "tl.disable_vectorize_256"
    """Disable usage of LDG/STG 256. Default: False"""

    TL_ENABLE_ASYNC_COPY = "tl.enable_async_copy"
    """Enable lowering eligible global->shared copies to PTX `cp.async`.

    When True (default), TileLang may lower:
    - `T.copy(global -> shared, ...)` to `ptx_cp_async + commit + wait`
    - `T.async_copy(global -> shared, ...)` to `ptx_cp_async + commit` (no wait)
    - plain user-written global->shared copy stores (e.g. in `T.Parallel`) to
      `ptx_cp_async + commit + wait`

    Important: Automatic cp.async lowering is gated by the surrounding loop
    context. TileLang will only auto-enable cp.async when the copy is observed
    inside a software-pipelined loop annotated with `num_stages > 0`
    (e.g. created by `T.Pipelined(..., num_stages=...)` or by pipeline planning).
    Outside such loops, TileLang will prefer synchronous copy lowering even when
    this flag is True.
    You can request local cp.async injection on a specific parallel loop via
    `T.Parallel(..., prefer_async=True)`.

    When False, TileLang will avoid the cp.async lowering path for `T.copy`.
    Explicit `T.async_copy` still requires cp.async support and may error if
    it cannot be lowered.

    Default: True
    """

    TL_ENABLE_LOWER_LDGSTG = "tl.enable_lower_ldgstg"
    """Enable non-predicated LDG/STG lowering for global memory access.
    When enabled, converts Ramp-based global buffer load/store to ldg/stg intrinsics.
    Default: False"""

    TL_ENABLE_LOWER_LDGSTG_PREDICATED = "tl.enable_lower_ldgstg_predicated"
    """Enable predicated LDG/STG lowering.
    When True, predicated loads (if_then_else with else=0) and
    predicated stores (IfThenElse with empty then case) are lowered to
    ldg/stg intrinsics. Default: False"""

    TL_ENABLE_VECTORIZE_PLANNER_VERBOSE = "tl.enable_vectorize_planner_verbose"
    """Enable verbose output for vectorize planner. When enabled, prints detailed
    information about each buffer's inferred vector size and which buffer determines
    the final vectorization factor. Useful for debugging vectorization issues.
    Default: False"""

    TL_DISABLE_WGMMA = "tl.disable_wgmma"
    """Disable usage of Hopper WGMMA. Default: False"""

    TL_DEBUG_MERGE_SHARED_MEMORY_ALLOCATIONS = "tl.debug_merge_shared_memory_allocations"
    """Enable debug information for merge shared memory allocations. Default: False"""

    TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE = "tl.enable_aggressive_shared_memory_merge"
    """Enable aggressive merge of shared memory allocations. Default: False"""

    TL_DISABLE_SHARED_MEMORY_REUSE = "tl.disable_shared_memory_reuse"
    """Disable shared memory reuse planning in MergeSharedMemoryAllocations.
    When enabled, shared memory allocations are still merged into a single
    allocation but each buffer gets its own dedicated region without lifetime-based
    reuse. Default: False"""

    TL_ENABLE_HOIST_COPY_ADDRESSES = "tl.enable_hoist_copy_addresses"
    """Enable hoisting copy address computations out of inner loops.
    When enabled, the HoistCopyAddresses pass lifts shared-memory
    address calculations out of inner loops to reduce integer ALU pressure.
    Default: False"""

    TL_ENABLE_COPY_STAGING_PAD = "tl.enable_copy_staging_pad"
    """Row-pad fragment→shared staging buffers to remove bank conflicts.
    When enabled, a 2D shared staging buffer whose row stride is a multiple of
    the 128-byte bank stride is laid out with a row padding of 128/element_size
    elements. Default: False"""

    TL_DISABLE_SHUFFLE_ELECT = "tl.disable_shuffle_elect"
    """Disable shuffle election optimization. Default: False"""

    TL_DISABLE_LOOP_UNSWITCHING = "tl.disable_loop_unswitching"
    """Disable loop unswitching optimization. Default: False"""

    TL_LOOP_UNSWITCHING_ALLOW_NON_TRIVIAL_ELSE = "tl.loop_unswitching_allow_non_trivial_else"
    """Allow loop unswitching even when the else-version of the loop body has side effects.

    This is more aggressive and may increase code size. Default: False.
    """

    TL_IF_STMT_BINDING_INLINE_REPLAYABLE_BINDS = "tl.if_stmt_binding_inline_replayable_binds"
    """Inline replayable scalar Bind statements while distributing if conditions.

    When True (default), IfStmtBinding may rewrite a guarded sequence such as
    ``if cond: idx = ids[i]; copy(idx); gemm()`` into separately guarded
    statements with ``idx`` substituted at each use, provided the Bind does not
    read a buffer written inside the same guarded body. This exposes copy and
    compute statements to pipeline planning while preserving non-replayable
    Bind scopes.
    """

    TL_DISABLE_THREAD_STORAGE_SYNC = "tl.disable_thread_storage_sync"
    """Disable thread storage synchronization pass. When enabled, disables the
    automatic insertion of thread synchronization barriers (e.g., __syncthreads())
    for shared memory access coordination. This can be useful for performance
    optimization in cases where manual synchronization is preferred or when
    synchronization is not needed. Default: False"""

    TL_FORCE_LET_INLINE = "tl.force_let_inline"
    """Force TileLang to inline let bindings during simplification. Default: False"""

    TL_AST_PRINT_ENABLE = "tl.ast_print_enable"
    """Enable TIR AST printing for debugging purposes. Default: False"""

    TL_LAYOUT_VISUALIZATION_ENABLE = "tl.layout_visualization_enable"
    """Enable layout inference visualization. Default: False"""

    TL_LAYOUT_VISUALIZATION_FORMATS = "tl.layout_visualization_formats"
    """Layout visualization formats.
    Acceptable values: "pdf", "png", "svg", "all"

    """

    TL_STORAGE_REWRITE_DETECT_INPLACE = "tl.storage_rewrite_detect_inplace"
    """Control StorageRewrite inplace detection.

    When False (default) StorageRewrite keeps distinct temporaries for patterns
    such as `dst[i] = f(src[i])`, avoiding implicit aliasing:

    ```
    read_buf = T.alloc_buffer((1,), T.int32, scope="local.var")
    write_buf = T.alloc_buffer((1,), T.int32, scope="local.var")
    write_buf[0] = read_buf[0] * 2
    f(write_buf[0])
    ```

    Setting the flag to True allows StorageRewrite to reuse the `read` buffer
    for the write when it can prove the update is safely inplace, producing IR
    like:

    ```
    read_buf = T.alloc_buffer((1,), T.int32, scope="local.var")
    read_buf[0] = read_buf[0] * 2
    f(read_buf[0])
    ```

    This reduces local memory usage but introduces aliasing between the buffers.

    Usage:

    ```python
    from tilelang.transform import PassContext, PassConfigKey

    with PassContext(
        config={PassConfigKey.TL_STORAGE_REWRITE_DETECT_INPLACE.value: True}
    ):
        mod = tilelang.transform.StorageRewrite()(mod)
    ```
    """

    # TIR related configs: TIR_XX

    TIR_ENABLE_EQUIV_TERMS_IN_CSE = "tir.enable_equiv_terms_in_cse_tir"
    """Enable equivalent terms in TIR Common Subexpression Elimination. Default: True"""

    TIR_DISABLE_CSE = "tirx.disable_cse_tir"
    """Disable TIR Common Subexpression Elimination. Default: False"""

    TIR_SIMPLIFY = "tirx.Simplify"
    """Enable/disable TIR simplification passes. Default: True"""

    TIR_DISABLE_STORAGE_REWRITE = "tirx.disable_storage_rewrite"
    """Disable storage rewrite optimization. Default: False"""

    TIR_DISABLE_VECTORIZE = "tirx.disable_vectorize"
    """Disable vectorization optimization. Default: False"""

    TIR_USE_ASYNC_COPY = "tirx.use_async_copy"
    """Enable asynchronous memory copy operations. Default: True"""

    TIR_ENABLE_DEBUG = "tirx.enable_debug"
    """Enable debug information in generated code. Default: False"""

    TL_USE_ASYNC_COP4 = "tl.use_async_cop4"
    """Emit cop4 (128-bit) async DMA inline asm in codegen for
    pts_load_async / pts_store_async calls. Disable to fall back to
    cop2-compatible async_load / async_store function calls.
    Default: False."""

    TL_DISABLE_REMOVE_REDUNDANT_SYNCS = "tl.disable_remove_redundant_syncs"
    """Disable RemoveRedundantSyncs, which drops thread barriers it can prove
    fence no shared-memory hazard. Set this when debugging a suspected data
    race: a kernel that is wrong with the pass on and correct with it off points
    at a barrier the pass should have kept. Default: False (the pass runs)."""

    TL_DISABLE_MERGE_LOOP = "tl.disable_merge_loop"
    """Disable MergeLoop, which fuses adjacent For loops over the same iteration
    space. Set this when a kernel is wrong with the pass on and correct with it
    off: MergeLoop runs before ThreadSync, so an illegal fusion removes the slot
    where a barrier would go, and it collapses a run of loops under a single
    AttrStmt wrapper. Also useful for attributing a perf change to the fusion.
    Default: False (the pass runs)."""

    TL_DISABLE_LOOP_PEELING = "tl.disable_loop_peeling"
    """Disable LoopPeeling, which peels the M/N tail of GEMM block copies when a
    matrix dimension is not divisible by the block size. Peeling is the
    correctness backstop for non-divisible shapes: the last grid block copies
    only the remainder instead of reading out of bounds. Set this when shapes are
    padded to tile multiples at the caller level so the peeled tail is never
    reached and the pass only adds dead code. Default: False (the pass runs)."""

    TL_DISABLE_GEMM_PAD = "tl.disable_gemm_pad"
    """Disable PadGemmTail, which pads the M/N/K tail of GEMM block copies when a
    matrix dimension is not divisible by the block size. Padding wraps the copy
    in an if/else so the main branch proves its loads in-bounds (removing the
    per-element predicate that makes non-divisible K slow) while the tail branch
    keeps the full block extent and reads 0 for out-of-bounds loads. Set this to
    fall back to LoopPeeling (or nothing) for A/B comparison. Default: False
    (the pass runs)."""

    TL_GEMM_PAD_M = "tl.gemm_pad_m"
    """Extra padding size (number of elements) for the M dimension. tilelang.jit
    pads A's M dim by this much before invoking the kernel. Default: 0 (no pad)."""

    TL_GEMM_PAD_N = "tl.gemm_pad_n"
    """Extra padding size (number of elements) for the N dimension. tilelang.jit
    pads B's N dim by this much. Default: 0 (no pad)."""

    TL_GEMM_PAD_K = "tl.gemm_pad_k"
    """Extra padding size (number of elements) for the K dimension. tilelang.jit
    pads A's K dim and B's K dim by this much. Default: 0 (no pad)."""

    TL_ENABLE_GEMM_PAD = "tl.enable_gemm_pad"
    """Master switch for data-layer GEMM padding. When True, tilelang.jit pads
    the A/B inputs by tl.gemm_pad_m / tl.gemm_pad_n before invoking the kernel.
    Default: False — data-layer padding is opt-in, so setting only the pad sizes
    (without this flag) leaves the kernel unchanged."""

    TIR_MERGE_STATIC_SMEM = "tirx.merge_static_smem"
    """Merge static shared memory allocations. Default: True"""

    TIR_ADD_LOWER_PASS = "tirx.add_lower_pass"
    """Additional lowering passes to be applied. Default: None"""

    TIR_NOALIAS = "tirx.noalias"
    """Enable pointer non-aliasing assumptions. Default: True"""

    # Output debugging options

    CUDA_KERNELS_OUTPUT_DIR = "cuda.kernels_output_dir"
    """Output directory for generated CUDA kernels. Default: empty string"""

    TL_DISABLE_OUT_OF_BOUND_WARNING = "tl.disable_out_of_bound_warning"
    """Disable out-of-bound access warnings in safe memory access legalization. Default: True"""

    TL_ENABLE_DUMP_IR = "tl.enable_dump_ir"
    """Enable dumping IR during lowering between passes. Default: False"""

    TL_DUMP_IR_DIR = "tl.dump_ir_path"
    """Path to the directory where IR will be dumped. Default: ./dump_ir/"""

    TL_PASS_PROFILE = "tl.pass_profile"
    """Enable per-pass timing profiling. Default: False"""

    TL_PASS_PROFILE_THRESHOLD_MS = "tl.pass_profile_threshold_ms"
    """Only show passes slower than this threshold (ms). 0 = show all. Default: 0"""


_DEPRECATED_PASS_CONFIG_MESSAGES = {
    PassConfigKey.TL_DISABLE_TMA_LOWER.value: (
        "`tl.disable_tma_lower` is deprecated and will be removed in v0.1.10. Use `T.copy(..., disable_tma=True)` per-copy instead."
    ),
}


def depythonize_pass_config_value(value: Any) -> Any:
    """Convert a pass-config value that came back through the FFI into plain Python.

    Values read out of a ``PassContext.config`` or a ``tilelang_pass_configs``
    PrimFunc attr are TVM objects, not the Python objects that were put in:
    ``True`` comes back as ``IntImm``, ``["-O3"]`` as ``ffi.Array``. Neither is
    JSON-serializable, so leaving them in ``pass_configs`` makes the
    ``json.dumps`` in the kernel-cache key raise ``TypeError`` (only observable
    with the cache enabled), and an ``ffi.Array`` also breaks the ``+``
    concatenation that appends compile flags.
    """
    # ffi.String is a str subclass; keep it as one rather than recursing into it
    # as a Sequence of characters.
    if isinstance(value, str):
        return str(value)
    if isinstance(value, Mapping):
        return {depythonize_pass_config_value(k): depythonize_pass_config_value(v) for k, v in value.items()}
    if isinstance(value, Sequence):
        return [depythonize_pass_config_value(v) for v in value]
    # IntImm / FloatImm / StringImm — .value is already a plain int/float/str.
    # A bool config arrives as IntImm(dtype="bool"), which must not degrade to 1.
    dtype = getattr(value, "dtype", None)
    if hasattr(value, "value") and dtype is not None:
        return bool(value.value) if dtype == "bool" else value.value

    # Plain Python scalars need no conversion and are already JSON-serializable.
    # They reach here rather than the FFI branch above because they have no
    # ``.dtype`` -- a config passed directly as ``{"tl.foo": True}`` never went
    # through the FFI. Exempt them before the warning below, which is only for
    # values nothing downstream can handle.
    if value is None or isinstance(value, (bool, int, float)):
        return value

    # Not a shape this function knows how to depythonize. Every registered
    # ``tl.*`` key is bool/int/str/Array, and PassContext type-checks those on
    # insertion, so the only way to get here is an exotic object riding in
    # through a ``tilelang_pass_configs`` func attr, which bypasses that
    # registry. Return it untouched and warn rather than coercing it: the
    # returned value is not only hashed into the cache key, it is fed straight
    # back into a real PassContext (see JITKernel._compile_and_create_adapter),
    # so a ``str(value)`` fallback would either be rejected there with a
    # confusing type error or -- on a String-typed key -- be silently accepted
    # and compile with a bogus config. Failing loudly at the cache key's
    # json.dumps is the better outcome; this warning just names the culprit,
    # which that TypeError does not.
    logger.warning(
        "Pass-config value of unsupported type %r left as-is: %r. Expected "
        "bool/int/float/str/list/dict; this will raise TypeError when the "
        "kernel-cache key is serialized.",
        type(value).__name__,
        value,
    )
    return value


def normalize_pass_configs(pass_configs: dict[str | PassConfigKey, Any] | None) -> dict[str, Any]:
    """Canonicalize known pass-config keys and emit compatibility warnings."""
    if pass_configs is None:
        return {}

    normalized: dict[str, Any] = {}
    warned_keys: set[str] = set()

    for key, value in pass_configs.items():
        normalized_key = key.value if isinstance(key, PassConfigKey) else key

        normalized[normalized_key] = value

        if normalized_key in _DEPRECATED_PASS_CONFIG_MESSAGES and normalized_key not in warned_keys:
            warnings.warn(_DEPRECATED_PASS_CONFIG_MESSAGES[normalized_key], DeprecationWarning, stacklevel=3)
            warned_keys.add(normalized_key)

    return normalized
