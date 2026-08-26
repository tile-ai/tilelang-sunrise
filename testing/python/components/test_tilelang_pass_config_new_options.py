"""Test new pass config keys:

- TL_USE_ASYNC_COP4: cop4 async DMA widening + __cache3__
- TL_DISABLE_REMOVE_REDUNDANT_SYNCS: barrier-preserving escape hatch
- TL_ENABLE_HOIST_COPY_ADDRESSES: copy address hoisting
- TL_DISABLE_MERGE_LOOP: adjacent-loop fusion escape hatch
"""

import json

import torch

import tilelang
import tilelang.language as T
from tilelang.transform.pass_config import PassConfigKey
from tilelang.utils.device import get_current_device


def _make_pipelined_gemm(
    M=512, N=1024, K=768, block_M=128, block_N=256, block_K=32, num_stages=2, k_step=None, num_threads=128, policy=None
):
    """Standard Pipelined GEMM kernel used for correctness verification."""

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=num_threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), T.float16)
            B_shared = T.alloc_shared((block_K, block_N), T.float16)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                gemm_args = {}
                if k_step is not None:
                    gemm_args["k_step"] = k_step
                if policy is not None:
                    gemm_args["policy"] = policy
                T.gemm(A_shared, B_shared, C_local, **gemm_args)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return kernel


def _make_async_dma_gemm(M=256, N=128, K=64, block_M=64, block_N=64, block_K=32, num_threads=128, k_step=4, num_stages=2):
    """2-stage async DMA GEMM with explicit async_scope for DMA + cop4 coverage.

    Used for source-level inspection only — the explicit async_scope triggers
    InjectPTSAsyncCopy and cop4 codegen when enabled.
    """

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=num_threads) as (bx, by):
            A_shared = T.alloc_buffer((block_M, block_K), T.float16, scope="shared")
            B_shared = T.alloc_buffer((block_K, block_N), T.float16, scope="shared")
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(C_local)
            num_iters = T.ceildiv(K, block_K)
            a_base = by * block_M
            b_base = bx * block_N
            with T.attr("default", "async_scope", 1):
                T.copy(A[a_base, 0 * block_K], A_shared)
                T.copy(B[0 * block_K, b_base], B_shared)
            T.gemm(A_shared, B_shared, C_local, k_step=k_step)
            for k in T.serial(1, num_iters):
                with T.attr("default", "async_scope", 1):
                    T.copy(A[a_base, k * block_K], A_shared)
                    T.copy(B[k * block_K, b_base], B_shared)
                T.gemm(A_shared, B_shared, C_local, k_step=k_step)
            C_shared = T.alloc_buffer((block_M, block_N), T.float16, scope="shared")
            T.copy(C_local, C_shared)
            with T.attr("default", "async_scope", 1):
                T.copy(C_shared, C[by * block_M, bx * block_N])

    return kernel


# ---------------------------------------------------------------------------
# TL_USE_ASYNC_COP4
# ---------------------------------------------------------------------------
def test_async_cop4_correctness():
    """Pipelined GEMM with cop4 enabled must produce correct results."""
    M, N, K = 512, 1024, 768
    dtype = torch.float16
    device = get_current_device()

    program = _make_pipelined_gemm(M, N, K)
    kernel = tilelang.compile(
        program,
        out_idx=[-1],
        execution_backend="tvm_ffi",
        pass_configs={PassConfigKey.TL_USE_ASYNC_COP4: True},
    )

    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)
    C = kernel(A, B)

    ref = torch.matmul(A.float(), B.float())
    assert C.shape == (M, N)
    assert torch.allclose(C.float().cpu(), ref.cpu(), atol=1e-2, rtol=1e-2), (
        f"cop4 kernel mismatch: max_diff={(C.float() - ref).abs().max().item():.6f}"
    )


def test_async_cop4_disabled_correctness():
    """Pipelined GEMM with cop4 disabled must produce correct results."""
    M, N, K = 512, 1024, 768
    dtype = torch.float16
    device = get_current_device()

    program = _make_pipelined_gemm(M, N, K)
    kernel = tilelang.compile(
        program,
        out_idx=[-1],
        execution_backend="tvm_ffi",
        pass_configs={PassConfigKey.TL_USE_ASYNC_COP4: False},
    )

    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)
    C = kernel(A, B)

    ref = torch.matmul(A.float(), B.float())
    assert C.shape == (M, N)
    assert torch.allclose(C.float().cpu(), ref.cpu(), atol=1e-2, rtol=1e-2), (
        f"cop4-off kernel mismatch: max_diff={(C.float() - ref).abs().max().item():.6f}"
    )


def test_async_cop4_source_contains_cache3():
    """Generated source with cop4 enabled must contain __cache3__."""
    M, N, K = 512, 1024, 768

    program = _make_pipelined_gemm(M, N, K)
    kernel = tilelang.compile(
        program,
        out_idx=[-1],
        execution_backend="tvm_ffi",
        pass_configs={PassConfigKey.TL_USE_ASYNC_COP4: True},
    )

    source = kernel.get_kernel_source()
    assert "__cache3__" in source, "Expected __cache3__ in generated source with cop4 enabled"


def test_async_cop4_source_contains_cop4():
    """Generated source with cop4 enabled must contain cop4 inline asm.

    Uses the 2-stage async DMA kernel which triggers InjectPTSAsyncCopy, so
    the codegen path emits cop4 inline asm when TL_USE_ASYNC_COP4 is on.
    """
    M, N, K = 256, 128, 64

    program = _make_async_dma_gemm(M, N, K)
    kernel = tilelang.compile(
        program,
        out_idx=[-1],
        execution_backend="tvm_ffi",
        pass_configs={PassConfigKey.TL_USE_ASYNC_COP4: True},
    )

    source = kernel.get_kernel_source()
    assert "cop4" in source, "Expected cop4 inline asm in generated source with cop4 enabled"


def test_async_cop4_source_no_cop4():
    """Generated source without cop4 must NOT contain cop4 inline asm.

    Uses the 2-stage async DMA kernel; verifies that cop4 asm is absent when
    TL_USE_ASYNC_COP4 is off.
    """
    M, N, K = 256, 128, 64

    program = _make_async_dma_gemm(M, N, K)
    kernel = tilelang.compile(
        program,
        out_idx=[-1],
        execution_backend="tvm_ffi",
        pass_configs={PassConfigKey.TL_USE_ASYNC_COP4: False},
    )

    source = kernel.get_kernel_source()
    assert "cop4" not in source, "cop4 inline asm must not appear when TL_USE_ASYNC_COP4 is off"


def test_async_cop4_disabled_no_cache3():
    """Generated source without cop4 must NOT contain __cache3__."""
    M, N, K = 512, 1024, 768

    program = _make_pipelined_gemm(M, N, K)
    kernel = tilelang.compile(
        program,
        out_idx=[-1],
        execution_backend="tvm_ffi",
    )

    source = kernel.get_kernel_source()
    assert "__cache3__" not in source, "__cache3__ must not appear when TL_USE_ASYNC_COP4 is off"


# ---------------------------------------------------------------------------
# TL_DISABLE_REMOVE_REDUNDANT_SYNCS
# ---------------------------------------------------------------------------
def test_remove_redundant_syncs_correctness():
    """Kernel with barrier removal pass ON must produce correct results.

    This is the primary regression test: if the pass drops a load-bearing
    barrier, this test will fail.
    """
    M, N, K = 512, 1024, 768
    dtype = torch.float16
    device = get_current_device()

    program = _make_pipelined_gemm(M, N, K)
    kernel = tilelang.compile(
        program,
        out_idx=[-1],
        execution_backend="tvm_ffi",
    )

    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)
    C = kernel(A, B)

    ref = torch.matmul(A.float(), B.float())
    assert C.shape == (M, N)
    assert torch.allclose(C.float().cpu(), ref.cpu(), atol=1e-2, rtol=1e-2), (
        f"barrier-removal kernel mismatch: max_diff={(C.float() - ref).abs().max().item():.6f}"
    )


def test_disable_remove_redundant_syncs_correctness():
    """Kernel with barrier removal DISABLED must produce correct results.

    If disabling the pass surfaces wrong results, the pass is dropping a
    barrier that fences a real hazard (load-bearing barrier).
    """
    M, N, K = 512, 1024, 768
    dtype = torch.float16
    device = get_current_device()

    program = _make_pipelined_gemm(M, N, K)
    kernel = tilelang.compile(
        program,
        out_idx=[-1],
        execution_backend="tvm_ffi",
        pass_configs={PassConfigKey.TL_DISABLE_REMOVE_REDUNDANT_SYNCS: True},
    )

    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)
    C = kernel(A, B)

    ref = torch.matmul(A.float(), B.float())
    assert C.shape == (M, N)
    assert torch.allclose(C.float().cpu(), ref.cpu(), atol=1e-2, rtol=1e-2), (
        f"barrier-preserving kernel mismatch: max_diff={(C.float() - ref).abs().max().item():.6f}"
    )


def test_disable_remove_redundant_syncs_consistent():
    """Enabled vs disabled pass must produce identical numerical results.

    Any discrepancy indicates the pass is removing a load-bearing barrier.
    """
    M, N, K = 512, 1024, 768
    dtype = torch.float16
    device = get_current_device()

    program = _make_pipelined_gemm(M, N, K)

    kernel_on = tilelang.compile(
        program,
        out_idx=[-1],
        execution_backend="tvm_ffi",
    )
    kernel_off = tilelang.compile(
        program,
        out_idx=[-1],
        execution_backend="tvm_ffi",
        pass_configs={PassConfigKey.TL_DISABLE_REMOVE_REDUNDANT_SYNCS: True},
    )

    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)

    C_on = kernel_on(A, B)
    C_off = kernel_off(A, B)

    assert torch.allclose(C_on.float().cpu(), C_off.float().cpu(), atol=1e-4, rtol=1e-4), (
        f"RemoveRedundantSyncs on/off mismatch: max_diff={(C_on.float() - C_off.float()).abs().max().item():.6e}"
    )


def test_remove_redundant_syncs_fullcol():
    """FullCol warp policy + k_step=4 exercises barrier layout with multiple
    warp columns, covering Pattern 3 (WAR guard on shared reads before sync)
    and adjacent-sync dedup in the GEMM body.
    """
    M, N, K = 512, 1024, 768
    dtype = torch.float16
    device = get_current_device()

    program = _make_pipelined_gemm(M, N, K, k_step=4, policy=T.GemmWarpPolicy.FullCol)
    kernel_on = tilelang.compile(
        program,
        out_idx=[-1],
        execution_backend="tvm_ffi",
    )
    kernel_off = tilelang.compile(
        program,
        out_idx=[-1],
        execution_backend="tvm_ffi",
        pass_configs={PassConfigKey.TL_DISABLE_REMOVE_REDUNDANT_SYNCS: True},
    )

    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)

    C_on = kernel_on(A, B)
    C_off = kernel_off(A, B)
    ref = torch.matmul(A.float(), B.float())

    assert torch.allclose(C_on.float().cpu(), ref.cpu(), atol=1e-2, rtol=1e-2), "FullCol+k_step=4 kernel mismatch vs ref"
    assert torch.allclose(C_on.float().cpu(), C_off.float().cpu(), atol=1e-4, rtol=1e-4), (
        f"FullCol+k_step=4 on/off mismatch: max_diff={(C_on.float() - C_off.float()).abs().max().item():.6e}"
    )


def test_remove_redundant_syncs_square():
    """Square warp policy + k_step=8 exercises warp-uniform barrier layout,
    covering Pattern 4 (leading sync removal at outermost scope) and
    Pattern 1 (three-barrier producer window) in the pipeline body.
    """
    M, N, K = 512, 1024, 768
    dtype = torch.float16
    device = get_current_device()

    program = _make_pipelined_gemm(M, N, K, k_step=8, policy=T.GemmWarpPolicy.Square, block_M=64, block_N=64, block_K=128, num_threads=256)
    kernel_on = tilelang.compile(
        program,
        out_idx=[-1],
        execution_backend="tvm_ffi",
    )
    kernel_off = tilelang.compile(
        program,
        out_idx=[-1],
        execution_backend="tvm_ffi",
        pass_configs={PassConfigKey.TL_DISABLE_REMOVE_REDUNDANT_SYNCS: True},
    )

    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)

    C_on = kernel_on(A, B)
    C_off = kernel_off(A, B)
    ref = torch.matmul(A.float(), B.float())

    assert torch.allclose(C_on.float().cpu(), ref.cpu(), atol=1e-2, rtol=1e-2), "Square+k_step=8 kernel mismatch vs ref"
    assert torch.allclose(C_on.float().cpu(), C_off.float().cpu(), atol=1e-4, rtol=1e-4), (
        f"Square+k_step=8 on/off mismatch: max_diff={(C_on.float() - C_off.float()).abs().max().item():.6e}"
    )


def test_remove_redundant_syncs_fullrow():
    """FullRow warp policy exercises barrier layout with multiple warp rows,
    covering adjacent-sync dedup and Pattern 1 in the row-dominant path.
    """
    M, N, K = 512, 1024, 768
    dtype = torch.float16
    device = get_current_device()

    program = _make_pipelined_gemm(M, N, K, k_step=4, policy=T.GemmWarpPolicy.FullRow)
    kernel_on = tilelang.compile(
        program,
        out_idx=[-1],
        execution_backend="tvm_ffi",
    )
    kernel_off = tilelang.compile(
        program,
        out_idx=[-1],
        execution_backend="tvm_ffi",
        pass_configs={PassConfigKey.TL_DISABLE_REMOVE_REDUNDANT_SYNCS: True},
    )

    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)

    C_on = kernel_on(A, B)
    C_off = kernel_off(A, B)
    ref = torch.matmul(A.float(), B.float())

    assert torch.allclose(C_on.float().cpu(), ref.cpu(), atol=1e-2, rtol=1e-2), "FullRow+k_step=4 kernel mismatch vs ref"
    assert torch.allclose(C_on.float().cpu(), C_off.float().cpu(), atol=1e-4, rtol=1e-4), (
        f"FullRow+k_step=4 on/off mismatch: max_diff={(C_on.float() - C_off.float()).abs().max().item():.6e}"
    )


def test_remove_redundant_syncs_3stage():
    """3-stage pipelined GEMM exercises deeper copy/compute overlap with more
    barriers, covering Pattern 1 (three-barrier windows) and multi-barrier
    adjacent dedup under higher barrier density.
    """
    M, N, K = 512, 1024, 768
    dtype = torch.float16
    device = get_current_device()

    program = _make_pipelined_gemm(M, N, K, num_stages=3, k_step=4)
    kernel_on = tilelang.compile(
        program,
        out_idx=[-1],
        execution_backend="tvm_ffi",
    )
    kernel_off = tilelang.compile(
        program,
        out_idx=[-1],
        execution_backend="tvm_ffi",
        pass_configs={PassConfigKey.TL_DISABLE_REMOVE_REDUNDANT_SYNCS: True},
    )

    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)

    C_on = kernel_on(A, B)
    C_off = kernel_off(A, B)
    ref = torch.matmul(A.float(), B.float())

    assert torch.allclose(C_on.float().cpu(), ref.cpu(), atol=1e-2, rtol=1e-2), "3-stage k_step=4 kernel mismatch vs ref"
    assert torch.allclose(C_on.float().cpu(), C_off.float().cpu(), atol=1e-4, rtol=1e-4), (
        f"3-stage k_step=4 on/off mismatch: max_diff={(C_on.float() - C_off.float()).abs().max().item():.6e}"
    )


# ---------------------------------------------------------------------------
# TL_ENABLE_HOIST_COPY_ADDRESSES
# ---------------------------------------------------------------------------
def test_hoist_copy_addresses_correctness():
    """Kernel compiled with copy-address hoisting must produce correct results."""
    M, N, K = 512, 1024, 768
    dtype = torch.float16
    device = get_current_device()

    program = _make_pipelined_gemm(M, N, K)
    kernel = tilelang.compile(
        program,
        out_idx=[-1],
        execution_backend="tvm_ffi",
        pass_configs={PassConfigKey.TL_ENABLE_HOIST_COPY_ADDRESSES: True},
    )

    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)
    C = kernel(A, B)

    ref = torch.matmul(A.float(), B.float())
    assert C.shape == (M, N)
    assert torch.allclose(C.float().cpu(), ref.cpu(), atol=1e-2, rtol=1e-2), (
        f"hoist-addr kernel mismatch: max_diff={(C.float() - ref).abs().max().item():.6f}"
    )


def test_hoist_copy_addresses_consistent():
    """Enabled vs disabled hoist must produce identical numerical results."""
    M, N, K = 512, 1024, 768
    dtype = torch.float16
    device = get_current_device()

    program = _make_pipelined_gemm(M, N, K)

    kernel_on = tilelang.compile(
        program,
        out_idx=[-1],
        execution_backend="tvm_ffi",
        pass_configs={PassConfigKey.TL_ENABLE_HOIST_COPY_ADDRESSES: True},
    )
    kernel_off = tilelang.compile(
        program,
        out_idx=[-1],
        execution_backend="tvm_ffi",
        pass_configs={PassConfigKey.TL_ENABLE_HOIST_COPY_ADDRESSES: False},
    )

    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)

    C_on = kernel_on(A, B)
    C_off = kernel_off(A, B)

    assert torch.allclose(C_on.float().cpu(), C_off.float().cpu(), atol=1e-4, rtol=1e-4), (
        f"HoistCopyAddresses on/off mismatch: max_diff={(C_on.float() - C_off.float()).abs().max().item():.6e}"
    )


def test_merge_loop_correctness():
    """MergeLoop (on by default) must not change GEMM results.

    The IR-level tests in test_tilelang_transform_merge_loop.py cover the
    dependency analysis on hand-built loop pairs; this exercises the pass in a
    real pipeline, where it runs before ThreadSync and StorageRewrite and an
    illegal fusion would strip a barrier or break shared-memory reuse.
    """
    M, N, K = 512, 1024, 768
    dtype = torch.float16
    device = get_current_device()

    program = _make_pipelined_gemm(M, N, K)
    kernel = tilelang.compile(
        program,
        out_idx=[-1],
        execution_backend="tvm_ffi",
        pass_configs={PassConfigKey.TL_DISABLE_MERGE_LOOP: False},
    )

    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)

    C = kernel(A, B)
    ref = A.float().cpu() @ B.float().cpu()

    assert torch.allclose(C.float().cpu(), ref, atol=1e-2, rtol=1e-2), (
        f"MergeLoop broke GEMM: max_diff={(C.float().cpu() - ref).abs().max().item():.6e}"
    )


def test_merge_loop_consistent():
    """Enabled vs disabled MergeLoop must produce identical numerical results.

    Fusing two adjacent loops reorders their iterations against each other
    (iteration i of the second body now runs before iteration i+1 of the
    first), so any missed dependence shows up here as a numerical mismatch.
    """
    M, N, K = 512, 1024, 768
    dtype = torch.float16
    device = get_current_device()

    program = _make_pipelined_gemm(M, N, K)

    kernel_on = tilelang.compile(
        program,
        out_idx=[-1],
        execution_backend="tvm_ffi",
        pass_configs={PassConfigKey.TL_DISABLE_MERGE_LOOP: False},
    )
    kernel_off = tilelang.compile(
        program,
        out_idx=[-1],
        execution_backend="tvm_ffi",
        pass_configs={PassConfigKey.TL_DISABLE_MERGE_LOOP: True},
    )

    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)

    C_on = kernel_on(A, B)
    C_off = kernel_off(A, B)

    assert torch.allclose(C_on.float().cpu(), C_off.float().cpu(), atol=1e-4, rtol=1e-4), (
        f"MergeLoop on/off mismatch: max_diff={(C_on.float() - C_off.float()).abs().max().item():.6e}"
    )


def test_merge_loop_consistent_across_warp_policies():
    """The fusion must hold across warp policies and pipeline depths.

    Each policy lays the shared-memory copies out differently, so a fusion that
    is legal under one can still be wrong under another; 3 stages adds a second
    shared buffer per operand, which is where a missed WAR would surface.

    Shapes and per-policy tile geometry match the RemoveRedundantSyncs variants
    above: Square needs the 64x64x128/256-thread tiling for k_step=8 to divide
    K/tile_size_k, otherwise gemm_tmma.h's inner_k static_assert fires.
    """
    M, N, K = 512, 1024, 768
    dtype = torch.float16
    device = get_current_device()

    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)
    ref = A.float().cpu() @ B.float().cpu()

    variants = [
        ("FullRow", dict(k_step=4, policy=T.GemmWarpPolicy.FullRow)),
        ("FullCol", dict(k_step=4, policy=T.GemmWarpPolicy.FullCol)),
        ("Square", dict(k_step=8, policy=T.GemmWarpPolicy.Square, block_M=64, block_N=64, block_K=128, num_threads=256)),
        ("3stage", dict(num_stages=3, k_step=4)),
    ]

    for label, kwargs in variants:
        program = _make_pipelined_gemm(M, N, K, **kwargs)
        kernel_on = tilelang.compile(
            program,
            out_idx=[-1],
            execution_backend="tvm_ffi",
            pass_configs={PassConfigKey.TL_DISABLE_MERGE_LOOP: False},
        )
        kernel_off = tilelang.compile(
            program,
            out_idx=[-1],
            execution_backend="tvm_ffi",
            pass_configs={PassConfigKey.TL_DISABLE_MERGE_LOOP: True},
        )
        C_on = kernel_on(A, B)
        C_off = kernel_off(A, B)

        assert torch.allclose(C_on.float().cpu(), ref, atol=1e-2, rtol=1e-2), (
            f"MergeLoop broke GEMM under {label}: max_diff={(C_on.float().cpu() - ref).abs().max().item():.6e}"
        )
        assert torch.allclose(C_on.float().cpu(), C_off.float().cpu(), atol=1e-4, rtol=1e-4), (
            f"MergeLoop on/off mismatch under {label}: max_diff={(C_on.float() - C_off.float()).abs().max().item():.6e}"
        )


def test_outer_pass_context_is_inherited_without_explicit_pass_configs():
    """A ``tl.*`` config set on an enclosing PassContext must reach the kernel
    even when compile() is given no explicit ``pass_configs``.

    Every other test here passes pass_configs= directly, which is why this path
    went uncovered: the None-to-{} normalization used to run *after* the merge
    loop, so ``key not in None`` raised TypeError and the except downgraded it to
    a warning — silently discarding the entire outer context for the common case
    of a kernel compiled without explicit configs.
    """
    program = _make_pipelined_gemm(256, 256, 256)

    with tilelang.transform.PassContext(config={PassConfigKey.TL_USE_ASYNC_COP4: True}):
        kernel = tilelang.compile(program, out_idx=[-1], execution_backend="tvm_ffi")

    assert kernel.pass_configs.get(PassConfigKey.TL_USE_ASYNC_COP4), (
        f"outer PassContext config was dropped; kernel saw {dict(kernel.pass_configs)}"
    )


def test_explicit_pass_configs_override_outer_pass_context():
    """An explicit pass_configs entry must win over the enclosing PassContext."""
    program = _make_pipelined_gemm(256, 256, 256)

    with tilelang.transform.PassContext(config={PassConfigKey.TL_USE_ASYNC_COP4: True}):
        kernel = tilelang.compile(
            program,
            out_idx=[-1],
            execution_backend="tvm_ffi",
            pass_configs={PassConfigKey.TL_USE_ASYNC_COP4: False},
        )

    assert not kernel.pass_configs.get(PassConfigKey.TL_USE_ASYNC_COP4), "explicit pass_configs must override the outer PassContext"


def _pc_strings(kernel):
    """kernel.pass_configs mixes PassConfigKey members (from a decorator) with
    plain strings (merged from the outer context); compare as strings."""
    return {k.value if hasattr(k, "value") else str(k): v for k, v in dict(kernel.pass_configs).items()}


def test_compile_does_not_mutate_caller_pass_configs():
    """compile() must not merge the outer PassContext into the caller's dict.

    For a @tilelang.jit kernel the dict handed to compile() *is* the decorator's
    own self.pass_configs, so merging in place made every inherited key a
    permanent part of the kernel's config and leaked it into later calls made
    under a different (or no) PassContext.
    """
    program = _make_pipelined_gemm(256, 256, 256)
    caller_configs = {PassConfigKey.TL_USE_ASYNC_COP4: True}
    before = dict(caller_configs)

    with tilelang.transform.PassContext(config={PassConfigKey.TL_DISABLE_REMOVE_REDUNDANT_SYNCS: True}):
        tilelang.compile(program, out_idx=[-1], execution_backend="tvm_ffi", pass_configs=caller_configs)

    assert caller_configs == before, f"compile() mutated the caller's pass_configs: {caller_configs} != {before}"


def test_outer_pass_context_is_part_of_the_kernel_cache_key():
    """Two identical @tilelang.jit calls under different PassContexts must give
    two different kernels.

    JITImpl keys its kernel cache on parse_args (argument shapes/dtypes), which
    does not include the enclosing PassContext. Without folding the context into
    that key, the first compiled kernel was returned for every later context —
    so `with PassContext(config=...)` around an unchanged call was silently
    ignored, and any A/B sweep over pass configs measured one kernel twice.
    """

    @tilelang.jit(out_idx=[-1], pass_configs={PassConfigKey.TL_USE_ASYNC_COP4: True})
    def kernel_factory(M, N, K, block_M, block_N, block_K):
        @T.prim_func
        def kernel(
            A: T.Tensor((M, K), T.float16),
            B: T.Tensor((K, N), T.float16),
            C: T.Tensor((M, N), T.float16),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), T.float16)
                B_shared = T.alloc_shared((block_K, block_N), T.float16)
                C_local = T.alloc_fragment((block_M, block_N), T.float32)
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, C[by * block_M, bx * block_N])

        return kernel

    args = (256, 256, 256, 64, 64, 32)

    # Compile once with no extra config, then again with one set. Identical call
    # form both times, so only the context distinguishes them.
    with tilelang.transform.PassContext(config={}):
        plain = kernel_factory(*args)
    with tilelang.transform.PassContext(config={PassConfigKey.TL_DISABLE_REMOVE_REDUNDANT_SYNCS: True}):
        gated = kernel_factory(*args)

    assert plain is not gated, "same call form under two PassContexts returned the same kernel object"

    key = PassConfigKey.TL_DISABLE_REMOVE_REDUNDANT_SYNCS.value
    assert key not in _pc_strings(plain), f"config leaked into the no-config kernel: {_pc_strings(plain)}"
    assert key in _pc_strings(gated), f"outer config missing from the second kernel: {_pc_strings(gated)}"


def test_inherited_pass_configs_are_json_serializable():
    """Inherited config values must be plain Python, not FFI objects.

    ``PassContext.current().config`` hands back what the FFI made of the values:
    ``True`` as ``IntImm``, ``["-O3"]`` as ``ffi.Array``. Merging those in
    verbatim made the ``json.dumps`` in KernelCache._generate_key raise
    ``TypeError: Object of type IntImm is not JSON serializable`` — a failure
    invisible to anyone running with ``TILELANG_DISABLE_CACHE=1``, since that
    path skips key generation entirely. Assert on the values directly so the
    check holds either way.
    """
    program = _make_pipelined_gemm(256, 256, 256)

    with tilelang.transform.PassContext(
        config={
            PassConfigKey.TL_USE_ASYNC_COP4: True,
            PassConfigKey.TL_CONFIG_INDEX_BITWIDTH: 32,
            PassConfigKey.TL_DEVICE_COMPILE_FLAGS: ["-O3"],
        }
    ):
        kernel = tilelang.compile(program, out_idx=[-1], execution_backend="tvm_ffi")

    configs = _pc_strings(kernel)
    json.dumps(configs, sort_keys=True)  # raises TypeError if an FFI object slipped through

    cop4 = configs[PassConfigKey.TL_USE_ASYNC_COP4.value]
    assert cop4 is True, f"bool config depythonized to {type(cop4).__name__} {cop4!r}"
    bitwidth = configs[PassConfigKey.TL_CONFIG_INDEX_BITWIDTH.value]
    assert bitwidth == 32 and type(bitwidth) is int, f"int config depythonized to {type(bitwidth).__name__} {bitwidth!r}"
    flags = configs[PassConfigKey.TL_DEVICE_COMPILE_FLAGS.value]
    assert flags == ["-O3"] and type(flags) is list, f"list config depythonized to {type(flags).__name__} {flags!r}"


def test_annotated_pass_configs_are_json_serializable():
    """Same requirement for T.annotate_pass_configs.

    Those land in a ``tilelang_pass_configs`` PrimFunc attr and are read back
    through the FFI too, so they hit the identical cache-key failure.
    """

    @T.prim_func
    def kernel(
        A: T.Tensor((256, 256), T.float16),
        B: T.Tensor((256, 256), T.float16),
        C: T.Tensor((256, 256), T.float16),
    ):
        T.annotate_pass_configs(
            {
                PassConfigKey.TL_USE_ASYNC_COP4: True,
                PassConfigKey.TL_DEVICE_COMPILE_FLAGS: ["-O3"],
            }
        )
        with T.Kernel(2, 2, threads=128) as (bx, by):
            C[bx, by] = A[bx, by] + B[bx, by]

    compiled = tilelang.compile(kernel, out_idx=[-1], execution_backend="tvm_ffi")

    configs = _pc_strings(compiled)
    json.dumps(configs, sort_keys=True)

    assert configs[PassConfigKey.TL_USE_ASYNC_COP4.value] is True
    flags = configs[PassConfigKey.TL_DEVICE_COMPILE_FLAGS.value]
    assert flags == ["-O3"] and type(flags) is list, f"annotated list config depythonized to {type(flags).__name__} {flags!r}"


if __name__ == "__main__":
    tilelang.testing.main()
