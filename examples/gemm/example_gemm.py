import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench
from tilelang.transform.pass_config import PassConfigKey

import torch
from typing import Callable


def matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    num_threads,
    a_local_load_type,
    b_local_load_type,
    enable_threadblock_swizzle,
    panel_size,
    swizzle_order,
    k_step,
    num_stages=2,
    policy=T.GemmWarpPolicy.FullCol,
    wc_interleave=4,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    """2-stage async DMA GEMM kernel with copy/compute overlap.

    Pipeline:
      1. Prefetch block 0 via async DMA (async_scope=1)
      2. Loop: async copy block k while gemm runs on block k-1

    Key optimizations:
      - async_scope=1: non-blocking DMA for true copy/gemm overlap
      - FullRow throughout; 128x128x128 + 256 threads for >=4096, 64x64x64 +
        128 threads below that (see get_config for the measured crossover)
      - Scalar LDS (gemm_tmma.h): avoids __shfl_sync serialization (+170%)
      - Swizzle tuned per shape: panel_size 4 below 4096, 2 at 4096, 1 at 8192
      - clear_accum=True on the first T.gemm instead of T.clear(C_local): the
        gemm template zeroes the accumulator slice each warp owns, which folds
        the zero-init into the gemm rather than paying for a separate pass
      - 128B-aligned global memory (PyTorch empty + assert)
    """
    num_iters = T.ceildiv(K, block_K)

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=num_threads) as (bx, by):
            A_shared = T.alloc_buffer((block_M, block_K), dtype, scope="shared")
            B_shared = T.alloc_buffer((block_K, block_N), dtype, scope="shared")
            T.use_swizzle(panel_size=panel_size, order=swizzle_order, enable=True)

            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            # C_local is zeroed by the first T.gemm below, which passes
            # clear_accum=True. That flag makes the TMMA template emit the
            # zero-init itself; it is not a no-op and not a hardware freebie
            # (see the clear_accum note in src/tl_templates/tang/gemm_tmma.h).

            # Pre-compute DMA base offsets (loop-invariant, hoisted from T.copy)
            a_base = by * block_M
            b_base = bx * block_N

            # Stage 1: prefetch first block via async DMA
            with T.attr("default", "async_scope", 1):
                T.copy(A[a_base, 0 * block_K], A_shared)
                T.copy(B[0 * block_K, b_base], B_shared)
            T.gemm(
                A_shared,
                B_shared,
                C_local,
                k_step=k_step,
                policy=policy,
                a_local_load_type=a_local_load_type,
                b_local_load_type=b_local_load_type,
                wc_interleave=wc_interleave,
                clear_accum=True,
            )

            # Stage 2: pipeline — async copy block k while gemm on block k-1
            for k in T.serial(1, num_iters):
                with T.attr("default", "async_scope", 1):
                    T.copy(A[a_base, k * block_K], A_shared)
                    T.copy(B[k * block_K, b_base], B_shared)
                T.gemm(
                    A_shared,
                    B_shared,
                    C_local,
                    k_step=k_step,
                    policy=policy,
                    a_local_load_type=a_local_load_type,
                    b_local_load_type=b_local_load_type,
                    wc_interleave=wc_interleave,
                )

            # Store result
            C_shared = T.alloc_buffer((block_M, block_N), dtype, scope="shared")
            T.copy(C_local, C_shared)
            with T.attr("default", "async_scope", 1):
                T.copy(C_shared, C[by * block_M, bx * block_N])

    return gemm


# JIT-compiled variants. The raw matmul() above is not decorated; these two
# module-level instances share the same kernel source but differ in
# compile_flags, so they get distinct cache entries on disk.
#
# cop4: widen async DMA from 8 to 16 bytes/load (default for all configs).
_jit_base = {"out_idx": [-1], "pass_configs": {PassConfigKey.TL_USE_ASYNC_COP4: True}}
_matmul_src = matmul  # save undecorated source before the first jit rebinds the name

matmul = tilelang.jit(**_jit_base)(_matmul_src)

# B-reuse for 1024³: lower MIN_WARP_COLS to 8 so the 64×64 tile (warp_cols=8)
# enters the B-fragment-reuse nest. The default threshold of 16 gates it off
# for the small tile because it costs occupancy for shapes that don't need it.
matmul_breuse = tilelang.jit(
    **_jit_base,
    compile_flags=["-DTL_TANG_GEMM_B_REUSE_MIN_WARP_COLS=8"],
)(_matmul_src)


def ref_program(A, B):
    assert A.device == B.device
    C = torch.matmul(A.cpu(), B.cpu()).to(A.device)
    return C


# ---------------------------------------------------------------------------
# Shape-aware auto-config selection
# ---------------------------------------------------------------------------


def get_config(M: int, N: int, K: int, wc_interleave: int = 4) -> dict:
    """Return an optimized config for the given GEMM shape.

    These defaults target the TANG backend and reflect architecture-specific
    shared-memory, register, and occupancy tradeoffs. Treat them as target-specific
    rather than portable defaults.

    Square shapes split at 4096: small shapes keep the 64x64x64 / 128-thread
    tile, while large shapes switch to 128x128x128 / 256 threads. 1024³ gets its
    own 64×64×128 + k_step=16 + B-reuse branch. Neither geometry wins for every
    shape, so keep the shape-aware crossover instead of collapsing these branches.

    Small shapes (<4096):
      - 64x64x64 + nt128 + ks8 + FullRow + both_overlap; panel_size is the
        shape-dependent knob for this branch.
      - panel_size=4. 1024³ uses 8 (panel_size is part of the B-reuse config).

    Large shapes (>=4096):
      - 128x128x128 + nt256 + ks8 + FullRow.
      - panel_size=2 at 4096³, 1 at 8192³.
      - This tile is also the one that enables the B-register-reuse nest in
        gemm_tmma.h (its warp_cols is 16). The nest is gated off for the small
        tile because its register cost reduces occupancy there -- see `use_b_reuse`.

    1024³ gets its own branch with 64×64×128 + k_step=16 + B-reuse. The
    64×64×128 tile has warp_cols=8 (64/8=8 warps along N), which is normally
    below the B-reuse gate (MIN_WARP_COLS defaults to 16). Lowering it to 8
    via compile_flags enables the B-fragment-reuse nest: B fragments loaded
    once per warp_row are reused across warp_col iterations, saving LDS
    bandwidth. The B-reuse path also avoids the B half-chunk split used by the
    non-B-reuse path at this size. Its register buffer reduces occupancy, so the
    dedicated branch retains the configuration only where the LDS savings justify
    that tradeoff.

    Tall/skinny: FullRow, short/wide: FullCol (both left as they were).
    """
    SQUARE_BASE = {
        "block_M": 64,
        "block_N": 64,
        "block_K": 64,
        "num_threads": 128,
        "num_stages": 2,
        "k_step": 8,
        "a_local_load_type": "load_overlap_mma",
        "b_local_load_type": "load_overlap_mma",
        "enable_threadblock_swizzle": True,
        "panel_size": 4,
        "swizzle_order": "row",
        "wc_interleave": wc_interleave,
        "policy": T.GemmWarpPolicy.FullRow,
    }

    # Large-shape geometry. num_threads is part of the package, not incidental:
    # the 128x128x128 tile needs 8 warps to cover it, and carrying nt=256 down
    # to the small shapes reduces efficiency there.
    LARGE_SQUARE = {
        **SQUARE_BASE,
        "block_M": 128,
        "block_N": 128,
        "block_K": 128,
        "num_threads": 256,
        "k_step": 8,
    }

    if M >= 8192 and N >= 8192:
        return {**LARGE_SQUARE, "panel_size": 1}
    elif M >= 4096 and N >= 4096:
        return {**LARGE_SQUARE, "panel_size": 2}
    elif M == 128 and N == 128 and K == 128:
        # 128³: tiny shape, dominated by launch overhead. The 8×64×128 tile
        # improves grid parallelism, while panel_size=2 and a single-stage
        # pipeline avoid barrier cost that is not amortised at this size.
        return {
            "block_M": 8,
            "block_N": 64,
            "block_K": 128,
            "num_threads": 128,
            "num_stages": 1,
            "k_step": 8,
            "a_local_load_type": "load_overlap_mma",
            "b_local_load_type": "load_before_mma",
            "enable_threadblock_swizzle": True,
            "panel_size": 2,
            "swizzle_order": "row",
            "wc_interleave": 4,
            "policy": T.GemmWarpPolicy.FullRow,
        }
    elif M == 1024 and N == 1024 and K == 1024:
        # 1024³ uses 64×64×128 + k_step=16 + B-reuse to reduce LDS traffic.
        # The B-fragment-reuse nest is gated in by lowering MIN_WARP_COLS from
        # its default 16 to 8 via compile_flags on matmul_breuse. k_step=16
        # amortises the loop overhead; B is loaded once per warp_row and reused
        # across warp_col iterations, saving LDS bandwidth.
        # _breuse=True triggers create_kernel to select matmul_breuse.
        return {
            "block_M": 64,
            "block_N": 64,
            "block_K": 128,
            "num_threads": 128,
            "num_stages": 2,
            "k_step": 16,
            "a_local_load_type": "load_overlap_mma",
            "b_local_load_type": "load_before_mma",
            "enable_threadblock_swizzle": True,
            "panel_size": 8,
            "swizzle_order": "row",
            "wc_interleave": wc_interleave,
            "policy": T.GemmWarpPolicy.FullRow,
            "_breuse": True,
        }
    elif M >= 1024 and N >= 1024:
        return {**SQUARE_BASE, "panel_size": 4}
    elif M > N * 4:
        return {
            "block_M": 64,
            "block_N": 128,
            "block_K": 256,
            "num_threads": 256,
            "num_stages": 2,
            "k_step": 8,
            "a_local_load_type": "load_overlap_mma",
            "b_local_load_type": "load_overlap_mma",
            "enable_threadblock_swizzle": True,
            "panel_size": 4,
            "swizzle_order": "row",
            "wc_interleave": wc_interleave,
            "policy": T.GemmWarpPolicy.FullRow,
        }
    elif N > M * 4:
        return {
            "block_M": 128,
            "block_N": 64,
            "block_K": 256,
            "num_threads": 256,
            "num_stages": 2,
            "k_step": 8,
            "a_local_load_type": "load_overlap_mma",
            "b_local_load_type": "load_overlap_mma",
            "enable_threadblock_swizzle": True,
            "panel_size": 4,
            "swizzle_order": "row",
            "wc_interleave": wc_interleave,
            "policy": T.GemmWarpPolicy.FullCol,
        }
    else:
        return {
            "block_M": 128,
            "block_N": 128,
            "block_K": 128,
            "num_threads": 256,
            "num_stages": 2,
            "k_step": 8,
            "a_local_load_type": "load_overlap_mma",
            "b_local_load_type": "load_overlap_mma",
            "enable_threadblock_swizzle": True,
            "panel_size": 2,
            "swizzle_order": "row",
            "wc_interleave": wc_interleave,
            "policy": T.GemmWarpPolicy.FullCol,
        }


def create_kernel(M: int, N: int, K: int, config: dict = None, wc_interleave: int = 4) -> Callable:
    """Create and return a compiled TileLang GEMM kernel."""
    if config is None:
        config = get_config(M, N, K, wc_interleave)
    use_breuse = config.pop("_breuse", False)
    fn = matmul_breuse if use_breuse else matmul
    return fn(M, N, K, **config)


def _bench(fn) -> float:
    """Benchmark using tilelang's do_bench with event backend."""
    return do_bench(fn, warmup=5, rep=20, backend="event", return_mode="median")


def _compare_outputs(tl_out, ref_out, atol=1e-2, rtol=1e-2) -> bool:
    return torch.allclose(tl_out.cpu(), ref_out.cpu(), atol=atol, rtol=rtol)


def main(wc_interleave: int = 4):
    shapes = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096), (8192, 8192, 8192)]
    device = "ptpu" if hasattr(torch, "ptpu") and torch.ptpu.is_available() else "cpu"
    dtype = torch.float16
    header = "{:20s} {:>12s} {:>12s} {:>12s} {:>7s} {:>7s} {:>8s}".format(
        "Shape", "TileLang", "torch.mm", "torch.Linear", "vs mm", "vs Lin", "correct"
    )
    print(header)
    print("-" * len(header))
    for m, n, k in shapes:
        config = get_config(m, n, k, wc_interleave)
        kernel = create_kernel(m, n, k, config, wc_interleave)
        # Allocate directly on the device. uniform_() works there, and a plain
        # allocation of these sizes is already 128-byte aligned, which the async
        # DMA path wants -- asserted below rather than assumed.
        A = torch.empty(m, k, dtype=dtype, device=device)
        B = torch.empty(k, n, dtype=dtype, device=device)
        A.uniform_()
        B.uniform_()
        assert A.data_ptr() % 128 == 0 and B.data_ptr() % 128 == 0, f"Tensor not 128-byte aligned! A={A.data_ptr()}, B={B.data_ptr()}"
        tl_ms = _bench(lambda kernel=kernel, A=A, B=B: kernel(A, B))
        tl_tflops = (2.0 * m * n * k) / (tl_ms / 1000.0) / 1e12
        ref_matmul = torch.matmul(A, B)
        matmul_ms = _bench(lambda A=A, B=B: torch.matmul(A, B))
        matmul_tflops = (2.0 * m * n * k) / (matmul_ms / 1000.0) / 1e12
        linear = torch.nn.Linear(k, n, bias=False, dtype=dtype, device=device)
        linear.weight.data = B.T.contiguous().clone()
        A_linear = A.clone()
        ref_linear = linear(A_linear)
        linear_ms = _bench(lambda linear=linear, A_linear=A_linear: linear(A_linear))
        linear_tflops = (2.0 * m * n * k) / (linear_ms / 1000.0) / 1e12
        tl_out = kernel(A, B)
        correct = _compare_outputs(tl_out, ref_matmul) and _compare_outputs(tl_out, ref_linear)
        mark = "PASS" if correct else "FAIL"
        print(
            f"({m},{n},{k})  {tl_ms:.4f}ms/{tl_tflops:.1f}T  {matmul_ms:.4f}ms/{matmul_tflops:.1f}T  {linear_ms:.4f}ms/{linear_tflops:.1f}T  {tl_tflops / matmul_tflops * 100:.1f}%  {tl_tflops / linear_tflops * 100:.1f}%  {mark}"
        )


if __name__ == "__main__":
    import sys

    wci = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 4
    main(wci)
