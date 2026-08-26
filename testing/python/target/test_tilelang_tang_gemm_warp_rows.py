"""Numerical correctness: warp_rows > 1 (M_Tile=128, num_warp_m=2 → warp_rows=8).

Verifies that the contiguous register layout in GemmTensorOp::body() produces
correct results when a single warp processes multiple rows of MMA tiles, each
with independent A-load indices and a shared contiguous C-local buffer.

Run: pytest testing/python/target/test_tilelang_tang_gemm_warp_rows.py -v
Requires PTPU hardware (S2/S3).
"""

import pytest
import torch
import tilelang
import tilelang.language as T


@pytest.mark.skipif(
    not (hasattr(torch, "ptpu") and torch.ptpu.is_available()),
    reason="PTPU hardware required for numerical correctness test",
)
def test_gemm_warp_rows_gt_one_correctness():
    """M=128, K=32, N=64, threads=64 → warp_rows=8. Run kernel and compare
    with torch.matmul."""
    M, N, K = 128, 64, 32
    block_M, block_N, block_K = 128, 64, 32
    num_threads = 64
    dtype = torch.float16

    @tilelang.jit(out_idx=[-1], target="tang")
    def gemm_kernel(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        num_threads,
        dtype,
    ):
        assert block_M == M and block_N == N and block_K == K, "single-tile test"

        @T.prim_func
        def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=num_threads) as (bx, by):
                a_shared = T.alloc_buffer((block_M, block_K), dtype, scope="shared")
                b_shared = T.alloc_buffer((block_K, block_N), dtype, scope="shared")
                c_local = T.alloc_fragment((block_M, block_N), dtype=T.float32)
                T.clear(c_local)
                T.copy(A[by * block_M, 0 * block_K], a_shared)
                T.copy(B[0 * block_K, bx * block_N], b_shared)
                T.gemm(a_shared, b_shared, c_local, clear_accum=False, policy=T.GemmWarpPolicy.FullCol)
                T.copy(c_local, C[by * block_M, bx * block_N])

        return main

    device = "ptpu"
    A = torch.empty(M, K, dtype=dtype, device=device)
    B = torch.empty(K, N, dtype=dtype, device=device)
    A.uniform_()
    B.uniform_()

    kernel = gemm_kernel(M, N, K, block_M, block_N, block_K, num_threads, dtype)
    C = kernel(A, B)
    ref = torch.matmul(A.float(), B.float())

    assert torch.allclose(C.float().cpu(), ref.cpu(), atol=1e-2, rtol=1e-2), (
        f"warp_rows=8 GEMM mismatch: max_diff={(C.float() - ref).abs().max().item():.6f}"
    )
