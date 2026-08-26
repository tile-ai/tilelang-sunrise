import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


PIPELINE_CONFIGS = [
    ((0, 1, 2), (0, 0, 1)),
    ((0, 2, 1), (0, 0, 1)),
    ((1, 0, 2), (0, 0, 1)),
    ((1, 2, 0), (0, 0, 1)),
    ((2, 0, 1), (0, 0, 1)),
    ((2, 1, 0), (0, 0, 1)),
    ((0, 1, 2), (0, 0, 2)),
    ((0, 2, 1), (0, 0, 2)),
    ((1, 0, 2), (0, 0, 2)),
    ((1, 2, 0), (0, 0, 2)),
    ((2, 0, 1), (0, 0, 2)),
    ((2, 1, 0), (0, 0, 2)),
    ((0, 1, 2), (0, 0, 3)),
    ((0, 2, 1), (0, 0, 3)),
    ((1, 0, 2), (0, 0, 3)),
    ((1, 2, 0), (0, 0, 3)),
    ((2, 0, 1), (0, 0, 3)),
    ((2, 1, 0), (0, 0, 3)),
]


@tilelang.jit(out_idx=[-1])
def matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    dtype="float16",
    accum_dtype="float",
    order=(0, 1, 2),
    stage=(0, 0, 1),
):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), order=order, stage=stage):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


@pytest.mark.parametrize("order,stage", PIPELINE_CONFIGS)
def test_gemm_single_pipeline(order, stage):
    m = n = k = 256
    block_M = block_N = block_K = 32
    kernel = matmul(m, n, k, block_M, block_N, block_K, order=order, stage=stage)
    a = torch.randn(m, k, dtype=torch.float16).ptpu()
    b = torch.randn(k, n, dtype=torch.float16).ptpu()
    c = kernel(a, b)
    ref_c = torch.matmul(a.cpu(), b.cpu()).half()
    try:
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-1)
    except AssertionError:
        pytest.fail(f"Pipeline config failed: order={order}, stage={stage}")


@tilelang.jit(out_idx=[-1])
def matmul_two_pipes(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    dtype="float16",
    accum_dtype="float",
    order=(0, 1, 2),
    stage=(0, 0, 1),
):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        A_new: T.Tensor((M, K), dtype),
        B_new: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)

            for k in T.Pipelined(T.ceildiv(K, block_K), order=order, stage=stage):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            for k in T.Pipelined(T.ceildiv(K, block_K), order=order, stage=stage):
                T.copy(A_new[by * block_M, k * block_K], A_shared)
                T.copy(B_new[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


@pytest.mark.parametrize("order,stage", PIPELINE_CONFIGS)
@pytest.mark.skip(reason="Multiple pipelines are not yet supported.")
def test_gemm_multi_pipeline(order, stage):
    m = n = k = 256
    block_M = block_N = block_K = 32
    kernel = matmul_two_pipes(m, n, k, block_M, block_N, block_K, order=order, stage=stage)
    a = torch.randn(m, k, dtype=torch.float16).ptpu()
    b = torch.randn(k, n, dtype=torch.float16).ptpu()
    a_new = torch.randn(m, k, dtype=torch.float16).ptpu()
    b_new = torch.randn(k, n, dtype=torch.float16).ptpu()
    c = kernel(a, b, a_new, b_new).cpu()
    ref_c = torch.matmul(a.cpu(), b.cpu()) + torch.matmul(a_new.cpu(), b_new.cpu())
    try:
        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-1)
    except AssertionError:
        pytest.fail(f"Multi-pipeline GEMM failed for order={order}, stage={stage}. Max diff: {(c - ref_c).abs().max().item():.4f}")


if __name__ == "__main__":
    tilelang.testing.main()
