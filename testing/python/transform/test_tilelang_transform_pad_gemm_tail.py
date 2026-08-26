"""End-to-end tests for PadGemmTail.

PadGemmTail wraps GEMM block copies whose K dimension is not divisible by
block_K in an if/else over the padded boundary. The main branch keeps the full
block extent and the analyzer proves `min + iv < K`, dropping the per-element
runtime predicate. The tail branch also keeps the full extent: its out-of-bounds
loads are legalized to 0 by the copy lowering, which is equivalent to
zero-padding the accumulation dimension.

These tests compile GEMMs with a non-divisible K and check the result against
torch. Without the pass the last pipeline iteration would read past the end of
A/B (garbage in the accumulator tail) or crash on the out-of-bounds access.

M/N tails are the responsibility of LoopPeeling, not PadGemmTail, so the primary
cases keep M/N divisible by the block size and vary only K. The last case pairs
a non-divisible K with non-divisible M/N to exercise both passes together.
"""

import torch

import tilelang
import tilelang.language as T
import tilelang.testing


def _gemm(M, N, K, block_M, block_N, block_K):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            B_shared = T.alloc_shared((block_K, block_N), "float16")
            C_local = T.alloc_fragment((block_M, block_N), "float16")
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def _run_and_check(M, N, K):
    kernel = tilelang.compile(_gemm(M, N, K, 128, 128, 64), out_idx=[2])
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        return torch.matmul(A.float(), B.float()).half()

    profiler.assert_allclose(ref_program, rtol=1e-2, atol=1e-2)


def test_pad_gemm_tail_nondivisible_k():
    # K=513 is not divisible by block_K=64: 513 = 8*64 + 1.
    _run_and_check(1024, 1024, 513)


def test_pad_gemm_tail_k_smaller_than_block():
    # K=32 < block_K=64: the whole K dim is a single padded tail iteration.
    _run_and_check(1024, 1024, 32)


def test_pad_gemm_tail_k_tail_small_remainder():
    # K=65 = 64 + 1: one full iteration plus a tail that pads 63 columns.
    _run_and_check(1024, 1024, 65)


def test_pad_gemm_tail_k_nondivisible_with_mn():
    # Non-divisible K together with non-divisible M and N: PadGemmTail handles K
    # while LoopPeeling handles M/N in the same kernel.
    _run_and_check(1023, 1022, 513)


if __name__ == "__main__":
    tilelang.testing.main()
