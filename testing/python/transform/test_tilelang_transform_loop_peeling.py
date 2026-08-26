"""End-to-end tests for LoopPeeling.

LoopPeeling rewrites GEMM block copies whose M/N dimension is not divisible by
the block size: the last grid block copies only the remainder instead of reading
past the end of the source tile. These tests compile non-divisible GEMMs and
check the result against torch, which fails loudly if the last block reads
out-of-bounds garbage (or crashes on the out-of-bounds access).

The pass is correctness-critical but slow on the peeled tail, so callers that
pad M/N to tile multiples may disable it via TL_DISABLE_LOOP_PEELING. That
disable path is intentionally *not* correctness-tested here: without peeling a
non-divisible shape reads out of bounds, which is exactly what the pass guards
against.
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


def test_loop_peeling_nondivisible_m():
    # M=1023 is not divisible by block_M=128: 1023 = 7*128 + 127.
    _run_and_check(1023, 1024, 512)


def test_loop_peeling_nondivisible_n():
    # N=1022 is not divisible by block_N=128.
    _run_and_check(1024, 1022, 512)


def test_loop_peeling_nondivisible_mn():
    # Both M and N leave a peeled tail.
    _run_and_check(1023, 1022, 512)


def test_loop_peeling_m_smaller_than_block():
    # M=64 < block_M=128: the whole M dim is a peeled tail (single grid block,
    # boundary == 0 so the main branch is dead and the tail covers every row).
    _run_and_check(64, 1024, 512)


def test_loop_peeling_n_smaller_than_block():
    # N=64 < block_N=128: the whole N dim is a peeled tail.
    _run_and_check(1024, 64, 512)


def test_loop_peeling_single_block_mn():
    # Both M and N are smaller than the block: one grid block, both dims peeled.
    _run_and_check(64, 64, 512)


def test_loop_peeling_small_remainder():
    # M=129 = 128 + 1: the tail block copies exactly one row (the extent-1
    # peeled dim is kept as a unit loop so the fragment layout stays invertible).
    _run_and_check(129, 1024, 512)


def test_loop_peeling_n_small_remainder():
    # N=129 = 128 + 1: symmetric unit tail on the N dim.
    _run_and_check(1024, 129, 512)


if __name__ == "__main__":
    tilelang.testing.main()
