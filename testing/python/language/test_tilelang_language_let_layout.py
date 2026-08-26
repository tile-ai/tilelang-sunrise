"""
Test buffer accesses through LetStmt expressions on the LLVM backend.

This test validates that TileLang correctly handles buffer accesses that flow
through let bindings. For example:

    block_mask_l = T.alloc_local((N_S,), T.int32)
    T.copy(BlockMask[by, :], block_mask_l)
    for i in T.Pipelined(N_S):
        a = block_mask_l[i]  # LetStmt: a is bound to a buffer load
        T.copy(A[a, 0], A_local)  # a is used as index in a copy

Key scenario tested:
1. Buffer access through let bindings
"""

import tilelang
import tilelang.language as T
import tilelang.testing
import torch


def blocksparse_copy_kernel(M, N, N_S, block_M, block_N, dtype=T.float16):
    """BlockSparse copy kernel using a local buffer for block mask indices."""
    block_mask_shape = (M // block_M, N_S)

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        BlockMask: T.Tensor(block_mask_shape, T.int32),
    ):
        with T.Kernel(T.ceildiv(M, block_M)) as by:
            A_local = T.alloc_local((block_M, block_N), dtype)
            B_local = T.alloc_local((block_M, block_N), dtype)
            block_mask_l = T.alloc_local((N_S,), T.int32)

            T.clear(B_local)
            T.copy(BlockMask[by, :], block_mask_l)
            for i in T.Pipelined(N_S):
                a = block_mask_l[i]  # LetStmt: buffer access
                if a >= 0:
                    T.copy(A[a, 0], A_local)
                    T.copy(A_local, B[by * block_M : (by + 1) * block_M, i * block_N : (i + 1) * block_N])

    return main


def ref_blocksparse_copy(A, B, BlockMask, M, N, N_S, block_M, block_N):
    """Reference implementation for blocksparse copy."""
    ref_B = B.clone()
    num_row_blocks = M // block_M

    for by in range(num_row_blocks):
        for i in range(N_S):
            src_row_start = BlockMask[by, i].item()
            ref_B[by * block_M : (by + 1) * block_M, i * block_N : (i + 1) * block_N] = A[
                src_row_start : src_row_start + block_M, 0:block_N
            ]

    return ref_B


def run_blocksparse_copy(M, N, block_M, block_N):
    """Run blocksparse copy test with given parameters."""
    N_S = N // block_N

    program = blocksparse_copy_kernel(M, N, N_S, block_M, block_N)
    kernel = tilelang.compile(
        program,
        out_idx=[1],
        target="llvm",
    )

    # Initialize tensors
    a = torch.randn(M, N, dtype=torch.float16)
    b = torch.zeros(M, N, dtype=torch.float16)

    # Create BlockMask with valid row indices
    num_row_blocks = M // block_M
    block_mask = torch.zeros((num_row_blocks, N_S), dtype=torch.int32)
    for by in range(num_row_blocks):
        for i in range(N_S):
            max_row_block = (M - block_M) // block_M
            block_mask[by, i] = torch.randint(0, max_row_block + 1, (1,)).item() * block_M

    # Run kernel
    c = kernel(a, block_mask)

    # Compute reference
    ref_c = ref_blocksparse_copy(a, b, block_mask, M, N, N_S, block_M, block_N)

    # Verify
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_llvm
def test_blocksparse_copy():
    """Test blocksparse copy with let-bound buffer indices on LLVM."""
    run_blocksparse_copy(M=1024, N=1024, block_M=128, block_N=128)


if __name__ == "__main__":
    tilelang.testing.main()
