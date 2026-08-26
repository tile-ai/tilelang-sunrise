"""Numerical correctness: clear_accum=True must actually zero the accumulator.

The obvious test for clear_accum -- run a gemm with it and compare against
torch -- passes even when the flag does nothing at all, because a freshly
launched kernel usually finds its registers already zero. That false negative is
not hypothetical: it is what let a broken clear_accum ship, with "0 mismatched
elements at 1024**3 and 2048**3" recorded as evidence.

So these tests POISON the accumulator instead of trusting its initial state:
C_local is pre-filled with a large non-zero value, and clear_accum=True must
erase it. If the flag is a no-op the error equals the poison exactly, which is a
much louder signal than a few mismatched elements.

Run: pytest testing/python/target/test_tilelang_tang_gemm_clear_accum.py -v
Requires PTPU hardware (S2/S3).
"""

import pytest
import torch
import tilelang
import tilelang.language as T
from tilelang.transform.pass_config import PassConfigKey

requires_ptpu = pytest.mark.skipif(
    not (hasattr(torch, "ptpu") and torch.ptpu.is_available()),
    reason="PTPU hardware required for numerical correctness test",
)

POISON = 1000.0


def _build(M, N, K, block_M, block_N, block_K, num_threads, k_step, policy, poison):
    """A k-looped gemm whose first call carries clear_accum=True.

    When poison != 0 the accumulator is pre-filled with it, so the kernel is
    only correct if clear_accum genuinely overwrites C_local.
    """

    @tilelang.jit(
        out_idx=[-1],
        target="tang",
        pass_configs={PassConfigKey.TL_USE_ASYNC_COP4: True},
    )
    def kernel():

        @T.prim_func
        def main(
            A: T.Tensor((M, K), T.float16),
            B: T.Tensor((K, N), T.float16),
            C: T.Tensor((M, N), T.float32),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=num_threads) as (bx, by):
                a_shared = T.alloc_buffer((block_M, block_K), T.float16, scope="shared")
                b_shared = T.alloc_buffer((block_K, block_N), T.float16, scope="shared")
                c_local = T.alloc_fragment((block_M, block_N), T.float32)

                if poison != 0.0:
                    T.fill(c_local, poison)

                for k in T.serial(T.ceildiv(K, block_K)):
                    T.copy(A[by * block_M, k * block_K], a_shared)
                    T.copy(B[k * block_K, bx * block_N], b_shared)
                    # Only the first gemm clears; the rest must accumulate.
                    if k == 0:
                        T.gemm(a_shared, b_shared, c_local, k_step=k_step, policy=policy, clear_accum=True)
                    else:
                        T.gemm(a_shared, b_shared, c_local, k_step=k_step, policy=policy)

                T.copy(c_local, C[by * block_M, bx * block_N])

        return main

    return kernel()


def _max_abs_err(M, N, K, poison, **cfg):
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.float16)
    B = torch.randn(K, N, dtype=torch.float16)
    ref = A.float() @ B.float()
    out = _build(M, N, K, poison=poison, **cfg)(A.ptpu(), B.ptpu()).float().cpu()
    return (out - ref).abs().max().item()


# Both shipping tile geometries. The 128**3 / 256-thread tile has warp_cols=16
# and so takes the B-register-reuse loop nest, which zeroes C on a different
# path from the 64**3 tile -- cover both or the reuse nest goes untested.
GEOMETRIES = [
    pytest.param(dict(block_M=64, block_N=64, block_K=64, num_threads=128, k_step=8, policy=T.GemmWarpPolicy.FullRow), id="64x64x64_nt128"),
    pytest.param(
        dict(block_M=128, block_N=128, block_K=128, num_threads=256, k_step=8, policy=T.GemmWarpPolicy.FullRow),
        id="128x128x128_nt256_breuse",
    ),
]


@requires_ptpu
@pytest.mark.parametrize("cfg", GEOMETRIES)
def test_clear_accum_erases_poisoned_accumulator(cfg):
    """clear_accum=True must overwrite a non-zero accumulator, not add to it.

    A no-op clear_accum leaves the error equal to POISON.
    """
    M = N = K = 512
    err = _max_abs_err(M, N, K, poison=POISON, **cfg)
    assert err < 1.0, (
        f"clear_accum=True did not clear the accumulator: max_abs_err={err:.3f}. "
        f"An error of ~{POISON} means the flag was ignored and the poison was "
        f"accumulated into the result."
    )


@requires_ptpu
@pytest.mark.parametrize("cfg", GEOMETRIES)
def test_clear_accum_matches_reference_when_unpoisoned(cfg):
    """Baseline: the same kernel with a zeroed accumulator is correct.

    Guards against the poison test passing for the wrong reason (e.g. a kernel
    that is broken for every initial value).
    """
    M = N = K = 512
    err = _max_abs_err(M, N, K, poison=0.0, **cfg)
    assert err < 1.0, f"gemm is wrong even with a zeroed accumulator: max_abs_err={err:.3f}"


if __name__ == "__main__":
    tilelang.testing.main()
