"""stcuv2 tcgen5 GEMM, **TS variant**: the A operand lives in tensor memory.

Routes through `mma_atmem` (A in TMEM, B in shared). A is staged by
`T.copy(fragment, tmem)` into a 16-bit `T.alloc_tmem`; that direction selects the
A-operand packed layout (`.32x32b`, lane == row, two 16-bit elements per cell),
which is unrelated to the accumulator's `.16x256b` layout.

Hardware constraints and the restrictions still in place are documented in
docs/s3_tmem_a_operand_atmem_layout.md. The first version supports single-warp,
K-major A, and 8/16-bit A only.

The frontend is always `T.tcgen05_gemm` (S3 only supports tcgen05 GEMM); `T.gemm`
is an early leftover spelling and is not used by new cases.

§1 Lowering assertions  test_atmem_lowering_*  (no ISS, regex over the source)
§2 Numeric (via ISS)    test_atmem_numeric_*
§3 Negative cases       test_atmem_*_rejected  (no ISS)
"""

import re

import pytest
import torch

import tilelang
import tilelang.testing
import tilelang.tang.language as T
from tilelang import tvm as tvm
from tilelang.jit import JITKernel

STCUV2_TARGET = {"kind": "tang", "arch": "stcuv2"}

_TL_DTYPE = {
    "float16": T.float16,
    "bfloat16": T.bfloat16,
    "int8": T.int8,
    "uint8": T.uint8,
}
_PT_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "uint8": torch.uint8,
}
# Integer MMA accumulates into int32, floating point into fp32.
_ACC_DTYPE = {
    "float16": (T.float32, torch.float32),
    "bfloat16": (T.float32, torch.float32),
    "int8": (T.int32, torch.int32),
    "uint8": (T.int32, torch.int32),
}


# ===========================================================================
# Shared kernels / helpers
# ===========================================================================


def _make_atmem_kernel(M, N, K, threads, dtype, trans_B=True):
    """A -> fragment -> TMEM(packed) -> mma_atmem(B in shared) -> TMEM -> global."""
    dt = _TL_DTYPE[dtype]
    acc, _ = _ACC_DTYPE[dtype]
    b_shape = (N, K) if trans_B else (K, N)

    @T.prim_func
    def kernel(A: T.Tensor((M, K), dt), B: T.Tensor(b_shape, dt), C: T.Tensor((M, N), acc)):
        with T.Kernel(1, 1, threads=threads):
            A_f = T.alloc_fragment((M, K), dt)
            A_t = T.alloc_tmem((M, K), dt)
            B_s = T.alloc_shared(b_shape, dt)
            C_t = T.alloc_tmem((M, N), acc)
            C_f = T.alloc_fragment((M, N), acc)
            T.copy(A, A_f)
            T.copy(A_f, A_t)
            T.copy(B, B_s)
            T.tcgen05_gemm(A_t, B_s, C_t, transpose_B=trans_B, clear_accum=True, mbar=None)
            T.tcgen05_ld(C_f, C_t)
            T.copy(C_f, C)

    return kernel


def _lower_source(kernel):
    with tvm.target.Target(STCUV2_TARGET):
        return tilelang.lower(kernel, target=STCUV2_TARGET).kernel_source


def _rand_operand(rows, cols, dtype):
    pt = _PT_DTYPE[dtype]
    if dtype == "int8":
        return torch.randint(-8, 8, (rows, cols), dtype=pt)
    if dtype == "uint8":
        return torch.randint(0, 16, (rows, cols), dtype=pt)
    return torch.randn(rows, cols, dtype=torch.float32).to(pt)


def _sim_run(kernel, M, N, K, dtype, trans_B=True):
    """Run on the ISS and return the relative error against a torch reference."""
    torch.manual_seed(0)
    A = _rand_operand(M, K, dtype)
    B = _rand_operand(N, K, dtype) if trans_B else _rand_operand(K, N, dtype)
    ref = A.float() @ (B.float().T if trans_B else B.float())
    jit = JITKernel(kernel, out_idx=[-1], target=STCUV2_TARGET, execution_backend="simulator", verbose=False)
    C = jit(A.ptpu(), B.ptpu()).cpu().float()
    return (C - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)


# ===========================================================================
# §1 Lowering assertions
# ===========================================================================


def test_atmem_lowering_emits_atmem_template():
    """TS must use mma_atmem, not the SS mma that takes a shared-memory
    descriptor."""
    src = _lower_source(_make_atmem_kernel(128, 64, 64, 32, "float16"))
    assert re.search(r"tang::ptx::mma_atmem<", src)
    # The SS mma must not appear alongside it; if it does, the kernel is
    # still going through the shared-A path.
    assert not re.search(r"tang::ptx::mma<", src)


def test_atmem_lowering_stages_a_with_packed_store():
    """Staging A must use the A-operand packed store, not the accumulator's
    16x256b store."""
    src = _lower_source(_make_atmem_kernel(128, 64, 64, 32, "float16"))
    assert "tang_tmem_st_a_operand<" in src
    assert "stt_16x256b" not in src


@pytest.mark.parametrize(
    "dtype,K,cells",
    [
        ("float16", 64, 32),  # 2 x 16-bit per cell
        ("bfloat16", 64, 32),
        ("int8", 128, 32),  # 4 x 8-bit per cell
        ("uint8", 128, 32),
        ("float16", 32, 16),
    ],
)
def test_atmem_lowering_cell_count_follows_dtype_packing(dtype, K, cells):
    """Cell count is K * element_bits / 32, not K -- packing follows the dtype."""
    src = _lower_source(_make_atmem_kernel(128, 64, K, 32, dtype))
    assert re.search(rf"tang_tmem_st_a_operand<{cells}>", src)


def test_atmem_lowering_one_store_per_32_row_block():
    """M=128 gives 4 blocks of 32 rows, one stt per block."""
    src = _lower_source(_make_atmem_kernel(128, 64, 64, 32, "float16"))
    assert len(re.findall(r"tang_tmem_st_a_operand<", src)) == 4


def test_atmem_lowering_a_and_c_share_one_tc_alloc():
    """A and the accumulator fold into a single tc_alloc: D takes the low
    columns and A follows above it."""
    src = _lower_source(_make_atmem_kernel(128, 64, 64, 32, "float16"))
    assert len(re.findall(r"tc_alloc<", src)) == 1


# ===========================================================================
# §2 Numeric (via ISS)
# ===========================================================================


@pytest.mark.parametrize(
    "M,N,K,dtype",
    [
        (128, 64, 64, "float16"),
        (128, 64, 64, "bfloat16"),
        (96, 64, 64, "float16"),
        (64, 64, 64, "float16"),
        (32, 64, 64, "float16"),
        (32, 32, 64, "float16"),
        (128, 64, 128, "int8"),
        (128, 64, 128, "uint8"),
        (64, 64, 128, "int8"),
        (32, 64, 128, "uint8"),
    ],
)
def test_atmem_numeric(M, N, K, dtype):
    """The TS result must match A@B^T."""
    rel = _sim_run(_make_atmem_kernel(M, N, K, 32, dtype), M, N, K, dtype)
    assert rel < 1e-2, f"rel_err too large: {rel:.3e}"


def test_atmem_large_tile_is_toolchain_limited():
    """Record the largest tile a single-warp TS drain sustains -- but that limit
    belongs to ptcc, not to TileLang.

    A 128x128 fp32 accumulator has to squeeze through the registers of 32 lanes.
    Whether it fits is decided by ptcc's register allocator: the ptcc shipped
    with the simulator reports 'ran out of registers', while the one on CI
    compiles it and produces correct numbers.

    So "must fail" cannot be written as an assertion. This case used to be a
    strict xfail, which turned into an XPASS(strict) failure the moment CI
    compiled it -- and the failure was reported against a reason that reads like
    a known limitation, which invites the conclusion that the drain broke.

    Both outcomes are accepted, and both keep an assertion: either the known
    register-allocation limit is hit, or the kernel compiles and the numbers are
    correct. A third outcome (compiles but wrong numbers) still fails.

    The limit itself belongs to the drain, not to atmem; TS is single-warp for
    now, so the multi-warp drain is not available either.
    """
    try:
        rel = _sim_run(_make_atmem_kernel(128, 128, 64, 32, "float16"), 128, 128, 64, "float16")
    except RuntimeError as exc:
        assert "ran out of registers" in str(exc), (
            f"expected the known register-allocation limit, but ptcc failed for another reason:\n{exc}"
        )
        return
    assert rel < 1e-2, f"rel_err too large: {rel:.3e}"


@pytest.mark.xfail(
    reason="K=32 with f16 leaves B's shared row at only 64 bytes, and ptcc "
    "reports 'Unsupported B matrix swizzle layout'. This is a pre-existing "
    "constraint on the B side, unrelated to A living in TMEM.",
    strict=True,
    raises=RuntimeError,
)
def test_atmem_numeric_short_k_breaks_b_swizzle():
    """Record the lower bound on B's shared row width."""
    rel = _sim_run(_make_atmem_kernel(64, 64, 32, 32, "float16"), 64, 64, 32, "float16")
    assert rel < 1e-2, f"rel_err too large: {rel:.3e}"


def test_atmem_b_still_caps_k():
    """A is exempt from the 128-byte row cap, but B is still in shared memory,
    so the cap on K is not raised.

    Recorded to prevent the assumption that TS can take a longer K.
    """
    with pytest.raises(Exception, match=r"operand B .*128-byte cap"):
        _lower_source(_make_atmem_kernel(128, 64, 128, 32, "float16"))


# ===========================================================================
# §3 Negative cases
# ===========================================================================


def test_atmem_multiwarp_rejected():
    """The first version is single-warp; multi-warp must fail loudly rather than
    silently computing as if it were single-warp."""
    with pytest.raises(Exception, match="single-warp"):
        _lower_source(_make_atmem_kernel(128, 64, 64, 128, "float16"))


def test_atmem_transposed_a_rejected():
    """A in TMEM is always K-major; there is no encoding for M-major."""
    dt = T.float16
    M, N, K = 128, 64, 64

    @T.prim_func
    def kernel(A: T.Tensor((K, M), dt), B: T.Tensor((N, K), dt), C: T.Tensor((M, N), T.float32)):
        with T.Kernel(1, 1, threads=32):
            A_f = T.alloc_fragment((K, M), dt)
            A_t = T.alloc_tmem((K, M), dt)
            B_s = T.alloc_shared((N, K), dt)
            C_t = T.alloc_tmem((M, N), T.float32)
            C_f = T.alloc_fragment((M, N), T.float32)
            T.copy(A, A_f)
            T.copy(A_f, A_t)
            T.copy(B, B_s)
            T.tcgen05_gemm(A_t, B_s, C_t, transpose_A=True, transpose_B=True, clear_accum=True, mbar=None)
            T.tcgen05_ld(C_f, C_t)
            T.copy(C_f, C)

    with pytest.raises(Exception, match="K-major"):
        _lower_source(kernel)


def test_atmem_subbyte_a_rejected():
    """Sub-byte A is not staged by this path yet; it must fail rather than be
    packed as if it were 8-bit."""
    M, N, K = 128, 64, 64

    @T.prim_func
    def kernel(A: T.Tensor((M, K), T.float4_e2m1fn), B: T.Tensor((N, K), T.float4_e2m1fn), C: T.Tensor((M, N), T.float32)):
        with T.Kernel(1, 1, threads=32):
            A_f = T.alloc_fragment((M, K), T.float4_e2m1fn)
            A_t = T.alloc_tmem((M, K), T.float4_e2m1fn)
            B_s = T.alloc_shared((N, K), T.float4_e2m1fn)
            C_t = T.alloc_tmem((M, N), T.float32)
            C_f = T.alloc_fragment((M, N), T.float32)
            T.copy(A, A_f)
            T.copy(A_f, A_t)
            T.copy(B, B_s)
            T.tcgen05_gemm(A_t, B_s, C_t, transpose_B=True, clear_accum=True, mbar=None)
            T.tcgen05_ld(C_f, C_t)
            T.copy(C_f, C)

    with pytest.raises(Exception, match="8/16/32-bit A"):
        _lower_source(kernel)


if __name__ == "__main__":
    tilelang.testing.main()
