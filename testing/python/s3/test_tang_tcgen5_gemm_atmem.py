"""stcuv2 tcgen5 GEMM, **TS variant**: the A operand lives in tensor memory.

Routes through `mma_atmem` (A in TMEM, B in shared). A is staged by
`T.copy(fragment, tmem)` into a 16-bit `T.alloc_tmem`; that direction selects the
A-operand packed layout (`.32x32b`, lane == row, two 16-bit elements per cell),
which is unrelated to the accumulator's `.16x256b` layout.

The implementation supports single-warp, K-major A with 8/16-bit elements.

The frontend is always `T.tcgen05_gemm` (S3 only supports tcgen05 GEMM); `T.gemm`
is an early leftover spelling and is not used by new cases.

§1 Lowering assertions  test_atmem_lowering_*  (regex over the source)
§3 Negative cases       test_atmem_*_rejected
"""

import re

import pytest
import torch

import tilelang
import tilelang.testing
import tilelang.tang.language as T
from tilelang import tvm as tvm

STCUV2_TARGET = {"kind": "tang", "arch": "stcuv2"}

_TL_DTYPE = {
    "float16": T.float16,
    "bfloat16": T.bfloat16,
    "int8": T.int8,
    "uint8": T.uint8,
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


# ===========================================================================
# §1 Lowering assertions
# ===========================================================================


def test_atmem_lowering_emits_atmem_template():
    """TS must use the atmem wrapper, not the SS one that takes a shared-memory
    descriptor."""
    src = _lower_source(_make_atmem_kernel(128, 64, 64, 32, "float16"))
    assert re.search(r"tl::tang_tcgen05_mma_ts\(", src)
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


def test_atmem_b_still_caps_k():
    """A is exempt from the 128-byte row cap, but the TS path still caps K.

    B is in shared memory, so a K-major B over the cap is normally split into
    128-byte segments; here A cannot follow, because a tensor-memory A is one
    indivisible operand. Both operands of an MMA have to cover the same K, so the
    cap stands on this path -- for the A side rather than the B side.
    """
    with pytest.raises(Exception, match=r"A in tensor memory is one indivisible operand"):
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


# ===========================================================================
# §4 A K window into a staged tile (the one operand slice stcuv2 can express)
# ===========================================================================


def _make_atmem_k_window_kernel(k0, M=64, N=64, K=64, KF=128):
    """A is staged whole (M, KF) into TMEM; the GEMM reads a K=K window of it.

    A in tensor memory is K-major and unswizzled, so the window is a plain
    column offset: ``offset_A`` elements become ``offset_A / elems_per_cell``
    32-bit cells added to A's TMEM address. Nothing else about the operand is
    region dependent (M/N/K come from the instruction descriptor), which is why
    this works where a shared descriptor cannot.
    """
    dt = T.float16

    @T.prim_func
    def kernel(
        A: T.Tensor((M, KF), dt),
        B: T.Tensor((N, K), dt),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(1, 1, threads=32):
            A_s = T.alloc_shared((M, KF), dt)
            A_t = T.alloc_tmem((M, KF), dt)
            B_s = T.alloc_shared((N, K), dt)
            C_t = T.alloc_tmem((M, N), T.float32)
            C_f = T.alloc_fragment((M, N), T.float32)
            T.copy(A, A_s)
            T.tcgen05_cp(A_t, A_s)
            T.copy(B, B_s)
            T.tcgen05_gemm(
                A_t[:, k0 : k0 + K],
                B_s,
                C_t,
                transpose_B=True,
                clear_accum=True,
                mbar=None,
            )
            T.tcgen05_ld(C_f, C_t)
            T.copy(C_f, C)

    return kernel


@pytest.mark.parametrize("k0,offset", [(0, None), (64, 32)])
def test_atmem_lowering_k_window_offsets_the_a_address(k0, offset):
    """The K origin reaches the MMA as a TMEM cell offset, not as nothing.

    f16 packs 2 elements per 32-bit cell, so K origin 64 is +32 cells. Before
    this was wired through, the origin was dropped entirely and the MMA quietly
    read from column 0.
    """
    src = _lower_source(_make_atmem_k_window_kernel(k0))
    assert re.search(r"tl::tang_tcgen05_mma_ts\(", src)
    if offset is None:
        assert not re.search(r"a_tmem = [^;]*\+", src)
    else:
        assert re.search(rf"a_tmem = [^;]*\+ {offset};", src)


def test_atmem_unaligned_k_origin_rejected():
    """The origin has to land on a cell boundary (f16: every 2 elements)."""
    with pytest.raises(Exception, match=r"not a multiple of 2"):
        _lower_source(_make_atmem_k_window_kernel(1))


def test_atmem_m_origin_rejected():
    """Only a K window is expressible: A's row (M) axis has no offset encoding."""
    dt = T.float16
    M, N, K, KF = 128, 64, 64, 128

    @T.prim_func
    def kernel(
        A: T.Tensor((M, KF), dt),
        B: T.Tensor((N, K), dt),
        C: T.Tensor((M // 2, N), T.float32),
    ):
        with T.Kernel(1, 1, threads=32):
            A_s = T.alloc_shared((M, KF), dt)
            A_t = T.alloc_tmem((M, KF), dt)
            B_s = T.alloc_shared((N, K), dt)
            C_t = T.alloc_tmem((M // 2, N), T.float32)
            C_f = T.alloc_fragment((M // 2, N), T.float32)
            T.copy(A, A_s)
            T.tcgen05_cp(A_t, A_s)
            T.copy(B, B_s)
            T.tcgen05_gemm(
                A_t[M // 2 : M, 0:K],
                B_s,
                C_t,
                transpose_B=True,
                clear_accum=True,
                mbar=None,
            )
            T.tcgen05_ld(C_f, C_t)
            T.copy(C_f, C)

    with pytest.raises(Exception, match=r"operand A is a slice of its buffer"):
        _lower_source(kernel)


def test_atmem_sliced_b_rejected():
    """B stays a shared descriptor, whose geometry is taken from the buffer."""
    dt = T.float16
    M, N, K, KF = 64, 64, 64, 128

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dt),
        B: T.Tensor((N, KF), dt),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(1, 1, threads=32):
            A_s = T.alloc_shared((M, K), dt)
            A_t = T.alloc_tmem((M, K), dt)
            B_s = T.alloc_shared((N, KF), dt)
            C_t = T.alloc_tmem((M, N), T.float32)
            C_f = T.alloc_fragment((M, N), T.float32)
            T.copy(A, A_s)
            T.tcgen05_cp(A_t, A_s)
            T.copy(B, B_s)
            T.tcgen05_gemm(
                A_t,
                B_s[:, 0:K],
                C_t,
                transpose_B=True,
                clear_accum=True,
                mbar=None,
            )
            T.tcgen05_ld(C_f, C_t)
            T.copy(C_f, C)

    with pytest.raises(Exception, match=r"operand B is a slice of its buffer"):
        _lower_source(kernel)


if __name__ == "__main__":
    tilelang.testing.main()
