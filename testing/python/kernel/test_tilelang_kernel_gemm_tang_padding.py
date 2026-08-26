"""TANG TMMA shared-memory padding regressions.

The TMMA GEMM pads its shared A/B tiles to avoid bank conflicts (per-tile where
the layout allows it, per-row otherwise). The padding is split across two sides
that must agree exactly -- the write side builds the layout
(``tilelang/tang/op/gemm/gemm_tmma.py``) and the read side addresses it
(``src/tl_templates/tang/gemm_tmma.h``) -- and it makes the layout's output
extent larger than its input extent, which downstream passes have to account
for.

Each test below pins one bug that the shipping configs did NOT catch, because
they all allocate shared width == block_K with fp16, which happens to be the
one combination where every buggy predicate accidentally agreed.
"""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.utils.device import is_ptpu_available

pytestmark = pytest.mark.skipif(not is_ptpu_available(), reason="TANG TMMA regressions need PTPU hardware")

_TL_DTYPE = {"float16": T.float16, "float32": T.float32}
_PT_DTYPE = {"float16": torch.float16, "float32": torch.float32}
# fp16 accumulates in fp32 but rounds inputs; fp32 (tf32 tensor cores) is tighter.
_TOL = {"float16": 2e-2, "float32": 1e-4}


def _run(func, M, N, K, dtype):
    jit = tilelang.compile(func, out_idx=[2], target="tang")
    torch.manual_seed(0)
    pt = _PT_DTYPE[dtype]
    a = torch.randn(M, K, dtype=pt)
    b = torch.randn(K, N, dtype=pt)
    got = jit(a.ptpu(), b.ptpu()).cpu().float()
    ref = a.float() @ b.float()
    rel = (got - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
    assert rel < _TOL[dtype], f"rel_max={rel:.4e} exceeds {_TOL[dtype]:.0e}"


def _gemm_wide_shared(M, N, K, block_M, block_N, block_K, dtype, width_mult):
    """GEMM whose shared buffers are ``width_mult`` times wider than the GEMM tile.

    ``width_mult > 1`` leaves slack columns, so the shared buffer extents stop
    agreeing with the GEMM tile dims.
    """
    tl_dt = _TL_DTYPE[dtype]

    @T.prim_func
    def main(A: T.Tensor((M, K), tl_dt), B: T.Tensor((K, N), tl_dt), C: T.Tensor((M, N), tl_dt)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K * width_mult), tl_dt, scope="shared")
            B_shared = T.alloc_shared((block_K, block_N * width_mult), tl_dt, scope="shared")
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(C_local)
            T.clear(A_shared)
            T.clear(B_shared)
            for ko in T.serial(T.ceildiv(K, block_K)):
                for i, k in T.Parallel(block_M, block_K):
                    A_shared[i, k] = A[by * block_M + i, ko * block_K + k]
                for k, j in T.Parallel(block_K, block_N):
                    B_shared[k, j] = B[ko * block_K + k, bx * block_N + j]
                T.gemm(A_shared[:, 0:block_K], B_shared[0:block_K, 0:block_N], C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def _gemm_pipelined(M, N, K, block_M, block_N, block_K, dtype, num_stages):
    tl_dt = _TL_DTYPE[dtype]

    @T.prim_func
    def main(A: T.Tensor((M, K), tl_dt), B: T.Tensor((K, N), tl_dt), C: T.Tensor((M, N), tl_dt)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), tl_dt)
            B_shared = T.alloc_shared((block_K, block_N), tl_dt)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(C_local)
            for k in T.Pipelined(K // block_K, num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


# ---------------------------------------------------------------------------
# 1. Per-tile padding gate must compare the GEMM TILE, not the buffer extents.
#
# The read side gates per-tile padding on ``isA ? M_Tile == K_Tile : K_Tile ==
# N_Tile`` (GEMM tile dims). Spelling the write-side gate as
# ``stride == inner_continuous`` (shared buffer extents) instead agrees only when
# the buffer is exactly as wide as block_K. With slack columns the two sides pick
# different schemes for the same operand: at 64x64x64 with a 128-wide buffer the
# read side sees M_Tile == K_Tile (per-tile) while the buffer form sees
# 64 != 128 (per-row), corrupting the result.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "block_M,block_N,block_K",
    [
        (64, 64, 64),  # the split case: M_Tile == K_Tile but buffer is 2x wide
        (32, 32, 32),
        (64, 128, 64),
    ],
)
def test_tmma_per_tile_gate_wide_shared_buffer(block_M, block_N, block_K):
    func = _gemm_wide_shared(256, 256, 128, block_M, block_N, block_K, "float16", width_mult=2)
    _run(func, 256, 256, 128, "float16")


# ---------------------------------------------------------------------------
# 2. Padding must stay off for fp32.
#
# Padding is intended for supported 16-bit operands. Applying it to fp32
# inflates the shared extent and can exceed the per-block resource limit.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "block_M,block_N,block_K",
    [
        (128, 128, 128),  # compile-fails if fp32 gets padded
        (64, 128, 128),
        (128, 128, 64),
    ],
)
def test_tmma_fp32_shared_fits_smem(block_M, block_N, block_K):
    func = _gemm_pipelined(256, 256, 256, block_M, block_N, block_K, "float32", num_stages=0)
    _run(func, 256, 256, 256, "float32")


# ---------------------------------------------------------------------------
# 3. A padded layout must survive T.Pipelined staging.
#
# Pipelining allocates num_stages copies of the tile, and lower_tile_op derives
# the stage count as buffer_extent / layout_extent. Dividing by the layout's
# padded OUTPUT extent truncates -- a 2-stage 64x64 fp16 buffer is 8192 elements
# against a padded output extent of 4600, giving 1 instead of 2, so the second
# stage gets no memory. The divisor has to be the layout's INPUT extent (4096).
#
# Staging also makes the buffer outrank the layout, so Layout::Forward returns a
# pass-through leading index and its result is wider than OutputShape; consumers
# must fold that back instead of asserting the arity matches.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("num_stages", [0, 2, 3])
@pytest.mark.parametrize(
    "block_M,block_N,block_K",
    [
        (64, 64, 64),
        (64, 128, 64),
        (128, 128, 64),
    ],
)
def test_tmma_padded_layout_with_pipelined_stages(num_stages, block_M, block_N, block_K):
    func = _gemm_pipelined(512, 512, 512, block_M, block_N, block_K, "float16", num_stages)
    _run(func, 512, 512, 512, "float16")


# ---------------------------------------------------------------------------
# 4. offset != 0 (column-sliced shared operand) keeps working.
#
# ``T.gemm(A_shared[:, block_K:], ...)`` gives the A operand a non-zero column
# offset. Both padding schemes are gated off in that case (the read side's
# has_stride_offset arm applies its own row-stride correction and never pads), so
# the two sides must agree on "no padding" here.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("block_M,block_N,block_K", [(32, 32, 32), (64, 64, 64)])
def test_tmma_column_offset_shared_operand(dtype, block_M, block_N, block_K):
    tl_dt = _TL_DTYPE[dtype]
    M = N = 128
    K = 64

    @T.prim_func
    def main(A: T.Tensor((M, K), tl_dt), B: T.Tensor((K, N), tl_dt), C: T.Tensor((M, N), tl_dt)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K * 2), tl_dt, scope="shared")
            B_shared = T.alloc_shared((block_K, block_N * 2), tl_dt, scope="shared")
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(C_local)
            T.clear(A_shared)
            T.clear(B_shared)
            for ko in T.serial(T.ceildiv(K, block_K)):
                for i, k in T.Parallel(block_M, block_K):
                    A_shared[i, k + block_K] = A[by * block_M + i, ko * block_K + k]
                for k, j in T.Parallel(block_K, block_N):
                    B_shared[k, j] = B[ko * block_K + k, bx * block_N + j]
                # offset_A = block_K
                T.gemm(A_shared[:, block_K:], B_shared[0:block_K, 0:block_N], C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    _run(main, M, N, K, dtype)


def _gemm_staged_epilogue(M, N, K, block_M, block_N, block_K, dtype):
    """GEMM whose result goes fragment -> shared -> global instead of straight out."""
    tl_dtype = _TL_DTYPE[dtype]

    @T.prim_func
    def main(
        A: T.Tensor((M, K), tl_dtype),
        B: T.Tensor((K, N), tl_dtype),
        C: T.Tensor((M, N), tl_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_buffer((block_M, block_K), tl_dtype, scope="shared")
            B_shared = T.alloc_buffer((block_K, block_N), tl_dtype, scope="shared")
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(C_local)
            for ko in T.serial(T.ceildiv(K, block_K)):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            C_shared = T.alloc_buffer((block_M, block_N), tl_dtype, scope="shared")
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize(
    "block_M,block_N,block_K",
    [
        (64, 64, 64),
        (64, 32, 64),
        (32, 64, 32),
    ],
)
def test_epilogue_staging_buffer_padded(dtype, block_M, block_N, block_K):
    """A staged epilogue must stay correct once its shared buffer is padded.

    ``src/tang/op/copy.cc`` gives a 2D shared destination of a fragment->shared
    copy a padded row stride so the row-major write avoids concentrating its
    accesses in one bank group.

    The risk that makes this worth a test: the same buffer is the source of the
    following shared->global bulk DMA, which computes its own row stride. If the
    layout pads but that stride does not follow, the store reads the wrong words
    -- the failure mode of the earlier column-offset bug above. block_N=32 also
    covers a row that is padded on a different multiple than 64.
    """
    M, N, K = 256, 256, 256
    func = _gemm_staged_epilogue(M, N, K, block_M, block_N, block_K, dtype)
    _run(func, M, N, K, dtype)


# ---------------------------------------------------------------------------
# 5. Copy-only roundtrip helper — exercises fragment→shared padding without GEMM.
# ---------------------------------------------------------------------------
_COPY_DTYPE_MAP = {
    "float16": (T.float16, torch.float16),
    "float32": (T.float32, torch.float32),
    "bfloat16": (T.bfloat16, torch.bfloat16),
    "int8": (T.int8, torch.int8),
}


def _staged_roundtrip(M, N, block_M, block_N, dtype):
    """global → shared → fragment → shared → global roundtrip.

    The fragment→shared copy triggers ``IsFragmentToSharedStage`` and gets
    padded when the row width is a multiple of 256 bits.
    """
    tl_dt, pt = _COPY_DTYPE_MAP[dtype]

    @T.prim_func
    def main(A: T.Tensor((M, N), tl_dt), B: T.Tensor((M, N), tl_dt)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_buffer((block_M, block_N), tl_dt, scope="shared")
            A_local = T.alloc_fragment((block_M, block_N), tl_dt)
            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(A_shared, A_local)
            T.copy(A_local, A_shared)
            T.copy(A_shared, B[by * block_M, bx * block_N])

    jit = tilelang.compile(main, out_idx=[1], target="tang")
    if dtype == "int8":
        a = torch.randint(-128, 127, (M, N), dtype=pt)
    else:
        torch.manual_seed(0)
        a = torch.randn(M, N, dtype=pt)
    got = jit(a.ptpu()).cpu()
    # int8 rounds in mixed-precision hardware; compare elementwise exact match.
    if dtype == "int8":
        assert (got == a).all().item(), f"int8 roundtrip mismatch: max_diff={(got.int() - a.int()).abs().max().item()}"
        return
    rel = (got.float() - a.float()).abs().max().item() / (a.float().abs().max().item() + 1e-6)
    assert rel < 1e-4, f"rel_max={rel:.4e}"


# ---------------------------------------------------------------------------
# 6. No-pad: row width is NOT a multiple of 256 bits — padding must stay off.
#
# fp16 × 8 cols = 128 bits < 256 → MakeTangRowPaddedLayout is a no-op.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "dtype,block_N,row_bits",
    [
        ("float16", 8, 128),
        ("float16", 4, 64),
        ("bfloat16", 8, 128),
        ("int8", 16, 128),
        ("int8", 8, 64),
    ],
)
def test_staged_copy_no_pad_narrow(dtype, block_N, row_bits):
    M, N = 128, 128
    _staged_roundtrip(M, N, 32, block_N, dtype)


# ---------------------------------------------------------------------------
# 7. Pad with int8: 32 cols × 8 bits = 256 bits → padding (16 elements).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "dtype,block_N,row_bits",
    [
        ("int8", 32, 256),
        ("int8", 64, 512),
    ],
)
def test_staged_copy_int8_pad(dtype, block_N, row_bits):
    M, N = 128, 128
    _staged_roundtrip(M, N, 32, block_N, dtype)


# ---------------------------------------------------------------------------
# 8. 1D fragment→shared copy — IsFragmentToSharedStage returns false
#    (shape.size() != 2), so the buffer keeps its unpadded layout.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_staged_copy_1d_no_pad(dtype):
    tl_dt, pt = _COPY_DTYPE_MAP[dtype]
    N = 256
    block_N = 64

    @T.prim_func
    def main(A: T.Tensor((N,), tl_dt), B: T.Tensor((N,), tl_dt)):
        with T.Kernel(T.ceildiv(N, block_N), threads=128) as (bx,):
            A_shared = T.alloc_buffer((block_N,), tl_dt, scope="shared")
            A_local = T.alloc_fragment((block_N,), tl_dt)
            T.copy(A[bx * block_N], A_shared)
            T.copy(A_shared, A_local)
            T.copy(A_local, A_shared)
            T.copy(A_shared, B[bx * block_N])

    jit = tilelang.compile(main, out_idx=[1], target="tang")
    torch.manual_seed(0)
    a = torch.randn(N, dtype=pt)
    got = jit(a.ptpu()).cpu().float()
    rel = (got - a.float()).abs().max().item() / (a.float().abs().max().item() + 1e-6)
    assert rel < 1e-6, f"1D roundtrip mismatch: rel_max={rel:.4e}"


if __name__ == "__main__":
    tilelang.testing.main()
