"""stcuv2 global <-> shared bulk-copy tests (TileLang -> S3, via ISS).

Covers ``T.copy(global, shared)`` / ``T.copy(shared, global)`` on the tang
stcuv2 subtarget.  Lowering (``LowerSTCUV2BulkCopy``) routes these into one of
three regimes, and this file exercises all three plus the rejections.

Why there are three regimes
---------------------------
``fcpg2s.3d`` and ``fcps2g.3d`` are *not* inverses.  A tile loaded swizzled and
stored back with the **same** SwizzleMode comes out permuted at atom
granularity::

    RTR = SW / atom
    out[r, c] == in[(r // RTR) * RTR + (c // atom) % RTR,
                    (r % RTR) * atom + (c % atom)]

The swizzled form is therefore only usable where both ends are known to agree,
which is what selects the regime:

  * shared buffer is a GEMM operand -> swizzled, because the MMA descriptor
    reads it back swizzled.  A *store* of such a buffer has no validated
    pairing and is rejected (§4).
  * ``tang_swizzle_atom_bytes`` annotated -> swizzled with that atom.  This is
    the validated cpt2s drain (atom 8, covered by
    ``test_tang_tcgen05_cpt2s.py``) and the deliberate escape hatch used by §3.
  * neither -> pure staging, unswizzled 1D flat byte copy, which is trivially
    its own inverse (§1, §2).

Test layout
-----------
  * §1  staging round-trip identity, contiguous tiles, multi-warp.
  * §2  staging round-trip for a sub-tile of a wider matrix (global row pitch
        differs from the shared row width, so the 1D path goes per row).
  * §3  regression pin: the annotated swizzled round-trip still permutes
        exactly per the formula above.
  * §4  rejections: a GEMM operand cannot be stored to global, and a staging
        shape that is not a multiple of the 128-byte 1D chunk is refused rather
        than silently overrunning.
  * §5  rank > 2 global buffers, where the row stride is not ``strides[0]``.
"""

import pytest
import torch

import tilelang.language as T
from tilelang.jit import JITKernel

STCUV2_TARGET = {"kind": "tang", "arch": "stcuv2"}

_TL_DTYPE = {"float16": T.float16, "bfloat16": T.bfloat16, "float32": T.float32}
_PT_DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

# fcpg2s/fcps2g default swizzle for fp16/bf16 is sw128a32:
# SW = 128 B, atom = 32 B, RTR = SW / atom = 4.
_ATOM_BYTES = 32
_RTR = 4


def _sim_jit(kernel):
    return JITKernel(kernel, out_idx=[-1], target=STCUV2_TARGET, execution_backend="simulator", verbose=False)


def _make_gs_roundtrip(M, N, dtype, threads=128, atom_bytes=None):
    """global -> shared -> global over a whole (M, N) tensor.

    ``atom_bytes=None`` leaves both copies unannotated, so the shared buffer is
    a pure staging stop and lowering picks the unswizzled 1D path.  Passing an
    atom size annotates *both* legs, forcing the swizzled 3D path (docs §16.2b).
    """
    tl_dt = _TL_DTYPE[dtype]
    ann = None if atom_bytes is None else {"tang_swizzle_atom_bytes": atom_bytes}

    @T.prim_func
    def kernel(In: T.Tensor((M, N), tl_dt), Out: T.Tensor((M, N), tl_dt)):
        with T.Kernel(1, threads=threads) as _bx:
            S = T.alloc_shared((M, N), tl_dt)
            T.copy(In, S, annotations=ann)
            T.copy(S, Out, annotations=ann)

    return kernel


def _randn(M, N, dtype):
    torch.manual_seed(0)
    return torch.randn(M, N, dtype=torch.float32).to(_PT_DTYPE[dtype])


# ===========================================================================
# §1  Staging round-trip identity (unswizzled 1D)
# ===========================================================================


@pytest.mark.parametrize(
    "M, N, dtype, threads",
    [
        pytest.param(128, 64, "float16", 128, id="fp16-128x64-1warp4"),
        pytest.param(128, 64, "bfloat16", 128, id="bf16-128x64"),
        pytest.param(64, 64, "float16", 128, id="fp16-64x64"),
        # 256 B rows: the swizzled path truncated these to the first 128 B
        # slice; the flat 1D copy has no slice notion so they round-trip.
        pytest.param(128, 128, "float16", 128, id="fp16-128x128-wide"),
        pytest.param(64, 256, "float16", 128, id="fp16-64x256-wide"),
        pytest.param(64, 32, "float32", 128, id="fp32-64x32"),
        # Multi-warp: rows are split across warps, each issuing its own copy.
        pytest.param(128, 64, "float16", 256, id="fp16-128x64-8warp"),
        pytest.param(256, 64, "float16", 512, id="fp16-256x64-16warp"),
    ],
)
def test_staging_roundtrip_is_identity(M, N, dtype, threads):
    """A staging buffer round-trips bit-exactly, for any row width."""
    jit = _sim_jit(_make_gs_roundtrip(M, N, dtype, threads=threads))
    In = _randn(M, N, dtype)
    Out = jit(In.ptpu()).cpu()

    assert tuple(Out.shape) == (M, N)
    assert torch.equal(Out.float(), In.float()), f"{int((Out != In).sum())}/{M * N} elements differ"


# ===========================================================================
# §2  Staging round-trip for a sub-tile of a wider matrix
# ===========================================================================


def _make_subtile_roundtrip(M, N, BN, threads=128):
    """Stage (M, BN) column blocks of a wider (M, N) tensor through shared.

    The global row pitch (N) differs from the shared row width (BN), so the 1D
    helper cannot do one flat transfer and issues one call per row instead.
    """

    @T.prim_func
    def kernel(In: T.Tensor((M, N), T.float16), Out: T.Tensor((M, N), T.float16)):
        with T.Kernel(1, threads=threads) as _bx:
            S = T.alloc_shared((M, BN), T.float16)
            for j in T.serial(N // BN):
                T.copy(In[0:M, j * BN : (j + 1) * BN], S)
                T.copy(S, Out[0:M, j * BN : (j + 1) * BN])

    return kernel


@pytest.mark.parametrize(
    "M, N, BN",
    [
        pytest.param(64, 128, 64, id="64x128-bn64"),
        pytest.param(128, 256, 128, id="128x256-bn128"),
        # Pitch not a multiple of the block width: exercises a partial trailing
        # block index without changing the per-row transfer size.
        pytest.param(64, 192, 64, id="64x192-bn64"),
    ],
)
def test_staging_roundtrip_subtile_is_identity(M, N, BN):
    jit = _sim_jit(_make_subtile_roundtrip(M, N, BN))
    In = _randn(M, N, "float16")
    Out = jit(In.ptpu()).cpu()

    assert torch.equal(Out.float(), In.float()), f"{int((Out != In).sum())}/{M * N} elements differ"


# ===========================================================================
# §3  Regression pin: the annotated swizzled round-trip still permutes
# ===========================================================================


def _permuted_reference(In: torch.Tensor, atom: int) -> torch.Tensor:
    """Formula from docs §13.1, confirmed at ISA level by probes P0/P1/P2."""
    M, N = In.shape
    r = torch.arange(M).unsqueeze(1).expand(M, N)
    c = torch.arange(N).unsqueeze(0).expand(M, N)
    src_row = (r // _RTR) * _RTR + (c // atom) % _RTR
    src_col = (r % _RTR) * atom + (c % atom)
    return In[src_row, src_col]


@pytest.mark.parametrize(
    "M, N, dtype",
    [
        pytest.param(128, 64, "float16", id="fp16-128x64"),
        pytest.param(128, 64, "bfloat16", id="bf16-128x64"),
        pytest.param(256, 64, "float16", id="fp16-256x64"),
        pytest.param(64, 64, "float16", id="fp16-64x64"),
    ],
)
def test_annotated_swizzled_roundtrip_matches_permutation(M, N, dtype):
    """Pin the exact permutation the swizzled round-trip performs.

    A row-identifiable input (row ``r`` filled with ``r + 1``) and a
    column-identifiable one (column ``c`` filled with ``c + 1``) together
    determine the source index of every output element, so comparing against
    ``_permuted_reference`` checks both components.  1-based fills keep "never
    written" (reads back as 0) distinguishable from "came from index 0".

    Rows here are exactly one SW wide (64 elems x 2 B = 128 B) so every element
    is written; the wide-row truncation is pinned separately below.
    """
    pt_dt = _PT_DTYPE[dtype]
    atom = _ATOM_BYTES // (torch.finfo(pt_dt).bits // 8)
    jit = _sim_jit(_make_gs_roundtrip(M, N, dtype, atom_bytes=_ATOM_BYTES))

    for probe in ("row", "col"):
        if probe == "row":
            In = (torch.arange(M, dtype=torch.float32) + 1).unsqueeze(1).expand(M, N)
        else:
            In = (torch.arange(N, dtype=torch.float32) + 1).unsqueeze(0).expand(M, N)
        In = In.contiguous().to(pt_dt)

        Out = jit(In.ptpu()).cpu().float()
        expect = _permuted_reference(In.float(), atom)

        assert (Out != 0).all(), f"{probe} probe: {int((Out == 0).sum())} elems unwritten"
        assert torch.equal(Out, expect), (
            f"{probe} probe: swizzled round-trip permutation changed; "
            f"first mismatch at {(Out != expect).nonzero()[0].tolist()}. "
            f"If it is now the identity, the ISA asymmetry in docs §15.2 was "
            f"fixed -- revisit the 1D staging decision."
        )


@pytest.mark.parametrize("M, N", [pytest.param(128, 128, id="fp16-128x128")])
def test_annotated_swizzled_roundtrip_truncates_wide_rows(M, N):
    """A 256 B shared row (2 SW slices) only gets its first slice stored.

    Unlike the permutation above, this one *is* a TileLang gap:
    ``fast_cp_async_bulk.h`` requires one ``fcps2g`` call per SW-wide slice
    (GMEM += SW, SMEM += RTR*512) and ``tang_bulk_s2g`` issues a single call.
    Probe P3 confirms the loop removes the truncation entirely; docs §19.1
    keeps it deferred because it does not make the round-trip usable -- the
    permutation remains, which is why staging uses 1D instead.
    """
    jit = _sim_jit(_make_gs_roundtrip(M, N, "float16", atom_bytes=_ATOM_BYTES))
    In = (torch.arange(M, dtype=torch.float32) + 1).unsqueeze(1).expand(M, N)
    Out = jit(In.contiguous().half().ptpu()).cpu().float()

    assert (Out[:, :64] != 0).all(), "first SW slice should be written"
    assert (Out[:, 64:] == 0).all(), "second SW slice is expected to stay untouched"


# ===========================================================================
# §4  Rejections
# ===========================================================================


def _make_gemm_operand_stored_to_global():
    """Store a GEMM A operand back to global -- the unvalidated pairing."""

    @T.prim_func
    def kernel(A: T.Tensor((128, 64), T.float16), B: T.Tensor((64, 128), T.float16), Out: T.Tensor((128, 64), T.float16)):
        with T.Kernel(1, threads=128) as _bx:
            A_s = T.alloc_shared((128, 64), T.float16)
            B_s = T.alloc_shared((64, 128), T.float16)
            C_t = T.alloc_tmem((128, 128), T.float32)
            T.copy(A, A_s)
            T.copy(B, B_s)
            T.tcgen05_gemm(A_s, B_s, C_t, clear_accum=True, mbar=None)
            T.copy(A_s, Out)

    return kernel


def test_gemm_operand_shared_to_global_is_rejected():
    """A swizzled MMA operand must not be streamed back to global.

    Its layout is dictated by the MMA descriptor, and the swizzled store does
    not invert the swizzled load, so this used to emit scrambled data with no
    diagnostic.
    """
    with pytest.raises(Exception) as err:
        _sim_jit(_make_gemm_operand_stored_to_global())

    msg = str(err.value)
    assert "shared operand layout" in msg, msg
    assert "tang_swizzle_atom_bytes" in msg, msg


@pytest.mark.parametrize(
    "M, N, dtype",
    [
        # 1 x 32 fp16 = 64 B total, contiguous -> below the 128 B chunk.
        pytest.param(1, 32, "float16", id="fp16-1x32-total64B"),
        # 4 x 16 fp16 = 32 B rows, 128 B total... but 3 rows x 32 B = 96 B.
        pytest.param(3, 16, "float16", id="fp16-3x16-total96B"),
    ],
)
def test_staging_shape_below_1d_chunk_is_rejected(M, N, dtype):
    """A staging tile smaller than the 128 B 1D chunk is refused.

    The 1D overloads round ``size`` up to 128 internally, which would write
    past the destination, so lowering rejects the shape instead of overrunning.
    """
    with pytest.raises(Exception) as err:
        _sim_jit(_make_gs_roundtrip(M, N, dtype))

    msg = str(err.value)
    assert "128-byte 1D transfer chunk" in msg, msg


def test_staging_shape_exactly_one_chunk_is_accepted():
    """The guard is a size check, not a blanket small-tile ban.

    8 x 8 fp16 is 16 B rows but exactly 128 B in total, and the contiguous
    transfer is one whole chunk, so it must be allowed.
    """
    jit = _sim_jit(_make_gs_roundtrip(8, 8, "float16"))
    In = _randn(8, 8, "float16")
    Out = jit(In.ptpu()).cpu()
    assert torch.equal(Out.float(), In.float())


# ===========================================================================
# §5  Rank > 2 global buffers
# ===========================================================================
#
# `gmem_row_bytes` is the distance between two rows of the 2D tile, i.e. the
# stride of whichever dimension indexes rows. Lowering used to read
# `global_strides[0]` unconditionally, which only coincides with that at rank 2;
# at rank 3 it is the batch stride, so every row past the first was fetched a
# whole matrix ahead. The shared side already skipped leading unit dims (the
# pipeline stage dim) -- the global side now skips the same ones.
#
# Failure mode was silent and shaped like "only row 0 survives": with a
# (3, 128, 64) fp32 tensor exactly 64 of 8192 slots per batch were right and the
# remaining rows read as 0. The row-identifiable fill below is what makes that
# visible, so keep it rather than comparing random data.


def _make_batched_roundtrip(batch_shape, M, N, dtype="float32"):
    """global -> shared -> global for one (M, N) tile per leading-index tuple.

    The shared staging buffer stays 2-D, so every leading mode of the global
    tensor is a unit-extent slice folded into the address -- exactly the shape
    that used to mis-derive the row stride.
    """
    tl_dt = _TL_DTYPE[dtype]
    shape = (*batch_shape, M, N)

    if len(batch_shape) == 1:

        @T.prim_func
        def kernel(In: T.Tensor(shape, tl_dt), Out: T.Tensor(shape, tl_dt)):
            with T.Kernel(1, threads=128) as _bx:
                S = T.alloc_shared((M, N), tl_dt)
                for i in T.serial(batch_shape[0]):
                    T.copy(In[i, :, :], S)
                    T.copy(S, Out[i, :, :])

        return kernel

    @T.prim_func
    def kernel(In: T.Tensor(shape, tl_dt), Out: T.Tensor(shape, tl_dt)):
        with T.Kernel(1, threads=128) as _bx:
            S = T.alloc_shared((M, N), tl_dt)
            for i in T.serial(batch_shape[0]):
                for j in T.serial(batch_shape[1]):
                    T.copy(In[i, j, :, :], S)
                    T.copy(S, Out[i, j, :, :])

    return kernel


def _row_identifiable(shape):
    """Row r of every tile is filled with r + 1, so a row read from the wrong
    offset is off by a whole row rather than by a plausible-looking value."""
    M, N = shape[-2], shape[-1]
    rows = (torch.arange(M, dtype=torch.float32) + 1).unsqueeze(1).expand(M, N)
    return rows.repeat(*shape[:-2], 1, 1).contiguous()


@pytest.mark.parametrize(
    "batch_shape, M, N, dtype",
    [
        pytest.param((3,), 128, 64, "float32", id="rank3-3x128x64-fp32"),
        pytest.param((3,), 128, 64, "float16", id="rank3-3x128x64-fp16"),
        # N != M: a row/col mix-up in the stride derivation cannot hide behind
        # a square tile.
        pytest.param((2,), 128, 32, "float32", id="rank3-2x128x32-fp32"),
        pytest.param((2, 2), 128, 64, "float32", id="rank4-2x2x128x64-fp32"),
    ],
)
def test_batched_staging_roundtrip_is_identity(batch_shape, M, N, dtype):
    """A rank > 2 staging round-trip is bit-exact for every leading index."""
    jit = _sim_jit(_make_batched_roundtrip(batch_shape, M, N, dtype))
    In = _row_identifiable((*batch_shape, M, N)).to(_PT_DTYPE[dtype])
    Out = jit(In.ptpu()).cpu()

    assert torch.equal(Out.float(), In.float()), (
        f"{int((Out != In).sum())}/{In.numel()} elements differ; "
        f"rows correct per tile: "
        f"{int((Out.float() == In.float()).all(dim=-1).sum())}"
        f"/{In.numel() // N}"
    )


def test_batched_roundtrip_reads_the_addressed_batch():
    """Each leading index must move its own tile, not batch 0's.

    Identity above would still pass if every batch happened to hold the same
    data, so give each batch a distinct value.
    """
    batch, M, N = 3, 128, 64
    jit = _sim_jit(_make_batched_roundtrip((batch,), M, N, "float32"))
    In = torch.stack([torch.full((M, N), float(b + 1)) for b in range(batch)])
    Out = jit(In.contiguous().ptpu()).cpu()

    for b in range(batch):
        assert torch.equal(Out[b], In[b]), f"batch {b} got {sorted(set(Out[b].flatten().tolist()))[:4]}, expected all {b + 1}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"] + sys.argv[1:]))
