"""stcuv2 multi-warp swizzled bulk copy correctness (TileLang -> S3, via ISS).

The global->shared bulk load (``tl::tang_bulk_g2s_*`` in
``tl_templates/tang/copy_global_shm.h``) partitions a tile's swizzle *stripes*
across ALL warps of the block: each warp copies a contiguous band of
``rows_per_stripe = 512/atom_bytes`` rows (16 rows for the sw128a32 path) and a
single ``__syncthreads`` publishes the resident tile. The warp count is a
codegen-time constant (``blockDim`` is unreliable on the stcuv2 ISS).

These tests exercise that multi-warp load and assert it is numerically correct:

  * §1  passthrough: an identity-B GEMM makes ``C == A`` exactly, so the output
        is literally the tile that the 4 warps cooperatively loaded. A wrong
        per-warp stripe offset shows up as a wrong 16/32-row band of C.
  * §2  numeric: random GEMM vs full-precision torch across dtypes, K-tiling
        (repeated multi-warp refills) and multi-block (each block loads
        independently -- where a bad warp count / concurrency bug first showed
        up as rel=1.0).
  * §3  determinism: re-run the same multi-block multi-warp kernel and require
        bit-identical, correct results, catching nondeterministic multi-warp
        DMA races.
  * §4  more warps: the tcgen5 GEMM consumer caps a block at 128 threads (its
        MMA + 128-lane TMEM drain), so §1-§3 only reach 4 warps. §4 drops the
        MMA and drives the g2s load on 1/2/4/8/16 warps, requiring bit-identical
        resident tiles -- transitively proving 8/16-warp loads are correct too.

The GEMM kernels here (§1-§3) launch with ``threads=128`` (= 4 warps), so a
128-row tile = 8 stripes is split 2-stripes-per-warp; the drain/MMA read path is
the already validated tcgen5 path, so any error localizes to the multi-warp
load. §4 pushes the load itself up to 16 warps.
"""

import pytest
import torch

import tilelang.language as T
from tilelang.jit import JITKernel

STCUV2_TARGET = {"kind": "tang", "arch": "stcuv2"}

_TL_DTYPE = {"float16": T.float16, "bfloat16": T.bfloat16, "float8_e4m3": T.float8_e4m3, "float8_e5m2": T.float8_e5m2}
_PT_DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float8_e4m3": torch.float8_e4m3fn, "float8_e5m2": torch.float8_e5m2}


def _sim_jit(kernel):
    return JITKernel(kernel, out_idx=[-1], target=STCUV2_TARGET, execution_backend="simulator", verbose=False)


def _rel_max(C, ref) -> float:
    return (C - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)


def _mk(pt, *shape):
    # randn has no fp8 kernel; draw in fp32 and cast down.
    if pt in (torch.float8_e4m3fn, torch.float8_e5m2):
        return torch.randn(*shape, dtype=torch.float32).to(pt)
    return torch.randn(*shape, dtype=pt)


def _make_tn_gemm(M, N, K, BM, BN, BK, dtype, stages=3):
    """Single-tile-per-block TN GEMM: C = A @ B^T, drained TMEM->global.

    A[by*BM : , k*BK :] and B[bx*BN : , k*BK :] are each loaded via the
    multi-warp swizzled bulk copy (all 4 warps of the block cooperate).
    """
    tl_dt = _TL_DTYPE[dtype]

    @T.prim_func
    def kernel(A: T.Tensor((M, K), tl_dt), B: T.Tensor((N, K), tl_dt), C: T.Tensor((M, N), T.float32)):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
            A_s = T.alloc_shared((BM, BK), tl_dt)
            B_s = T.alloc_shared((BN, BK), tl_dt)
            C_t = T.alloc_tmem((BM, BN), T.float32)
            for k in T.Pipelined(K // BK, num_stages=stages):
                T.copy(A[by * BM, k * BK], A_s)
                T.copy(B[bx * BN, k * BK], B_s)
                T.gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=(k == 0))
            T.copy(C_t, C[by * BM, bx * BN])

    return kernel


def _make_g2s_readback(M, N, threads):
    """global -> shared (multi-warp bulk load) -> fragment -> global.

    Exercises ONLY the multi-warp g2s bulk load on ``threads/32`` warps. The
    ``shared -> fragment -> global`` drain is a plain register-path copy spread
    over all ``threads`` threads; it does NOT invert the hardware bulk swizzle,
    so ``Out != In`` -- but ``Out`` is a deterministic function of the shared
    bytes the load produced, which is exactly what we compare across warp counts.
    """

    @T.prim_func
    def kernel(In: T.Tensor((M, N), T.float16), Out: T.Tensor((M, N), T.float16)):
        with T.Kernel(1, threads=threads):
            S = T.alloc_shared((M, N), T.float16)
            F = T.alloc_fragment((M, N), T.float16)
            T.copy(In, S)  # g2s multi-warp swizzled bulk load
            T.copy(S, F)  # shared -> fragment (register path, all threads)
            T.copy(F, Out)  # fragment -> global

    return kernel


# ===========================================================================
# §1  Passthrough: identity B => C == A == the multi-warp-loaded tile
# ===========================================================================


@pytest.mark.parametrize(
    "dtype, tol",
    [
        pytest.param("float16", 1e-3, id="fp16"),
        pytest.param("bfloat16", 1e-2, id="bf16"),
    ],
)
def test_multiwarp_copy_passthrough(dtype, tol):
    """A@I^T == A, so C is exactly the tile the 4 warps cooperatively loaded.

    A is given a per-row ramp (row i -> constant value i) so each 16-row
    swizzle stripe / 32-row warp band is a distinct value: a mis-partitioned
    multi-warp load (wrong stripe offset, dropped stripe, warp/stride bug)
    corrupts a whole band and is caught row-exactly.
    """
    # fp16/bf16 need a 128-byte swizzle row (BK=64); K==N is required for an
    # identity B, so K-tile K=128 over BK=64 (identity split across K-tiles
    # still sums to A@I == A exactly).
    M = N = K = 128
    BK = 64
    pt = _PT_DTYPE[dtype]
    # Row-structured A: row i is the constant i/16 (distinct per 16-row stripe,
    # small enough to be exact in fp16/bf16).
    A = (torch.arange(M, dtype=torch.float32).view(M, 1) / 16.0).repeat(1, K)
    A = A.to(pt)
    B = torch.eye(N, K, dtype=torch.float32).to(pt)  # B^T == I

    jit = _sim_jit(_make_tn_gemm(M, N, K, M, N, BK, dtype, stages=3))
    C = jit(A.ptpu(), B.ptpu()).cpu().float()

    ref = A.float()  # C == A
    assert tuple(C.shape) == (M, N)
    rel = _rel_max(C, ref)
    assert rel < tol, f"rel_diff={rel:.5f} >= tol={tol}"
    # Per-row check: every 32-row warp band must be exactly right.
    for r0 in range(0, M, 32):
        band = (C[r0 : r0 + 32] - ref[r0 : r0 + 32]).abs().max().item()
        assert band < tol, f"warp band rows[{r0}:{r0 + 32}] wrong (max={band})"


# ===========================================================================
# §2  Numeric: random GEMM vs torch (single / K-tiling / multi-block)
# ===========================================================================


# (M, N, K, BM, BN, BK, dtype, tol)
_CASES = [
    # single block, single K-tile: 4 warps each load one 128x* tile
    pytest.param(128, 128, 64, 128, 128, 64, "float16", 2e-2, id="fp16-1block"),
    pytest.param(128, 128, 64, 128, 128, 64, "bfloat16", 3e-2, id="bf16-1block"),
    pytest.param(128, 128, 128, 128, 128, 128, "float8_e4m3", 2e-2, id="fp8e4m3-1block"),
    # K-tiling: multiple sequential multi-warp loads + refills into the tile
    pytest.param(128, 128, 128, 128, 128, 64, "float16", 2e-2, id="fp16-Ktile2"),
    pytest.param(128, 128, 192, 128, 128, 64, "float16", 2e-2, id="fp16-Ktile3"),
    pytest.param(128, 128, 256, 128, 128, 128, "float8_e4m3", 2e-2, id="fp8e4m3-Ktile2"),
    # multi-block: each block's 4 warps load independently (concurrency)
    pytest.param(256, 256, 64, 128, 128, 64, "float16", 2e-2, id="fp16-grid2x2"),
    pytest.param(128, 256, 32, 128, 128, 32, "float32", 2e-2, id="fp32-grid1x2"),
]


@pytest.mark.parametrize("M, N, K, BM, BN, BK, dtype, tol", _CASES)
def test_multiwarp_copy_gemm_numeric(M, N, K, BM, BN, BK, dtype, tol):
    """Random-input TN GEMM whose A/B tiles are loaded by the multi-warp copy.

    Any per-warp / per-stripe corruption of the cooperative load surfaces as a
    numeric error vs the full-precision torch reference.
    """
    # float32 is stored as tf32 by the tcgen5 path; map to the T.float32 dtype.
    tl_dt = T.float32 if dtype == "float32" else _TL_DTYPE[dtype]
    pt = torch.float32 if dtype == "float32" else _PT_DTYPE[dtype]

    @T.prim_func
    def kernel(A: T.Tensor((M, K), tl_dt), B: T.Tensor((N, K), tl_dt), C: T.Tensor((M, N), T.float32)):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
            A_s = T.alloc_shared((BM, BK), tl_dt)
            B_s = T.alloc_shared((BN, BK), tl_dt)
            C_t = T.alloc_tmem((BM, BN), T.float32)
            for k in T.Pipelined(K // BK, num_stages=3):
                T.copy(A[by * BM, k * BK], A_s)
                T.copy(B[bx * BN, k * BK], B_s)
                T.gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=(k == 0))
            T.copy(C_t, C[by * BM, bx * BN])

    torch.manual_seed(0)
    A = _mk(pt, M, K)
    B = _mk(pt, N, K)
    ref = A.float() @ B.float().T

    jit = _sim_jit(kernel)
    C = jit(A.ptpu(), B.ptpu())
    assert tuple(C.shape) == (M, N)
    rel = _rel_max(C.cpu().float(), ref)
    assert rel < tol, f"rel_diff={rel:.4f} >= tol={tol}"


# ===========================================================================
# §3  Determinism: repeated multi-warp loads must be stable and correct
# ===========================================================================


def test_multiwarp_copy_deterministic():
    """Re-run a multi-block multi-warp GEMM; require identical, correct output.

    Concurrent per-warp bulk DMAs to disjoint stripes must not race: every
    repetition has to be bit-identical to the first and match torch. A
    nondeterministic multi-warp bug would show up as run-to-run divergence.
    """
    M, N, K = 256, 256, 64
    dtype, tol, reps = "float16", 2e-2, 5

    jit = _sim_jit(_make_tn_gemm(M, N, K, 128, 128, 64, dtype, stages=3))
    torch.manual_seed(0)
    A = _mk(_PT_DTYPE[dtype], M, K)
    B = _mk(_PT_DTYPE[dtype], N, K)
    ref = A.float() @ B.float().T
    A_d, B_d = A.ptpu(), B.ptpu()

    first = None
    for it in range(reps):
        C = jit(A_d, B_d).cpu().float()
        rel = _rel_max(C, ref)
        assert rel < tol, f"iter {it}: rel_diff={rel:.4f} >= tol={tol}"
        if first is None:
            first = C
        else:
            assert torch.equal(C, first), f"iter {it}: multi-warp load nondeterministic (max delta {(C - first).abs().max().item()})"


# ===========================================================================
# §4  More warps: multi-warp g2s load is warp-count invariant (=> 8/16 warps
#     are correct, transitively from the validated 4-warp identity-GEMM path)
# ===========================================================================


@pytest.mark.parametrize(
    "M, N",
    [
        pytest.param(128, 64, id="128x64-8stripes"),
        pytest.param(256, 64, id="256x64-16stripes"),
    ],
)
def test_multiwarp_copy_warp_count_invariant(M, N):
    """Drive the g2s bulk load on 1/2/4/8/16 warps; require bit-identical tiles.

    The tcgen5 GEMM consumer (the swizzle-correct reader) issues its MMA on one
    warp and drains TMEM over a fixed 128-lane layout, so it caps a block at 128
    threads (= 4 warps); the other tests here therefore only reach 4 warps. To
    exercise *more* warps in the cooperative load we drop the MMA and use
    ``_make_g2s_readback`` (global -> shared multi-warp bulk load -> plain
    fragment drain -> global).

    The plain drain does not invert the hardware bulk swizzle, so the output is
    not equal to the input -- but it is a deterministic function of the shared
    bytes the load wrote. Running the identical load on nwarps = 1,2,4,8,16 and
    getting BIT-IDENTICAL output proves every warp count produces the exact same
    resident tile (no dropped, duplicated, or mis-offset stripe). Because the
    4-warp load is independently proven exactly correct by the identity-GEMM
    passthrough (:func:`test_multiwarp_copy_passthrough`), byte-equality at 8 and
    16 warps proves those larger warp counts load correctly too.
    """
    In = (torch.arange(M, dtype=torch.float32).view(M, 1) / 16.0).repeat(1, N)
    In = In.to(torch.float16)
    In_d = In.ptpu()

    baseline = None
    baseline_nwarps = None
    for threads in (32, 64, 128, 256, 512):
        nwarps = threads // 32
        out = _sim_jit(_make_g2s_readback(M, N, threads))(In_d).cpu()
        if baseline is None:
            baseline, baseline_nwarps = out, nwarps
        else:
            assert torch.equal(out, baseline), (
                f"nwarps={nwarps} produced a different resident tile than the "
                f"nwarps={baseline_nwarps} baseline (max delta "
                f"{(out.float() - baseline.float()).abs().max().item()}): the "
                f"multi-warp g2s stripe partition is not warp-count invariant"
            )


# NOTE: the shared->global (s2g) bulk-copy roundtrip defect (g2s->s2g does not
# reconstruct the input) lives in its own focused file,
# testing/python/s3/test_tang_gs_copy.py, so all s2g coverage is in one place.


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"] + sys.argv[1:]))
