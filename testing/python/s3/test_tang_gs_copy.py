"""stcuv2 global <-> shared bulk-copy lowering tests (TileLang -> S3).

Covers ``T.copy(global, shared)`` / ``T.copy(shared, global)`` on the tang
stcuv2 subtarget.  Lowering (``LowerSTCUV2BulkCopy``) routes these into one of
three regimes; this file pins the lowered shape of the unswizzled staging path.

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
    pairing and is rejected.
  * ``tang_swizzle_atom_bytes`` annotated -> swizzled with that atom.  This is
    the validated cpt2s drain (atom 8, covered by
    ``test_tang_tcgen05_cpt2s.py``) and the deliberate escape hatch.
  * neither -> pure staging, unswizzled 1D flat byte copy, which is trivially
    its own inverse.

Test layout
-----------
  * §1  a constant global pitch keeps its compile-time split.
  * §2  a pipelined (multi-stage) staging loop keeps its block barriers.
"""

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm

STCUV2_TARGET = {"kind": "tang", "arch": "stcuv2"}

_TL_DTYPE = {"float16": T.float16, "bfloat16": T.bfloat16, "float32": T.float32}


def _make_gs_roundtrip(M, N, dtype, threads=128, atom_bytes=None):
    """global -> shared -> global over a whole (M, N) tensor.

    ``atom_bytes=None`` leaves both copies unannotated, so the shared buffer is
    a pure staging stop and lowering picks the unswizzled 1D path.  Passing an
    atom size annotates *both* legs, forcing the swizzled 3D path.
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


# ===========================================================================
# §1  Runtime (non-constant) geometry is refused
# ===========================================================================
#
# Lowering folds a copy's row count and both pitches. Geometry that does not
# fold would require a runtime pitch predicate choosing between the flat
# transfer and the per-row loop.
#
#     if (n * 2 == BN * 2) { one flat fcpg2s } else { for (row) fcpg2s(+row*BN*2) }
#
# That form does not compile on stcuv2: ptcc requires the set_mem preamble and
# its bulk-copy consumer to remain physically adjacent, which a mixed branch
# cannot guarantee.
#
# The runtime form is therefore gone and such a copy fails at lowering, by name.
# The trigger is a global whose row pitch does not fold: with the last extent a
# `T.dynamic` var, `gmem_row_bytes` stays an expression while the constant
# shared tile folds. The global pitch is therefore the frontend's supported
# dynamic-geometry trigger.


def test_constant_geometry_still_folds():
    """The shapes every kernel uses keep their compile-time split.

    Pins both constant arms, so a future change cannot quietly turn a fold back
    into a runtime branch: matching pitches stay one flat call, a constant pitch
    mismatch stays the per-row loop, and neither grows a pitch predicate.
    """

    def lower(kernel):
        with tvm.target.Target(STCUV2_TARGET):
            return tilelang.lower(kernel, target=STCUV2_TARGET).kernel_source

    flat = lower(_make_gs_roundtrip(128, 128, "float16"))
    assert flat.count("fcpg2s(") == 1 and flat.count("fcps2g(") == 1, flat
    assert "for (int row" not in flat, "matching pitches fold to one transfer"

    per_row = lower(_make_subtile_roundtrip(128, 256, 128))
    assert "for (int row" in per_row, "a constant pitch mismatch stays per-row"
    assert "?" not in per_row, "a folded pitch must not become a runtime select"


# ===========================================================================
# §2  Pipelined 1D staging: bulk copy into shared, consumed by plain loads
# ===========================================================================
#
# This shape is outside the async handshake pass's current consumer model. Here a
# `T.Pipelined` loop stages into shared and consumes the tile with plain loads,
# and the handshake pass only registers a buffer whose consumer is an MMA
# (`inject_s3_async_copy.cc`: `epilogue_stage_` is filled from MMA reads alone),
# so the copy stays blocking.
#
# The loop therefore carries two block barriers per iteration, inserted by `ThreadSync
# ("shared")` (tilelang/tang/pipeline.py:223). Both are real: the read of
# version k is spread over every warp's byte range, and the copy that follows it
# rewrites version k, so the first is the cross-warp publish of the previous
# generation and the second is a genuine WAR. The assertions below therefore pin
# the *barrier count*: a change that drops either barrier is a race, and nothing
# else in the tree would report it.


def _make_pipelined_staging_roundtrip(M, K, BM, BK, stages=2, threads=128):
    """`T.Pipelined` over K: stage a (BM, BK) tile through shared, copy it out.

    The consumer of the staging buffer is `T.copy(S, F)` -- fragment loads, not
    MMA operands -- which is what keeps the 1D copy on the blocking path. `M` is
    meant to equal `BM`: the copy stages only the first `BM` rows, so a round
    trip over a taller tensor would leave the rest untouched.
    """

    @T.prim_func
    def kernel(In: T.Tensor((M, K), T.float16), Out: T.Tensor((M, K), T.float16)):
        with T.Kernel(1, threads=threads) as _bx:
            S = T.alloc_shared((BM, BK), T.float16)
            F = T.alloc_fragment((BM, BK), T.float16)
            for k in T.Pipelined(K // BK, num_stages=stages):
                T.copy(In[0, k * BK], S)
                T.copy(S, F)
                T.copy(F, Out[0, k * BK])

    return kernel


def _pipelined_loop(src):
    """The body of the `for (int k ...)` software pipeline, braces matched."""
    start = src.index("for (int k ")
    open_at = src.index("{", start)
    depth = 0
    for i in range(open_at, len(src)):
        depth += (src[i] == "{") - (src[i] == "}")
        if depth == 0 and i > open_at:
            return src[open_at : i + 1]
    raise AssertionError("unbalanced braces in the generated source")


def test_pipelined_staging_keeps_two_block_barriers():
    """Pin two required block barriers and the absence of handshake ops.

    Not a throughput assertion but a shape one -- `ThreadSync` inserts both
    barriers and no pass may remove them; if either disappears, the loop races.
    """
    with tvm.target.Target(STCUV2_TARGET):
        src = tilelang.lower(_make_pipelined_staging_roundtrip(64, 256, 64, 64), target=STCUV2_TARGET).kernel_source
    body = _pipelined_loop(src)

    assert body.count("__syncthreads()") == 2, body
    assert "sync_wait(" not in body, body
    assert "fence_g2s_arrive" not in body, body
    assert "fcpg2s(" in body, "the staging load must still be the bulk copy"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"] + sys.argv[1:]))
