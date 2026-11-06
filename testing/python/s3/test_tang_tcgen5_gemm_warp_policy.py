"""How a tcgen5 GEMM's warp partition is really chosen.

``T.gemm(policy=...)`` cannot express the interesting case. The policy enum has
only ``Square`` / ``FullRow`` / ``FullCol``, and ``from_warp_partition(1, k)``
collapses to ``FullCol``, whose ``n_warp`` the backend recomputes as the block's
warp count -- so a caller cannot ask for "the MMA on 2 of my 4 warps" that way.

What the emitter drives is a 1 x num_warps split of N, and ``num_warps`` is the
*thread region* the gemm sits in. Narrowing the region is therefore what narrows
the MMA, which is exactly what a warp-specialised kernel wants; ``Square`` and
``FullRow`` are refused loudly rather than silently becoming the full-width
split (they would need an M split, which no ISS test covers yet).
"""

import re


import tilelang
import tilelang.tang.language as T
from tilelang import tvm

STCUV2_TARGET = {"kind": "tang", "arch": "stcuv2"}

M = N = 256
K = 512
BM = BN = 128
BK = 64


def _build(policy=None, region_warps=None):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((N, K), T.float16),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(2, 2, threads=128) as (bx, by):
            A_s = T.alloc_shared((BM, BK), T.float16)
            B_s = T.alloc_shared((BN, BK), T.float16)
            C_t = T.alloc_tmem((BM, BN), T.float32)
            for k in T.Pipelined(K // BK, num_stages=2):
                T.copy(A[by * BM, k * BK], A_s)
                T.copy(B[bx * BN, k * BK], B_s)
                extra = {} if policy is None else {"policy": policy}
                if region_warps is None:
                    T.gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=(k == 0), **extra)
                else:
                    with T.ws(0, warp_group_size=32 * region_warps):
                        T.gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=(k == 0), **extra)
            T.copy(C_t, C[by * BM, bx * BN])

    return kernel


def _lower(kernel):
    with tvm.target.Target(STCUV2_TARGET):
        return tilelang.lower(kernel, target=STCUV2_TARGET).kernel_source


def _sub_tiles(src):
    """The (M, N, K) of every tcgen5 instruction descriptor, in order."""
    return re.findall(r"mma_desc<[^>]*?, (\d+), (\d+), (\d+)>", src)


def _barrier_inits(src):
    """The ``mbarrier[i].init(count)`` table of the lowering, in order."""
    return [(int(i), int(c)) for i, c in re.findall(r"mbarrier\[(\d+)\]\.init\((\d+)\)", src)]


def test_block_partition_splits_n_over_all_four_warps():
    src = _lower(_build())
    subs = _sub_tiles(src)
    assert subs, src
    assert set(subs) == {("128", "32", "64")}, subs  # N=128 over 4 warps


def test_narrowing_the_region_narrows_the_mma():
    """The supported way to run the MMA on fewer warps."""
    src = _lower(_build(region_warps=2))
    subs = _sub_tiles(src)
    assert subs, src
    assert set(subs) == {("128", "64", "64")}, subs  # N=128 over 2 warps


def test_the_region_scoped_barrier_owns_its_own_slot():
    """Every barrier slot has exactly one owner, and the region-scoped producer
    is allocated above the pipeline handshake's range.

    The three passes that publish on the shared mbarrier array used to size it
    by construction instead of by asking the IR: ``MaterializeTangMbarriers`` and
    ``InjectTangMbarrierPartials`` both numbered from slot 0, which is where the
    handshake pass (``InjectS3AsyncCopy``) starts too. Two producers on one slot
    means two ``.init`` calls on it, so the second count overwrites the first and
    every wait on that slot either never flips or releases early. The fix
    (``FreeMbarrierSlot`` in src/tang/transform/s3_mbarrier_utils.h) makes each
    producer allocate above the highest slot already requested in the IR; this
    shape is the live case that reaches two producers at once, the pipeline
    handshake plus the 2-warp region's partial rendezvous.

    The numbers the assertions key on come from the shape's own parameters
    (128 threads / 32 = the block's warps, ``region_warps`` = the region's), not
    from the current allocation: a producer that went back to numbering from 0
    fails uniqueness, and one that raises its base without raising the array size
    fails the declaration check.
    """
    src = _lower(_build(region_warps=2))
    inits = _barrier_inits(src)
    assert inits, src
    indices = [i for i, _ in inits]
    # One init per slot: two producers sharing a base initialise one slot twice,
    # with two different counts.
    assert len(indices) == len(set(indices)), inits
    # The declared array has to cover the highest slot handed out, or the init
    # writes past its end.
    declared = re.findall(r"__shared__ uint64_t mbarrier_mem\[(\d+)\]", src)
    assert declared, src
    assert int(declared[0]) == max(indices) + 1, (declared, inits)
    # The handshake arrives once per warp of the *block*, the region-scoped
    # rendezvous once per warp of the *region* only, and every region-scoped slot
    # has to sit above the handshake's whole range.
    block_warps, region_warps = 128 // 32, 2
    handshake = [i for i, count in inits if count == block_warps]
    region = [i for i, count in inits if count == region_warps]
    assert handshake and region, inits
    assert max(handshake) < min(region), (handshake, region)


def test_full_col_is_the_supported_policy(capfd):
    src = _lower(_build(policy=T.GemmWarpPolicy.FullCol))
    assert set(_sub_tiles(src)) == {("128", "32", "64")}
    assert "not supported" not in capfd.readouterr().err


def test_square_and_full_row_are_reported(capfd):
    """Split-M policies must not silently become the full-width N split."""
    src = _lower(_build(policy=T.GemmWarpPolicy.Square))
    assert set(_sub_tiles(src)) == {("128", "32", "64")}
    err = capfd.readouterr().err
    assert "policy 0 is not supported" in err, err
    assert "narrow the thread region" in err, err
