"""The TMEM drain must not write rows past a dynamic row bound.

A drain writes a whole `BM`-row tile. When the destination's row extent is
dynamic and not a multiple of the tile -- the last token tile of `num_tokens` --
the tail rows belong to whatever follows the tensor: the next batch, split or
row tile. The shared -> global copy path predicates such a store itself; the
drain writes raw buffer stores, so it needs the row count.

These cases pin the contract in the lowering, because they are deterministic
and need no ISS run.
"""

import re


import tilelang
import tilelang.language as T

STCUV2_TARGET = {"kind": "tang", "arch": "stcuv2"}

TB, M3, K, NS = 32, 24, 128, 2
N32 = 32


def _make_kernel(dynamic_rows: bool):
    """Two splits over the token axis; the row extent is dynamic or fixed."""

    nt = T.dynamic("num_tokens") if dynamic_rows else 64

    @T.prim_func
    def kernel(x: T.Tensor[(nt, K), T.bfloat16], fn: T.Tensor[(N32, K), T.float32], out: T.Tensor[(NS, nt, M3), T.float32]) -> None:
        with T.Kernel(T.ceildiv(nt, TB), NS, threads=128) as (px, py):
            acc = T.alloc_fragment((TB, N32), T.float32)
            xs16 = T.alloc_shared((TB, K), T.bfloat16)
            xs = T.alloc_shared((TB, K), T.float32)
            fs = T.alloc_shared((N32, K), T.float32)
            T.clear(acc)
            T.copy(x[px * TB, 0], xs16)
            T.copy(fn, fs)
            for jj in T.serial(K // 4):
                for i, j in T.Parallel(TB, 4):
                    xs[i, jj * 4 + j] = T.cast(xs16[i, jj * 4 + j], T.float32)
            T.gemm(xs, fs, acc, transpose_A=False, transpose_B=True, clear_accum=False, k_step=4)
            # Only the accumulator's first M3 columns have a destination.
            T.copy(acc[:, 0:M3], out[py, px * TB, 0])

    return kernel


def _if_conditions(flat: str) -> list[str]:
    """Every ``if ( ... )`` condition in flat, paren-matched.

    Balanced matching rather than a pattern over the arithmetic: the guard is
    whatever condition the pass emits (``row_hi < rows_owned``, spelled with
    ``k * tiles`` and ``num_tokens`` today), and only the *presence* of the row
    bound in it is the contract. Reading the condition off the source keeps the
    assertion independent of how the row index is written.
    """
    conditions = []
    for match in re.finditer(r"if \s*\(", flat):
        depth = 0
        for pos in range(match.end() - 1, min(match.end() + 500, len(flat))):
            if flat[pos] == "(":
                depth += 1
            elif flat[pos] == ")":
                depth -= 1
                if depth == 0:
                    conditions.append(flat[match.end() : pos])
                    break
    return conditions


def _lower(dynamic_rows: bool) -> str:
    return tilelang.lower(_make_kernel(dynamic_rows), target=STCUV2_TARGET).kernel_source


def test_drain_guards_rows_against_a_dynamic_bound():
    """A dynamic row extent puts a per-strip row guard on the drain's stores."""
    src = _lower(dynamic_rows=True)
    flat = " ".join(src.split())
    assert re.search(r"ldt_16x256b_x\d+\(", flat), "expected an LDT drain"
    assert "num_tokens" in flat, "the drain must consult the destination's row extent:\n" + src
    # The guard is per 16-row strip, not per store: one comparison decides the
    # whole strip, and a boundary strip takes the narrow branch that keeps only
    # the lower half -- so two row-bound comparisons per `ldt` group, whatever
    # the tile. Counted as "conditions that mention the row bound", not as a
    # pattern over the row arithmetic, so a change in how the row index is
    # spelled cannot fail the test on its own.
    # Scoped to the drain: the staging load further up is predicated on the same
    # dynamic extent, and its predicate is not one of these guards.
    drain = flat[flat.index("ldt_16x256b") :]
    guards = [cond for cond in _if_conditions(drain) if "num_tokens" in cond]
    ldt_groups = flat.count("ldt_16x256b_x")
    assert len(guards) == 2 * ldt_groups, (
        "expected one strip guard plus its lower-half branch per LDT group "
        "(%d groups), found %d guarded conditions:\n%s" % (ldt_groups, len(guards), src)
    )
    # ...and the stores they gate are the drain's, so the guard is not somewhere
    # unrelated in the kernel.
    assert "out[" in drain, src


def test_drain_has_no_guard_when_the_tile_provably_fits():
    """In-bounds drains keep the stores they always emitted (no extra code)."""
    src = _lower(dynamic_rows=False)
    flat = " ".join(src.split())
    assert re.search(r"ldt_16x256b_x\d+\(", flat), "expected an LDT drain"
    assert "num_tokens" not in flat, "a fixed in-bounds tile must not grow a row guard:\n" + src
