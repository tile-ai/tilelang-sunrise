"""Tests for the HoistCopyAddresses pass.

The pass turns a loop-varying copy address into a loop-carried running scalar.
These tests pin down which storage scopes and copy directions qualify, since
that decision is what distinguishes a rewritten copy from an untouched one.
"""

from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing


def _run(func):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    return tl.transform.HoistCopyAddresses()(mod)["main"]


def _addr_scalars(func):
    """Names of the running-address scalars the pass allocated, in source order.

    The pass may hoist one address per (buffer, stride, base) per loop level, so
    an inner loop gets its own scalar with a de-duplicated name (`A_addr_0_1`).
    """
    import re

    return re.findall(r'(\w+) = T\.alloc_buffer\(\(1,\), "int32", scope="local"\)', func.script())


def _check_unchanged(func):
    """The pass must leave this function structurally alone."""
    after = _run(func)
    tvm.ir.assert_structural_equal(after, func.with_attr("global_symbol", "main"), map_free_vars=True)


def test_global_to_shared():
    """Both sides are memory-resident, so both get a running address."""

    @T.prim_func
    def before(A: T.Tensor((1024, 1024), T.float16)):
        As = T.alloc_shared((8, 64), T.float16)
        for k in T.serial(16):
            for i in T.serial(8):
                As[i, 0] = A[i, k * 64]

    # Outer k-loop hoists A's column address; the inner i-loop hoists the row
    # address of each side.
    assert _addr_scalars(_run(before)) == ["A_addr_0", "A_addr_0_1", "As_addr_1"]


def test_shared_to_global():
    """shared→global rewrites the read and the write independently."""

    @T.prim_func
    def before(C: T.Tensor((1024, 1024), T.float16)):
        Cs = T.alloc_shared((8, 64), T.float16)
        for k in T.serial(16):
            for i in T.serial(8):
                C[i, k * 64] = Cs[i, 0]

    assert _addr_scalars(_run(before)) == ["C_addr_0", "Cs_addr_0", "C_addr_1"]


def test_register_side_is_not_rewritten():
    """A fragment index must stay affine, or the array spills out of registers.

    Only the global side of this copy may become a running scalar; `Af[i]` has
    to survive verbatim.
    """

    @T.prim_func
    def before(A: T.Tensor((1024, 1024), T.float16)):
        Af = T.alloc_fragment((8,), T.float16)
        for k in T.serial(16):
            for i in T.serial(8):
                Af[i] = A[i, k * 64]

    after = _run(before)
    assert _addr_scalars(after) == ["A_addr_0", "A_addr_0_1"]
    assert "Af[i]" in after.script()


def test_fragment_to_global():
    """register→global is supported; only the global side is rewritten."""

    @T.prim_func
    def before(C: T.Tensor((1024, 1024), T.float16)):
        Cf = T.alloc_fragment((8,), T.float16)
        for k in T.serial(16):
            for i in T.serial(8):
                C[i, k * 64] = Cf[i]

    after = _run(before)
    assert _addr_scalars(after) == ["C_addr_0", "C_addr_0_1"]
    assert "Cf[i]" in after.script()


def test_shared_to_shared_unsupported():
    """Not a copy direction this pass handles."""

    @T.prim_func
    def before():
        As = T.alloc_shared((8, 64), T.float16)
        Bs = T.alloc_shared((8, 64), T.float16)
        for k in T.serial(16):
            for i in T.serial(8):
                Bs[i, k * 4] = As[i, k * 4]

    _check_unchanged(before)


def test_global_to_global_unsupported():
    """Not a copy direction this pass handles."""

    @T.prim_func
    def before(A: T.Tensor((1024, 1024), T.float16), B: T.Tensor((1024, 1024), T.float16)):
        for k in T.serial(16):
            for i in T.serial(8):
                B[i, k * 64] = A[i, k * 64]

    _check_unchanged(before)


def test_vectorized_copy_keeps_ramp():
    """A Ramp index is split on its base and re-wrapped as a Ramp."""

    @T.prim_func
    def before(A: T.Tensor((1024, 1024), T.float16)):
        As = T.alloc_shared((8, 64), T.float16)
        for k in T.serial(16):
            for i in T.serial(8):
                for v in T.vectorized(4):
                    As[i, v] = A[i, k * 64 + v]

    after = _run(before)
    assert _addr_scalars(after) == ["A_addr_0", "A_addr_0_1", "As_addr_1"]
    # The lane offset stays inside the Ramp: only its base became a scalar.
    assert "A[A_addr_0_1[0], A_addr_0[0] + v]" in after.script()


def test_parallel_loop_skipped():
    """A parallel loop has no sequential carry to chain a running address through."""

    @T.prim_func
    def before(A: T.Tensor((1024, 1024), T.float16)):
        As = T.alloc_shared((8, 64), T.float16)
        for k in T.parallel(16):
            for i in T.serial(8):
                As[i, 0] = A[i, k * 64]

    after = _run(before)
    # The parallel k-loop itself is skipped; the inner serial i-loop still
    # qualifies, so only its addresses are hoisted.
    assert "A_addr" in after.script()
    assert "for k in T.parallel(16)" in after.script()


# ---------------------------------------------------------------------------
# Address splitting: stride / base / residual
# ---------------------------------------------------------------------------


def test_gemm_address_matches_docstring():
    """The documented GEMM address, in the loop shape this pass actually sees.

    The pass runs after VectorizeLoop/UnrollLoop, so a copy's inner loop is
    vectorized by then: `i` is bound *inside* the rewritten `k` loop and its term
    cannot be hoisted. Result is exactly the form the file docstring gives,
    `addr[0] + ((k+1)%2*8192 + i*512)`, with `tx*2` folded into the base.
    """

    @T.prim_func
    def before(A: T.Tensor((1024, 1024), T.float16)):
        As = T.alloc_shared((16384,), T.float16)
        for tx in T.thread_binding(128, thread="threadIdx.x"):
            for k in T.serial(16):
                for i in T.vectorized(8):
                    As[i] = A[0, (k + 1) % 2 * 8192 + i * 512 + tx * 2 + k * 64]

    script = _run(before).script()
    assert "A_addr_0[0] = tx * 2" in script  # base
    assert "A[0, A_addr_0[0] + ((k + 1) % 2 * 8192 + i * 512)]" in script  # residual
    assert "A_addr_0[0] = A_addr_0[0] + 64" in script  # stride


def test_invariant_terms_fold_into_initialiser():
    """A *serial* inner loop is claimed by the pass itself, leaving no residual.

    Same address as test_gemm_address_matches_docstring, but with `i` serial: the
    inner loop is rewritten on its own, so `i * 512` becomes its stride and every
    other term — k-linear part included — is invariant w.r.t. `i` and folds into
    the initialiser. This shape does not survive VectorizeLoop in the real
    pipeline; it is here to pin the per-loop behaviour.
    """

    @T.prim_func
    def before(A: T.Tensor((1024, 1024), T.float16)):
        As = T.alloc_shared((16384,), T.float16)
        for tx in T.thread_binding(128, thread="threadIdx.x"):
            for k in T.serial(16):
                for i in T.serial(8):
                    As[i] = A[0, (k + 1) % 2 * 8192 + i * 512 + tx * 2 + k * 64]

    script = _run(before).script()
    assert "A_addr_0[0] = (k + 1) % 2 * 8192 + tx * 2 + k * 64" in script
    assert "A_addr_0[0] = A_addr_0[0] + 512" in script
    # Fully folded: nothing is left to recompute at the use site.
    assert "A[0, A_addr_0[0]]" in script


def test_residual_stays_in_place():
    """A term bound inside the loop can neither fold into stride nor into base.

    `tx` is thread-bound *inside* the k loop, so `tx * 2` must be recomputed at
    the use site as `addr[0] + tx * 2` while the k-linear part still hoists.
    """

    @T.prim_func
    def before(A: T.Tensor((1024, 1024), T.float16)):
        As = T.alloc_shared((256,), T.float16)
        for k in T.serial(16):
            for tx in T.thread_binding(128, thread="threadIdx.x"):
                As[tx] = A[0, k * 64 + tx * 2]

    script = _run(before).script()
    assert "A[0, A_addr_0[0] + tx * 2]" in script
    assert "A_addr_0[0] = A_addr_0[0] + 64" in script


def test_multiple_linear_terms_accumulate():
    """Several loop_var terms sum into a single stride (64 + 8 = 72)."""

    @T.prim_func
    def before(A: T.Tensor((1024, 1024), T.float16)):
        As = T.alloc_shared((8,), T.float16)
        for k in T.serial(16):
            As[0] = A[0, k * 64 + k * 8]

    assert "A_addr_0[0] = A_addr_0[0] + 72" in _run(before).script()


def test_cancelling_coefficients_not_rewritten():
    """stride == 0 means there is no running sum worth building."""

    @T.prim_func
    def before(A: T.Tensor((1024, 1024), T.float16)):
        As = T.alloc_shared((8,), T.float16)
        for k in T.serial(16):
            As[0] = A[0, k * 64 + k * -64 + 5]

    _check_unchanged(before)


def test_no_linear_term_not_rewritten():
    """A loop-invariant address has nothing to carry."""

    @T.prim_func
    def before(A: T.Tensor((1024, 1024), T.float16)):
        As = T.alloc_shared((8,), T.float16)
        for _ in T.serial(16):
            As[0] = A[0, 7]

    _check_unchanged(before)


def test_symbolic_coefficient_not_rewritten():
    """LinearCoeff only accepts an IntImm multiplier, so `k * n` is left alone."""

    @T.prim_func
    def before(A: T.Tensor((1024, 1024), T.float16), n: T.int32):
        As = T.alloc_shared((8,), T.float16)
        for k in T.serial(16):
            As[0] = A[0, k * n]

    _check_unchanged(before)


def test_nonzero_loop_min_offsets_initialiser():
    """init = base + min * stride, so a loop starting at 3 must pre-skip 3 steps.

    base 100 + min 3 * stride 64 = 292; getting this wrong would read the wrong
    address on every iteration.
    """

    @T.prim_func
    def before(A: T.Tensor((1024, 1024), T.float16)):
        As = T.alloc_shared((8,), T.float16)
        for k in T.serial(3, 16):
            As[0] = A[0, k * 64 + 100]

    script = _run(before).script()
    assert "A_addr_0[0] = 292" in script
    assert "for k in range(3, 16)" in script


# ---------------------------------------------------------------------------
# Sharing and increment placement
# ---------------------------------------------------------------------------


def test_distinct_buffers_get_distinct_scalars():
    """Buffer identity is part of the dedup key: same stride/base must not collapse."""

    @T.prim_func
    def before(A: T.Tensor((1024, 1024), T.float16), B: T.Tensor((1024, 1024), T.float16)):
        As = T.alloc_shared((8,), T.float16)
        Bs = T.alloc_shared((8,), T.float16)
        for k in T.serial(16):
            As[0] = A[0, k * 64]
            Bs[0] = B[0, k * 64]

    script = _run(before).script()
    assert _addr_scalars(_run(before)) == ["A_addr_0", "B_addr_1"]
    assert "A[0, A_addr_0[0]]" in script
    assert "B[0, B_addr_1[0]]" in script


def test_same_base_stride_shares_one_scalar():
    """Matching (buffer, stride, base) shares a scalar even when residuals differ."""

    @T.prim_func
    def before(A: T.Tensor((1024, 1024), T.float16)):
        As = T.alloc_shared((256,), T.float16)
        for k in T.serial(16):
            for tx in T.thread_binding(128, thread="threadIdx.x"):
                As[tx] = A[0, k * 64 + tx * 2]
                As[tx + 128] = A[0, k * 64 + tx * 4]

    after = _run(before)
    # One scalar, two use sites with different residuals.
    assert _addr_scalars(after) == ["A_addr_0"]
    script = after.script()
    assert "A[0, A_addr_0[0] + tx * 2]" in script
    assert "A[0, A_addr_0[0] + tx * 4]" in script


def test_reads_in_one_iteration_agree():
    """The bump is at the loop tail, so both reads observe the same address.

    Incrementing mid-body would hand the second read the next iteration's value.
    """

    @T.prim_func
    def before(A: T.Tensor((1024, 1024), T.float16)):
        As = T.alloc_shared((8,), T.float16)
        Bs = T.alloc_shared((8,), T.float16)
        for k in T.serial(16):
            As[0] = A[0, k * 64]
            Bs[0] = A[0, k * 64]

    lines = [ln.strip() for ln in _run(before).script().splitlines()]
    body = lines[lines.index("for k in range(16):") + 1 :]
    reads = [i for i, ln in enumerate(body) if "A[0, A_addr_0[0]]" in ln]
    bump = next(i for i, ln in enumerate(body) if "A_addr_0[0] + 64" in ln)
    assert len(reads) == 2, body
    assert max(reads) < bump, f"increment must follow every read: {body}"


if __name__ == "__main__":
    tilelang.testing.main()
