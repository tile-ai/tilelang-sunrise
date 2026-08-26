"""IR-level checks for MergeLoop's dependency analysis (IsFusionLegal).

Each test builds a two-loop SeqStmt by hand, runs tl.transform.MergeLoop, and
asserts whether the two adjacent loops fused into one. MergeLoop may only fuse
loop bodies that are disjoint at allocation granularity; anything with a
write/read overlap on the same buffer must be left as two loops.

The shared producer/consumer case is the pipeline hazard motivating the guard:
MergeLoop runs before ThreadSync("shared"), so fusing a producer loop with the
consumer loop that reads the same shared allocation would strip the slot where
ThreadSync inserts the __syncthreads.
"""

import tilelang
import tilelang.testing
from tilelang import tvm
import tilelang.language as T
from tvm import IRModule, tirx


def _count_top_loops(func):
    """Number of loops at the top of the function body, counting a loop whether
    it is bare or wrapped in an AttrStmt (e.g. async_scope). A single merged
    loop under its wrapper counts as one."""

    def _is_loop(s):
        return isinstance(s, tirx.For) or (isinstance(s, tirx.AttrStmt) and isinstance(s.body, tirx.For))

    body = func.body
    if isinstance(body, tirx.SeqStmt):
        return sum(1 for s in body.seq if _is_loop(s))
    return 1 if _is_loop(body) else 0


def _merge(func, pass_configs=None):
    if pass_configs is None:
        # MergeLoop is default-disabled (TL_DISABLE_MERGE_LOOP defaults to
        # True), so enable it explicitly to exercise the fusion logic under test.
        pass_configs = {tilelang.PassConfigKey.TL_DISABLE_MERGE_LOOP: False}
    mod = IRModule.from_expr(func.with_attr("global_symbol", "main"))
    with tilelang.transform.PassContext(config=pass_configs):
        out = tilelang.transform.MergeLoop()(mod)
    return _count_top_loops(out["main"])


def _fused_body_var_names(func, pass_configs=None):
    """Names of the vars actually referenced inside the single fused loop.

    _count_top_loops only sees the loop *count*, so it stays green even if the
    var substitution in MergeLoopRewriter is dropped entirely: the fused body
    would still be one loop, just one referencing a loop var that no longer
    exists. Merging loops with different loop vars requires rewriting every
    subsequent loop's var to the first loop's, so assert on the body instead.
    """
    if pass_configs is None:
        # MergeLoop is default-disabled; enable it explicitly (see _merge).
        pass_configs = {tilelang.PassConfigKey.TL_DISABLE_MERGE_LOOP: False}
    mod = IRModule.from_expr(func.with_attr("global_symbol", "main"))
    with tilelang.transform.PassContext(config=pass_configs):
        out = tilelang.transform.MergeLoop()(mod)
    body = out["main"].body
    assert isinstance(body, tirx.For), f"expected a single fused loop, got {type(body)}"
    names = set()
    tirx.stmt_functor.post_order_visit(body.body, lambda n: names.add(n.name) if isinstance(n, tirx.Var) else None)
    return body.loop_var.name, names


def test_merge_loop_fuses_independent_buffers():
    @T.prim_func
    def independent(
        A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32"), C: T.Buffer((128,), "float32"), D: T.Buffer((128,), "float32")
    ):
        for i in range(128):
            C[i] = A[i]
        for j in range(128):
            D[j] = B[j]

    assert _merge(independent) == 1


def test_merge_loop_substitutes_second_loop_var():
    # The fused loop keeps the *first* loop's var, so the second body must be
    # rewritten onto it. A leftover `jjj` reference would be a free var.
    @T.prim_func
    def differing_names(
        A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32"), C: T.Buffer((128,), "float32"), D: T.Buffer((128,), "float32")
    ):
        for i in range(128):
            C[i] = A[i]
        for jjj in range(128):
            D[jjj] = B[jjj]

    loop_var, used = _fused_body_var_names(differing_names)
    assert loop_var == "i"
    assert used == {"i"}, f"stale loop var left in fused body: {used}"


def test_merge_loop_substitutes_every_var_in_a_run():
    # A run of three fuses into one loop, so the rewrite must run for *each*
    # subsequent loop (k = 1..n-1), not just the second.
    @T.prim_func
    def three_names(
        A: T.Buffer((128,), "float32"),
        B: T.Buffer((128,), "float32"),
        C: T.Buffer((128,), "float32"),
        D: T.Buffer((128,), "float32"),
        E: T.Buffer((128,), "float32"),
        F: T.Buffer((128,), "float32"),
    ):
        for aaa in range(128):
            C[aaa] = A[aaa]
        for bbb in range(128):
            D[bbb] = B[bbb]
        for ccc in range(128):
            F[ccc] = E[ccc]

    loop_var, used = _fused_body_var_names(three_names)
    assert loop_var == "aaa"
    assert used == {"aaa"}, f"stale loop vars left in fused body: {used}"


def test_merge_loop_substitutes_var_inside_nested_loop():
    # The substitution must recurse into nested loops: the inner body references
    # the outer var being rewritten, so a shallow rewrite would leave `zzz` free.
    @T.prim_func
    def nested_names(
        A: T.Buffer((128, 4), "float32"),
        B: T.Buffer((128, 4), "float32"),
        C: T.Buffer((128, 4), "float32"),
        D: T.Buffer((128, 4), "float32"),
    ):
        for i in range(128):
            for u in range(4):
                C[i, u] = A[i, u]
        for zzz in range(128):
            for w in range(4):
                D[zzz, w] = B[zzz, w]

    loop_var, used = _fused_body_var_names(nested_names)
    assert loop_var == "i"
    assert "zzz" not in used, f"stale outer var survived in nested body: {used}"
    assert used == {"i", "u", "w"}, used


def test_merge_loop_rejects_loop_carried_raw():
    @T.prim_func
    def carried_raw(A: T.Buffer((129,), "float32"), B: T.Buffer((128,), "float32")):
        for i in range(128):
            A[i] = T.float32(1.0)
        for j in range(128):
            B[j] = A[j + 1]

    assert _merge(carried_raw) == 2


def test_merge_loop_rejects_identical_index_ignoring_loop_var():
    # Same index on both sides, but the index ignores the fused loop var, so the
    # read must observe the LAST iteration's value: fusion would change results.
    @T.prim_func
    def index_ignores_loop_var(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
        S = T.decl_buffer((4,), "float32", scope="local")
        for i in range(128):
            for u in range(4):
                S[u] = A[i]
        for j in range(128):
            for v in range(4):
                B[j] = S[v]

    assert _merge(index_ignores_loop_var) == 2


def test_merge_loop_rejects_waw():
    @T.prim_func
    def waw(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
        for i in range(128):
            A[i] = T.float32(1.0)
        for j in range(128):
            A[j] = B[j]

    assert _merge(waw) == 2


def test_merge_loop_fuses_shared_read_only():
    @T.prim_func
    def shared_read(A: T.Buffer((128,), "float32"), C: T.Buffer((128,), "float32"), D: T.Buffer((128,), "float32")):
        for i in range(128):
            C[i] = A[i]
        for j in range(128):
            D[j] = A[j]

    assert _merge(shared_read) == 1


def test_merge_loop_rejects_shared_producer_consumer_raw():
    # Pipeline hazard: a producer loop fills a shared allocation and a consumer
    # loop reads it. ThreadSync would insert a __syncthreads between them; if
    # MergeLoop fused them first that barrier loses its slot.
    @T.prim_func
    def shared_producer_consumer(A: T.Buffer((128,), "float32"), C: T.Buffer((128,), "float32")):
        S = T.decl_buffer((128,), "float32", scope="shared")
        for i in range(128):
            S[i] = A[i]
        for j in range(128):
            C[j] = S[j]

    assert _merge(shared_producer_consumer) == 2


def test_merge_loop_rejects_drain_then_refill_to_keep_shared_reuse():
    # Reuse hazard (distinct from the barrier hazard above): the first loop
    # drains shared buffer S1 and the second starts filling a *different* shared
    # buffer S2. Nothing aliases, so fusion is dependence-legal — but the two
    # lifetimes are adjacent rather than overlapping, so
    # MergeSharedMemoryAllocations would give S1 and S2 the same arena offset.
    # Fusing makes them overlap (iteration i+1 re-reads S1 after iteration i has
    # written S2), costing a whole buffer of shared memory to save one loop.
    @T.prim_func
    def drain_then_refill(
        A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32"), C: T.Buffer((128,), "float32"), D: T.Buffer((128,), "float32")
    ):
        S1 = T.decl_buffer((128,), "float32", scope="shared")
        S2 = T.decl_buffer((128,), "float32", scope="shared")
        for i in range(128):
            S1[i] = A[i]
        for j in range(128):
            C[j] = S1[j]
        for k in range(128):
            S2[k] = B[k]
        for m in range(128):
            D[m] = S2[m]

    # No pair fuses: S1 fill/drain is a producer/consumer RAW, the drain/refill
    # pair is held back to preserve reuse, and S2's is another RAW.
    assert _merge(drain_then_refill) == 4


def test_merge_loop_fuses_two_shared_fills():
    # The write/write counterpart, and the case that must keep fusing: a GEMM
    # prologue filling two shared tiles that are both live into the consumer.
    # There is no reuse to lose here, so the reuse guard must not fire.
    @T.prim_func
    def two_shared_fills(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
        S1 = T.decl_buffer((128,), "float32", scope="shared")
        S2 = T.decl_buffer((128,), "float32", scope="shared")
        for i in range(128):
            S1[i] = A[i]
        for j in range(128):
            S2[j] = B[j]

    assert _merge(two_shared_fills) == 1


def test_merge_loop_rejects_nontrivial_step():
    # The merged loop keeps only the first loop's step, so fusing a step=1 loop
    # with a step=2 loop would silently run the second body over iterations it
    # never had. TVMScript's `range` cannot express a non-unit step, so build
    # the two For nodes by hand (independent buffers, so only the step differs).
    from tvm.tirx import For, ForKind, Var, SeqStmt, BufferStore, decl_buffer, IntImm

    A = decl_buffer((8,), "float32", name="A")
    C = decl_buffer((8,), "float32", name="C")
    i = Var("i", "int32")
    j = Var("j", "int32")
    loop_a = For(i, 0, 8, ForKind.SERIAL, BufferStore(A, T.float32(1.0), [i]))
    loop_b = For(j, 0, 8, ForKind.SERIAL, BufferStore(C, T.float32(2.0), [j]), step=IntImm("int32", 2))
    func = tirx.PrimFunc([], SeqStmt([loop_a, loop_b]))

    assert _merge(func) == 2


def _two_loop_func(wrap_first=None, wrap_second=None, node="default"):
    """Build a two-loop SeqStmt over independent buffers, each loop optionally
    wrapped in AttrStmt(node, key, 1). Independent buffers isolate the test
    to wrapper handling (IsFusionLegal alone would allow the fusion).

    ``node`` selects the AttrStmt node: a literal (the default, a scope-like
    node shared by both wrappers) or "own_loop_var", which makes each wrapper's
    node that loop's own loop_var, the way LowerOpaqueBlock emits pragmas."""
    from tvm.tirx import For, ForKind, Var, SeqStmt, BufferStore, decl_buffer, IntImm, AttrStmt

    A = decl_buffer((8,), "float32", name="A")
    C = decl_buffer((8,), "float32", name="C")
    i = Var("i", "int32")
    j = Var("j", "int32")
    a = For(i, 0, 8, ForKind.SERIAL, BufferStore(A, T.float32(1.0), [i]))
    b = For(j, 0, 8, ForKind.SERIAL, BufferStore(C, T.float32(2.0), [j]))
    node_a, node_b = (i, j) if node == "own_loop_var" else (node, node)
    s0 = AttrStmt(node_a, wrap_first, IntImm("int32", 1), a) if wrap_first else a
    s1 = AttrStmt(node_b, wrap_second, IntImm("int32", 1), b) if wrap_second else b
    return tirx.PrimFunc([], SeqStmt([s0, s1]))


def test_merge_loop_fuses_matching_async_scope():
    # Both loops share the same wrapper: fusing under one async_scope is
    # semantically identical (the GEMM A+B copy case). Must fuse.
    assert _merge(_two_loop_func("async_scope", "async_scope")) == 1


def test_merge_loop_rejects_async_scope_then_plain():
    # First loop in async_scope, second plain. Fusing under the async wrapper
    # would pull the plain copy into the async scope. Must NOT fuse.
    assert _merge(_two_loop_func("async_scope", None)) == 2


def test_merge_loop_rejects_plain_then_async_scope():
    # First plain, second in async_scope. Fusing into a bare loop would drop
    # the async scope, degrading an async copy to synchronous. Must NOT fuse.
    assert _merge(_two_loop_func(None, "async_scope")) == 2


def test_merge_loop_rejects_mismatched_attr_key():
    # An unrelated attr key must not be treated as a compatible wrapper. Must
    # NOT fuse.
    assert _merge(_two_loop_func("pragma_import_c", None)) == 2


def test_merge_loop_fuses_pragma_wrapper_with_own_loop_var_node():
    # LowerOpaqueBlock emits AttrStmt(node=<the loop's own loop_var>, ...) for
    # loop pragmas, so two distinct loops NEVER have structurally equal nodes.
    # Comparing the node made every pragma-wrapped pair incompatible, which
    # silently reduced the pass to a no-op on real kernels (essentially every
    # lowered loop carries pragma_unroll_explicit). A self-referential node
    # names the loop rather than describing scope, so it must be exempt.
    assert _merge(_two_loop_func("pragma_unroll_explicit", "pragma_unroll_explicit", node="own_loop_var")) == 1


def test_merge_loop_rejects_self_node_paired_with_scope_node():
    # One wrapper's node is its own loop_var, the other's is a real scope node.
    # Same key and value, but these are not the same wrapper: exempting the node
    # comparison must not collapse this into a match. Must NOT fuse.
    from tvm.tirx import For, ForKind, Var, SeqStmt, BufferStore, decl_buffer, IntImm, AttrStmt

    A = decl_buffer((8,), "float32", name="A")
    C = decl_buffer((8,), "float32", name="C")
    i = Var("i", "int32")
    j = Var("j", "int32")
    a = For(i, 0, 8, ForKind.SERIAL, BufferStore(A, T.float32(1.0), [i]))
    b = For(j, 0, 8, ForKind.SERIAL, BufferStore(C, T.float32(2.0), [j]))
    s0 = AttrStmt(i, "async_scope", IntImm("int32", 1), a)  # node = own loop_var
    s1 = AttrStmt("default", "async_scope", IntImm("int32", 1), b)  # node = scope
    func = tirx.PrimFunc([], SeqStmt([s0, s1]))
    assert _merge(func) == 2


def test_merge_loop_rejects_pragma_wrapper_with_mismatched_value():
    # Self-referential nodes are exempt, but attr_key and value are still
    # compared: unroll_explicit False and True are different directives.
    from tvm.tirx import For, ForKind, Var, SeqStmt, BufferStore, decl_buffer, IntImm, AttrStmt

    A = decl_buffer((8,), "float32", name="A")
    C = decl_buffer((8,), "float32", name="C")
    i = Var("i", "int32")
    j = Var("j", "int32")
    a = For(i, 0, 8, ForKind.SERIAL, BufferStore(A, T.float32(1.0), [i]))
    b = For(j, 0, 8, ForKind.SERIAL, BufferStore(C, T.float32(2.0), [j]))
    s0 = AttrStmt(i, "pragma_unroll_explicit", IntImm("int32", 0), a)
    s1 = AttrStmt(j, "pragma_unroll_explicit", IntImm("int32", 1), b)
    func = tirx.PrimFunc([], SeqStmt([s0, s1]))
    assert _merge(func) == 2


def test_merge_loop_default_disabled_and_disable_config_is_identity():
    # MergeLoop is default-disabled (STCU/ptcc miscompiles the fused body), so
    # disabling must leave the IR exactly unchanged — the escape hatch for
    # bisecting a suspected bad fusion.
    @T.prim_func
    def independent(
        A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32"), C: T.Buffer((128,), "float32"), D: T.Buffer((128,), "float32")
    ):
        for i in range(128):
            C[i] = A[i]
        for j in range(128):
            D[j] = B[j]

    # Default (no config): disabled, exact identity.
    mod = IRModule.from_expr(independent.with_attr("global_symbol", "main"))
    with tilelang.transform.PassContext(config={}):
        out = tilelang.transform.MergeLoop()(mod)
    assert _count_top_loops(out["main"]) == 2
    assert tvm.ir.structural_equal(out["main"], mod["main"])

    # Explicit disable: also an exact identity.
    configs = {tilelang.PassConfigKey.TL_DISABLE_MERGE_LOOP: True}
    assert _merge(independent, pass_configs=configs) == 2

    mod = IRModule.from_expr(independent.with_attr("global_symbol", "main"))
    with tilelang.transform.PassContext(config=configs):
        out = tilelang.transform.MergeLoop()(mod)
    assert tvm.ir.structural_equal(out["main"], mod["main"])

    # Explicit enable fuses the two loops into one.
    enable = {tilelang.PassConfigKey.TL_DISABLE_MERGE_LOOP: False}
    assert _merge(independent, pass_configs=enable) == 1


if __name__ == "__main__":
    tilelang.testing.main()
