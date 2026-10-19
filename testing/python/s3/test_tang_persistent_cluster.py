"""S3/TANG persistent + cluster support — tests for §1-§5b.

Layering (each section is independent, cheaper first):

  §1  DSL exports           TestClusterDSL
       Pure Python: `T.*` namespace exposes the cluster primitives,
       ``_normalize_cluster_dims`` handles int/tuple/None correctly.
       Does not lower, does not touch a device.

  §2  Scheduler.coord       TestClusterSchedulerCoord
       PersistentTileScheduler.coord() is a pure PrimExpr builder — for a
       Python ``int`` tile_id we can constant-fold it via arith.Analyzer
       and compare against the expected (m_row, n_col) decomposition.
       Verifies cluster-reshape (M' = ceildiv(num_m_tiles, cluster_size))
       and swizzle interaction.

  §3  Lowering assertions   TestClusterLowering
       Stop at ``lower_to_host_device_ir`` — this is the last stage before
       target codegen, so it works TODAY even though TANG codegen is missing
       cluster handlers. Asserts:
         - cluster_dims attr is on the device PrimFunc
         - the device IR still contains the cluster ops as CallNodes
         - alloc_cluster_barrier lowers to a shared allocation with
           the ``shared.cluster_barrier`` storage scope

  §4  Codegen dump          TestClusterCodegenDump
       Runs the full ``tilelang.lower`` (through TANG codegen) and greps
       the generated C++ source. Encodes the Phase 1 codegen contract for
       cluster ops (``tl::block_rank_in_cluster``, ``tl::cluster_*`` and
       the ``cluster.h`` include gate). All green as of Phase 1 landing.

  §5a Ptcc compile-only     TestClusterPtccCompile
       Runs ptcc on the generated source. Verifies the TANG asm in
       ``src/tl_templates/tang/cluster.h`` is accepted (``sync.all.bkcls``,
       ``__stvm_read_sreg_cluster_ctarank`` builtin). Needs ptcc; does not
       need iss / ptpu.

  §5b ISS numeric           TestClusterNumericISS
       Full simulator pipeline (codegen → ptcc → loader → stcu_rm). Verifies
       ``block_rank_in_cluster`` reads ``cluster_ctarank`` and the two CTAs
       see ranks ``[0, 1]``. Needs ptcc + iss + loader + ptpu.

Run:
    python -m pytest testing/python/s3/test_tang_persistent_cluster.py -v
"""

from __future__ import annotations

import re

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.engine.lower import lower_to_host_device_ir
from tilelang.backend.module import create_backend_context
from tilelang.language.kernel import _normalize_cluster_dims
from tvm import arith, tirx


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Dict-form target: the ``"tang -arch=stcuv2"`` CLI string form was rejected
# by TVM after a recent upgrade; the dict form is the supported way.
STCUV2 = tvm.target.Target({"kind": "tang", "arch": "stcuv2"})


def _ceildiv(a: int, b: int) -> int:
    return -(-a // b)


def _fold_int(expr) -> int:
    """Constant-fold a PrimExpr built by ``coord()`` for an int tile_id."""
    simplified = arith.Analyzer().simplify(expr)
    assert isinstance(simplified, tirx.IntImm), f"expected constant-foldable, got {type(simplified).__name__}: {simplified}"
    return int(simplified.value)


def _walk_calls(stmt, op_name: str) -> list[tirx.Call]:
    """Collect every CallNode in ``stmt`` whose Op name matches ``op_name``."""
    found: list[tirx.Call] = []

    def visitor(node):
        if isinstance(node, tirx.Call):
            op = node.op
            if hasattr(op, "name") and op.name == op_name:
                found.append(node)

    tirx.stmt_functor.post_order_visit(stmt, visitor)
    return found


def _device_func(mod: tvm.IRModule) -> tirx.PrimFunc:
    """Extract the single device PrimFunc from a lowered module."""
    fns = list(mod.functions.items())
    assert len(fns) == 1, f"expected 1 device fn, got {len(fns)}: {[g.name_hint for g, _ in fns]}"
    return fns[0][1]


# ===========================================================================
# §1  DSL exports (pure Python)
# ===========================================================================
class TestClusterDSL:
    """Cluster primitives are exposed on ``tilelang.language`` (``T.*``)."""

    @pytest.mark.parametrize(
        "name",
        [
            "ClusterKernel",
            "Kernel",
            "block_rank_in_cluster",
            "cluster_arrive_relaxed",
            "cluster_arrive",
            "cluster_wait",
            "cluster_sync",
            pytest.param("alloc_cluster_barrier", marks=pytest.mark.skip(reason="barrier support removed (handled elsewhere)")),
            "PersistentTileScheduler",
        ],
    )
    def test_cluster_primitive_in_namespace(self, name):
        assert hasattr(T, name), f"tilelang.language.{name} missing"
        assert callable(getattr(T, name)), f"tilelang.language.{name} not callable"

    @pytest.mark.parametrize(
        "raw,expected",
        [
            (None, None),
            (1, None),  # (1,1,1) collapses to None
            (2, [2, 1, 1]),
            ((2, 1, 1), [2, 1, 1]),
            ([2, 1, 1], [2, 1, 1]),
            ((4, 2, 1), [4, 2, 1]),
        ],
    )
    def test_normalize_cluster_dims(self, raw, expected):
        assert _normalize_cluster_dims(raw) == expected

    def test_normalize_cluster_dims_rejects_bad(self):
        with pytest.raises(ValueError):
            _normalize_cluster_dims("2")  # not int/tuple/list/None


# ===========================================================================
# §2  Scheduler.coord() — pure PrimExpr, constant-foldable
# ===========================================================================
class TestClusterSchedulerCoord:
    """Verifies PersistentTileScheduler.coord() semantics for cluster_size >= 1.

    Setup notes:
      - ``num_workers`` MUST be passed explicitly, else the constructor probes
        the GPU (``driver.get_num_sms()``) which fails on CPU-only hosts.
      - We test ``stateful=False`` so ``coord()`` builds pure expressions we
        can constant-fold via arith.Analyzer.
    """

    @staticmethod
    def _sched(m_tiles, n_tiles, cluster_size, swizzle):
        return T.PersistentTileScheduler(
            m_tiles,
            n_tiles,
            num_workers=8,  # arbitrary; coord() only uses reshape sizes
            cluster_size=cluster_size,
            swizzle_size=swizzle,
            stateful=False,
        )

    @pytest.mark.parametrize(
        "m_tiles,n_tiles,cluster_size,swizzle",
        [
            (4, 2, 1, 1),  # baseline (no cluster, no swizzle)
            (4, 2, 2, 1),  # cluster=2, M reshapes to 2
            (8, 2, 2, 1),  # cluster=2, M reshapes to 4
            (8, 2, 4, 1),  # cluster=4, M reshapes to 2
            (4, 2, 2, 2),  # cluster + swizzle
            (5, 2, 2, 1),  # uneven M — ceildiv rounds up (M'=3)
        ],
    )
    def test_coord_covers_reshaped_grid(self, m_tiles, n_tiles, cluster_size, swizzle):
        """Every tile_id in [0, m_clusters*n_tiles) maps to a distinct
        (m_row, n_col) inside the reshaped grid, and covers it entirely."""
        m_clusters = _ceildiv(m_tiles, cluster_size)
        total = m_clusters * n_tiles

        seen: set[tuple[int, int]] = set()
        sched = self._sched(m_tiles, n_tiles, cluster_size, swizzle)
        for tid in range(total):
            m_expr, n_expr = sched.coord(tid)
            m, n = _fold_int(m_expr), _fold_int(n_expr)
            assert 0 <= m < m_clusters, f"tid={tid}: m={m} out of cluster-row range [0,{m_clusters})"
            assert 0 <= n < n_tiles, f"tid={tid}: n={n} out of n range [0,{n_tiles})"
            assert (m, n) not in seen, f"tid={tid} maps to already-seen ({m},{n})"
            seen.add((m, n))

        assert len(seen) == total, f"coord() should be a bijection onto the reshaped grid; got {len(seen)} unique of {total}"

    def test_coord_cluster_row_semantic(self):
        """For cluster_size=2 the caller does ``bx = m_row * cluster_size +
        cta_rank`` — so ``m_row`` from coord() must be a cluster-row index
        (i.e. m_row * cluster_size + rank stays in [0, M) for rank<cluster_size).
        """
        M, N, C = 8, 3, 2
        sched = self._sched(M, N, C, 1)
        m_clusters = _ceildiv(M, C)
        for tid in range(m_clusters * N):
            m_row = _fold_int(sched.coord(tid)[0])
            # every rank in the cluster maps back into the M range
            for rank in range(C):
                assert m_row * C + rank < M or m_row == m_clusters - 1, f"tid={tid} cluster_row={m_row}: rank {rank} out of range"

    def test_coord_baseline_matches_row_major(self):
        """Without cluster / swizzle, coord() reproduces standard row-major."""
        M, N = 4, 3
        sched = self._sched(M, N, cluster_size=1, swizzle=1)
        # column_major=True (default) with swizzle=1 means M is fastest.
        # tid = m + n * M
        for tid in range(M * N):
            m = _fold_int(sched.coord(tid)[0])
            n = _fold_int(sched.coord(tid)[1])
            expected_m = tid % M
            expected_n = tid // M
            assert (m, n) == (expected_m, expected_n), f"tid={tid}: got ({m},{n}), expected ({expected_m},{expected_n})"


# ===========================================================================
# §3  Lowering assertions — before backend codegen, works on develop today
# ===========================================================================

# --- kernel factories: defined at module level so @T.prim_func can inspect
#     source (eager mode uses ``inspect.getsourcelines``).


@T.prim_func
def _cluster_rank_kernel(A: T.Tensor((128,), "int32")):
    """A cluster kernel exercising every basic cluster primitive."""
    with T.ClusterKernel(128, threads=128, cluster_dims=(2, 1, 1)) as (bx,):
        T.cluster_arrive()
        T.cluster_wait()
        rank = T.block_rank_in_cluster()
        if T.get_thread_binding() == 0:
            A[bx] = rank
        T.cluster_sync()


@T.prim_func
def _cluster_barrier_kernel(A: T.Tensor((128,), "int32")):
    """A cluster kernel that allocates a cluster mbarrier.

    Uses ``T.mbarrier_wait_parity`` — which HAS a TANG codegen handler
    (``tl::mbarrier_wait_parity``) — so the alloc survives lowering DCE.
    We deliberately avoid ``T.mbarrier_arrive`` here because on a
    ``shared.cluster_barrier`` buffer that lowers to
    ``tl.ptx_arrive_cluster_barrier``, whose TANG codegen is out of
    Phase 1 scope (it needs a cluster-aware ``.arrive(cta_id)`` handler
    that will be added in a follow-up).
    """
    with T.ClusterKernel(128, threads=128, cluster_dims=(4, 1, 1)) as (bx,):
        mbar = T.alloc_cluster_barrier([256])
        T.cluster_sync()
        T.mbarrier_wait_parity(mbar, 0)
        if T.block_rank_in_cluster() == 0:
            A[bx] = 0
        T.cluster_sync()


def _lower_ir(kernel):
    """Run passes up to (but not including) backend codegen."""
    context = create_backend_context(STCUV2)
    _, device_mod, _, _, _ = lower_to_host_device_ir(kernel, context)
    return device_mod


class TestClusterLowering:
    """Lowering-only assertions (target=stcuv2, but stopping before codegen).

    These pass on the current develop branch — they do NOT depend on the
    (still-missing) TANG codegen cluster handlers.
    """

    def test_cluster_dims_reaches_device_func(self):
        device_mod = _lower_ir(_cluster_rank_kernel)
        fn = _device_func(device_mod)
        assert "cluster_dims" in fn.attrs, f"cluster_dims not on device func attrs: {dict(fn.attrs).keys()}"
        dims = list(fn.attrs["cluster_dims"])
        assert [int(d) for d in dims] == [2, 1, 1], f"cluster_dims mismatch: {dims}"

    def test_cluster_ops_preserved_in_device_ir(self):
        """The lowered device body still contains the cluster op CallNodes.

        This is what codegen will consume — if the ops survive to here but
        codegen doesn't have a handler, the failure is squarely at codegen.
        """
        fn = _device_func(_lower_ir(_cluster_rank_kernel))
        for op_name in [
            "tl.cluster_arrive",
            "tl.cluster_wait",
            "tl.cluster_sync",
            "tl.block_rank_in_cluster",
        ]:
            calls = _walk_calls(fn.body, op_name)
            assert calls, f"{op_name} missing from lowered device IR"

    @pytest.mark.skip(reason="barrier support removed (handled elsewhere)")
    def test_alloc_cluster_barrier_scope(self):
        """T.alloc_cluster_barrier lowers to a shared allocation with
        the ``shared.cluster_barrier`` storage scope. This scope drives
        ``lower_shared_barrier`` to emit ``storage_sync('cluster')``."""
        fn = _device_func(_lower_ir(_cluster_barrier_kernel))
        scopes: list[str] = []

        def visitor(node):
            # scopes may appear either as an Allocate scope or via
            # tir.attr(..., "storage_scope", "shared.cluster_barrier") depending
            # on which pass ran last — collect from either shape.
            if isinstance(node, tirx.AttrStmt) and node.attr_key == "storage_scope":
                val = node.value
                if isinstance(val, (tirx.StringImm, str)):
                    scopes.append(str(val))
            elif hasattr(node, "buffer_var") and hasattr(node, "annotations"):
                ann = getattr(node, "annotations", None) or {}
                if "storage_scope" in ann:
                    scopes.append(str(ann["storage_scope"]))

        tirx.stmt_functor.post_order_visit(fn.body, visitor)
        # Also print the IR text as fallback signal — makes failures easy to
        # diagnose without re-running.
        body_txt = str(fn.body)
        assert any("shared.cluster_barrier" in s for s in scopes) or "shared.cluster_barrier" in body_txt, (
            f"shared.cluster_barrier scope not found in lowered IR; scopes seen: {scopes}\n---body---\n{body_txt[:1500]}"
        )


# ===========================================================================
# §4  Codegen dump — TANG codegen contract
# ===========================================================================
#
# These tests RUN the full ``tilelang.lower`` pipeline (through TANG codegen)
# and inspect the generated C++ source. They encode the actual codegen
# contract Phase 1 must satisfy.
#
# Before Phase 1 landed they XFAIL-ed with ``Unresolved call
# tl.cluster_arrive`` because ``src/tang/codegen/codegen_tang.cc`` had no
# ``VisitExpr_(CallNode*)`` handler for the cluster ops. The xfail
# decorators used strict=True so an XPASS forced their removal — that
# happened once we added the handlers, and this section is now all-green.


def _codegen_source(kernel) -> str:
    """Run the full lowering + codegen pipeline and return the TANG source."""
    artifact = tilelang.lower(kernel, target=STCUV2)
    assert artifact.kernel_source is not None, "kernel_source is None"
    return artifact.kernel_source


class TestClusterCodegenDump:
    """Assertions on the TANG-generated C++ source — the Phase 1 codegen
    contract, now met."""

    # ------------------------------------------------------------------
    # 4.1 Per-op call emission
    # ------------------------------------------------------------------
    def test_block_rank_in_cluster_emits_tl_call(self):
        """T.block_rank_in_cluster() → `tl::block_rank_in_cluster()`."""
        src = _codegen_source(_cluster_rank_kernel)
        assert "tl::block_rank_in_cluster()" in src, f"missing tl::block_rank_in_cluster() in generated source\n{src[:2000]}"

    def test_cluster_arrive_emits_tl_call(self):
        """T.cluster_arrive() → `tl::cluster_arrive();` statement."""
        src = _codegen_source(_cluster_rank_kernel)
        assert re.search(r"tl::cluster_arrive\s*\(\)\s*;", src), f"missing tl::cluster_arrive();\n{src[:2000]}"

    def test_cluster_wait_emits_tl_call(self):
        """T.cluster_wait() → `tl::cluster_wait();` statement."""
        src = _codegen_source(_cluster_rank_kernel)
        assert re.search(r"tl::cluster_wait\s*\(\)\s*;", src), f"missing tl::cluster_wait();\n{src[:2000]}"

    def test_cluster_sync_emits_tl_call(self):
        """T.cluster_sync() → `tl::cluster_sync();` statement."""
        src = _codegen_source(_cluster_rank_kernel)
        assert re.search(r"tl::cluster_sync\s*\(\)\s*;", src), f"missing tl::cluster_sync();\n{src[:2000]}"

    def test_cluster_arrive_relaxed_emits_tl_call(self):
        """T.cluster_arrive_relaxed() → `tl::cluster_arrive_relaxed();`."""

        @T.prim_func
        def kern(A: T.Tensor((32,), "int32")):
            with T.ClusterKernel(32, threads=128, cluster_dims=(2, 1, 1)) as (bx,):
                T.cluster_arrive_relaxed()
                T.cluster_wait()
                A[bx] = 0

        src = _codegen_source(kern)
        assert re.search(r"tl::cluster_arrive_relaxed\s*\(\)\s*;", src), f"missing tl::cluster_arrive_relaxed();\n{src[:2000]}"

    # ------------------------------------------------------------------
    # 4.2 cluster.h header wiring
    # ------------------------------------------------------------------
    def test_cluster_h_header_included(self):
        """When any cluster op appears, the generated source must
        `#include <tl_templates/tang/cluster.h>`."""
        src = _codegen_source(_cluster_rank_kernel)
        assert re.search(r"#include\s*<tl_templates/tang/cluster\.h>", src), f"cluster.h not included\n{src[:2000]}"

    # NOTE: no xfail — negative regression guard, passes today AND after
    # Phase 1. Locks that the include stays gated on need_cluster_h_.
    def test_cluster_h_not_included_when_unused(self):
        """A kernel without cluster ops must NOT include cluster.h — the
        include is gated on need_cluster_h_ (== `false` by default)."""

        @T.prim_func
        def plain(A: T.Tensor((32,), "int32")):
            with T.Kernel(32, threads=128) as (bx,):
                if T.get_thread_binding() == 0:
                    A[bx] = bx

        src = _codegen_source(plain)
        assert "tl_templates/tang/cluster.h" not in src, f"cluster.h leaked into a non-cluster kernel\n{src[:1500]}"

    # ------------------------------------------------------------------
    # 4.3 shared.cluster_barrier scope
    # ------------------------------------------------------------------
    @pytest.mark.skip(reason="barrier support removed (handled elsewhere)")
    def test_cluster_barrier_uses_shared_scope(self):
        """T.alloc_cluster_barrier lowers to an mbarrier array that codegen
        must emit into ``__shared__`` storage (same class as regular shared
        barrier). This exercises PrintStorageScope's ``shared.cluster_barrier``
        branch AND the ``VisitStmt_(AllocBufferNode*)`` scope dispatch.

        Layout: ``__shared__ uint64_t <name>_mem[N]; auto <name> =
        reinterpret_cast<Barrier*>(<name>_mem);`` — so we assert the
        ``__shared__`` storage + a following ``Barrier*`` reinterpret, which
        together are the codegen contract for the cluster barrier alloc.
        """
        src = _codegen_source(_cluster_barrier_kernel)
        assert re.search(r"__shared__[^\n]*_mem\s*\[", src), (
            f"no __shared__ mbarrier backing storage; shared.cluster_barrier scope handling missing?\n---source---\n{src[:2500]}"
        )
        assert re.search(r"reinterpret_cast<\s*Barrier\s*\*\s*>", src), (
            "no Barrier* reinterpret for the cluster mbarrier; "
            "AllocBufferNode dispatch for shared.cluster_barrier not matched\n"
            f"---source---\n{src[:2500]}"
        )

    # ------------------------------------------------------------------
    # 4.4 sanity — no unresolved cluster calls in output
    # ------------------------------------------------------------------
    def test_no_unresolved_cluster_call_in_source(self):
        """Sanity — once codegen handles cluster ops, the generated source
        should not contain the raw Op names (``tl.cluster_arrive``) that
        would indicate a fallback path was taken."""
        src = _codegen_source(_cluster_rank_kernel)
        for raw in (
            "tl.cluster_arrive",
            "tl.cluster_wait",
            "tl.cluster_sync",
            "tl.block_rank_in_cluster",
        ):
            assert raw not in src, (
                f"raw op name {raw!r} leaked into generated source — codegen fell through to a default handler?\n{src[:2000]}"
            )

    # ------------------------------------------------------------------
    # 4.5 cluster_dims attribute emission (Phase 3 launch-config hook)
    # ------------------------------------------------------------------
    def test_cluster_dims_attribute_emitted(self):
        """T.ClusterKernel(cluster_dims=(2,1,1)) must emit
        ``__cluster_dims__(2, 1, 1)`` in the kernel signature, so ptcc records
        the ``tang-cluster-dims`` metadata that the ISS cluster launch path
        consumes (kernel_cfg.in: is_explicit_cluster, cluster_nctaid.*).

        This is the first half of the Phase 3 launch-config chain — the
        second half (loader emitting the cluster keys from that metadata)
        lives in the external STCU_loader.py and is tracked separately.
        """
        src = _codegen_source(_cluster_rank_kernel)
        assert re.search(r"__cluster_dims__\(\s*2\s*,\s*1\s*,\s*1\s*\)", src), (
            f"missing __cluster_dims__(2, 1, 1) in kernel signature — cluster_dims attr not emitted by codegen\n---source---\n{src[:2000]}"
        )

    # ------------------------------------------------------------------
    # 4.6 cluster-aware threadblock swizzle (WithCluster)
    # ------------------------------------------------------------------
    @pytest.mark.parametrize(
        ("order", "expected"),
        [
            (
                "row",
                "const dim3 blockIdx = tl::rasterization2DRowWithCluster<2, 2>();",
            ),
            (
                "column",
                "const dim3 blockIdx = tl::rasterization2DColumnWithCluster<2, 2>();",
            ),
        ],
    )
    def test_cluster_swizzle_emits_withcluster(self, order, expected):
        """``T.use_swizzle`` inside a ``ClusterKernel`` must emit the
        cluster-aware ``...WithCluster<panel_size/cluster_dim_x,
        cluster_dim_x>()`` variant (mirrors the CUDA codegen cluster_dims
        branch in ``VisitStmt_(AttrStmtNode*)``). With ``cluster_dims=(2,1,1)``
        and ``panel_size=4`` this is ``WithCluster<2, 2>``."""

        @T.prim_func
        def kern(A: T.Tensor((64,), "int32")):
            with T.ClusterKernel(64, threads=1, cluster_dims=(2, 1, 1)):
                T.use_swizzle(panel_size=4, order=order)
                A[0] = 0

        src = _codegen_source(kern)
        assert expected in src, f"missing cluster-aware swizzle {expected!r}\n---source---\n{src[:2000]}"


# ===========================================================================
# §5a  Ptcc compile-only — kernel source actually accepted by ptcc
# ===========================================================================
#
# Codegen tests (§4) prove we emit the right *string*. The next question
# is whether ptcc actually accepts that string. We can answer it on any
# machine that has ptcc — no ISS / no ptpu needed.
#
# Two paths to a working ptcc are supported (same order the adapter uses):
#   1. env TANG_S3_PTCC_PATH points at a ptcc binary, OR
#   2. a system-installed ptcc under /usr/local/tangrt/bin.
#
# Real numeric tests (§5b) are separate and skipped when the full ISS
# toolchain is not available.

from pathlib import Path  # noqa: E402
from tilelang.jit.adapter import simulator as _sim  # noqa: E402


def _ptcc_available() -> bool:
    """Just ptcc — a subset of _simulator_available (which also needs iss + loader)."""
    try:
        return _sim._ptcc_path().exists()
    except Exception:
        return False


_REQUIRES_PTCC = pytest.mark.skipif(
    not _ptcc_available(),
    reason="ptcc not found — set TANG_S3_PTCC_PATH to a working ptcc binary (e.g. LLVM/build/bin/ptcc for cluster support)",
)


def _ptcc_compile(source: str, tmp_path: Path) -> tuple[bool, str, Path]:
    """Compile ``source`` as a TANG kernel using the adapter's own ptcc
    routine. Returns (ok, diagnostics, o_path)."""
    src_path = tmp_path / "kernel.t"
    src_path.write_text(source)
    try:
        o_path = _sim._compile_with_ptcc(src_path, tmp_path, verbose=False)
        return True, "", o_path
    except RuntimeError as e:
        return False, str(e), tmp_path / "source.o"


@T.prim_func
def _cluster_sync_only_kernel(A: T.Tensor((128,), "int32")):
    """A cluster kernel using ONLY the two zero-risk cluster ops that have
    landed TANG-native asm (cluster_sync → sync.all.bkcls,
    block_rank_in_cluster → __stvm_read_sreg_cluster_ctarank builtin).

    Deliberately avoids T.cluster_arrive / T.cluster_wait — those still
    carry the CUDA-form asm placeholder and will trip ptcc until the
    hardware team confirms the sync-reg operand convention (Phase 3).
    """
    with T.ClusterKernel(128, threads=128, cluster_dims=(4, 1, 1)) as (bx,):
        T.cluster_sync()
        if T.get_thread_binding() == 0:
            A[bx] = T.block_rank_in_cluster()
        T.cluster_sync()


class TestClusterPtccCompile:
    """§5a — the generated .t source is accepted by ptcc.

    Splits the ISS numeric test into two: ptcc compile (this class, runs
    wherever ptcc is available) and full ISS numeric (§5b, needs iss +
    loader + ptpu, deferred).
    """

    @_REQUIRES_PTCC
    def test_cluster_kernel_compiles_with_ptcc(self, tmp_path):
        """The generated source for a cluster kernel using cluster_sync +
        block_rank_in_cluster compiles cleanly under ptcc — this is the
        first end-to-end check that our TANG-native cluster.h asm is right.
        """
        src = _codegen_source(_cluster_sync_only_kernel)
        ok, diag, _ = _ptcc_compile(src, tmp_path)
        assert ok, (
            f"ptcc rejected cluster kernel source. cluster.h asm may be broken.\n---diag---\n{diag[:2500]}\n---source---\n{src[:2500]}"
        )

    @_REQUIRES_PTCC
    def test_baseline_kernel_still_compiles(self, tmp_path):
        """Baseline — a non-cluster kernel must keep compiling. Guards
        against cluster.h leaking into or breaking normal kernels.
        """

        @T.prim_func
        def plain(A: T.Tensor((32,), "int32")):
            with T.Kernel(32, threads=128) as (bx,):
                if T.get_thread_binding() == 0:
                    A[bx] = bx

        src = _codegen_source(plain)
        ok, diag, _ = _ptcc_compile(src, tmp_path)
        assert ok, f"ptcc rejected plain kernel:\n{diag[:2000]}\n---source---\n{src[:1500]}"

    # cluster_arrive + cluster_wait now emit real TANG asm (aligned to
    # LLVM origin/dev_12599: warpsync + fence.mem + sync.arrive.bkcls 0 for
    # arrive; sync.wait.bkcls 0, n, n + fence.mem for wait), so this kernel
    # compiles today. Was xfail-ed pending the sync-reg convention, which
    # dev_12599 resolved (barrier id 0, nwarps = cluster_nctarank * warps/blk).
    @_REQUIRES_PTCC
    def test_split_arrive_wait_compiles(self, tmp_path):
        src = _codegen_source(_cluster_rank_kernel)  # uses arrive+wait
        ok, diag, _ = _ptcc_compile(src, tmp_path)
        assert ok, diag[:2000]


# ===========================================================================
# §5b  ISS numeric — cluster kernel through the full simulator pipeline
# ===========================================================================
#
# The full chain (codegen → ptcc → STCU_loader → stcu_rm) is wired end to
# end and verified 2026-08-26:
#   - codegen emits __cluster_dims__(2,1,1)
#   - ptcc records "tang-cluster-dims"="2,1,1"
#   - new loader (pengliren Aug 13) writes kernel_cfg.in cluster keys
#       (is_explicit_cluster=1, cluster_nctaid.x=2, cluster_nctarank=2, ...)
#   - ISS runs both CTAs; block_rank_in_cluster returns [0, 1]
#
# NOTE: this test was previously xfail(strict=False) under the
# "ISS scalar-register-as-value" theory. That turned out to be a test
# artifact, not an ISS bug: the kernel had an unused input param (A) that
# got DCE'd, leaving user_config.json (2 args) out of sync with
# kernel_info.args_num (1 arg), so the loader mapped the output to the wrong
# address and the dump read garbage. A single-buffer kernel (no dead input)
# returns the correct ranks — see docs/s3_cluster_debug_report.md.


def _simulator_available() -> bool:
    try:
        return _sim._ptcc_path().exists() and _sim._iss_path().exists() and _sim._loader_path().exists()
    except Exception:
        return False


def _ptpu_available() -> bool:
    try:
        import torch
        import torch_ptpu  # noqa: F401

        torch.zeros(1).ptpu(4)
        return True
    except Exception:
        return False


_REQUIRES_ISS = pytest.mark.skipif(
    not (_simulator_available() and _ptpu_available()),
    reason="ISS/loader/ptpu unavailable (set TANG_S3_{PTCC,ISS,LOADER}_PATH)",
)


def _sim_jit(kernel, out_idx):
    """JITKernel on the simulator backend, patching the post-upgrade
    Target.canon_target ABI drift that simulator.py still depends on."""
    from tvm.target import Target as _TvmTarget

    if not hasattr(_TvmTarget, "canon_target"):
        _TvmTarget.canon_target = staticmethod(lambda t: t if isinstance(t, _TvmTarget) else _TvmTarget(t))
    from tilelang.jit import JITKernel

    return JITKernel(kernel, out_idx=out_idx, target={"kind": "tang", "arch": "stcuv2"}, execution_backend="simulator", verbose=False)


@T.prim_func
def _cluster_rank_numeric_kernel(B: T.Tensor((64,), "int32")):
    with T.ClusterKernel(2, threads=32, cluster_dims=(2, 1, 1)) as (bx,):
        T.cluster_sync()
        for i in T.serial(32):
            B[bx * 32 + i] = T.block_rank_in_cluster()
        T.cluster_sync()


class TestClusterNumericISS:
    """§5b — full simulator numeric test (codegen → ptcc → loader → stcu_rm).

    ``block_rank_in_cluster`` reads the ``cluster_ctarank`` special register
    and stores it as a value; the two CTAs must see ranks ``[0, 1]``.
    """

    @_REQUIRES_ISS
    def test_block_rank_in_cluster_numeric(self):
        kernel = _sim_jit(_cluster_rank_numeric_kernel, out_idx=[0])
        B = kernel().cpu()
        # each 32-element slice is one CTA's rank
        ranks = [int(B[b * 32]) for b in range(2)]
        assert ranks == [0, 1], f"got {ranks}, expected [0, 1]"


# ===========================================================================
# §5c  copy_cluster (DSM / TMA multicast)
# ===========================================================================
class TestClusterCopyUnsupported:
    """``copy_cluster(dst_block=...)`` lowers to an element-wise DSM store
    (``ptx_cluster_store`` → ``tl::cluster_store`` → ``mapa.shared.cluster`` +
    ``st.shared``) when the LLVM dev_dshm branch is present. TMA multicast
    (``cluster_mask=...``) still needs ``cp.async.bulk.shared::cluster`` and
    must fail loudly."""

    def test_copy_cluster_dst_block_lowers(self):
        @T.prim_func
        def kern(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
            with T.ClusterKernel(2, threads=32, cluster_dims=(2, 1, 1)):
                s_src = T.alloc_shared((64,), "float32")
                s_dst = T.alloc_shared((64,), "float32")
                T.copy_cluster(s_src, s_dst, dst_block=1)

        with STCUV2:
            artifact = tilelang.lower(kern, target=STCUV2)
        src = artifact.kernel_source
        assert "tl::cluster_store(" in src, f"copy_cluster(dst_block=...) did not lower to tl::cluster_store\n---source---\n{src[:2000]}"

    def test_copy_cluster_mask_rejected(self):
        @T.prim_func
        def kern(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
            with T.ClusterKernel(2, threads=32, cluster_dims=(2, 1, 1)):
                s_src = T.alloc_shared((64,), "float32")
                s_dst = T.alloc_shared((64,), "float32")
                T.copy_cluster(s_src, s_dst, cluster_mask=3)

        with STCUV2, pytest.raises(tvm.error.InternalError, match="does not support"):
            tilelang.lower(kern, target=STCUV2)


# ===========================================================================
# §5c-numeric  copy_cluster(dst_block) DSM 数值验证（full simulator）
# ===========================================================================
class TestClusterCopyNumericISS:
    """``copy_cluster(dst_block=1, remote_barrier=bar)`` 的 DSM 数据搬运端到端
    数值验证：rank 0 CTA 把 A 逐元素 DSM store 到 rank 1 CTA 的 s_dst，通过
    remote_barrier 通知，rank 1 wait 后读 s_dst 写 B。期望 B == A。

    依赖 LLVM dev_dshm 分支（mvlut / st.shared w1 / mbarrier.arrive.shared w1）
    和 TANG LowerSharedBarrier pass（barrier 初始化）。"""

    @pytest.mark.skip(reason="barrier support removed (handled elsewhere)")
    @_REQUIRES_ISS
    def test_copy_cluster_dst_block_numeric(self):
        import torch

        @T.prim_func
        def kern(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
            with T.ClusterKernel(2, threads=32, cluster_dims=(2, 1, 1)) as (bx,):
                s_src = T.alloc_shared((64,), "float32")
                s_dst = T.alloc_shared((64,), "float32")
                bar = T.alloc_cluster_barrier(1)
                if T.block_rank_in_cluster() == 0:
                    for i in T.parallel(64):
                        s_src[i] = A[i]
                    T.copy_cluster(s_src, s_dst, dst_block=1, remote_barrier=bar[0])
                if T.block_rank_in_cluster() == 1:
                    T.mbarrier_wait_parity(bar[0], 0)
                    for i in T.parallel(64):
                        B[i] = s_dst[i]

        kernel = _sim_jit(kern, out_idx=[1])
        A = torch.arange(64, dtype=torch.float32).ptpu(4)
        B = kernel(A).cpu()
        A_cpu = A.cpu()
        assert torch.equal(B, A_cpu), f"copy_cluster(dst_block) DSM mismatch\n---got---\n{B}\n---expected---\n{A_cpu}"

    @pytest.mark.skip(reason="barrier support removed (handled elsewhere)")
    @_REQUIRES_ISS
    def test_copy_cluster_with_cluster_sync(self):
        """3-CTA cluster：remote_barrier（点对点数据信号）+ cluster_sync（全局
        栅栏）搭配。rank0 → rank1 搬数据，rank2 第一阶段空转、第二阶段由
        cluster_sync 拉齐，三个 CTA 各写自己的 rank。"""
        import torch

        @T.prim_func
        def kern(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32"), C: T.Tensor((3,), "int32")):
            with T.ClusterKernel(3, threads=32, cluster_dims=(3, 1, 1)) as (bx,):
                s_src = T.alloc_shared((64,), "float32")
                s_dst = T.alloc_shared((64,), "float32")
                bar = T.alloc_cluster_barrier(1)
                rank = T.block_rank_in_cluster()

                if rank == 0:
                    for i in T.parallel(64):
                        s_src[i] = A[i]
                    T.copy_cluster(s_src, s_dst, dst_block=1, remote_barrier=bar[0])

                if rank == 1:
                    T.mbarrier_wait_parity(bar[0], 0)
                    for i in T.parallel(64):
                        B[i] = s_dst[i]

                T.cluster_sync()
                C[bx] = rank

        kernel = _sim_jit(kern, out_idx=[1, 2])
        A = torch.arange(64, dtype=torch.float32).ptpu(4)
        B, C = kernel(A)
        B_cpu, C_cpu = B.cpu(), C.cpu()
        assert torch.equal(B_cpu, A.cpu()), f"copy_cluster DSM mismatch:\n{B_cpu}\nvs\n{A.cpu()}"
        assert C_cpu.tolist() == [0, 1, 2], f"cluster_sync rank mismatch: {C_cpu.tolist()}, expected [0, 1, 2]"
