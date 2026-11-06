"""S3/TANG persistent + cluster support — tests for §1-§5.

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
       target codegen. Asserts:
         - cluster_dims attr is on the device PrimFunc
         - the device IR still contains the cluster ops as CallNodes
         - alloc_cluster_barrier lowers to a shared allocation with
           the ``shared.cluster_barrier`` storage scope

  §4  Codegen dump          TestClusterCodegenDump
       Runs the full ``tilelang.lower`` (through TANG codegen) and greps
       the generated C++ source. Checks cluster ops
       (``tl::block_rank_in_cluster``, ``tl::cluster_*``), the
       ``cluster.h`` include gate and launch attributes.

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
            "alloc_cluster_barrier",
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
    We avoid ``T.mbarrier_arrive`` because on a
    ``shared.cluster_barrier`` buffer it lowers to
    ``tl.ptx_arrive_cluster_barrier``, which has no TANG codegen handler.
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
    """Lowering-only assertions (target=stcuv2, stopping before codegen)."""

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
# These tests run the full ``tilelang.lower`` pipeline and inspect the
# generated C++ source.


def _codegen_source(kernel) -> str:
    """Run the full lowering + codegen pipeline and return the TANG source."""
    artifact = tilelang.lower(kernel, target=STCUV2)
    assert artifact.kernel_source is not None, "kernel_source is None"
    return artifact.kernel_source


class TestClusterCodegenDump:
    """Assertions on the TANG-generated C++ source."""

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
    # 4.5 cluster_dims attribute emission
    # ------------------------------------------------------------------
    def test_cluster_dims_attribute_emitted(self):
        """T.ClusterKernel(cluster_dims=(2,1,1)) must emit
        ``__cluster_dims__(2, 1, 1)`` in the kernel signature, so ptcc records
        the ``tang-cluster-dims`` metadata that the ISS cluster launch path
        consumes (kernel_cfg.in: is_explicit_cluster, cluster_nctaid.*).
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
# §5  copy_cluster (DSM / TMA multicast)
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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"] + sys.argv[1:]))
