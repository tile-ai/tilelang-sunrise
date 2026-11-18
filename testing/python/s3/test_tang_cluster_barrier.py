"""S3/TANG cluster barrier support — codegen tests.

The cluster barrier (``T.alloc_cluster_barrier`` + ``T.copy_cluster(...,
remote_barrier=...)``) synchronises *data arrival* across CTAs, unlike the
``T.cluster_sync`` / ``T.cluster_arrive`` / ``T.cluster_wait`` family which
synchronise *thread arrival* through the fixed hardware barrier id 0.

Layering:

  §1  Codegen dump  TestClusterBarrierCodegenDump
       Runs the full ``tilelang.lower`` (through TANG codegen) and asserts the
       generated C++:
         - ``alloc_cluster_barrier`` lowers to a ``Barrier*`` array whose init
           is published with ``tl::cluster_sync()`` (cluster-scoped storage
           sync, not ``__syncthreads``);
         - ``copy_cluster(dst_block=..., remote_barrier=...)`` lowers to a
           ``tl::cluster_arrive_barrier(&bar[i], rank)`` remote mbarrier arrive.

Run:
    python -m pytest testing/python/s3/test_tang_cluster_barrier.py -v
"""

from __future__ import annotations

import os

# TANG_ARCH must match the stcuv2 target these cases lower for.
os.environ.setdefault("TANG_ARCH", "stcuv2")

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm

STCUV2 = tvm.target.Target({"kind": "tang", "arch": "stcuv2"})


def _codegen_source(kernel) -> str:
    artifact = tilelang.lower(kernel, target=STCUV2)
    assert artifact.kernel_source is not None, "kernel_source is None"
    return artifact.kernel_source


@T.prim_func
def _cluster_barrier_copy_kernel(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
    """2-CTA cluster: rank 0 copies A into its shared buffer and pushes it into
    rank 1's shared buffer, signalling completion with a remote mbarrier arrive;
    rank 1 waits on the barrier parity and writes the result out."""
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


class TestClusterBarrierCodegenDump:
    """Assertions on the TANG-generated C++ source."""

    def test_alloc_cluster_barrier_publishes_init_with_cluster_sync(self):
        """alloc_cluster_barrier lowers to a Barrier* array whose init is
        published cluster-wide (tl::cluster_sync), not with __syncthreads."""
        src = _codegen_source(_cluster_barrier_copy_kernel)
        # Cluster header gate: cluster_sync + cluster_arrive_barrier both live
        # in cluster.h.
        assert "#include <tl_templates/tang/cluster.h>" in src
        # Barrier array + init prologue.
        assert "reinterpret_cast<Barrier*>" in src
        assert ".init(1);" in src
        # The init publish is cluster-scoped: LowerSharedBarrier emits
        # storage_sync("cluster") -> tl::cluster_sync() for cluster barriers.
        assert "tl::cluster_sync();" in src
        # The wait side is a block-scoped mbarrier parity wait.
        assert ".wait(0);" in src

    def test_copy_cluster_remote_barrier_lowers_remote_arrive(self):
        """copy_cluster(remote_barrier=...) appends a single-thread remote
        mbarrier arrive (tl::cluster_arrive_barrier) after the DSM stores."""
        src = _codegen_source(_cluster_barrier_copy_kernel)
        assert "tl::cluster_arrive_barrier(" in src
        # The arrive is guarded to a single elected thread and targets rank 1.
        assert "tl::cluster_arrive_barrier(&(bar[0]), (int)(1));" in src

    def test_cluster_barrier_scope_not_rejected_by_pipeline(self):
        """The pipeline no longer raises for shared.cluster_barrier; lowering
        completes and produces source (this is the regression guard for the
        removed ValueError)."""
        src = _codegen_source(_cluster_barrier_copy_kernel)
        assert src.strip(), "cluster barrier kernel should lower to source"
