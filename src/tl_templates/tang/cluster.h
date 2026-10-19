// Cluster primitives for TANG (STCU / STCUV2)
//
// The cluster barrier sync now maps directly to the LLVM dev_12599 builtins
// that have landed on develop:
//   - __cluster_barrier_arrive() / __cluster_barrier_arrive_relaxed() /
//     __cluster_barrier_wait()    (CUDA-style wrappers, auto-included via
//     __clang_tang_builtin_vars.h)
//   - __stvm_sync_all_cluster(id) (fused arrive+wait, for cluster_sync)
//
// Memory-order contract (dev_12599 / CUDA semantics):
//   cluster_arrive()          release:   warpsync -> fence.mem ->
//   sync.arrive.bkcls 0 cluster_arrive_relaxed()  relaxed:   warpsync ->
//   sync.arrive.bkcls 0  (no fence) cluster_wait()            acquire:
//   sync.wait.bkcls 0, n, n -> fence.mem cluster_sync()            fused:
//   sync.all.bkcls 0
//
// Cluster barrier id is fixed to 0 (__stvm_cluster_barrier_id).

#pragma once

#include "common.h"

namespace tl {

TL_DEVICE void cluster_arrive_relaxed() { __cluster_barrier_arrive_relaxed(); }

TL_DEVICE void cluster_arrive() { __cluster_barrier_arrive(); }

TL_DEVICE void cluster_wait() { __cluster_barrier_wait(); }

TL_DEVICE void cluster_sync() {
  __stvm_sync_all_cluster(__stvm_cluster_barrier_id);
}

TL_DEVICE int block_rank_in_cluster() {
  // Preferred form — Clang exposes __stvm_read_sreg_cluster_ctarank() as
  // a builtin (see LLVM STCUV2Intrinsics.td:36 and
  // clang/lib/Headers/cccl/tang/__ptx/instructions/generated/get_sreg.h).
  return static_cast<int>(__stvm_read_sreg_cluster_ctarank());
}

// Distributed shared memory (DSM) store: write `value` into the shared memory
// of CTA `rank` within this cluster. Equivalent to CUDA's
// cooperative_groups::cluster_group::map_shared_rank + a store.
//
// Requires the LLVM dev_dshm branch: __cluster_map_shared_rank returns an AS7
// (shared::cluster) pointer and the assignment lowers to st.shared with a peer
// handle (S3, `..., rs3`) or st.shared.dshm (S5). The b8/b16 scalar widths are
// an ISA gap on DSHM; use 32/64-bit element types (or f16x2/bf16x2 atoms).
template <typename T> TL_DEVICE void cluster_store(T *smem, int rank, T value) {
  auto *remote = __cluster_map_shared_rank(smem, static_cast<unsigned>(rank));
  *remote = value;
}

} // namespace tl
