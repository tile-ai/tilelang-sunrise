#pragma once

#include "common.h"

namespace tl {

namespace detail {

// Provide architecture-specific defaults so callers may omit arguments.
TL_DEVICE constexpr int default_warp_size() { return 32; }

TL_DEVICE constexpr int default_warps_per_group() { return 32; }

TL_DEVICE int linear_thread_idx_in_block() {
  return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
}

} // namespace detail

TL_DEVICE int get_lane_idx(int warp_size = detail::default_warp_size()) {
  warp_size = warp_size > 0 ? warp_size : detail::default_warp_size();
  return detail::linear_thread_idx_in_block() % warp_size;
}

TL_DEVICE int get_warp_idx_sync(int warp_size = detail::default_warp_size()) {
  warp_size = warp_size > 0 ? warp_size : detail::default_warp_size();
  return detail::linear_thread_idx_in_block() / warp_size;
}

TL_DEVICE int get_warp_idx(int warp_size = detail::default_warp_size()) {
  warp_size = warp_size > 0 ? warp_size : detail::default_warp_size();
  return detail::linear_thread_idx_in_block() / warp_size;
}

TL_DEVICE int
get_warp_group_idx(int warp_size = detail::default_warp_size(),
                   int warps_per_group = detail::default_warps_per_group()) {
  // On S2, there is no warp group. Here we return 0 for language support only.
  return 0;
}

// Elect exactly one representative thread within each group of `thread_extent`
// threads; thread_extent == 0 is the special case "one thread in the whole
// block". Used by LowerSharedBarrier to pick the thread that initialises the
// shared mbarriers.
//
// CUDA implements the intra-warp election with elect.sync, which picks one of
// the *active* lanes. Here lane 0 is chosen unconditionally instead, so the two
// agree only where lane 0 is active. That holds for every current caller (the
// election sits in uniform control flow), and TANG has no equivalent of the
// register-allocation benefit that motivates elect.sync on CUDA.
template <int thread_extent> TL_DEVICE bool tl_shuffle_elect() {
  if constexpr (thread_extent == 0) {
    return get_lane_idx() == 0 && get_warp_idx() == 0;
  } else if constexpr (thread_extent == detail::default_warp_size()) {
    return get_lane_idx() == 0;
  } else {
    constexpr int warp_extent =
        (thread_extent + detail::default_warp_size() - 1) /
        detail::default_warp_size();
    static_assert(warp_extent > 0);
    return get_lane_idx() == 0 && (get_warp_idx() % warp_extent) == 0;
  }
}

} // namespace tl

// __match_any_sync for TANG/PTCC: delegate to clang's builtin match_any_sync.
// Must be at global scope (outside tl namespace) so generated code can find it.
#ifndef __match_any_sync
static inline __device__ unsigned int __match_any_sync(unsigned int __mask,
                                                       int __val) {
  return match_any_sync(__mask, __val);
}
#endif
