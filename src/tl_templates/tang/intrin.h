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

} // namespace tl

// __match_any_sync for TANG/PTCC: delegate to clang's builtin match_any_sync.
// Must be at global scope (outside tl namespace) so generated code can find it.
#ifndef __match_any_sync
static inline __device__ unsigned int __match_any_sync(unsigned int __mask,
                                                       int __val) {
  return match_any_sync(__mask, __val);
}
#endif
