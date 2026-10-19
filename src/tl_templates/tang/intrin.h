#pragma once

#include "common.h"

#include <type_traits>

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

// ---------------------------------------------------------------------------
// __match_any_sync for TANG/PTCC.
//
// The toolchain has used two names for this builtin. Older revisions declare
// `match_any_sync`; newer ones declare `__match_any_sync`, the CUDA spelling
// that codegen already emits. Both are plain function declarations, so the
// `#ifndef __match_any_sync` this bridge used to be guarded by could never
// detect anything: the name is not a macro, so the guard was always true.
//
// Probe for the declaration instead, and bridge only when it is missing. The
// bridge is a template on purpose. A template cannot redeclare the toolchain's
// non-template overload -- that is the "static declaration of
// '__match_any_sync' follows non-static declaration" error -- and once SFINAE
// removes it the call resolves straight to the toolchain's own function.
//
// Must stay at global scope, outside namespace tl, so generated code finds it.
// ---------------------------------------------------------------------------
namespace tl {
namespace detail {

// True when the toolchain declares `__match_any_sync` itself. The probe has to
// sit above the bridge, so that it only ever sees the toolchain's declaration.
// TVal{} keeps the expression dependent, which is what makes a missing
// declaration a substitution failure rather than a hard error.
template <class TVal, class = void>
struct has_vendor_match_any_sync : std::false_type {};

template <class TVal>
struct has_vendor_match_any_sync<
    TVal, std::void_t<decltype(__match_any_sync(0u, TVal{}))>>
    : std::true_type {};

} // namespace detail
} // namespace tl

// The condition has to sit in the return type AND depend on TVal. A default
// template argument is not part of the immediate context, and a condition that
// does not depend on TVal is evaluated when the template is declared rather
// than during substitution -- GCC reports either one as a hard error instead of
// a substitution failure.
template <class TVal>
static inline __device__
    std::enable_if_t<!tl::detail::has_vendor_match_any_sync<TVal>::value,
                     unsigned int>
    __match_any_sync(unsigned int __mask, TVal __val) {
  return match_any_sync(__mask, __val);
}
