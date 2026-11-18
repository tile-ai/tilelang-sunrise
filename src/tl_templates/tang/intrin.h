#pragma once

#include "common.h"

#include <type_traits>

namespace tl {

// Pair/quad views used by this header. They are spelled here instead of reused
// from the toolchain headers because those typedef names are not stable across
// PTCC releases (the toolchain's own bf16 pair/quad typedefs are missing from
// some header vintages even though `__bf16` and `ext_vector_type` work), and
// every kernel TU includes this file, so an unknown name breaks kernels that
// use neither atomics nor pair math. The overloads these replace
// (`::atomicAdd`, the builtins) resolve on the underlying vector type, so
// aliases match.
using f16x2_vec = __fp16 __attribute__((ext_vector_type(2)));
using bf16x2_vec = __bf16 __attribute__((ext_vector_type(2)));
using bf16x4_vec = __bf16 __attribute__((ext_vector_type(4)));
using f32x2_vec = float __attribute__((ext_vector_type(2)));
using f32x4_vec = float __attribute__((ext_vector_type(4)));

namespace detail {

template <typename To, typename From> TL_DEVICE To bit_cast(const From &value) {
  static_assert(sizeof(To) == sizeof(From), "bit_cast requires equal sizes");
  return *reinterpret_cast<const To *>(&value);
}

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

#if defined(__Tang_BUILTIN_VARS_H) && defined(__Tang__) &&                     \
    (!defined(__Tang_ARCH__) || (__Tang_ARCH__ >= 200))
namespace detail {

// STCUV2 has no dedicated atomic load/store instruction. This is the fallback
// contract used by tang::atomic_ref: naturally aligned volatile accesses
// provide the indivisible transfer, while fence.mem supplies the requested
// ordering.
TL_DEVICE void atomic_memory_fence() {
  asm volatile("fence.mem;" ::: "memory");
}

constexpr int kAtomicConsume = 1;
constexpr int kAtomicAcquire = 2;
constexpr int kAtomicRelease = 3;
constexpr int kAtomicSeqCst = 5;

template <typename Carrier, typename Vector, typename Scalar>
TL_DEVICE Carrier vector_atomic_add(Scalar *dst, const Scalar *src) {
  Vector value = *reinterpret_cast<const Vector *>(src);
  Vector previous = ::atomicAdd(reinterpret_cast<Vector *>(dst), value);
  return bit_cast<Carrier>(previous);
}

} // namespace detail

template <typename T> TL_DEVICE T atomic_load(const T *src, int memory_order) {
  static_assert(sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8,
                "STCUV2 atomic_load supports 16/32/64-bit scalar values");

  // seq_cst load: fence -> volatile load -> fence
  if (memory_order == detail::kAtomicSeqCst) {
    detail::atomic_memory_fence();
  }
  T value = *reinterpret_cast<const volatile T *>(src);
  // consume is implemented with acquire strength, matching tang::atomic_ref.
  if (memory_order == detail::kAtomicConsume ||
      memory_order == detail::kAtomicAcquire ||
      memory_order == detail::kAtomicSeqCst) {
    detail::atomic_memory_fence();
  }
  return value;
}

template <typename T, typename U>
TL_DEVICE void atomic_store(T *dst, U value, int memory_order) {
  static_assert(sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8,
                "STCUV2 atomic_store supports 16/32/64-bit scalar values");

  // Both release and seq_cst store fence before publishing the value.
  if (memory_order == detail::kAtomicRelease ||
      memory_order == detail::kAtomicSeqCst) {
    detail::atomic_memory_fence();
  }
  *reinterpret_cast<volatile T *>(dst) = static_cast<T>(value);
}

// STCUV2 vector atomics require naturally aligned consecutive storage:
// f16/bf16 x2: 4 B, f16/bf16 x4: 8 B, f32 x2: 8 B, f32 x4: 16 B.
TL_DEVICE float2 atomic_addx2_f32(float *dst, const float *src) {
  return detail::vector_atomic_add<float2, f32x2_vec>(dst, src);
}

TL_DEVICE unsigned atomic_addx2_f16(__fp16 *dst, const __fp16 *src) {
  f16x2_vec previous = ::atomicAdd(reinterpret_cast<f16x2_vec *>(dst),
                                   *reinterpret_cast<const f16x2_vec *>(src));
  return detail::bit_cast<unsigned>(previous);
}

TL_DEVICE unsigned atomic_addx2_bf16(__bf16 *dst, const __bf16 *src) {
  bf16x2_vec previous = ::atomicAdd(reinterpret_cast<bf16x2_vec *>(dst),
                                    *reinterpret_cast<const bf16x2_vec *>(src));
  return detail::bit_cast<unsigned>(previous);
}

TL_DEVICE void atomic_addx2_ret_f16(__fp16 *dst, const __fp16 *src,
                                    uint1 *result) {
  f16x2_vec previous = ::atomicAdd(reinterpret_cast<f16x2_vec *>(dst),
                                   *reinterpret_cast<const f16x2_vec *>(src));
  *reinterpret_cast<f16x2_vec *>(result) = previous;
}

TL_DEVICE void atomic_addx2_ret_bf16(__bf16 *dst, const __bf16 *src,
                                     uint1 *result) {
  bf16x2_vec previous = ::atomicAdd(reinterpret_cast<bf16x2_vec *>(dst),
                                    *reinterpret_cast<const bf16x2_vec *>(src));
  *reinterpret_cast<bf16x2_vec *>(result) = previous;
}

TL_DEVICE float4 atomic_addx4_f32(float *dst, const float *src) {
  return detail::vector_atomic_add<float4, f32x4_vec>(dst, src);
}

TL_DEVICE unsigned long long atomic_addx4_f16(__fp16 *dst, const __fp16 *src) {
  half4_vec previous = ::atomicAdd(reinterpret_cast<half4_vec *>(dst),
                                   *reinterpret_cast<const half4_vec *>(src));
  return detail::bit_cast<unsigned long long>(previous);
}

TL_DEVICE unsigned long long atomic_addx4_bf16(__bf16 *dst, const __bf16 *src) {
  bf16x4_vec previous = ::atomicAdd(reinterpret_cast<bf16x4_vec *>(dst),
                                    *reinterpret_cast<const bf16x4_vec *>(src));
  return detail::bit_cast<unsigned long long>(previous);
}

TL_DEVICE void atomic_addx4_ret_f16(__fp16 *dst, const __fp16 *src,
                                    uint2 *result) {
  half4_vec previous = ::atomicAdd(reinterpret_cast<half4_vec *>(dst),
                                   *reinterpret_cast<const half4_vec *>(src));
  *reinterpret_cast<half4_vec *>(result) = previous;
}

TL_DEVICE void atomic_addx4_ret_bf16(__bf16 *dst, const __bf16 *src,
                                     uint2 *result) {
  bf16x4_vec previous = ::atomicAdd(reinterpret_cast<bf16x4_vec *>(dst),
                                    *reinterpret_cast<const bf16x4_vec *>(src));
  *reinterpret_cast<bf16x4_vec *>(result) = previous;
}
#endif

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

// ---------------------------------------------------------------------------
// __match_all_sync for TANG/PTCC.
//
// The same two-spelling history as __match_any_sync above, one signature
// further: the LLVM21 toolchain's `__clang_tang_builtin_vars.h` declares
// `match_all_sync(mask, val, pred)` with a predicate out-parameter, while newer
// ones declare the CUDA spelling `__match_all_sync` that codegen emits. Without
// this bridge the generated source fails to compile on the older toolchain with
// "use of undeclared identifier '__match_all_sync'".
//
// Probe for the prefixed name and bridge to the unprefixed one only when it is
// missing. A template, for the reason stated above: it cannot redeclare the
// toolchain's non-template overload, and once SFINAE removes it the call
// resolves straight to the toolchain's own function.
//
// Must stay at global scope, outside namespace tl, so generated code finds it.
// ---------------------------------------------------------------------------
namespace tl {
namespace detail {

// True when the toolchain declares `__match_all_sync` itself.
template <class TVal, class = void>
struct has_vendor_match_all_sync : std::false_type {};

template <class TVal>
struct has_vendor_match_all_sync<
    TVal, std::void_t<decltype(__match_all_sync(0u, TVal{}, (int *)nullptr))>>
    : std::true_type {};

} // namespace detail
} // namespace tl

// Condition in the return type and dependent on TVal, exactly as above.
template <class TVal>
static inline __device__
    std::enable_if_t<!tl::detail::has_vendor_match_all_sync<TVal>::value,
                     unsigned int>
    __match_all_sync(unsigned int __mask, TVal __val, int *__pred) {
  return match_all_sync(__mask, __val, __pred);
}

// ---------------------------------------------------------------------------
// Packed pair (x2) math: name the vector operation, let the backend pick the
// instruction.
//
// `T.add2` and friends are two-lane elementwise math on one 32-bit (f16/bf16)
// or 64-bit (f32) value. How many instructions that becomes is the backend's
// decision, so these helpers do not expand or fuse anything themselves: they
// only express the operation on the toolchain's own vector typedefs
// (`f16x2_vec`, `bf16x2_vec`, `f32x2_vec`) and move the value through the
// storage types codegen already declares (`uint1` for the 16-bit pairs,
// `float2` for f32x2). A per-lane `__h*2`-style expansion is exactly the shape
// the backend does not merge, which is why the vector form is the entry point.
//
// ---------------------------------------------------------------------------
namespace tl {

namespace detail {
// `max2_nan` / `min2_nan` must keep a NaN operand (CUDA's `__hmax2_nan` /
// `__hmin2_nan`). `llvm.maximum/minimum` says that in one vector op, but ptcc
// 2.9.4 does not terminate while selecting it for a 2-lane vector (the compile
// is killed by `timeout`; 2.10.0 compiles it fine), so the NaN test is spelled
// per lane here. Switch back to `__builtin_elementwise_maximum/minimum` once
// the toolchain baseline is >= 2.10.0.
template <bool kIsMax, typename Vec>
TL_DEVICE Vec nan_aware_pair(const Vec &a, const Vec &b) {
  Vec out;
  for (int i = 0; i < 2; ++i) {
    auto x = a[i];
    auto y = b[i];
    out[i] =
        (x != x)
            ? x
            : ((y != y) ? y : (kIsMax ? (x > y ? x : y) : (x < y ? x : y)));
  }
  return out;
}
} // namespace detail

TL_DEVICE uint1 packed_add2_f16(uint1 a, uint1 b) {
  return detail::bit_cast<uint1>(detail::bit_cast<f16x2_vec>(a) +
                                 detail::bit_cast<f16x2_vec>(b));
}

TL_DEVICE uint1 packed_sub2_f16(uint1 a, uint1 b) {
  return detail::bit_cast<uint1>(detail::bit_cast<f16x2_vec>(a) -
                                 detail::bit_cast<f16x2_vec>(b));
}

TL_DEVICE uint1 packed_mul2_f16(uint1 a, uint1 b) {
  return detail::bit_cast<uint1>(detail::bit_cast<f16x2_vec>(a) *
                                 detail::bit_cast<f16x2_vec>(b));
}

TL_DEVICE uint1 packed_fma2_f16(uint1 a, uint1 b, uint1 c) {
  return detail::bit_cast<uint1>(__builtin_elementwise_fma(
      detail::bit_cast<f16x2_vec>(a), detail::bit_cast<f16x2_vec>(b),
      detail::bit_cast<f16x2_vec>(c)));
}

TL_DEVICE uint1 packed_max2_f16(uint1 a, uint1 b) {
  return detail::bit_cast<uint1>(__builtin_elementwise_max(
      detail::bit_cast<f16x2_vec>(a), detail::bit_cast<f16x2_vec>(b)));
}

TL_DEVICE uint1 packed_min2_f16(uint1 a, uint1 b) {
  return detail::bit_cast<uint1>(__builtin_elementwise_min(
      detail::bit_cast<f16x2_vec>(a), detail::bit_cast<f16x2_vec>(b)));
}

// `max2_nan` / `min2_nan` are the NaN-propagating flavours (CUDA:
// `__hmax2_nan` / `__hmin2_nan`): a NaN operand wins. They go through
// `__builtin_elementwise_maximum/minimum` -> `llvm.maximum/minimum`, which the
// backend lowers to a bare `max`/`min`, so unlike `max2` they do not depend on
// a NaN fixup being present in the toolchain.
TL_DEVICE uint1 packed_max2_nan_f16(uint1 a, uint1 b) {
  return detail::bit_cast<uint1>(detail::nan_aware_pair<true>(
      detail::bit_cast<f16x2_vec>(a), detail::bit_cast<f16x2_vec>(b)));
}

TL_DEVICE uint1 packed_min2_nan_f16(uint1 a, uint1 b) {
  return detail::bit_cast<uint1>(detail::nan_aware_pair<false>(
      detail::bit_cast<f16x2_vec>(a), detail::bit_cast<f16x2_vec>(b)));
}

TL_DEVICE uint1 packed_abs2_f16(uint1 a) {
  return detail::bit_cast<uint1>(
      __builtin_elementwise_abs(detail::bit_cast<f16x2_vec>(a)));
}

TL_DEVICE uint1 packed_add2_bf16(uint1 a, uint1 b) {
  return detail::bit_cast<uint1>(detail::bit_cast<bf16x2_vec>(a) +
                                 detail::bit_cast<bf16x2_vec>(b));
}

TL_DEVICE uint1 packed_sub2_bf16(uint1 a, uint1 b) {
  return detail::bit_cast<uint1>(detail::bit_cast<bf16x2_vec>(a) -
                                 detail::bit_cast<bf16x2_vec>(b));
}

TL_DEVICE uint1 packed_mul2_bf16(uint1 a, uint1 b) {
  return detail::bit_cast<uint1>(detail::bit_cast<bf16x2_vec>(a) *
                                 detail::bit_cast<bf16x2_vec>(b));
}

TL_DEVICE uint1 packed_fma2_bf16(uint1 a, uint1 b, uint1 c) {
  return detail::bit_cast<uint1>(__builtin_elementwise_fma(
      detail::bit_cast<bf16x2_vec>(a), detail::bit_cast<bf16x2_vec>(b),
      detail::bit_cast<bf16x2_vec>(c)));
}

TL_DEVICE uint1 packed_max2_bf16(uint1 a, uint1 b) {
  return detail::bit_cast<uint1>(__builtin_elementwise_max(
      detail::bit_cast<bf16x2_vec>(a), detail::bit_cast<bf16x2_vec>(b)));
}

TL_DEVICE uint1 packed_min2_bf16(uint1 a, uint1 b) {
  return detail::bit_cast<uint1>(__builtin_elementwise_min(
      detail::bit_cast<bf16x2_vec>(a), detail::bit_cast<bf16x2_vec>(b)));
}

TL_DEVICE uint1 packed_max2_nan_bf16(uint1 a, uint1 b) {
  return detail::bit_cast<uint1>(detail::nan_aware_pair<true>(
      detail::bit_cast<bf16x2_vec>(a), detail::bit_cast<bf16x2_vec>(b)));
}

TL_DEVICE uint1 packed_min2_nan_bf16(uint1 a, uint1 b) {
  return detail::bit_cast<uint1>(detail::nan_aware_pair<false>(
      detail::bit_cast<bf16x2_vec>(a), detail::bit_cast<bf16x2_vec>(b)));
}

TL_DEVICE uint1 packed_abs2_bf16(uint1 a) {
  return detail::bit_cast<uint1>(
      __builtin_elementwise_abs(detail::bit_cast<bf16x2_vec>(a)));
}

TL_DEVICE float2 packed_add2_f32(float2 a, float2 b) {
  return detail::bit_cast<float2>(detail::bit_cast<f32x2_vec>(a) +
                                  detail::bit_cast<f32x2_vec>(b));
}

TL_DEVICE float2 packed_sub2_f32(float2 a, float2 b) {
  return detail::bit_cast<float2>(detail::bit_cast<f32x2_vec>(a) -
                                  detail::bit_cast<f32x2_vec>(b));
}

TL_DEVICE float2 packed_mul2_f32(float2 a, float2 b) {
  return detail::bit_cast<float2>(detail::bit_cast<f32x2_vec>(a) *
                                  detail::bit_cast<f32x2_vec>(b));
}

TL_DEVICE float2 packed_fma2_f32(float2 a, float2 b, float2 c) {
  return detail::bit_cast<float2>(__builtin_elementwise_fma(
      detail::bit_cast<f32x2_vec>(a), detail::bit_cast<f32x2_vec>(b),
      detail::bit_cast<f32x2_vec>(c)));
}

TL_DEVICE float2 packed_max2_f32(float2 a, float2 b) {
  return detail::bit_cast<float2>(__builtin_elementwise_max(
      detail::bit_cast<f32x2_vec>(a), detail::bit_cast<f32x2_vec>(b)));
}

TL_DEVICE float2 packed_min2_f32(float2 a, float2 b) {
  return detail::bit_cast<float2>(__builtin_elementwise_min(
      detail::bit_cast<f32x2_vec>(a), detail::bit_cast<f32x2_vec>(b)));
}

TL_DEVICE float2 packed_max2_nan_f32(float2 a, float2 b) {
  return detail::bit_cast<float2>(detail::nan_aware_pair<true>(
      detail::bit_cast<f32x2_vec>(a), detail::bit_cast<f32x2_vec>(b)));
}

TL_DEVICE float2 packed_min2_nan_f32(float2 a, float2 b) {
  return detail::bit_cast<float2>(detail::nan_aware_pair<false>(
      detail::bit_cast<f32x2_vec>(a), detail::bit_cast<f32x2_vec>(b)));
}

TL_DEVICE float2 packed_abs2_f32(float2 a) {
  return detail::bit_cast<float2>(
      __builtin_elementwise_abs(detail::bit_cast<f32x2_vec>(a)));
}

} // namespace tl
