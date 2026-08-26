#pragma once

#include "common.h"
#include <limits>

namespace tl {

// Select a wider accumulator type for improved numerical accuracy.
// Default: accumulate in the same type. Specialize FP16/BF16 to float.
// template <typename T> struct AccType {
//   using type = T;
// };
// template <> struct AccType<half_t> {
//   using type = float;
// };
// template <> struct AccType<bfloat16_t> {
//   using type = float;
// };

struct SumOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return x + y;
  }
};

struct MaxOp {
  // Primary template: integer / bit-op paths (compare + select).
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return (x > y ? x : y);
  }
  // Float / double specialisations: NaN-ignoring via manual NaN check
  // because TANG's fmaxf/fmax does not follow IEEE754 NaN semantics
  // (fmaxf(NaN, x) returns NaN instead of x).
  TL_DEVICE float operator()(float const &x, float const &y) {
    if (y != y)
      return x; // y is NaN, keep x
    if (x != x)
      return y; // x is NaN, use y
    return x > y ? x : y;
  }
  TL_DEVICE double operator()(double const &x, double const &y) {
    if (y != y)
      return x;
    if (x != x)
      return y;
    return x > y ? x : y;
  }
};

struct MaxOpNan {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return x != x ? x : (y != y ? y : (x > y ? x : y));
  }
  // Float / double specialisations: TANG's fmaxf/fmax do not follow
  // IEEE754 NaN semantics (neither NaN-ignoring nor NaN-propagation is
  // reliable), so use explicit NaN checks.  NaN wins.
  TL_DEVICE float operator()(float const &x, float const &y) {
    if (x != x)
      return x; // x is NaN, propagate
    if (y != y)
      return y; // y is NaN, propagate
    return x > y ? x : y;
  }
  TL_DEVICE double operator()(double const &x, double const &y) {
    if (x != x)
      return x;
    if (y != y)
      return y;
    return x > y ? x : y;
  }
};

struct MinOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return (x < y ? x : y);
  }
  // Float / double specialisations: NaN-ignoring via manual NaN check
  // because TANG's fminf/fmin does not follow IEEE754 NaN semantics.
  TL_DEVICE float operator()(float const &x, float const &y) {
    if (y != y)
      return x;
    if (x != x)
      return y;
    return x < y ? x : y;
  }
  TL_DEVICE double operator()(double const &x, double const &y) {
    if (y != y)
      return x;
    if (x != x)
      return y;
    return x < y ? x : y;
  }
};

struct MinOpNan {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return x != x ? x : (y != y ? y : (x < y ? x : y));
  }
  // Float / double specialisations: TANG's fminf/fmin do not follow
  // IEEE754 NaN semantics, so use explicit NaN checks.  NaN wins.
  TL_DEVICE float operator()(float const &x, float const &y) {
    if (x != x)
      return x;
    if (y != y)
      return y;
    return x < y ? x : y;
  }
  TL_DEVICE double operator()(double const &x, double const &y) {
    if (x != x)
      return x;
    if (y != y)
      return y;
    return x < y ? x : y;
  }
};

struct BitAndOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return x & y;
  }
};

struct BitOrOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return x | y;
  }
};

struct BitXorOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return x ^ y;
  }
};

// Neutral element used by lanes that fall outside the warp-partial range in
// the CUB two-stage path below. The primary template covers every integral
// type (int32/uint32/int64/uint64/...) through arithmetic on the type itself;
// specialisations handle the floating-point cases whose identity for max/min
// is +/-infinity rather than the smallest/largest representable value.
template <class Op, typename T> struct ReducerIdentity;

template <typename T> struct ReducerIdentity<SumOp, T> {
  TL_DEVICE static T value() { return T(0); }
};
template <typename T> struct ReducerIdentity<MaxOp, T> {
  TL_DEVICE static T value() {
    using U = unsigned long long;
    return T(U(1) << (sizeof(T) * 8 - 1));
  }
};
template <typename T> struct ReducerIdentity<MaxOpNan, T> {
  TL_DEVICE static T value() {
    using U = unsigned long long;
    return T(U(1) << (sizeof(T) * 8 - 1));
  }
};
template <typename T> struct ReducerIdentity<MinOp, T> {
  TL_DEVICE static T value() {
    using U = unsigned long long;
    return T(~(U(1) << (sizeof(T) * 8 - 1)));
  }
};
template <typename T> struct ReducerIdentity<MinOpNan, T> {
  TL_DEVICE static T value() {
    using U = unsigned long long;
    return T(~(U(1) << (sizeof(T) * 8 - 1)));
  }
};
template <typename T> struct ReducerIdentity<BitAndOp, T> {
  TL_DEVICE static T value() { return T(~T(0)); }
};
template <typename T> struct ReducerIdentity<BitOrOp, T> {
  TL_DEVICE static T value() { return T(0); }
};
template <typename T> struct ReducerIdentity<BitXorOp, T> {
  TL_DEVICE static T value() { return T(0); }
};

template <> struct ReducerIdentity<MaxOp, float> {
  TL_DEVICE static float value() { return -TANGRT_INF_F; }
};
template <> struct ReducerIdentity<MaxOpNan, float> {
  TL_DEVICE static float value() { return -TANGRT_INF_F; }
};
template <> struct ReducerIdentity<MinOp, float> {
  TL_DEVICE static float value() { return TANGRT_INF_F; }
};
template <> struct ReducerIdentity<MinOpNan, float> {
  TL_DEVICE static float value() { return TANGRT_INF_F; }
};
template <> struct ReducerIdentity<MaxOp, __fp16> {
  TL_DEVICE static __fp16 value() { return __fp16(-TANGRT_INF_F); }
};

template <> struct ReducerIdentity<MaxOp, unsigned int> {
  TL_DEVICE static unsigned int value() { return 0u; }
};
template <> struct ReducerIdentity<MaxOpNan, unsigned int> {
  TL_DEVICE static unsigned int value() { return 0u; }
};
template <> struct ReducerIdentity<MinOp, unsigned int> {
  TL_DEVICE static unsigned int value() { return ~0u; }
};
template <> struct ReducerIdentity<MinOpNan, unsigned int> {
  TL_DEVICE static unsigned int value() { return ~0u; }
};
template <> struct ReducerIdentity<MaxOp, unsigned long> {
  TL_DEVICE static unsigned long value() { return 0ul; }
};
template <> struct ReducerIdentity<MaxOpNan, unsigned long> {
  TL_DEVICE static unsigned long value() { return 0ul; }
};
template <> struct ReducerIdentity<MinOp, unsigned long> {
  TL_DEVICE static unsigned long value() { return ~0ul; }
};
template <> struct ReducerIdentity<MinOpNan, unsigned long> {
  TL_DEVICE static unsigned long value() { return ~0ul; }
};
template <> struct ReducerIdentity<MaxOp, unsigned long long> {
  TL_DEVICE static unsigned long long value() { return 0ull; }
};
template <> struct ReducerIdentity<MaxOpNan, unsigned long long> {
  TL_DEVICE static unsigned long long value() { return 0ull; }
};
template <> struct ReducerIdentity<MinOp, unsigned long long> {
  TL_DEVICE static unsigned long long value() { return ~0ull; }
};
template <> struct ReducerIdentity<MinOpNan, unsigned long long> {
  TL_DEVICE static unsigned long long value() { return ~0ull; }
};

// Per-Reducer "expensive" flag. `Sum/BitAnd/BitOr/BitXor` map to a simple
// arithmetic or logical operation, while `Max/Min` require compare-and-select
// work. The block-reduce dispatcher uses this to choose between raking (fewer
// op invocations but a longer warp-0 critical path) and warp reductions (more
// invocations distributed across warps). The crossover is hardware-dependent;
// classify compare-and-select reducers as expensive so the dispatcher can
// favor the more parallel reduction path.
template <class Op> struct ReducerIsExpensive {
  static constexpr bool value = false;
};
template <> struct ReducerIsExpensive<MaxOp> {
  static constexpr bool value = true;
};
template <> struct ReducerIsExpensive<MaxOpNan> {
  static constexpr bool value = true;
};
template <> struct ReducerIsExpensive<MinOp> {
  static constexpr bool value = true;
};
template <> struct ReducerIsExpensive<MinOpNan> {
  static constexpr bool value = true;
};

// -------------------------------------------------------------------------
// Sequential per-thread reduction.
//
// Fully unrolled fold of `LEN` array elements seeded with `prefix`. Used
// both as a primitive for callers that hand multiple items per thread AND
// as the serial rake stage inside the block-level raking algorithm below.
// The prefix seed lets raking threads splice their own register partial
// into the fold without extra shared-memory traffic.
// -------------------------------------------------------------------------
template <int LEN, typename T, typename Op>
TL_DEVICE T thread_reduce(const T *arr, Op op, T prefix) {
  T r = prefix;
#pragma unroll
  for (int i = 0; i < LEN; ++i) {
    r = op(r, arr[i]);
  }
  return r;
}

// -------------------------------------------------------------------------
// Padded raking grid layout (CUB's BlockRakingLayout).
//
// SEGMENT_LENGTH = ceil(SHARED_ELEMENTS / 32) -- the number of scalars each
// raking thread walks in phase 2.
//
// USE_SEGMENT_PADDING = 1 when SEGMENT_LENGTH is even and > 2: injecting one
// dead slot between segments makes the raker's sequential reads land in
// distinct banks. SEGMENT_LENGTH == 2 stays un-padded because a 2-wide
// vector load is naturally conflict-free; odd SEGMENT_LENGTHs are
// co-prime with the bank count.
//
// placement_ptr: thread `linear_tid` writes at
//   offset = linear_tid + (padded ? linear_tid / SEGMENT_LENGTH : 0)
// so a full 32-thread warp's writes stay one-per-bank.
//
// raking_ptr: raking thread `rid` reads a contiguous run of SEGMENT_LENGTH
// scalars starting at rid * (SEGMENT_LENGTH + padding).
// -------------------------------------------------------------------------
template <int SHARED_ELEMENTS> struct BlockRakingLayout {
  static constexpr int WARP_THREADS = 32;
  static constexpr int MAX_RAKING_THREADS =
      (SHARED_ELEMENTS < WARP_THREADS) ? SHARED_ELEMENTS : WARP_THREADS;
  static constexpr int SEGMENT_LENGTH =
      (SHARED_ELEMENTS + MAX_RAKING_THREADS - 1) / MAX_RAKING_THREADS;
  static constexpr int RAKING_THREADS =
      (SHARED_ELEMENTS + SEGMENT_LENGTH - 1) / SEGMENT_LENGTH;
  static constexpr int USE_SEGMENT_PADDING =
      (((SEGMENT_LENGTH & 1) == 0) && (SEGMENT_LENGTH > 2)) ? 1 : 0;
  static constexpr int GRID_ELEMENTS =
      RAKING_THREADS * (SEGMENT_LENGTH + USE_SEGMENT_PADDING);
  static constexpr bool UNGUARDED = (SHARED_ELEMENTS % RAKING_THREADS == 0);

  template <typename T>
  TL_DEVICE static T *placement_ptr(T *buff, int linear_tid) {
    int offset = linear_tid;
    if constexpr (USE_SEGMENT_PADDING > 0) {
      offset += offset / SEGMENT_LENGTH;
    }
    return buff + offset;
  }

  template <typename T>
  TL_DEVICE static T *raking_ptr(T *buff, int raking_tid) {
    return buff + raking_tid * (SEGMENT_LENGTH + USE_SEGMENT_PADDING);
  }
};

// Forward declaration: used by AllReduce methods below.
template <typename T, typename ReduceOp>
TL_DEVICE T warp_reduce(T value, ReduceOp op);

// -------------------------------------------------------------------------
// AllReduce.
//
// Algorithm pick (compile-time, in order):
//   * threads == scale              base case (each group has one lane).
//   * scale > 1 (interleaved)       butterfly XOR. Raking assumes
//                                   contiguous participants; interleaved
//                                   layouts must stay on butterfly.
//   * threads <= 32                 butterfly XOR via pure __shfl_xor.
//                                   Zero smem, zero __syncthreads.
//   * threads >  32                 raking (CUB's BLOCK_REDUCE_RAKING).
//                                   One __syncthreads, warp-0 serial fold
//                                   over SEGMENT_LENGTH scalars, warp
//                                   shuffle. Handles any warp count.
// -------------------------------------------------------------------------
template <class Reducer, int threads, int scale, int thread_offset = 0,
          int all_threads = threads, int batch_size = 1,
          int workspace_stride = 0>
struct AllReduce {
  static_assert(threads == 1024 or threads == 512 or threads == 256 or
                threads == 128 or threads == 64 or threads == 32 or
                threads == 16 or threads == 8 or threads == 4 or threads == 2 or
                threads == 1);
  static_assert(threads % scale == 0);

  // Algorithm choice (compile-time). Three paths available for
  // scale == 1 && threads > 128:
  //   * kUseWarpRed   —— CUB BLOCK_REDUCE_WARP_REDUCTIONS. Every warp shfl-
  //                     reduces in parallel; warp 0 folds the WARPS
  //                     partials. Higher parallel op count, shorter
  //                     critical path. Wins for expensive ops (Max/Min).
  //   * kUseRaking    —— CUB BLOCK_REDUCE_RAKING. warp 0 alone rakes
  //                     SEG_LEN partials serially, then shfl-reduces.
  //                     Fewer total op invocations, longer critical path.
  //                     Wins for cheap ops (Sum, bit-ops).
  //   * butterfly     —— fallback for scale > 1 (interleaved participants)
  //                     and threads <= 32 (pure __shfl_xor, no smem, zero
  //                     __syncthreads).
#ifdef TL_REDUCE_FORCE_BUTTERFLY
  // Debug / profiling override
  static constexpr bool kUseWarpRed = false;
  static constexpr bool kUseRaking = false;
#else
  static constexpr bool kUseWarpRed = (scale == 1) && (threads >= 256) &&
                                      (threads / 32 <= 32) &&
                                      ReducerIsExpensive<Reducer>::value;
  static constexpr bool kUseRaking =
      (scale == 1) && (threads > 128) && !kUseWarpRed;
#endif

  // Scalar AllReduce.
  template <typename T, bool Broadcast = true>
  static TL_DEVICE T run(T x, T *red_buf = nullptr, int row_offset = 0) {
    // Offset into a per-row slice of the workspace when multiple
    // rows share the same block (block_m > 1).  Defaults to 0 so
    // existing single-row callers are unaffected.
    if (red_buf)
      red_buf += row_offset;
    if constexpr (threads == scale) {
      return x;
    } else if constexpr (kUseWarpRed) {
      return warp_reductions_reduce<T, Broadcast>(x, red_buf);
    } else if constexpr (kUseRaking) {
      return raking_reduce<T, Broadcast>(x, red_buf);
    } else {
      return butterfly_reduce(x, red_buf);
    }
  }

  // Batched AllReduce via butterfly (runtime batch_size).
  // For CUB paths (raking / warp_reductions), use the multi-channel
  // overload below with a compile-time N instead.
  template <typename T>
  static TL_DEVICE void run_batch(T *x, T *red_buf = nullptr) {
    if constexpr (threads == scale) {
      return;
    } else {
      butterfly_reduce_batch(x, red_buf);
    }
  }

  // Multi-channel overload: reduce N independent scalars in one call.
  // Barrier count stays at (1 + Broadcast) total for the whole array
  // instead of (1 + Broadcast) * N. `red_buf` must be sized for the
  // chosen algorithm's per-channel layout: N * GRID_ELEMENTS entries per
  // reduce group for raking, N * WARPS entries for warp_reductions.
  template <typename T, int N, bool Broadcast = true>
  static TL_DEVICE void run(T (&xs)[N], T *red_buf) {
    if constexpr (threads == scale) {
      return;
    } else if constexpr (kUseWarpRed) {
      warp_reductions_reduce_multi<T, N, Broadcast>(xs, red_buf);
    } else if constexpr (kUseRaking) {
      raking_reduce_multi<T, N, Broadcast>(xs, red_buf);
    } else {
// Butterfly path: fall back to per-channel loop.
#pragma unroll
      for (int i = 0; i < N; ++i) {
        xs[i] = butterfly_reduce(xs[i], red_buf);
      }
    }
  }

private:
  // Only the last shared-memory level touching red_buf needs a trailing
  // barrier, to fence its reads from a later reuse of the offset; inner
  // levels are already fenced from the next level by their leading
  // __syncthreads(). That level is the recursion terminator (offset == scale)
  // or offset == 32, past which remaining levels are pure __shfl_xor.
  static constexpr bool kNeedsTailBarrier =
      (threads / 2 >= 32) && ((threads / 2 == scale) || (threads / 2 == 32));

  template <typename T> static TL_DEVICE T butterfly_reduce(T x, T *red_buf) {
    constexpr int offset = threads / 2;
    if constexpr (offset >= 32) {
      __syncthreads();
      red_buf[threadIdx.x - thread_offset] = x;
      __syncthreads();
      x = Reducer()(x, red_buf[(threadIdx.x - thread_offset) ^ offset]);
      if constexpr (kNeedsTailBarrier) {
        __syncthreads();
      }
    } else {
      x = Reducer()(x, T(__shfl_xor(x, offset)));
    }
    if constexpr (offset == scale) {
      return x;
    } else {
      return AllReduce<Reducer, offset, scale, thread_offset, all_threads,
                       batch_size, workspace_stride>::run(x, red_buf);
    }
  }

  template <typename T>
  static TL_DEVICE void butterfly_reduce_batch(T *x, T *red_buf) {
    constexpr int offset = threads / 2;
    if constexpr (offset >= 32) {
      __syncthreads();
#pragma unroll
      for (int i = 0; i < batch_size; ++i) {
        red_buf[(threadIdx.x - thread_offset) + i * workspace_stride] = x[i];
      }
      __syncthreads();
#pragma unroll
      for (int i = 0; i < batch_size; ++i) {
        x[i] =
            Reducer()(x[i], red_buf[((threadIdx.x - thread_offset) ^ offset) +
                                    i * workspace_stride]);
      }
      if constexpr (kNeedsTailBarrier) {
        __syncthreads();
      }
    } else {
#pragma unroll
      for (int i = 0; i < batch_size; ++i) {
        x[i] = Reducer()(x[i], T(__shfl_xor(x[i], offset)));
      }
    }
    if constexpr (offset != scale) {
      AllReduce<Reducer, offset, scale, thread_offset, all_threads, batch_size,
                workspace_stride>::run_batch(x, red_buf);
    }
  }

  // Warp-reductions block reduce (CUB BLOCK_REDUCE_WARP_REDUCTIONS).
  //   Stage 1: every warp shfl-reduces its 32 lanes in parallel (no smem).
  //   Stage 2: each warp's lane 0 publishes its partial. One __syncthreads.
  //   Stage 3: warp 0 folds the WARPS partials, unused lanes contribute
  //            the reduce identity so warp_reduce doesn't see stale data.
  //   Stage 4 (Broadcast): thread 0 publishes result, __syncthreads,
  //            every thread reads. Skipped when Broadcast=false.
  template <typename T, bool Broadcast>
  static TL_DEVICE T warp_reductions_reduce(T x, T *red_buf) {
    constexpr int WARPS = threads / 32;
    const int tid = static_cast<int>(threadIdx.x) - thread_offset;
    const int gid = tid / threads;
    const int lid = tid - gid * threads;
    const int lane = lid & 31;
    const int wid = lid >> 5;
    T *rb = red_buf + gid * WARPS;

    // Stage 1: intra-warp reduce, all warps in parallel.
    x = warp_reduce<T>(x, Reducer());

    // Stage 2: warp leaders publish partials.
    if (lane == 0)
      rb[wid] = x;
    __syncthreads();

    // Stage 3: warp 0 folds partials; lanes >= WARPS contribute identity.
    T folded;
    if (wid == 0) {
      folded = (lane < WARPS) ? rb[lane] : ReducerIdentity<Reducer, T>::value();
      folded = warp_reduce<T>(folded, Reducer());
    }

    if constexpr (Broadcast) {
      if (wid == 0 && lane == 0)
        rb[0] = folded;
      __syncthreads();
      T result = rb[0];
      // Trailing barrier: red_buf is reused by the next reduction in the
      // same kernel (the workspace allocator may hand out the same shared
      // offset). Without it, a fast thread's phase-1 deposit can overwrite
      // rb[0] before a slow thread has broadcast-read it.
      __syncthreads();
      return result;
    } else {
      return folded;
    }
  }

  // Multi-channel warp_reductions. Channels share the WARPS-per-group
  // partial slots via an interleaved layout `rb[wid * N + c]`, so one
  // __syncthreads covers all N channels.
  template <typename T, int N, bool Broadcast>
  static TL_DEVICE void warp_reductions_reduce_multi(T (&xs)[N], T *red_buf) {
    constexpr int WARPS = threads / 32;
    const int tid = static_cast<int>(threadIdx.x) - thread_offset;
    const int gid = tid / threads;
    const int lid = tid - gid * threads;
    const int lane = lid & 31;
    const int wid = lid >> 5;
    T *rb = red_buf + gid * WARPS * N;

// Stage 1: intra-warp reduce, all channels, no barriers.
#pragma unroll
    for (int c = 0; c < N; ++c) {
      xs[c] = warp_reduce<T>(xs[c], Reducer());
    }

    // Stage 2: every warp's lane 0 publishes its N partials.
    if (lane == 0) {
#pragma unroll
      for (int c = 0; c < N; ++c) {
        rb[wid * N + c] = xs[c];
      }
    }
    __syncthreads();

    // Stage 3: warp 0 folds each channel independently.
    T folded[N];
    if (wid == 0) {
#pragma unroll
      for (int c = 0; c < N; ++c) {
        T v = (lane < WARPS) ? rb[lane * N + c]
                             : ReducerIdentity<Reducer, T>::value();
        folded[c] = warp_reduce<T>(v, Reducer());
      }
    }

    if constexpr (Broadcast) {
      if (wid == 0 && lane == 0) {
#pragma unroll
        for (int c = 0; c < N; ++c)
          rb[c] = folded[c];
      }
      __syncthreads();
#pragma unroll
      for (int c = 0; c < N; ++c)
        xs[c] = rb[c];
      // Trailing barrier: see the scalar path — red_buf may be reused by a
      // following reduction that shares the same shared-memory offset.
      __syncthreads();
    } else {
#pragma unroll
      for (int c = 0; c < N; ++c)
        xs[c] = folded[c];
    }
  }

  // Raking block reduce (CUB BLOCK_REDUCE_RAKING).
  //   Phase 1: every thread deposits its partial into the padded grid.
  //   Phase 2: __syncthreads.
  //   Phase 3: raking warp (lane < RAKING_THREADS) serially folds
  //            SEGMENT_LENGTH scalars, then warp-shuffle reduces.
  //   Phase 4 (Broadcast): thread 0 publishes result, __syncthreads,
  //            every thread reads. Skipped when Broadcast=false.
  template <typename T, bool Broadcast>
  static TL_DEVICE T raking_reduce(T x, T *red_buf) {
    using Layout = BlockRakingLayout<threads>;
    constexpr int RAKING_THREADS = Layout::RAKING_THREADS;
    constexpr int SEG = Layout::SEGMENT_LENGTH;

    const int tid = static_cast<int>(threadIdx.x) - thread_offset;
    const int gid = tid / threads;
    const int lid = tid - gid * threads;
    T *rb = red_buf + gid * Layout::GRID_ELEMENTS;

    // Phase 1: deposit partial into the padded raking grid.
    *Layout::template placement_ptr<T>(rb, lid) = x;
    __syncthreads();

    T folded;
    if (lid < RAKING_THREADS) {
      T *seg = Layout::template raking_ptr<T>(rb, lid);
      folded = seg[0];
      if constexpr (SEG > 1) {
        folded = thread_reduce<SEG - 1, T>(seg + 1, Reducer(), folded);
      }
      folded = warp_reduce<T>(folded, Reducer());
    }

    if constexpr (Broadcast) {
      if (lid == 0)
        rb[0] = folded;
      __syncthreads();
      T result = rb[0];
      // Trailing barrier: red_buf is reused by the next reduction in the
      // same kernel (the workspace allocator may hand out the same shared
      // offset). Without it, a fast thread's phase-1 deposit can overwrite
      // rb[0] before a slow thread has broadcast-read it.
      __syncthreads();
      return result;
    } else {
      return folded;
    }
  }

  // Multi-channel raking. Layout: channels are interleaved at grid stride.
  // Lane l writes rb[c * GRID_ELEMENTS + placement_offset(l)] for each
  // channel c, so one __syncthreads covers all N channels.
  template <typename T, int N, bool Broadcast>
  static TL_DEVICE void raking_reduce_multi(T (&xs)[N], T *red_buf) {
    using Layout = BlockRakingLayout<threads>;
    constexpr int RAKING_THREADS = Layout::RAKING_THREADS;
    constexpr int SEG = Layout::SEGMENT_LENGTH;
    constexpr int GRID = Layout::GRID_ELEMENTS;

    const int tid = static_cast<int>(threadIdx.x) - thread_offset;
    const int gid = tid / threads;
    const int lid = tid - gid * threads;
    T *rb = red_buf + gid * GRID * N;

// Phase 1: deposit all N partials, one placement per channel.
#pragma unroll
    for (int c = 0; c < N; ++c) {
      *Layout::template placement_ptr<T>(rb + c * GRID, lid) = xs[c];
    }
    __syncthreads();

    T folded[N];
    if (lid < RAKING_THREADS) {
#pragma unroll
      for (int c = 0; c < N; ++c) {
        T *seg = Layout::template raking_ptr<T>(rb + c * GRID, lid);
        T f = seg[0];
        if constexpr (SEG > 1) {
          f = thread_reduce<SEG - 1, T>(seg + 1, Reducer(), f);
        }
        folded[c] = warp_reduce<T>(f, Reducer());
      }
    }

    if constexpr (Broadcast) {
      if (lid == 0) {
#pragma unroll
        for (int c = 0; c < N; ++c) {
          rb[c * GRID] = folded[c];
        }
      }
      __syncthreads();
#pragma unroll
      for (int c = 0; c < N; ++c) {
        xs[c] = rb[c * GRID];
      }
      // Trailing barrier: see the scalar path — red_buf may be reused by a
      // following reduction that shares the same shared-memory offset.
      __syncthreads();
    } else {
#pragma unroll
      for (int c = 0; c < N; ++c) {
        xs[c] = folded[c];
      }
    }
  }
};

template <int threads, bool reverse = false> struct CumSum1D {
  static_assert(threads == 1024 or threads == 512 or threads == 256 or
                threads == 128 or threads == 64 or threads == 32);
  template <typename T, int SEG = 32>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                            int N) {
    if (N <= 0)
      return;

    constexpr unsigned MASK = 0xffffffff;
    const int tid = threadIdx.x;
    const int lane = tid % SEG;

    if (tid >= SEG)
      return;

    T carry = (T)0;

    if (reverse) {
      const int num_segments = (N + SEG - 1) / SEG;
      for (int seg = num_segments - 1; seg >= 0; --seg) {
        const int idx = seg * SEG + lane;
        T val = (idx < N) ? src[idx] : (T)0;

#pragma unroll
        for (int off = 1; off < SEG; off <<= 1) {
          T n = (T)__shfl_down_sync(MASK, val, off);
          if (lane < SEG - off)
            val += n;
        }

        val += carry;

        if (idx < N)
          dst[idx] = val;

        T segSum = (T)__shfl_sync(MASK, val, 0);
        if (lane == 0)
          carry = segSum;
        carry = (T)__shfl_sync(MASK, carry, 0);
      }
    } else {
      const int num_segments = (N + SEG - 1) / SEG;
      for (int seg = 0; seg < num_segments; ++seg) {
        const int idx = seg * SEG + lane;
        T val = (idx < N) ? src[idx] : (T)0;

#pragma unroll
        for (int off = 1; off < SEG; off <<= 1) {
          T n = (T)__shfl_up_sync(MASK, val, off);
          if (lane >= off)
            val += n;
        }

        val += carry;

        if (idx < N)
          dst[idx] = val;

        T segSum = (T)__shfl_sync(MASK, val, SEG - 1);
        if (lane == SEG - 1)
          carry = segSum;
        carry = (T)__shfl_sync(MASK, carry, SEG - 1);
      }
    }
  }
};

template <int threads, int Axis = 0, bool reverse = false> struct CumSum2D {
  static_assert(threads == 1024 or threads == 512 or threads == 256 or
                threads == 128 or threads == 64 or threads == 32);
  template <typename T, int SEG = 32>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                            int H, int W, int src_stride, int dst_stride) {

    constexpr int TILE_H = threads / SEG;
    constexpr unsigned MASK = 0xffffffff;
    const int outer = Axis == 0 ? W : H;
    const int num_blocks = (outer + TILE_H - 1) / TILE_H;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int row = tid / 32;

    for (int b = 0; b < num_blocks; ++b) {
      const int gRow = b * TILE_H + row;
      if (gRow >= outer)
        return;

      T carry = (T)0;

      if (reverse) {
        // Start from the last segment for reverse mode
        for (int seg = ((Axis == 0 ? H : W) + SEG - 1) / SEG - 1; seg >= 0;
             --seg) {
          const int col = seg * SEG + lane;

          const int real_row = Axis == 1 ? gRow : col;
          const int real_col = Axis == 1 ? col : gRow;

          T val = (col < (Axis == 0 ? H : W))
                      ? src[real_row * src_stride + real_col]
                      : (T)0;

#pragma unroll
          for (int off = 1; off < SEG; off <<= 1) {
            T n = (T)__shfl_down_sync(MASK, val, off);
            if (lane < SEG - off)
              val += n;
          }

          val += carry;

          if (real_row < H && real_col < W)
            dst[real_row * dst_stride + real_col] = val;

          T segSum = (T)__shfl_sync(MASK, val, (T)0);
          if (lane == 0)
            carry = segSum;
          carry = (T)__shfl_sync(MASK, carry, (T)0);
        }
      } else {
        for (int seg = 0; seg * SEG < (Axis == 0 ? H : W); ++seg) {
          const int col = seg * SEG + lane;

          const int real_row = Axis == 1 ? gRow : col;
          const int real_col = Axis == 1 ? col : gRow;

          T val = (col < (Axis == 0 ? H : W))
                      ? src[real_row * src_stride + real_col]
                      : (T)0;

#pragma unroll
          for (int off = 1; off < SEG; off <<= 1) {
            T n = (T)__shfl_up_sync(MASK, val, off);
            if (lane >= off)
              val += n;
          }

          val += carry;

          if ((Axis == 1 ? (gRow < H && col < W) : (col < H && gRow < W))) {
            dst[real_row * dst_stride + real_col] = val;
          }

          T segSum = (T)__shfl_sync(MASK, val, SEG - 1);
          if (lane == SEG - 1)
            carry = segSum;
          carry = (T)__shfl_sync(MASK, carry, SEG - 1);
        }
      }
    }
  }
};

template <int threads, bool reverse = false> struct CumMax1D {
  static_assert(threads == 1024 or threads == 512 or threads == 256 or
                threads == 128 or threads == 64 or threads == 32);
  template <typename T, int SEG = 32>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                            int N) {
    if (N <= 0)
      return;

    constexpr unsigned MASK = 0xffffffff;
    const int tid = threadIdx.x;
    const int lane = tid % SEG;

    if (tid >= SEG)
      return;

    const T lowest = std::numeric_limits<T>::lowest();
    T carry = lowest;

    if (reverse) {
      const int num_segments = (N + SEG - 1) / SEG;
      for (int seg = num_segments - 1; seg >= 0; --seg) {
        const int idx = seg * SEG + lane;
        T val = (idx < N) ? src[idx] : lowest;

#pragma unroll
        for (int off = 1; off < SEG; off <<= 1) {
          T n = (T)__shfl_down_sync(MASK, val, off);
          if (lane < SEG - off)
            val = (n > val) ? n : val;
        }

        val = (carry > val) ? carry : val;

        if (idx < N)
          dst[idx] = val;

        T segMax = (T)__shfl_sync(MASK, val, 0);
        if (lane == 0)
          carry = segMax;
        carry = (T)__shfl_sync(MASK, carry, 0);
      }
    } else {
      const int num_segments = (N + SEG - 1) / SEG;
      for (int seg = 0; seg < num_segments; ++seg) {
        const int idx = seg * SEG + lane;
        T val = (idx < N) ? src[idx] : lowest;

#pragma unroll
        for (int off = 1; off < SEG; off <<= 1) {
          T n = (T)__shfl_up_sync(MASK, val, off);
          if (lane >= off)
            val = (n > val) ? n : val;
        }

        val = (carry > val) ? carry : val;

        if (idx < N)
          dst[idx] = val;

        T segMax = (T)__shfl_sync(MASK, val, SEG - 1);
        if (lane == SEG - 1)
          carry = segMax;
        carry = (T)__shfl_sync(MASK, carry, SEG - 1);
      }
    }
  }
};

template <int threads, int Axis = 0, bool reverse = false> struct CumMax2D {
  static_assert(threads == 1024 or threads == 512 or threads == 256 or
                threads == 128 or threads == 64 or threads == 32);
  template <typename T, int SEG = 32>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                            int H, int W, int src_stride, int dst_stride) {

    constexpr int TILE_H = threads / SEG;
    constexpr unsigned MASK = 0xffffffff;
    const int outer = Axis == 0 ? W : H;
    const int num_blocks = (outer + TILE_H - 1) / TILE_H;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int row = tid / 32;

    for (int b = 0; b < num_blocks; ++b) {
      const int gRow = b * TILE_H + row;
      if (gRow >= outer)
        return;

      const T lowest = std::numeric_limits<T>::lowest();
      T carry = lowest;

      if (reverse) {
        for (int seg = ((Axis == 0 ? H : W) + SEG - 1) / SEG - 1; seg >= 0;
             --seg) {
          const int col = seg * SEG + lane;

          const int real_row = Axis == 1 ? gRow : col;
          const int real_col = Axis == 1 ? col : gRow;

          T val = (col < (Axis == 0 ? H : W))
                      ? src[real_row * src_stride + real_col]
                      : lowest;

#pragma unroll
          for (int off = 1; off < SEG; off <<= 1) {
            T n = (T)__shfl_down_sync(MASK, val, off);
            if (lane < SEG - off)
              val = (n > val) ? n : val;
          }

          val = (carry > val) ? carry : val;

          if (real_row < H && real_col < W)
            dst[real_row * dst_stride + real_col] = val;

          T segMax = (T)__shfl_sync(MASK, val, 0);
          if (lane == 0)
            carry = segMax;
          carry = (T)__shfl_sync(MASK, carry, 0);
        }
      } else {
        for (int seg = 0; seg * SEG < (Axis == 0 ? H : W); ++seg) {
          const int col = seg * SEG + lane;

          const int real_row = Axis == 1 ? gRow : col;
          const int real_col = Axis == 1 ? col : gRow;

          T val = (col < (Axis == 0 ? H : W))
                      ? src[real_row * src_stride + real_col]
                      : lowest;

#pragma unroll
          for (int off = 1; off < SEG; off <<= 1) {
            T n = (T)__shfl_up_sync(MASK, val, off);
            if (lane >= off)
              val = (n > val) ? n : val;
          }

          val = (carry > val) ? carry : val;

          if ((Axis == 1 ? (gRow < H && col < W) : (col < H && gRow < W))) {
            dst[real_row * dst_stride + real_col] = val;
          }

          T segMax = (T)__shfl_sync(MASK, val, SEG - 1);
          if (lane == SEG - 1)
            carry = segMax;
          carry = (T)__shfl_sync(MASK, carry, SEG - 1);
        }
      }
    }
  }
};

template <typename T, typename ReduceOp>
TL_DEVICE T warp_reduce(T value, ReduceOp op) {
  value = op(value, __shfl_xor(value, 16));
  value = op(value, __shfl_xor(value, 8));
  value = op(value, __shfl_xor(value, 4));
  value = op(value, __shfl_xor(value, 2));
  value = op(value, __shfl_xor(value, 1));
  return value;
}

template <typename T> TL_DEVICE T warp_reduce_sum(T value) {
  return warp_reduce<T>(value, SumOp());
}

template <typename T> TL_DEVICE T warp_reduce_max(T value) {
  return warp_reduce<T>(value, MaxOp());
}

template <typename T> TL_DEVICE T warp_reduce_min(T value) {
  return warp_reduce<T>(value, MinOp());
}

template <typename T> TL_DEVICE T warp_reduce_bitand(T value) {
  return warp_reduce<T>(value, BitAndOp());
}

template <typename T> TL_DEVICE T warp_reduce_bitor(T value) {
  return warp_reduce<T>(value, BitOrOp());
}

} // namespace tl
