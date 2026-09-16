#pragma once

#include "common.h"
// Public PTX umbrella header instead of the private
// <cccl/tang/__ptx/instructions/mbarrier_*.h>; the `__ptx` prefix marks those
// as internal. It re-exports mbarrier_{init,arrive,expect_tx,wait} and the
// tang::ptx::__mbarrier_t type.
#include <cccl/tang/ptx>

// ---------------------------------------------------------------------------
// TANG stcuv2 (S3) mbarrier support.
//
// Only four mbarrier atomics exist in hardware -- expect_tx, complete_tx,
// arrive and arrive_drop. init, inval and every wait flavour are emulated by
// the device library: init is a plain store of (count << 20) | count followed
// by fence.mem, and the waits poll the phase field in the top bits of the
// 64-bit barrier word. Two consequences shape this file:
//
//   * There is no blocking wait instruction, so Barrier::wait spins on the
//     try_wait_parity predicate, matching the recipe in the vendor mbarrier
//     guide. try_wait_parity returns true once the phase has flipped away from
//     the parity passed in, which is the same convention TileLang and CUDA use
//     (a barrier starts at phase 0, so the first completion is awaited with
//     parity 0, the second with parity 1, ...).
//   * There is no fused arrive.expect_tx and no cluster/cta_id arrive. Those
//     TileLang builtins are rejected in the TANG code generator rather than
//     emulated here: splitting a fused arrive.expect_tx into two calls would
//     quietly drop its atomicity.
//
// expect_tx is wired up because the instruction exists, but pairing it with an
// async bulk copy is NOT supported on current silicon -- the transaction
// counter that complete_tx bumps is documented as unreliable, and the bulk-copy
// overloads taking an __mbarrier_t* are deprecated upstream. Publish producer
// completion with a fence group bound to the barrier, or with a sync barrier,
// instead. See docs/tang_mbarrier.md.
//
// Everything here is gated on stcuv2 by construction: the underlying builtins
// are compiled only for __Tang_ARCH__ >= 200, and the code generator emits this
// header only for kernels that actually use a barrier.
// ---------------------------------------------------------------------------

namespace tl {

TL_DEVICE void mbarrier_init(::tang::ptx::__mbarrier_t &smem_barrier,
                             uint32_t arrive_count) {
  ::tang::ptx::mbarrier_init(&smem_barrier, arrive_count);
}

TL_DEVICE void mbarrier_inval(::tang::ptx::__mbarrier_t &smem_barrier) {
  ::tang::ptx::mbarrier_inval(&smem_barrier);
}

TL_DEVICE void mbarrier_arrive(::tang::ptx::__mbarrier_t &smem_barrier,
                               uint32_t count = 1) {
  // The returned token is only useful for the test_wait/try_wait(token) forms;
  // the parity waits read the phase field directly, so it is dropped here.
  (void)::tang::ptx::mbarrier_arrive(&smem_barrier, count);
}

TL_DEVICE bool mbarrier_try_wait(::tang::ptx::__mbarrier_t &smem_barrier,
                                 uint32_t phase) {
  return ::tang::ptx::mbarrier_try_wait_parity(&smem_barrier, phase);
}

TL_DEVICE void mbarrier_wait(::tang::ptx::__mbarrier_t &smem_barrier,
                             uint32_t phase) {
  // mbarrier_try_wait_parity is an opaque device-library call taking a volatile
  // pointer, so the poll cannot be hoisted out of this loop. There is also a
  // suspendTimeHint overload that sleeps between polls; it is not used here
  // because the hint units are not part of the documented contract.
  while (!mbarrier_try_wait(smem_barrier, phase)) {
  }
}

TL_DEVICE void mbarrier_expect_tx(::tang::ptx::__mbarrier_t &smem_barrier,
                                  uint32_t transaction_bytes) {
  (void)::tang::ptx::mbarrier_expect_tx(&smem_barrier, transaction_bytes);
}

// Bind the calling warp's tensor-core fence group to a barrier: once the work
// outstanding in that group retires, hardware posts one arrive on the barrier.
// This is the producer half of an async MMA -- the counterpart of Blackwell's
// tcgen05.commit.mbarrier, and the reason a TANG MMA can publish its own
// completion at all.
//
// Two properties differ from a plain mbarrier_arrive and both matter:
//
//   * It does not block. fence_tc<fg>() stalls the issuing warp until the
//     group drains; this only registers the arrival, so the warp runs on and
//     whoever waits on the barrier's parity observes the completion instead.
//   * The arrive is per warp, and exactly one (.noinc). A fence group is
//     private to the warp that issued into it ("synchronize operations in
//     current warp" -- cccl/tang/__ptx/instructions/fence.h), so one warp's
//     fence says nothing about a sibling warp's MMA: N issuing warps must each
//     call this, producing N arrives. The barrier's arrive count has to match
//     that warp count, not the thread count.
//
// The TC fence group id is 0 or 1 (narrower than the 0-7 of the mem domain).
template <int FenceGroup = 0>
TL_DEVICE void
fence_tc_arrive_mbarrier(::tang::ptx::__mbarrier_t &smem_barrier) {
  static_assert(FenceGroup == 0 || FenceGroup == 1,
                "TANG tensor-core fence group id must be 0 or 1");
  ::tang::ptx::fence_tc_arrive_mbarrier(
      &smem_barrier, static_cast<::tang::ptx::FenceGroup>(FenceGroup));
}

// CUDA needs fence.mbarrier_init.release.cluster between the init store and the
// first use of a barrier. On TANG the device library's mbarrier_init already
// ends in fence.mem, and LowerSharedBarrier emits a __syncthreads() right after
// this call to publish the init to the rest of the block, so there is nothing
// left for this fence to order.
TL_DEVICE void fence_barrier_init() {}

} // namespace tl

// Global scope, matching the CUDA template: the code generator emits
// `reinterpret_cast<Barrier*>(...)` with an unqualified name.
struct alignas(8) Barrier {
  using ValueType = ::tang::ptx::__mbarrier_t;

private:
  ValueType barrier_;

  TL_DEVICE ValueType &storage() const {
    return *const_cast<ValueType *>(&barrier_);
  }

public:
  Barrier() = delete;

  TL_DEVICE void init(uint32_t arrive_count) const {
    tl::mbarrier_init(storage(), arrive_count);
  }

  TL_DEVICE void inval() const { tl::mbarrier_inval(storage()); }

  TL_DEVICE void arrive() const { tl::mbarrier_arrive(storage()); }

  TL_DEVICE void wait(uint32_t phase) const {
    tl::mbarrier_wait(storage(), phase);
  }

  TL_DEVICE bool try_wait(uint32_t phase) const {
    return tl::mbarrier_try_wait(storage(), phase);
  }

  TL_DEVICE void expect_transaction(uint32_t transaction_bytes) const {
    tl::mbarrier_expect_tx(storage(), transaction_bytes);
  }

  // Publish the calling warp's tensor-core completion into this barrier; see
  // tl::fence_tc_arrive_mbarrier for the per-warp arrive count.
  template <int FenceGroup = 0> TL_DEVICE void arrive_on_tc_fence() const {
    tl::fence_tc_arrive_mbarrier<FenceGroup>(storage());
  }
};

// The code generator backs a Barrier array with `__shared__ uint64_t[N]`, so
// the object must stay a bare 8-byte word.
static_assert(sizeof(Barrier) == sizeof(uint64_t));
static_assert(alignof(Barrier) == alignof(uint64_t));
