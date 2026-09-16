#pragma once

#include "common.h"
// Public PTX umbrella header instead of the private
// <cccl/tang/__ptx/instructions/tc_cp.h>; the `__ptx` prefix marks that as
// internal.
#include <cccl/tang/ptx>

// ---------------------------------------------------------------------------
// TANG stcuv2 tensor memory -> shared memory copy (cpt2s).
//
// cpt2s streams tensor memory straight into shared memory without going
// through the register file, so a whole accumulator tile can be staged for a
// bulk s2g store without paying any register pressure. It is the reverse of
// cps2t (shared -> tmem) and is a *mem-domain* operation: completion is drained
// with fence_mem on the matching fence group (fg_t2s_default), NOT with
// fence_tmem (that one only covers the ldt/stt register path).
//
// Layout. One cpt2s_16x64b_2d call moves 64 tensor memory rows and writes
// 512 * (SW / atom) bytes of shared memory. Inside every atom the copy engine
// interleaves atom/8 tensor memory rows, because each row contributes exactly
// 64 bits per atom. With atom = 8 bytes that interleave degenerates to a single
// row per atom, which is the only configuration that lands a plain row-major
// tile with an SW-byte row pitch; atom = 32 would scatter 4 accumulator rows
// into each 32-byte atom instead. Hence the atom == 8 restriction below.
//
// This is the 32-bit passthrough form (no pack_16b). pack_16b keeps the LOW 16
// bits of every 32-bit tensor memory word, which is a bit-level pack rather
// than a numeric narrowing, so it must not be used to turn an fp32 accumulator
// into fp16/bf16 -- that would silently emit mantissa bits as the result.
// ---------------------------------------------------------------------------

namespace tl {

// Rows drained per cpt2s_16x64b call (dim1 sweeps 4 x 16 rows).
constexpr uint32_t kTangCpt2sRowsPerCall = 64;

// Caller contract -- none of this is checkable here, since the helper only sees
// a void* and a pre-encoded address. LowerTangTmemToSharedCopy
// (src/tang/op/copy.cc) enforces all of it:
//   * `smem` points at a tile whose row is exactly one 128-byte swizzle unit
//     (32 lanes of a 32-bit type), so kBytesPerCall lands whole rows;
//   * `rows` is a multiple of kTangCpt2sRowsPerCall;
//   * `tmem_addr` is the (row << 16) | col encoding produced by
//     LowerSharedTmem::GetTmemOffset from a BufferLoad on the tmem buffer.
//
// The row-in-the-high-half format is TANG's own, not one carried over from
// CUDA: cccl/tang/__ptx/instructions/tc_cp.h gives the cpt2s dim descriptors
// TMEM strides of 0x100000 (dim1) and 0x400000 (dim2), i.e. 16 << 16 and
// 64 << 16. One 16x64b super-tile is dim1 x STN = 64 rows, so the 0x400000
// dim2 stride is exactly "advance to the next 64 rows" -- which is what the
// (r << 16) stepping below reproduces. Doing it by hand is not a choice: tang
// only exposes a 3D (hardware-stepped) form for the pack_16b variant, and this
// is the 32-bit passthrough one. It does assume the buffer's row mapping is the
// identity (it is; see the layout assigned in src/tang/op/copy.cc).
template <tang::ptx::SwizzleMode SW>
TL_DEVICE void tang_cp_tmem_to_shared(void *smem, uint32_t tmem_addr,
                                      uint32_t rows) {
  static_assert(SW == tang::ptx::sw128a8,
                "tang_cp_tmem_to_shared currently only supports sw128a8: an "
                "8-byte atom is what makes the staged tile row-major (wider "
                "atoms interleave atom/8 tensor memory rows per atom)");
  // One call writes 512 * (swizzle width / atom) bytes. Derived rather than
  // spelled out so it tracks SW if the static_assert above is ever relaxed.
  constexpr uint32_t kSwizzleBytes = 128u; // sw128a8: the "128" half
  constexpr uint32_t kAtomBytes = 8u;      // sw128a8: the "a8" half
  constexpr uint32_t kBytesPerCall = 512u * (kSwizzleBytes / kAtomBytes);

  if (__warpid() == 0) {
    for (uint32_t r = 0; r < rows; r += kTangCpt2sRowsPerCall) {
      // The tensor memory row offset lives in the taddr high bits.
      tang::ptx::cpt2s_16x64b_2d<SW, tang::ptx::fg_t2s_default>(
          static_cast<char *>(smem) +
              (r / kTangCpt2sRowsPerCall) * kBytesPerCall,
          tmem_addr + (r << 16));
    }
    tang::ptx::fence_mem(tang::ptx::fg_t2s_default);
  }
  // The staged tile is consumed by the whole block (typically by a bulk s2g),
  // so publish warp 0's writes before returning.
  __syncthreads();
}

// ---- named wrapper (referenced by codegen_tang.cc by swizzle suffix) ----
TL_DEVICE void tang_cp_tmem_to_shared_sw128a8(void *smem, uint32_t tmem_addr,
                                              uint32_t rows) {
  tang_cp_tmem_to_shared<tang::ptx::sw128a8>(smem, tmem_addr, rows);
}

template <bool SwizzledSource>
TL_DEVICE void tang_cp_shared_to_tmem(void *smem, uint32_t tmem_addr,
                                      uint32_t rows, uint32_t cols,
                                      uint32_t row_words) {
  // TODO: implement MMA A-operand staging.
  (void)smem;
  (void)tmem_addr;
  (void)rows;
  (void)cols;
  (void)row_words;
}

TL_DEVICE void tang_cp_shared_to_tmem_linear(void *smem, uint32_t tmem_addr,
                                             uint32_t rows, uint32_t cols,
                                             uint32_t row_words) {
  tang_cp_shared_to_tmem<false>(smem, tmem_addr, rows, cols, row_words);
}

TL_DEVICE void tang_cp_shared_to_tmem_sw128a32(void *smem, uint32_t tmem_addr,
                                               uint32_t rows, uint32_t cols,
                                               uint32_t row_words) {
  tang_cp_shared_to_tmem<true>(smem, tmem_addr, rows, cols, row_words);
}

} // namespace tl
