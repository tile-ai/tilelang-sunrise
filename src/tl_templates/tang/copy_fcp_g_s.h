#pragma once

#include "common.h"
#include <cccl/tang/ptx>

namespace tl {

// ---- swizzle-mode-templated cores ----
template <tang::ptx::SwizzleMode SW>
TL_DEVICE void tang_bulk_g2s(void *smem, const void *gmem, uint32_t rows,
                             uint32_t smem_row_bytes, uint32_t gmem_row_bytes) {
  // TODO: reimplement fcpg2s<SW>-based global->shared bulk copy.
  (void)smem;
  (void)gmem;
  (void)rows;
  (void)smem_row_bytes;
  (void)gmem_row_bytes;
}

template <tang::ptx::SwizzleMode SW>
TL_DEVICE void tang_bulk_s2g(void *gmem, const void *smem, uint32_t rows,
                             uint32_t smem_row_bytes, uint32_t gmem_row_bytes) {
  // TODO: reimplement fcps2g<SW>-based shared->global bulk copy.
  (void)gmem;
  (void)smem;
  (void)rows;
  (void)smem_row_bytes;
  (void)gmem_row_bytes;
}

// ---- named wrappers (referenced by codegen_tang.cc by swizzle suffix) ----
TL_DEVICE void tang_bulk_g2s_sw128a32(void *smem, const void *gmem,
                                      uint32_t rows, uint32_t smem_row_bytes,
                                      uint32_t gmem_row_bytes) {
  // TODO: reimplement tang_bulk_g2s<sw128a32> wrapper.
  (void)smem;
  (void)gmem;
  (void)rows;
  (void)smem_row_bytes;
  (void)gmem_row_bytes;
}

TL_DEVICE void tang_bulk_s2g_sw128a32(void *gmem, const void *smem,
                                      uint32_t rows, uint32_t smem_row_bytes,
                                      uint32_t gmem_row_bytes) {
  // TODO: reimplement tang_bulk_s2g<sw128a32> wrapper.
  (void)gmem;
  (void)smem;
  (void)rows;
  (void)smem_row_bytes;
  (void)gmem_row_bytes;
}

TL_DEVICE void tang_bulk_g2s_sw128a64(void *smem, const void *gmem,
                                      uint32_t rows, uint32_t smem_row_bytes,
                                      uint32_t gmem_row_bytes) {
  // TODO: reimplement tang_bulk_g2s<sw128a64> wrapper.
  (void)smem;
  (void)gmem;
  (void)rows;
  (void)smem_row_bytes;
  (void)gmem_row_bytes;
}

TL_DEVICE void tang_bulk_s2g_sw128a64(void *gmem, const void *smem,
                                      uint32_t rows, uint32_t smem_row_bytes,
                                      uint32_t gmem_row_bytes) {
  // TODO: reimplement tang_bulk_s2g<sw128a64> wrapper.
  (void)gmem;
  (void)smem;
  (void)rows;
  (void)smem_row_bytes;
  (void)gmem_row_bytes;
}

} // namespace tl
