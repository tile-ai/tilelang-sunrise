#pragma once

#include "common.h"
#include <cccl/tang/ptx>

namespace tl {

template <tang::ptx::SwizzleMode SW,
          tang::ptx::PackPaddingMode PP = tang::ptx::no_pack>
TL_DEVICE void tang_bulk_g2s(void *smem, const void *gmem, uint32_t rows,
                             uint32_t smem_row_bytes, uint32_t gmem_row_bytes,
                             uint32_t nwarps) {
  // TODO: implement STCUV2 bulk copy.
  (void)smem;
  (void)gmem;
  (void)rows;
  (void)smem_row_bytes;
  (void)gmem_row_bytes;
  (void)nwarps;
}

template <tang::ptx::SwizzleMode SW>
TL_DEVICE void tang_bulk_s2g(void *gmem, const void *smem, uint32_t rows,
                             uint32_t smem_row_bytes, uint32_t gmem_row_bytes,
                             uint32_t nwarps) {
  // TODO: implement STCUV2 bulk copy.
  (void)smem;
  (void)gmem;
  (void)rows;
  (void)smem_row_bytes;
  (void)gmem_row_bytes;
  (void)nwarps;
}

TL_DEVICE void tang_bulk_g2s_1d(void *smem, const void *gmem, uint32_t rows,
                                uint32_t smem_row_bytes,
                                uint32_t gmem_row_bytes, uint32_t nwarps) {
  // TODO: implement STCUV2 bulk copy.
  (void)smem;
  (void)gmem;
  (void)rows;
  (void)smem_row_bytes;
  (void)gmem_row_bytes;
  (void)nwarps;
}

TL_DEVICE void tang_bulk_s2g_1d(void *gmem, const void *smem, uint32_t rows,
                                uint32_t smem_row_bytes,
                                uint32_t gmem_row_bytes, uint32_t nwarps) {
  // TODO: implement STCUV2 bulk copy.
  (void)smem;
  (void)gmem;
  (void)rows;
  (void)smem_row_bytes;
  (void)gmem_row_bytes;
  (void)nwarps;
}

} // namespace tl
