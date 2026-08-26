#pragma once

#include "common.h"

namespace tl {

__device__ __forceinline__ unsigned int ceil_div(unsigned int a,
                                                 unsigned int b) {
  return (a + b - 1) / b;
}

template <int panel_width> TL_DEVICE dim3 rasterization2DRow() {
  const unsigned int gx = gridDim.x;
  const unsigned int gy = gridDim.y;
  const unsigned int block_idx = blockIdx.x + blockIdx.y * gx;
  if constexpr (panel_width > 1) {
    const unsigned int panel_size = panel_width * gx;
    const unsigned int panel_idx = block_idx / panel_size;
    const unsigned int panel_offset = block_idx - panel_idx * panel_size;
    // total_panel = ceil_div(gx*gy, panel_width*gx) == ceil_div(gy,
    // panel_width), so the division is by the compile-time constant panel_width
    // rather than by the runtime panel_size.
    const unsigned int total_panel = (gy + panel_width - 1) / panel_width;
    // The final panel may be short (gy not a multiple of panel_width), in which
    // case its stride is the remaining rows. stride_last simplifies to
    // gy - panel_width * panel_idx: the general form
    // (grid_size - panel_idx * panel_size) / gx has gx as a factor of every
    // term in the numerator, so the division is exact and cancels.
    //
    // The comparison must be panel_idx + 1 < total_panel, not
    // panel_idx < total_panel: the latter treats the last panel as full,
    // leaving stride = panel_width on a short panel so row_idx runs past gy
    // - 1. That overruns the C tile and, as a past bug in this file showed, can
    // clobber the input B. Verified exhaustively against the pre-simplification
    // formula over panel_width in {1,2,3,4,5,8,16} x gx,gy in 1..40: zero
    // mismatches.
    const unsigned int stride = (panel_idx + 1 < total_panel)
                                    ? panel_width
                                    : gy - panel_width * panel_idx;
    const unsigned int col_idx = (panel_idx & 1)
                                     ? gx - 1 - (panel_offset / stride)
                                     : panel_offset / stride;
    const unsigned int row_idx =
        panel_offset % stride + panel_idx * panel_width;
    return {col_idx, row_idx, blockIdx.z};
  } else {
    // panel_width == 1: every panel is exactly one grid row, so the panel is
    // never short and stride is always 1. That collapses panel_offset / stride
    // to panel_offset and panel_offset % stride to 0, removing both divisions.
    const unsigned int panel_idx = block_idx / gx;
    const unsigned int panel_offset = block_idx - panel_idx * gx;
    const unsigned int col_idx =
        (panel_idx & 1) ? gx - 1 - panel_offset : panel_offset;
    const unsigned int row_idx = panel_idx;
    return {col_idx, row_idx, blockIdx.z};
  }
}

template <int panel_width> TL_DEVICE dim3 rasterization2DColumn() {
  const unsigned int gx = gridDim.x;
  const unsigned int gy = gridDim.y;
  const unsigned int block_idx = blockIdx.x + blockIdx.y * gx;
  const unsigned int panel_size = panel_width * gy;
  const unsigned int panel_idx = block_idx / panel_size;
  const unsigned int panel_offset = block_idx - panel_idx * panel_size;
  const unsigned int grid_size = gx * gy;
  const unsigned int total_panel = (grid_size + panel_size - 1) / panel_size;
  const unsigned int stride = (panel_idx + 1 < total_panel)
                                  ? panel_width
                                  : (grid_size - panel_idx * panel_size) / gy;
  const unsigned int row_idx =
      (panel_idx & 1) ? gy - 1 - panel_offset / stride : panel_offset / stride;
  const unsigned int col_idx = panel_offset % stride + panel_idx * panel_width;
  return {col_idx, row_idx, blockIdx.z};
}

} // namespace tl
