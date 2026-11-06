#pragma once

namespace tileops {
__device__ __forceinline__ void load16(void* dst, const void* src) {
  __builtin_memcpy(dst, src, 16);
}
__device__ __forceinline__ void load16_evict_first(void* dst, const void* src) {
  load16(dst, src);
}
__device__ __forceinline__ float load_f32_evict_last(const float* src) {
  return *src;
}
__device__ __forceinline__ unsigned vmaxu2(unsigned a, unsigned b) {
  unsigned lo = (a & 65535u) > (b & 65535u) ? (a & 65535u) : (b & 65535u);
  unsigned hi = (a >> 16) > (b >> 16) ? (a >> 16) : (b >> 16);
  return lo | (hi << 16);
}
}  // namespace tileops
