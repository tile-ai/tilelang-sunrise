#pragma once

namespace tileops {

__device__ __forceinline__ void threadfence_all() { __threadfence_all(); }

}  // namespace tileops
