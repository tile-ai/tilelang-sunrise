#pragma once

#include <type_traits>

#include "barrier.h"
#include "common.h"
#include "tang_fp8.h"
#include <cccl/tang/ptx>

namespace tl {

// ---- A/B element data type -> tang::ptx::EleType ----
template <typename T> struct tcgen5_ele_type;
template <> struct tcgen5_ele_type<__fp16> {
  static constexpr tang::ptx::EleType value = tang::ptx::eFP16;
};
template <> struct tcgen5_ele_type<__bf16> {
  static constexpr tang::ptx::EleType value = tang::ptx::eBF16;
};
template <> struct tcgen5_ele_type<float> {
  static constexpr tang::ptx::EleType value = tang::ptx::eTF32;
};
template <> struct tcgen5_ele_type<int8_t> {
  static constexpr tang::ptx::EleType value = tang::ptx::eS8;
};
template <> struct tcgen5_ele_type<uint8_t> {
  static constexpr tang::ptx::EleType value = tang::ptx::eU8;
};
template <> struct tcgen5_ele_type<fp8_e4_t> {
  static constexpr tang::ptx::EleType value = tang::ptx::eFP8_E4M3;
};
template <> struct tcgen5_ele_type<fp8_e5_t> {
  static constexpr tang::ptx::EleType value = tang::ptx::eFP8_E5M2;
};

// ---- C accumulator data type -> tang::ptx::TmemType ----
template <typename T> struct tcgen5_tmem_type;
template <> struct tcgen5_tmem_type<float> {
  static constexpr tang::ptx::TmemType value = tang::ptx::tFP32;
};
template <> struct tcgen5_tmem_type<__fp16> {
  static constexpr tang::ptx::TmemType value = tang::ptx::tFP16;
};
template <> struct tcgen5_tmem_type<__bf16> {
  static constexpr tang::ptx::TmemType value = tang::ptx::tBF16;
};
template <> struct tcgen5_tmem_type<int> {
  static constexpr tang::ptx::TmemType value = tang::ptx::tS32;
};

template <int CELLS>
TL_DEVICE void tang_tmem_stt_cells(uint32_t (&in)[CELLS], uint32_t taddr) {
  // TODO: implement scale-factor staging.
  (void)in;
  (void)taddr;
}

template <int CELLS, typename T>
TL_DEVICE void tang_tmem_st_a_operand(T *frag, uint32_t taddr) {
  // TODO: implement MMA A-operand staging.
  (void)frag;
  (void)taddr;
}

} // namespace tl
