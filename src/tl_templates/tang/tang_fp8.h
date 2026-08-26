#pragma once
#include <cstdint>

// Minimal FP8 storage types for the TANG (stcuv2) tcgen5 GEMM path.
//
// The tensor-core MMA consumes FP8 operands purely as raw bytes living in
// shared memory; the actual numeric interpretation (E4M3 / E5M2) is selected by
// the EleType field of the MMA instruction descriptor (see gemm_tcgen5.h and
// cccl/tang/__ptx/instructions/tc_mma.h), NOT by the C++ element type. The
// shared-memory bulk copy (copy_fcp_g_s.h) moves the operands byte-for-byte. So
// a distinct 1-byte storage wrapper per format is all the GEMM path needs.
//
// The wrappers are intentionally *distinct* C++ types (not (u)int8_t) so the
// tcgen5_ele_type<> trait can map them to eFP8_E4M3 / eFP8_E5M2 instead of the
// integer eS8 / eU8.
//
// codegen_tang.cc (GetFP8Type) emits these exact names: scalar `fp8_e4_t` /
// `fp8_e5_t` and packed `fp8_e{4,5}_{2,4,8,16,32}_t`. They are declared in the
// global namespace because codegen emits them unqualified.

#define TL_TANG_FP8_DEF(NAME, NLANES, ALIGN)                                   \
  struct __attribute__((aligned(ALIGN))) NAME {                                \
    uint8_t __x[NLANES];                                                       \
  }

// Scalar (1 byte).
struct fp8_e4_t {
  uint8_t __x;
};
struct fp8_e5_t {
  uint8_t __x;
};

// Packed vector variants (aligned to their byte width, capped at 16B).
TL_TANG_FP8_DEF(fp8_e4_2_t, 2, 2);
TL_TANG_FP8_DEF(fp8_e4_4_t, 4, 4);
TL_TANG_FP8_DEF(fp8_e4_8_t, 8, 8);
TL_TANG_FP8_DEF(fp8_e4_16_t, 16, 16);
TL_TANG_FP8_DEF(fp8_e4_32_t, 32, 16);
TL_TANG_FP8_DEF(fp8_e5_2_t, 2, 2);
TL_TANG_FP8_DEF(fp8_e5_4_t, 4, 4);
TL_TANG_FP8_DEF(fp8_e5_8_t, 8, 8);
TL_TANG_FP8_DEF(fp8_e5_16_t, 16, 16);
TL_TANG_FP8_DEF(fp8_e5_32_t, 32, 16);

#undef TL_TANG_FP8_DEF

static_assert(sizeof(fp8_e4_t) == 1, "fp8_e4_t must be 1 byte");
static_assert(sizeof(fp8_e5_t) == 1, "fp8_e5_t must be 1 byte");
