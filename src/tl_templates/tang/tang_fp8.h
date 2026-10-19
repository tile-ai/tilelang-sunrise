#pragma once
#include <cstdint>

// FP8<->BF16 conversion wrappers need TANG bfloat16 + fp8 types and builtins.
#include <__clang_tang_bf16.h>
#include <__clang_tang_fp8.h>

// Minimal FP8 storage types for the TANG (stcuv2) tcgen5 GEMM path.
//
// The tensor-core MMA consumes FP8 operands purely as raw bytes living in
// shared memory; the actual numeric interpretation (E4M3 / E5M2) is selected by
// the EleType field of the MMA instruction descriptor (see gemm_tcgen05.h and
// cccl/tang/__ptx/instructions/tc_mma.h), NOT by the C++ element type. The
// shared-memory bulk copy (copy_global_shm.h) moves the operands byte-for-byte.
// So a distinct 1-byte storage wrapper per format is all the GEMM path needs.
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
//
// The GEMM operand path only ever moves these as bytes, but a plain
// elementwise kernel (`A[i] = T.cast(x, "float8_e4m3")`) makes codegen emit a
// direct `fp8_e4_t(x)` / `(fp8_e4_t)x` cast, so the scalar types also carry
// numeric converting constructors. They delegate to the vendor builtin
// __tang_cvt_float_to_fp8 (round-to-nearest-even, saturating), which is what
// picks the E4M3 / E5M2 encoding here -- unlike the MMA path, where the
// descriptor's EleType does. The reverse direction (fp8 -> float) needs nothing
// here: codegen emits __tang_cvt_fp8_to_halfraw on the reinterpreted storage.
//
// `__x` stays public and the default constructor stays trivial, so raw-byte use
// (`fp8_e4_t v; v.__x = b;`, reinterpret_cast to __tang_fp8_storage_t, arrays
// of the type) is unaffected and the types remain 1-byte and trivially
// copyable. Following the vendor's own __tang_fp8_e4m3, the conversions are
// `explicit` and there is deliberately no `unsigned char` overload, which would
// be ambiguous with raw-byte storage.
#define TL_TANG_FP8_SCALAR_DEF(NAME, INTERP)                                   \
  struct NAME {                                                                \
    uint8_t __x;                                                               \
                                                                               \
    NAME() = default;                                                          \
                                                                               \
    __device__ explicit NAME(const float val) {                                \
      __x = __tang_cvt_float_to_fp8(val, __TANG_SATFINITE, INTERP);            \
    }                                                                          \
    __device__ explicit NAME(const double val)                                 \
        : NAME(static_cast<float>(val)) {}                                     \
    __device__ explicit NAME(const short int val)                              \
        : NAME(static_cast<float>(val)) {}                                     \
    __device__ explicit NAME(const unsigned short int val)                     \
        : NAME(static_cast<float>(val)) {}                                     \
    __device__ explicit NAME(const int val) : NAME(static_cast<float>(val)) {} \
    __device__ explicit NAME(const unsigned int val)                           \
        : NAME(static_cast<float>(val)) {}                                     \
    __device__ explicit NAME(const long int val)                               \
        : NAME(static_cast<float>(val)) {}                                     \
    __device__ explicit NAME(const unsigned long int val)                      \
        : NAME(static_cast<float>(val)) {}                                     \
    __device__ explicit NAME(const long long int val)                          \
        : NAME(static_cast<float>(val)) {}                                     \
    __device__ explicit NAME(const unsigned long long int val)                 \
        : NAME(static_cast<float>(val)) {}                                     \
                                                                               \
    /* Non-finite values must bypass the constructors above: those use         \
     * __TANG_SATFINITE (matching the vendor fp8 types), which clamps an       \
     * infinite input down to MAXNORM -- right for arithmetic overflow, wrong  \
     * for materialising an explicit infinity. __TANG_NOSAT preserves e5m2's   \
     * inf/nan encodings; e4m3 has no infinity, so the vendor conversion       \
     * yields NaN there (documented in __clang_tang_fp8.h). Static factory     \
     * rather than a constructor overload so it cannot be picked up by         \
     * implicit conversion, and so the type stays trivially copyable. */       \
    __device__ static NAME from_float_nosat(const float val) {                 \
      NAME out;                                                                \
      out.__x = __tang_cvt_float_to_fp8(val, __TANG_NOSAT, INTERP);            \
      return out;                                                              \
    }                                                                          \
  }

TL_TANG_FP8_SCALAR_DEF(fp8_e4_t, __TANG_E4M3);
TL_TANG_FP8_SCALAR_DEF(fp8_e5_t, __TANG_E5M2);

#undef TL_TANG_FP8_SCALAR_DEF

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

// ============================================================================
// FP8 <-> BFloat16 Conversion Wrappers (stcuv2)
// ============================================================================
// These wrappers call the TANG compiler builtins (__tang_cvt_*), which the
// ptcc compiler recognises without an explicit #include <__clang_tang_fp8.h>.
//
// bf16 -> fp8  (direct HW: CVT_e4m3_bf16_rn / CVT_e5m2_bf16_rn)
// fp8  -> bf16 (composed: fp8->half->float->bf16, all HW instructions)

// bfloat162 -> fp8x2 (direct: __tang_cvt_bfloat16raw2_to_fp8x2)
typedef unsigned char __tl_fp8x2_storage_t;
static inline __tl_fp8x2_storage_t
__tl_cvt_bfloat162_to_fp8x2(const __tang_bfloat162 src,
                            const __tang_fp8_interpretation_t interp) {
  // Use __tang_bfloat162's `operator __tang_bfloat162_raw()` rather than a
  // reinterpret_cast: the latter type-puns __tang_bfloat162 as its _raw POD and
  // is strict-aliasing UB. The conversion operator is the vendor's intended
  // (and equivalent) path and is well-defined.
  __tang_bfloat162_raw raw = src;
  return __tang_cvt_bfloat16raw2_to_fp8x2(raw, __TANG_SATFINITE, interp);
}

// fp8_e4m3x2 -> bfloat162 (composed: fp8->half->float->bf16, all HW)
static inline __tang_bfloat162
__tl_cvt_e4m3x2_to_bfloat162(const __tl_fp8x2_storage_t src) {
  __half2_raw h = __tang_cvt_fp8x2_to_halfraw2(src, __TANG_E4M3);
  // half2(h) invokes __half2's non-explicit `__half2(const __half2_raw&)` ctor
  // instead of reinterpret_cast-ing the _raw POD to half2 (strict-aliasing UB).
  float2 f = __half22float2(half2(h));
  return __float22bfloat162_rn(f);
}

// fp8_e5m2x2 -> bfloat162 (composed: fp8->half->float->bf16, all HW)
static inline __tang_bfloat162
__tl_cvt_e5m2x2_to_bfloat162(const __tl_fp8x2_storage_t src) {
  __half2_raw h = __tang_cvt_fp8x2_to_halfraw2(src, __TANG_E5M2);
  // half2(h): non-explicit __half2(const __half2_raw&) ctor, not a UB pun.
  float2 f = __half22float2(half2(h));
  return __float22bfloat162_rn(f);
}
