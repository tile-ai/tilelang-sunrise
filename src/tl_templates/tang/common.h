#pragma once

#ifndef __TANGCC_RTC__
#include <tang_runtime.h>
#endif

#include <stdio.h>

#define TANGRT_INF_F __int_as_float(0x7f800000U)
#define TANGRT_NAN_F __int_as_float(0x7fffffffU)
#define TANGRT_INF __longlong_as_double(0x7ff0000000000000ULL)
#define TANGRT_NAN __longlong_as_double(0xfff8000000000000ULL)

#define uint unsigned int
#define uchar unsigned char
#define ushort unsigned short

#define TL_DEVICE __forceinline__ __device__
#define TL_DEVICE_NOINLINE __noinline__ __device__
#define TL_PATCH

#define NOT_COMPILATION

#define TILELANG_CHECK(stmt)                                                   \
  do {                                                                         \
    tangError_t __err = (stmt);                                                \
    if (__err != tangSuccess) {                                                \
      snprintf(error_buf, ERROR_BUF_SIZE, "%s:%d: %s - %s", __FILE__,          \
               __LINE__, tangGetErrorName(__err), tangGetErrorString(__err));  \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define TILELANG_CHECK_LAST_ERROR(kernel_name)                                 \
  do {                                                                         \
    tangError_t __err = tangGetLastError();                                    \
    if (__err != tangSuccess) {                                                \
      snprintf(error_buf, ERROR_BUF_SIZE, kernel_name ": %s - %s",             \
               tangGetErrorName(__err), tangGetErrorString(__err));            \
      return -1;                                                               \
    }                                                                          \
  } while (0)

// Bridge half/BF16 operations that PTCC exposes only for float32.
TL_DEVICE half hpow(half x, half y) { return half(powf(float(x), float(y))); }

TL_DEVICE __bf16 hpow(__bf16 x, __bf16 y) {
  return __bf16(powf(float(x), float(y)));
}

TL_DEVICE half hfmod(half x, half y) { return half(fmodf(float(x), float(y))); }

TL_DEVICE __bf16 hfmod(__bf16 x, __bf16 y) {
  return __bf16(fmodf(float(x), float(y)));
}

TL_DEVICE half hsinh(half x) { return half(sinhf(float(x))); }

TL_DEVICE __bf16 hsinh(__bf16 x) { return __bf16(sinhf(float(x))); }

TL_DEVICE half hcosh(half x) { return half(coshf(float(x))); }

TL_DEVICE __bf16 hcosh(__bf16 x) { return __bf16(coshf(float(x))); }

TL_DEVICE half hatan(half x) { return half(atanf(float(x))); }

TL_DEVICE __bf16 hatan(__bf16 x) { return __bf16(atanf(float(x))); }

TL_DEVICE half herf(half x) { return half(erff(float(x))); }

TL_DEVICE __bf16 herf(__bf16 x) { return __bf16(erff(float(x))); }

TL_DEVICE half hnearbyint(half x) { return half(nearbyintf(float(x))); }

TL_DEVICE __bf16 hnearbyint(__bf16 x) { return __bf16(nearbyintf(float(x))); }

TL_DEVICE half htanh(half x) { return half(tanhf(float(x))); }

TL_DEVICE __bf16 htanh(__bf16 x) { return __bf16(tanhf(float(x))); }

// Pack two half values.
TL_DEVICE unsigned __pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// Pack two __bf16 values.
TL_DEVICE unsigned __pack_bfloat162(const __bf16 x, const __bf16 y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// Pack four char values
TL_DEVICE int make_int(signed char x0, signed char x1, signed char x2,
                       signed char x3) {
  return (x3 << 24) | (x2 << 16) | (x1 << 8) | x0;
}

// Pack four unsigned char values.
TL_DEVICE unsigned int make_uint(unsigned char x0, unsigned char x1,
                                 unsigned char x2, unsigned char x3) {
  return (x3 << 24) | (x2 << 16) | (x1 << 8) | x0;
}

namespace tl {
TL_DEVICE float RoundTiesAwayFromZero(float x) { return roundf(x); }

TL_DEVICE double RoundTiesAwayFromZero(double x) { return round(x); }

TL_DEVICE half RoundTiesAwayFromZero(half x) { return half(roundf(float(x))); }

TL_DEVICE __bf16 RoundTiesAwayFromZero(__bf16 x) {
  return __bf16(roundf(float(x)));
}

// Any
template <typename T> TL_DEVICE bool Any(T *a, int size) {
  for (int i = 0; i < size; i++) {
    if (a[i]) {
      return true;
    }
  }
  return false;
}

// All
template <typename T> TL_DEVICE bool All(T *a, int size) {
  for (int i = 0; i < size; i++) {
    if (!a[i]) {
      return false;
    }
  }
  return true;
}

// Pow of int
template <int y = 1, typename T> TL_DEVICE T pow_of_int(T x) {
  T result = x;
  for (int i = 1; i < y; i++) {
    result *= x;
  }
  return result;
}

} // namespace tl
