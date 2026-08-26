#pragma once

#ifndef __TANGCC_RTC__
#include <tang_runtime.h>
#endif

#include <stdio.h>

#define TANGRT_INF_F __int_as_float(0x7f800000U)
#define TANGRT_NAN_F __int_as_float(0x7fffffffU)
#define TANGRT_INF __longlong_as_double(0x7ff0000000000000ULL)
#define TANGRT_NAN __longlong_as_double(0xfff8000000000000ULL)

#define hpow powf

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
