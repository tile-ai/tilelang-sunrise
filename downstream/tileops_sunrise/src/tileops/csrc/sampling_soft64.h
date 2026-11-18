#pragma once

#include <stdint.h>
#if defined(TILEOPS_SOFT64_HOST_TEST)
#define TILEOPS_S64_INLINE inline
#else
#define TILEOPS_S64_INLINE __device__ __forceinline__
#endif
namespace tileops {
namespace sampling_soft64 {
static_assert(sizeof(int64_t) == 8 && sizeof(uint64_t) == 8 &&
                  sizeof(float) == 4,
              "sampling CDF words must have explicit 8-byte storage");
TILEOPS_S64_INLINE int64_t word(uint64_t u) {
  union {
    uint64_t u;
    int64_t i;
  } v;
  v.u = u;
  return v.i;
}
TILEOPS_S64_INLINE bool nan(uint64_t u) {
  return (u & UINT64_C(0x7fffffffffffffff)) > UINT64_C(0x7ff0000000000000);
}
TILEOPS_S64_INLINE uint64_t jam(uint64_t u, unsigned n) {
  if (n == 0) return u;
  if (n >= 64) return u != 0;
  return (u >> n) | ((u << (64 - n)) != 0);
}
// sig has three guard/round/sticky bits and an effective biased exponent >= 1.
TILEOPS_S64_INLINE int64_t pack(uint64_t sign, int exponent, uint64_t sig) {
  uint64_t low = sig & 7;
  uint64_t mantissa = sig >> 3;
  if (low > 4 || (low == 4 && (mantissa & 1))) ++mantissa;
  if (mantissa >= (UINT64_C(1) << 53)) {
    mantissa >>= 1;
    ++exponent;
  }
  if (exponent >= 2047) return word(sign | UINT64_C(0x7ff0000000000000));
  if (exponent == 1 && mantissa < (UINT64_C(1) << 52)) exponent = 0;
  return word(sign | (uint64_t(exponent) << 52) |
              (mantissa & UINT64_C(0x000fffffffffffff)));
}
TILEOPS_S64_INLINE int64_t from_f32(float x) {
  union {
    float f;
    uint32_t u;
  } v;
  v.f = x;
  uint64_t sign = uint64_t(v.u >> 31) << 63;
  unsigned exponent = (v.u >> 23) & 255;
  uint32_t mantissa = v.u & 0x7fffffU;
  if (exponent == 255)
    return word(sign | UINT64_C(0x7ff0000000000000) |
                (uint64_t(mantissa) << 29) |
                (mantissa ? UINT64_C(0x0008000000000000) : 0));
  if (exponent == 0) {
    if (mantissa == 0) return word(sign);
    int e = -126;
    while ((mantissa & 0x800000U) == 0) {
      mantissa <<= 1;
      --e;
    }
    return word(sign | (uint64_t(e + 1023) << 52) |
                (uint64_t(mantissa & 0x7fffffU) << 29));
  }
  return word(sign | (uint64_t(int(exponent) - 127 + 1023) << 52) |
              (uint64_t(mantissa) << 29));
}
TILEOPS_S64_INLINE int64_t add(int64_t a, int64_t b) {
  uint64_t x = uint64_t(a), y = uint64_t(b);
  const uint64_t sign_mask = UINT64_C(0x8000000000000000);
  const uint64_t fraction = UINT64_C(0x000fffffffffffff);
  if (nan(x)) return word(x | UINT64_C(0x0008000000000000));
  if (nan(y)) return word(y | UINT64_C(0x0008000000000000));
  uint64_t ax = x & ~sign_mask, ay = y & ~sign_mask;
  if (ax == UINT64_C(0x7ff0000000000000)) {
    if (ay == ax && ((x ^ y) & sign_mask))
      return word(UINT64_C(0x7ff8000000000000));
    return word(x);
  }
  if (ay == UINT64_C(0x7ff0000000000000)) return word(y);
  if (ax == 0 && ay == 0) return word(x & y & sign_mask);
  if (ax == 0) return word(y);
  if (ay == 0) return word(x);
  if (ax < ay) {
    uint64_t t = x;
    x = y;
    y = t;
  }
  int ex = int((x >> 52) & 2047), ey = int((y >> 52) & 2047);
  uint64_t sx = ((x & fraction) | (ex ? UINT64_C(1) << 52 : 0)) << 3;
  uint64_t sy = ((y & fraction) | (ey ? UINT64_C(1) << 52 : 0)) << 3;
  if (ex == 0) ex = 1;
  if (ey == 0) ey = 1;
  sy = jam(sy, unsigned(ex - ey));
  uint64_t sig;
  if (((x ^ y) & sign_mask) == 0) {
    sig = sx + sy;
    if (sig & (UINT64_C(1) << 56)) {
      sig = jam(sig, 1);
      ++ex;
    }
  } else {
    sig = sx - sy;
    if (sig == 0) return 0;
    while (sig < (UINT64_C(1) << 55) && ex > 1) {
      sig <<= 1;
      --ex;
    }
  }
  return pack(x & sign_mask, ex, sig);
}
TILEOPS_S64_INLINE int64_t sub(int64_t a, int64_t b) {
  return add(a, word(uint64_t(b) ^ UINT64_C(0x8000000000000000)));
}
TILEOPS_S64_INLINE int64_t mul(int64_t a, int64_t b) {
  uint64_t x = uint64_t(a), y = uint64_t(b);
  const uint64_t mask = UINT64_C(0x000fffffffffffff);
  uint64_t sign = (x ^ y) & UINT64_C(0x8000000000000000);
  if (nan(x)) return word(x | UINT64_C(0x0008000000000000));
  if (nan(y)) return word(y | UINT64_C(0x0008000000000000));
  int ex = int((x >> 52) & 2047), ey = int((y >> 52) & 2047);
  uint64_t sx = x & mask, sy = y & mask;
  bool zx = ex == 0 && sx == 0, zy = ey == 0 && sy == 0;
  if (ex == 2047 || ey == 2047) {
    if (zx || zy) return word(UINT64_C(0x7ff8000000000000));
    return word(sign | UINT64_C(0x7ff0000000000000));
  }
  if (zx || zy) return word(sign);
  if (ex) {
    sx |= UINT64_C(1) << 52;
    ex -= 1023;
  } else {
    ex = -1022;
    while (sx < (UINT64_C(1) << 52)) {
      sx <<= 1;
      --ex;
    }
  }
  if (ey) {
    sy |= UINT64_C(1) << 52;
    ey -= 1023;
  } else {
    ey = -1022;
    while (sy < (UINT64_C(1) << 52)) {
      sy <<= 1;
      --ey;
    }
  }
  // Exact 53x53 product, using four 32-bit-limb products and explicit carry.
  uint64_t p0 = (sx & 0xffffffffU) * (sy & 0xffffffffU);
  uint64_t p1 = (sx & 0xffffffffU) * (sy >> 32);
  uint64_t p2 = (sx >> 32) * (sy & 0xffffffffU);
  uint64_t p3 = (sx >> 32) * (sy >> 32);
  uint64_t middle = (p0 >> 32) + (p1 & 0xffffffffU) + (p2 & 0xffffffffU);
  uint64_t lo = (middle << 32) | (p0 & 0xffffffffU);
  uint64_t hi = p3 + (p1 >> 32) + (p2 >> 32) + (middle >> 32);
  unsigned shift = (hi & (UINT64_C(1) << 41)) ? 50 : 49;
  int exponent = ex + ey + 1023 + int(shift == 50);
  uint64_t sig =
      (hi << (64 - shift)) | (lo >> shift) | ((lo << (64 - shift)) != 0);
  if (exponent < 1) {
    sig = jam(sig, unsigned(1 - exponent));
    exponent = 1;
  }
  return pack(sign, exponent, sig);
}
TILEOPS_S64_INLINE bool gt(int64_t a, int64_t b) {
  uint64_t x = uint64_t(a), y = uint64_t(b);
  if (nan(x) || nan(y)) return false;
  if (((x | y) & UINT64_C(0x7fffffffffffffff)) == 0) return false;
  if ((x ^ y) >> 63) return (y >> 63) != 0;
  return (x >> 63) ? x < y : x > y;
}
// T.max(point,+0): ignore NaN, prefer +0 over -0, retain positive infinity.
TILEOPS_S64_INLINE int64_t max_zero(int64_t a) { return gt(a, 0) ? a : 0; }
}  // namespace sampling_soft64
}  // namespace tileops
#undef TILEOPS_S64_INLINE
