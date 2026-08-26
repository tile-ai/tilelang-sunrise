#pragma once

#include "common.h"

#include <stdint.h>

namespace tl {

// =============================================================================
// Shared helpers
// =============================================================================
//
// splitmix64-style state mixer. Used to derive per-thread initial state for
// every generator below from (seed, seq, off). Same triple => same state,
// different triples => avalanche-mixed independent state.
TL_DEVICE unsigned long long rng_mix64(unsigned long long x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

// Box-Muller transform shared by every normal_float_* helper. The lower bound
// on u1 prevents log(0) producing inf when the uniform happens to round to 0.
TL_DEVICE float rng_box_muller_float(float u1, float u2) {
  u1 = u1 > 1.1754943508222875e-38f ? u1 : 1.1754943508222875e-38f;
  return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530717958647692f * u2);
}

// =============================================================================
// Philox4_32_10 (Salmon et al. 2011; matches curandStatePhilox4_32_10_t).
//
// Counter-based RNG. State is a 128-bit counter + 64-bit key; each step does
// 10 rounds of (multiply-high-low + xor + key-bump) and produces 4 uint32
// outputs which we cache and dispense one-by-one.
//
// Algorithm constants are public (cppreference std::philox_engine, Random123
// reference). Initialisation derivation is splitmix64 from (seed, seq, off),
// NOT curand's internal skip-ahead -- so we do not promise bit-exact equality
// with curand even though the per-counter algorithm is identical.
// =============================================================================

struct TangRNGStatePhilox {
  unsigned int counter[4];
  unsigned int key[2];
  unsigned int out[4];
  int idx; // 0..3; 4 means buffer exhausted, refill next call.
};

TL_DEVICE void rng_philox_round(unsigned int *ctr, unsigned int *key) {
  const unsigned int kPhiloxM0 = 0xD2511F53u;
  const unsigned int kPhiloxM1 = 0xCD9E8D57u;
  unsigned long long p0 = (unsigned long long)kPhiloxM0 * ctr[0];
  unsigned long long p1 = (unsigned long long)kPhiloxM1 * ctr[2];
  unsigned int lo0 = static_cast<unsigned int>(p0);
  unsigned int hi0 = static_cast<unsigned int>(p0 >> 32);
  unsigned int lo1 = static_cast<unsigned int>(p1);
  unsigned int hi1 = static_cast<unsigned int>(p1 >> 32);
  unsigned int new0 = hi1 ^ ctr[1] ^ key[0];
  unsigned int new1 = lo1;
  unsigned int new2 = hi0 ^ ctr[3] ^ key[1];
  unsigned int new3 = lo0;
  ctr[0] = new0;
  ctr[1] = new1;
  ctr[2] = new2;
  ctr[3] = new3;
}

TL_DEVICE void rng_philox_bumpkey(unsigned int *key) {
  key[0] += 0x9E3779B9u; // golden ratio
  key[1] += 0xBB67AE85u; // sqrt(3) - 1
}

TL_DEVICE void rng_philox_refill(TangRNGStatePhilox *rng) {
  unsigned int ctr[4] = {rng->counter[0], rng->counter[1], rng->counter[2],
                         rng->counter[3]};
  unsigned int key[2] = {rng->key[0], rng->key[1]};
  // 10 rounds.
  for (int i = 0; i < 9; ++i) {
    rng_philox_round(ctr, key);
    rng_philox_bumpkey(key);
  }
  rng_philox_round(ctr, key);
  rng->out[0] = ctr[0];
  rng->out[1] = ctr[1];
  rng->out[2] = ctr[2];
  rng->out[3] = ctr[3];
  // Advance 128-bit input counter for the next refill.
  if (++rng->counter[0] == 0u)
    if (++rng->counter[1] == 0u)
      if (++rng->counter[2] == 0u)
        ++rng->counter[3];
  rng->idx = 0;
}

TL_DEVICE void rng_init_philox(TangRNGStatePhilox *rng, unsigned long long seed,
                               unsigned long long seq, unsigned long long off) {
  unsigned long long w0 = rng_mix64(seed ^ 0xa0761d6478bd642fULL);
  unsigned long long w1 = rng_mix64(seq ^ 0xe7037ed1a0b428dbULL);
  unsigned long long w2 = rng_mix64(off ^ 0x8ebc6af09c88c6e3ULL);
  rng->key[0] = static_cast<unsigned int>(w0);
  rng->key[1] = static_cast<unsigned int>(w0 >> 32);
  rng->counter[0] = static_cast<unsigned int>(w1);
  rng->counter[1] = static_cast<unsigned int>(w1 >> 32);
  rng->counter[2] = static_cast<unsigned int>(w2);
  rng->counter[3] = static_cast<unsigned int>(w2 >> 32);
  rng->idx = 4; // force refill on first draw
}

TL_DEVICE unsigned int rng_rand_philox(TangRNGStatePhilox *rng) {
  if (rng->idx >= 4)
    rng_philox_refill(rng);
  return rng->out[rng->idx++];
}

TL_DEVICE float rng_uniform_float_philox(TangRNGStatePhilox *rng) {
  // 24-bit mantissa -> float in [0, 1).
  return static_cast<float>(rng_rand_philox(rng) >> 8) * (1.0f / 16777216.0f);
}

TL_DEVICE float rng_normal_float_philox(TangRNGStatePhilox *rng) {
  float u1 = rng_uniform_float_philox(rng);
  float u2 = rng_uniform_float_philox(rng);
  return rng_box_muller_float(u1, u2);
}

// =============================================================================
// MRG32k3a (L'Ecuyer 1999; matches curandStateMRG32k3a_t).
//
// Combined multiple-recursive generator: two parallel order-3 recursions
// modulo m1=2^32-209 and m2=2^32-22853, output = (s1 - s2) mod m1.
//
// Tang's ptcc treats double as float, so the classical double-mod
// implementation loses precision. We use an integer formulation that stays in
// uint64 arithmetic with explicit mod, and converts the 32-bit result to float
// at the end. Algorithm identical to curand; initialisation derived from
// splitmix64 so NOT bit-exact with curand's skip-ahead.
// =============================================================================

struct TangRNGStateMRG32k3a {
  unsigned long long s10, s11, s12;
  unsigned long long s20, s21, s22;
};

#define TL_MRG_M1 4294967087ULL // 2^32 - 209
#define TL_MRG_M2 4294944443ULL // 2^32 - 22853

TL_DEVICE unsigned int rng_rand_mrg32k3a(TangRNGStateMRG32k3a *rng) {
  // Component 1: x1n = (1403580 * s11 - 810728 * s10) mod m1
  // Rewritten to keep operands non-negative: add a multiple of m1 before sub.
  unsigned long long p1 =
      1403580ULL * rng->s11 +
      (TL_MRG_M1 * 810728ULL - 810728ULL * rng->s10) % TL_MRG_M1;
  p1 %= TL_MRG_M1;
  rng->s10 = rng->s11;
  rng->s11 = rng->s12;
  rng->s12 = p1;
  // Component 2: x2n = (527612 * s22 - 1370589 * s20) mod m2
  unsigned long long p2 =
      527612ULL * rng->s22 +
      (TL_MRG_M2 * 1370589ULL - 1370589ULL * rng->s20) % TL_MRG_M2;
  p2 %= TL_MRG_M2;
  rng->s20 = rng->s21;
  rng->s21 = rng->s22;
  rng->s22 = p2;
  // Output: (p1 - p2) mod m1, mapped to uint32.
  unsigned long long out;
  if (p1 <= p2)
    out = p1 + TL_MRG_M1 - p2;
  else
    out = p1 - p2;
  return static_cast<unsigned int>(out);
}

TL_DEVICE void rng_init_mrg32k3a(TangRNGStateMRG32k3a *rng,
                                 unsigned long long seed,
                                 unsigned long long seq,
                                 unsigned long long off) {
  // Derive six non-zero seeds, taken mod (m-1) and shifted by 1 to avoid the
  // degenerate all-zero state. Avalanche each input separately.
  unsigned long long w0 = rng_mix64(seed ^ 0x6a09e667f3bcc908ULL);
  unsigned long long w1 = rng_mix64(seed + 0xbb67ae8584caa73bULL);
  unsigned long long w2 = rng_mix64(seq ^ 0x3c6ef372fe94f82bULL);
  unsigned long long w3 = rng_mix64(seq + 0xa54ff53a5f1d36f1ULL);
  unsigned long long w4 = rng_mix64(off ^ 0x510e527fade682d1ULL);
  unsigned long long w5 = rng_mix64(off + 0x9b05688c2b3e6c1fULL);
  rng->s10 = (w0 % (TL_MRG_M1 - 1ULL)) + 1ULL;
  rng->s11 = (w1 % (TL_MRG_M1 - 1ULL)) + 1ULL;
  rng->s12 = (w2 % (TL_MRG_M1 - 1ULL)) + 1ULL;
  rng->s20 = (w3 % (TL_MRG_M2 - 1ULL)) + 1ULL;
  rng->s21 = (w4 % (TL_MRG_M2 - 1ULL)) + 1ULL;
  rng->s22 = (w5 % (TL_MRG_M2 - 1ULL)) + 1ULL;
}

TL_DEVICE float rng_uniform_float_mrg32k3a(TangRNGStateMRG32k3a *rng) {
  // 24-bit mantissa -> float in [0, 1). Output range is [0, m1-1].
  return static_cast<float>(rng_rand_mrg32k3a(rng) >> 8) * (1.0f / 16777216.0f);
}

TL_DEVICE float rng_normal_float_mrg32k3a(TangRNGStateMRG32k3a *rng) {
  float u1 = rng_uniform_float_mrg32k3a(rng);
  float u2 = rng_uniform_float_mrg32k3a(rng);
  return rng_box_muller_float(u1, u2);
}

#undef TL_MRG_M1
#undef TL_MRG_M2

// =============================================================================
// XORWOW (Marsaglia 2003; matches curandStateXORWOW_t).
//
// 5-word xorshift register + Weyl-style counter. Period ~ 2^192. Algorithm
// is fully public; initialisation derived from splitmix64 so NOT bit-exact
// with curand's skip-ahead.
// =============================================================================

struct TangRNGStateXORWOW {
  unsigned int x[5];
  unsigned int d;
};

TL_DEVICE unsigned int rng_rand_xorwow(TangRNGStateXORWOW *rng) {
  unsigned int t = rng->x[0] ^ (rng->x[0] >> 2);
  rng->x[0] = rng->x[1];
  rng->x[1] = rng->x[2];
  rng->x[2] = rng->x[3];
  rng->x[3] = rng->x[4];
  rng->x[4] = (rng->x[4] ^ (rng->x[4] << 4)) ^ (t ^ (t << 1));
  rng->d += 362437u;
  return rng->x[4] + rng->d;
}

TL_DEVICE void rng_init_xorwow(TangRNGStateXORWOW *rng, unsigned long long seed,
                               unsigned long long seq, unsigned long long off) {
  unsigned long long w0 = rng_mix64(seed ^ 0x243f6a8885a308d3ULL);
  unsigned long long w1 = rng_mix64(seq ^ 0x13198a2e03707344ULL);
  unsigned long long w2 = rng_mix64(off ^ 0xa4093822299f31d0ULL);
  rng->x[0] = static_cast<unsigned int>(w0);
  rng->x[1] = static_cast<unsigned int>(w0 >> 32);
  rng->x[2] = static_cast<unsigned int>(w1);
  rng->x[3] = static_cast<unsigned int>(w1 >> 32);
  rng->x[4] = static_cast<unsigned int>(w2);
  rng->d = static_cast<unsigned int>(w2 >> 32);
  // Guard against the all-zero degenerate state.
  if ((rng->x[0] | rng->x[1] | rng->x[2] | rng->x[3] | rng->x[4]) == 0u) {
    rng->x[0] = 1u;
  }
}

TL_DEVICE float rng_uniform_float_xorwow(TangRNGStateXORWOW *rng) {
  return static_cast<float>(rng_rand_xorwow(rng) >> 8) * (1.0f / 16777216.0f);
}

TL_DEVICE float rng_normal_float_xorwow(TangRNGStateXORWOW *rng) {
  float u1 = rng_uniform_float_xorwow(rng);
  float u2 = rng_uniform_float_xorwow(rng);
  return rng_box_muller_float(u1, u2);
}

} // namespace tl
