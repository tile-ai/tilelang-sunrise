#pragma once

// GEMM kernel for the STCU tensor cores.
//
// The shared->register load path uses bank-conflict-aware shared-memory
// padding (per-tile where the layout allows it, per-row otherwise) and
// stateless linear offset computation.
// Loads are single-buffered and interleaved with the MMA chain; a
// double-buffered register prefetch was tried and reverted because the B-side
// prefetch corrupts results on the transpose_B=True path (see below) while
// measuring no faster than the single-buffered form.

namespace tl {

// NOTE: Helper utility for compile-time calculation
template <typename T> constexpr T compile_time_max(T first, T second) {
  return (first > second) ? first : second;
}

template <typename T> constexpr T compile_time_min(T first, T second) {
  return (first < second) ? first : second;
}

template <typename A_type, typename B_type, typename C_type, bool TransposeA,
          bool TransposeB>
TL_DEVICE void call_mma(C_type *D, const A_type *A_frag, const B_type *B_frag,
                        const C_type *C_frag) {
  constexpr int index = 2 * TransposeA + TransposeB;
  // FP32 cases
  if constexpr (std::is_same_v<A_type, float> &&
                std::is_same_v<B_type, float>) {
    if constexpr (std::is_same_v<C_type, float>) {
      __tmma_m8n8k4_mma_f32f32f32(D, A_frag, B_frag, C_frag, index, 1);
    } else {
      static_assert(sizeof(C_type) == 0,
                    "Unsupported C_type for fp32 Tensor Core MMA");
    }
  }
  // FP16 cases
  else if constexpr (std::is_same_v<A_type, __fp16> &&
                     std::is_same_v<B_type, __fp16>) {
    if constexpr (std::is_same_v<C_type, float>) {
      __tmma_m8n8k8_mma_f32f16(D, A_frag, B_frag, C_frag, index, 1);
    } else if constexpr (std::is_same_v<C_type, __fp16>) {
      __tmma_m8n8k8_mma_f16f16(D, A_frag, B_frag, C_frag, index);
    } else {
      static_assert(sizeof(C_type) == 0,
                    "Unsupported C_type for fp16 Tensor Core MMA");
    }
  }
  // BF16 cases
  else if constexpr (std::is_same_v<A_type, __bf16> &&
                     std::is_same_v<B_type, __bf16>) {
    if constexpr (std::is_same_v<C_type, float>) {
      __tmma_m8n8k8_mma_f32bf16(D, A_frag, B_frag, C_frag, index, 1);
    } else if constexpr (std::is_same_v<C_type, __bf16>) {
      // NOTE: uses hardcoded 1 (no-transpose) unlike the FP16 path which
      // passes 'index'; required by the __tmma_m8n8k8_mma_bf16bf16 intrinsic.
      __tmma_m8n8k8_mma_bf16bf16(D, A_frag, B_frag, C_frag, 1);
    } else {
      static_assert(sizeof(C_type) == 0,
                    "Unsupported C_type for bf16 Tensor Core MMA");
    }
  }
  // signed int8
  else if constexpr (std::is_same_v<A_type, int8_t> &&
                     std::is_same_v<B_type, int8_t>) {
    auto A_frag_int32 = reinterpret_cast<const int *>(A_frag);
    auto B_frag_int32 = reinterpret_cast<const int *>(B_frag);
    __tmma_m8n8k16_mma_s32s8(D, A_frag_int32, B_frag_int32, C_frag, index, 1);
  }
  // unsigned int8
  else if constexpr (std::is_same_v<A_type, uint8_t> &&
                     std::is_same_v<B_type, uint8_t>) {
    auto A_frag_int32 = reinterpret_cast<const int *>(A_frag);
    auto B_frag_int32 = reinterpret_cast<const int *>(B_frag);
    __tmma_m8n8k16_mma_s32u8(D, A_frag_int32, B_frag_int32, C_frag, index, 1);
  }
  // Unsupported A/B types
  else {
    static_assert(
        sizeof(A_type) == 0 || sizeof(B_type) == 0,
        "Only fp16 or bf16 types supported for A/B in Tensor Core MMA");
  }
}

template <int M, int N, int K, int num_warp_m, int num_warp_n, int stride_a,
          int stride_b, int offset_a, int offset_b, bool TransposeA,
          bool TransposeB, bool clear_accum, typename A_type, typename B_type,
          typename C_type>
class GemmTensorOp {
public:
  // ---- clear_accum: zeroed here, NOT by the hardware ----
  // The flag means "this gemm defines C rather than accumulating into it, so
  // the caller need not zero C first". It is implemented below by an explicit
  // store of 0 over C_local, because the hardware does NOT provide it.
  //
  // An earlier revision of this comment claimed the opposite: that ptcc -O3
  // lowers every chain to a leading `tensor.mul` (D = A*B) which does not read
  // C, so the first MMA "defines" the accumulator and no zero-init is needed.
  // That is wrong, and the reasoning behind it was wrong in a way worth
  // recording, because the mistake is easy to repeat.
  //
  // `tensor.mul` is real and it does start every chain, but it feeds a tensor
  // *latch*, not C. The chain ends in `tensor.macc.out`, which drains the
  // latch, and the compiler then folds the incoming accumulator back in with a
  // separate `add.f32` per C element:
  //     tensor.mul   <- A*B, ignores C   (this is the instruction that misled)
  //     tensor.macc  x (n-2)
  //     tensor.macc.out t26_t27, ...     <- drains the latch
  //     add.f32 t18, t26, w1             <- w1 is the OLD C. C is read here.
  // So C is live across the call regardless of the chain's leading opcode.
  // Reading the tensor-op mix alone cannot show this; the accumulator fold-in
  // is in the scalar `add.f32` count, which that analysis never looked at.
  //
  // The claim that an opaque initial value produces the same behavior as a
  // zero accumulator was an artifact. The uninitialized form still emits the
  // accumulator fold-in and therefore reads whatever C held.
  //
  // A freshly launched kernel can find registers already zero, hiding the
  // missing initialization. Initializing the accumulator to a nonzero sentinel
  // makes the resulting error follow that sentinel, confirming that the old
  // value is read rather than overwritten.
  //
  // Hence the explicit zeroing below. It is not defensive coding: without it
  // the kernel is wrong, and wrong in the way that hides in testing and appears
  // when register state happens to be dirty (a preceding kernel on the same
  // core, different launch geometry, a future toolchain that allocates
  // differently).

  // Shape processed by each MMA instruction.
  static constexpr uint32_t tile_size_m = 8;
  static constexpr uint32_t tile_size_n = 8;
  static constexpr uint32_t tile_size_k = 16 / sizeof(A_type);
  // vec_size: number of A_type/B_type elements per uint32_t word (word = 4
  // bytes)
  static constexpr uint32_t vec_size = 4 / sizeof(A_type);

  // Shape processed by each thread block.
  static constexpr uint32_t M_Tile = M;
  static constexpr uint32_t N_Tile = N;
  static constexpr uint32_t K_Tile = K;

  static constexpr uint32_t inner_k = K_Tile / tile_size_k;
  static constexpr uint32_t warp_rows = M_Tile / (num_warp_m * tile_size_m);
  static constexpr uint32_t warp_cols = N_Tile / (num_warp_n * tile_size_n);

  static constexpr uint32_t warp_size = 32;
  static constexpr uint32_t local_size_a =
      (tile_size_m * tile_size_k) / warp_size;
  static constexpr uint32_t local_size_b =
      (tile_size_n * tile_size_k) / warp_size;
  static constexpr uint32_t local_size_c =
      (tile_size_m * tile_size_n) / warp_size;

  static constexpr uint32_t k_tile_4bytes = (K_Tile / vec_size);
  static constexpr uint32_t offset_a_4bytes = offset_a / vec_size;
  static constexpr uint32_t offset_b_4bytes = offset_b / vec_size;

  static constexpr uint32_t stride_real_a = (stride_a - offset_a);
  static constexpr uint32_t stride_real_b = (stride_b - offset_b);

  static constexpr uint32_t ta = TransposeA ? tile_size_m : tile_size_k;
  static constexpr uint32_t stride_tile_m = stride_real_a / ta;

  static constexpr uint32_t tb = TransposeB ? tile_size_k : tile_size_n;
  static constexpr uint32_t stride_tile_k = stride_real_b / tb;

  // ---- Flat (non-interleaved) shared-to-register load ----
  // `Transpose` controls the tile_idx layout:
  //   - false: A tile_idx has K inner, B tile_idx has N inner
  //   - true:  A tile_idx has M inner, B tile_idx has K inner
  //
  // Addressing is *stateless*: every iteration derives its address from the
  // immutable tile_idx plus compile-time multiples of the step, rather than
  // advancing `tile_idx`/`lin_off` accumulators. The mutating form creates a
  // serial dependency chain across the unrolled body — each iteration's address
  // waits on the previous iteration's adds — and keeps every intermediate live.
  // With i*tile_step and i*lin_step folded to constants by the unroller, the
  // compiler can reuse a temporary for the offset chain and issue the loads
  // independently. Kept for the shorter dependency chain and lower register
  // pressure, not as a throughput claim.
  template <uint32_t kstep, uint32_t Stride, uint32_t TileSizeK,
            uint32_t Offset4Bytes, uint32_t KTile4Bytes, bool isA,
            bool Transpose>
  static TL_DEVICE void LoadFromShared(uint32_t *local_base,
                                       uint32_t *shared_base,
                                       uint32_t tile_idx) {
    constexpr uint32_t tile_step =
        (Transpose == isA) ? (Stride / TileSizeK) : 1;
    constexpr uint32_t lin_step = tile_step << 5; // * 32 (warp_size)
    constexpr bool has_stride_offset = (Offset4Bytes != 0);

    // ---- Shared-memory padding (must match makeTensorCoreSwizzleLayoutTANG in
    // tilelang/tang/op/gemm/gemm_tmma.py) ----
    // Two schemes, selected at compile time; the write side picks the same one
    // per operand, and any divergence silently corrupts that operand.
    //   PER-TILE (gated): a gap of PadWords is inserted after every 32-word
    //     (8x8 fp16) tile, so the physical word is AFFINE in the unroll index:
    //       w = laneid + (tile_idx + i*tile_step) * (32 + PadWords)
    //     and i*tile_step*(32+PadWords) folds into the LDS [imm] field -- no
    //     per-element div/mask. Mirrors the layout's tile-linear offset
    //     tile_idx*(tile_size+pad)+idx_in_tile (words: PadWords =
    //     pad/vec_size).
    //   PER-ROW (fallback): a gap after every `Stride` elements. The correction
    //     w += (w/StrideWords)*PadWords is non-affine for A (the gap crossing
    //     depends on laneid); kept for non-fp16 / offset!=0, where the layout
    //     also still pads per row.
    //
    // The per-tile gate is PER-OPERAND: `isA ? (M_Tile == K_Tile) : (K_Tile ==
    // N_Tile)` is exactly the write side's `stride == inner_continuous` for
    // that operand. A unified gate (e.g. M_Tile == K_Tile for both) splits B's
    // write side (per-tile) from its read side (per-row) at M != N tiles --
    // e.g. 2048's 64x128 A tile is per-row while its 128x128 B tile is
    // per-tile.
    constexpr uint32_t StrideWords = Stride / vec_size;
    // Padding is restricted to 16-bit operands (vec_size == 2). Applying the
    // same scheme to fp32 inflates the shared extent and can exceed the
    // available resource limit. Other element widths stay on their existing
    // unpadded paths.
    // The write side (gemm_tmma.py) gates on element_bits == 16 to match; both
    // sides must agree per operand or that operand is silently corrupted.
    constexpr uint32_t PadWords =
        (vec_size == 2 && StrideWords % 8 == 0) ? uint32_t(4) : uint32_t(0);
    constexpr bool per_tile_pad =
        (vec_size == 2) && !has_stride_offset && (PadWords != 0) &&
        (isA ? (M_Tile == K_Tile) : (K_Tile == N_Tile));

    if constexpr (per_tile_pad) {
      constexpr uint32_t phys_tile = 32 + PadWords;
      const uint32_t base = __laneid() + tile_idx * phys_tile;
#pragma unroll
      for (uint32_t idx = 0; idx < kstep; idx++) {
        local_base[idx] = shared_base[base + idx * (tile_step * phys_tile)];
      }
      return;
    }

    // ---- Warp-ALU pipeline (fp16, StrideWords==32) ----
    // tile_idx is warp-uniform (derived from warpid / k-loop index), so its
    // padded base tile_idx*(32+PadWords)=tile_idx*36=(t<<5)+(t<<2) pipelines on
    // the warp scalar ALU in parallel with the per-lane laneid work on the
    // vector ALU. The pre-add correction folds the per-tile padding into
    // uni_base, which is only exact when StrideWords==32 (laneid∈[0,32) stays
    // within one tile row). For 1024³ this eliminates the per-element div/mask
    // that the fallback path pays inside the unrolled loop.
    if constexpr (!has_stride_offset && StrideWords == 32 && PadWords == 4) {
      constexpr uint32_t step_pad = (lin_step / StrideWords) * PadWords;
      uint32_t uni_base = (tile_idx << 5) + (tile_idx << 2);
      uint32_t base_off = __laneid() + uni_base;
#pragma unroll
      for (uint32_t idx = 0; idx < kstep; idx++) {
        local_base[idx] = shared_base[base_off + idx * (lin_step + step_pad)];
      }
      return;
    }

    const uint32_t lin_base = tile_idx << 5;

#pragma unroll
    for (uint32_t idx = 0; idx < kstep; idx++) {
      uint32_t offset = __laneid() + lin_base + idx * lin_step;

      if constexpr (has_stride_offset) {
        // Stride-offset path: scalar LDS load with row-stride adjustment.
        // new_row = offset / KTile4Bytes, implemented as a shift. This is only
        // equal to true division when KTile4Bytes is a power of two; for a
        // non-power-of-two K tile the row index (and thus the shared address)
        // would be wrong. Enforce it at compile time on this path only — the
        // no-offset path below does not use the shift and is unaffected.
        static_assert((KTile4Bytes & (KTile4Bytes - 1)) == 0,
                      "GemmTensorOp stride-offset load requires K/vec_size to "
                      "be a power of two (shift-based row division); use a "
                      "power-of-two block_K for offset/strided shared regions");
        constexpr uint32_t log2_kt = __builtin_ctz(KTile4Bytes);
        uint32_t new_row = (offset >> log2_kt) + 1;
        local_base[idx] = shared_base[offset + new_row * Offset4Bytes];
      } else {
        // Per-row padding: one gap of PadWords after every StrideWords words.
        // Non-affine in idx (the gap crossing depends on laneid), which is why
        // the per-tile branch above is preferred where the write side allows
        // it.
        if constexpr (PadWords != 0)
          offset += (offset / StrideWords) * PadWords;
        local_base[idx] = shared_base[offset];
      }
    }
  }

  template <uint32_t kstep, uint32_t Stride, uint32_t StrideTileM, uint32_t Ta,
            uint32_t Offset4Bytes, uint32_t KTile4Bytes>
  static TL_DEVICE void LoadA(uint32_t *A_local_base, uint32_t *A_shared_base,
                              int tile_idx_m, int tile_idx_k) {
    uint32_t tile_idx = TransposeA ? (tile_idx_m + tile_idx_k * StrideTileM)
                                   : (tile_idx_k + tile_idx_m * StrideTileM);
    LoadFromShared<kstep, Stride, Ta, Offset4Bytes, KTile4Bytes, true,
                   TransposeA>(A_local_base, A_shared_base, tile_idx);
  }

  template <uint32_t kstep, uint32_t Stride, uint32_t StrideTileK, uint32_t Tb,
            uint32_t Offset4Bytes, uint32_t KTile4Bytes>
  static TL_DEVICE void LoadB(uint32_t *B_local_base, uint32_t *B_shared_base,
                              int tile_idx_k, int tile_idx_n) {
    uint32_t tile_idx = TransposeB ? (tile_idx_k + tile_idx_n * StrideTileK)
                                   : (tile_idx_n + tile_idx_k * StrideTileK);
    LoadFromShared<kstep, Stride, Tb, Offset4Bytes, KTile4Bytes, false,
                   TransposeB>(B_local_base, B_shared_base, tile_idx);
  }

  static TL_DEVICE void body(A_type *__restrict__ A_shared,
                             B_type *__restrict__ B_shared,
                             C_type *__restrict__ C_local) {
    // Zero the accumulator this warp owns when the caller declared that this
    // gemm defines C instead of accumulating into it. See the clear_accum note
    // at the top of this class for why the hardware does not do this for us.
    // Each warp writes only its own slice (warp_rows * warp_cols fragments of
    // local_size_c), which is the same region its MMA chains below write.
    if constexpr (clear_accum) {
#pragma unroll
      for (uint32_t i = 0; i < warp_rows * warp_cols * local_size_c; ++i) {
        C_local[i] = static_cast<C_type>(0);
      }
    }

    // Warp ID decomposition using warp ALUs (lower latency than integer ALUs)
    //
    // FullRow specialization: when num_warp_n == 1 every warp is assigned to
    // the M dimension, so warp_n == warpid / num_warp_m is identically 0 and
    // warp_m == warpid & (num_warp_m - 1) is just warpid (num_warp_m equals the
    // warp count here). Emitting the div is then pure overhead, and
    // tile_base_n == 0 lets the compiler constant-fold every B swizzle address
    // that derives from it. Both shipping square configs (block_M == block_N
    // with FullRow) take this branch.
    // The and.u32 is unconditional: warpid is a special register, so reading it
    // into a variable takes an asm either way, and the mask is what that read
    // rides on (it degenerates to a copy when num_warp_n == 1 and to a
    // materialized 0 when num_warp_m == 1). Only the div is branch-dependent.
    uint32_t warp_n = 0, warp_m;
    if constexpr (num_warp_n != 1) {
      asm("div.u32 %0, warpid, %1;" : "=wr"(warp_n) : "n"(num_warp_m));
    }
    asm("and.u32 %0, warpid, %1;" : "=wr"(warp_m) : "n"(num_warp_m - 1));
    const uint32_t tile_base_n = warp_n * warp_cols;
    const uint32_t tile_base_m = warp_m * warp_rows;

    constexpr uint32_t astep =
        local_size_a; // stride within one MMA's A fragment
    constexpr uint32_t bstep =
        local_size_b; // stride within one MMA's B fragment
    constexpr uint32_t cstep = warp_cols * local_size_c;

    uint32_t *A_shared_base = reinterpret_cast<uint32_t *>(A_shared);
    uint32_t *B_shared_base = reinterpret_cast<uint32_t *>(B_shared);

    // ---- Single-buffered register operands ----
    // A double-buffered variant that prefetches the next row's A / next
    // column's B while the current MMA chain runs was reverted:
    //   - B-side prefetch can corrupt transpose_B=True because the rotating
    //     buffer is overwritten before the MMA chain has consumed it;
    //   - A-side prefetch adds register pressure without a demonstrated
    //   benefit.
    // Keep both loads single-buffered until the B-side hazard is understood.
    uint32_t A_buf[1];
    uint32_t B_buf[1];

// Cap the k-loop unroll factor. Fully unrolling this loop lets the whole
// k-tile's worth of A/B fragments and addresses stay live at once, which
// can spill at large kStep and severely degrade throughput. Capping avoids
// pathological resource pressure, though the tradeoff can vary by config. The
// shipping block_K=64 config is unaffected because its k-loop runs once.
#pragma unroll 4
    for (int tile_idx_k = 0; tile_idx_k < inner_k; tile_idx_k += 1) {

#pragma unroll
      for (int warp_row_idx = 0; warp_row_idx < warp_rows; ++warp_row_idx) {

        auto acc_ptr =
            reinterpret_cast<C_type *>(C_local) + warp_row_idx * cstep;
        uint32_t tile_idx_m = tile_base_m + warp_row_idx;

        LoadA<1, stride_real_a, stride_tile_m, ta, offset_a_4bytes,
              k_tile_4bytes>(A_buf, A_shared_base, tile_idx_m, tile_idx_k);

        auto a_cur_ptr = reinterpret_cast<A_type *>(A_buf);

#pragma unroll
        for (int warp_col_idx = 0; warp_col_idx < warp_cols;
             ++warp_col_idx, acc_ptr += local_size_c) {

          uint32_t tile_idx_n = tile_base_n + warp_col_idx;

          auto b_cur_ptr = reinterpret_cast<B_type *>(B_buf);

          C_type C_reg[local_size_c];
#pragma unroll
          for (uint32_t i = 0; i < local_size_c; ++i)
            C_reg[i] = acc_ptr[i];

          LoadB<1, stride_real_b, stride_tile_k, tb, offset_b_4bytes,
                k_tile_4bytes>(B_buf, B_shared_base, tile_idx_k, tile_idx_n);

#pragma unroll
          for (uint32_t kii = 0; kii < 1; ++kii) {
            tl::call_mma<A_type, B_type, C_type, TransposeA, TransposeB>(
                C_reg, a_cur_ptr + kii * astep, b_cur_ptr + kii * bstep, C_reg);
          }

#pragma unroll
          for (uint32_t i = 0; i < local_size_c; ++i)
            acc_ptr[i] = C_reg[i];
        }
      }
    }
  }
};

template <int M, int N, int K, int num_warp_m, int num_warp_n, int stride_a,
          int stride_b, int offset_a, int offset_b, bool trans_A, bool trans_B,
          bool clear_accum = false, typename A_type, typename B_type,
          typename C_type>
TL_DEVICE void gemm_tang(A_type *__restrict__ pA, B_type *__restrict__ pB,
                         C_type *__restrict__ accum) {
  using Compute = GemmTensorOp<M, N, K, num_warp_m, num_warp_n, stride_a,
                               stride_b, offset_a, offset_b, trans_A, trans_B,
                               clear_accum, A_type, B_type, C_type>;
  Compute::body(pA, pB, accum);
}

} // namespace tl
