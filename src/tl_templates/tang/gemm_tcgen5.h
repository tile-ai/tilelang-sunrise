#pragma once

#include <type_traits>

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

// ---- sub-byte tensor-core element tags (fp6 / fp4) ----
// fp6/fp4 operands have no native C++ dtype: they are carried through the GEMM
// as raw uint8 byte buffers and their numeric interpretation is selected purely
// by the MMA descriptor's EleType field. We expose distinct 1-byte tag types so
// the codegen can pick the element type *by name* (mirroring fp8_e4_t /
// fp8_e5_t) and resolve it through the same tcgen5_ele_type<> trait, instead of
// threading a raw integer EleType code. These tags are never used as buffer
// element types; only as the AEle/BEle selector of gemm_tang_tcgen5[_scale].
struct fp6_e2m3_t {
  uint8_t raw;
}; // eFP6_E2M3
struct fp6_e3m2_t {
  uint8_t raw;
}; // eFP6_E3M2
struct fp4_e2m1_t {
  uint8_t raw;
}; // eFP4_E2M1     (mxf4, ue8m0 scale)
struct fp4_e2m1_nv_t {
  uint8_t raw;
}; // eNVFP4_E2M1   (nvfp4, ue4m3 scale)
struct fp4_e2m1_mix_t {
  uint8_t raw;
}; // eFP4_E2M1_MIX (fp4 mixed w/ non-fp4)

template <> struct tcgen5_ele_type<fp6_e2m3_t> {
  static constexpr tang::ptx::EleType value = tang::ptx::eFP6_E2M3;
};
template <> struct tcgen5_ele_type<fp6_e3m2_t> {
  static constexpr tang::ptx::EleType value = tang::ptx::eFP6_E3M2;
};
template <> struct tcgen5_ele_type<fp4_e2m1_t> {
  static constexpr tang::ptx::EleType value = tang::ptx::eFP4_E2M1;
};
template <> struct tcgen5_ele_type<fp4_e2m1_nv_t> {
  static constexpr tang::ptx::EleType value = tang::ptx::eNVFP4_E2M1;
};
template <> struct tcgen5_ele_type<fp4_e2m1_mix_t> {
  static constexpr tang::ptx::EleType value = tang::ptx::eFP4_E2M1_MIX;
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

// ---- shared-memory swizzle mode for the MMA descriptor ----
// The descriptor swizzle MUST match the physical layout produced by the
// shared-memory bulk copy in copy_fcp_g_s.h (tang_bulk_g2s_sw128a32 /
// _sw128a64). The required swizzle depends on the operand element size AND its
// major-ness:
//   * fp16/bf16 (2 bytes), any major          -> sw128bytes_atom32bytes
//   * tf32 (4 bytes) / int8 (1 byte), K-major -> sw128bytes_atom32bytes
//   * tf32 (4 bytes) / int8 (1 byte), MN-major-> sw128bytes_atom64bytes
// The MN-major / atom64 case is the NN GEMM B operand (B given as (K,N), so its
// outer N dimension is contiguous). The K-major / atom32 case is the TN B
// operand and the usual M-K row-major A operand.
//
// mn_major mirrors the descriptor's a_major / b_major flag (1 == MN_Major).
constexpr tang::ptx::SwizzleMode tcgen5_operand_swizzle(int elem_bytes,
                                                        bool mn_major) {
  // TODO: reimplement per-element-size / majorness swizzle selection.
  (void)elem_bytes;
  (void)mn_major;
  return tang::ptx::sw128bytes_atom32bytes;
}

// ---- Helper: compute SBO from LBO ----
// SBO (Swizzle Block Offset / stride byte offset) for the sw128 family equals
// one physical SMEM swizzle row (SG = 512 bytes, = 4 banks * 128B), independent
// of LBO and of a_major/b_major. This is verified against the golden MMA
// references (gver/unit_compiler_test_case/testcase_mma_without_scale_46,47),
// where BOTH the atom32 A descriptor and the atom64 B descriptor use SBO=512.
// (The previous "atom32 -> 4*LBO" form only coincided with 512 because the
// K-major contiguous row was exactly 128 bytes; atom64 with a 512-byte LBO must
// still use SBO=512, not 2*LBO=1024.)
constexpr uint32_t tcgen5_sbo(uint32_t lbo, tang::ptx::SwizzleMode sw) {
  // TODO: reimplement SBO-from-LBO computation (sw128 family -> 512).
  (void)lbo;
  (void)sw;
  return 512u;
}

// ---- gemm_tang_tcgen5 ----
// clear_accum is a *runtime* argument: when true the accumulator (D in TMEM) is
// overwritten (D = A*B); when false the existing accumulator is read back and
// added (D = A*B + D). For a K-tiled GEMM the first K-tile passes
// clear_accum=true and the remaining tiles clear_accum=false so partial
// products accumulate. Because the K-tile predicate (k == 0) is not a
// compile-time constant, clear_accum cannot be a template parameter; we branch
// on it at runtime and dispatch to the corresponding mma<enable_input_d>
// instantiation (enable_input_d == !clear_accum).
template <int M, int N, int K, int warp_m, int warp_n, int stride_a,
          int stride_b, int offset_a, int offset_b, bool TransposeA,
          bool TransposeB, int kPack, int kStep, bool a_major, bool b_major,
          typename AEle = void, typename BEle = void, typename A_type,
          typename B_type, typename C_type>
TL_DEVICE void gemm_tang_tcgen5(A_type *A_smem, B_type *B_smem, C_type *C_tmem,
                                bool clear_accum) {
  // TODO: reimplement tcgen5 MMA (SS/TS) body; currently a no-op stub.
  (void)A_smem;
  (void)B_smem;
  (void)C_tmem;
  (void)clear_accum;
}

// ---- scale-staging helper: store CELLS uint32 (32x32b) into TMEM ----
template <int CELLS>
TL_DEVICE void tcgen5_stt_cells(uint32_t (&in)[CELLS], uint32_t taddr) {
  // TODO: reimplement scale-factor staging (stt_32x32b_x1/x2/x4).
  (void)in;
  (void)taddr;
}

// ---- gemm_tang_tcgen5_scale ----
// Block-scaled tcgen5 MMA (mxf8f6f4 / mxf4 / nvfp4): D = (A*sfa) @ (B*sfb).
//
// The per-block scale factors are consumed by mma_scale through TMEM operands,
// so they must first be materialized in tensor memory.  We stage them directly
// global->TMEM with a warp-collective stt (each of warp-0's 32 lanes carries
// one row), avoiding a swizzled shared-memory scale copy.  SFA occupies columns
// [0,16) and SFB columns [16,32) of the caller-provided SF_tmem region (a
// second T.alloc_tmem that LowerSharedTmem folds into the accumulator's
// tc_alloc).
//
//   scale_vec    : ScaleVecType enum value (1=X1, 2=X2, 3=X4)
//   scale_format : 1 = UE8M0, 0 = UE4M3
//   sf_block     : elements per scale block (32 for mxf8f6f4/mxf4, 16 for
//   nvfp4) stride_sfa/b : per-row byte stride of the GLOBAL scale tensors (=
//   full
//                  K/sf_block of the un-tiled scale operand)
//   AEle/BEle    : tensor-core element tag types for A/B (fp8_e*_t / fp6_*_t /
//                  fp4_*_t), resolved to tang::ptx::EleType through the shared
//                  tcgen5_ele_type<> trait -- covers the mixed-precision tags
//                  (fp4_e2m1_mix_t etc.) that differ from a dtype inference.
template <int M, int N, int K, int warp_m, int warp_n, int stride_a,
          int stride_b, int offset_a, int offset_b, bool TransposeA,
          bool TransposeB, int kPack, int kStep, bool a_major, bool b_major,
          int scale_vec, int scale_format, int sf_block, int stride_sfa,
          int stride_sfb, typename AEle, typename BEle, typename A_type,
          typename B_type, typename C_type>
TL_DEVICE void gemm_tang_tcgen5_scale(A_type *A_smem, B_type *B_smem,
                                      const void *SFA_gmem_,
                                      const void *SFB_gmem_, C_type *C_tmem,
                                      uint32_t *SF_tmem, bool clear_accum) {
  // TODO: reimplement block-scaled tcgen5 MMA (mxf8f6f4/mxf4/nvfp4) body;
  // currently a no-op stub.
  (void)A_smem;
  (void)B_smem;
  (void)SFA_gmem_;
  (void)SFB_gmem_;
  (void)C_tmem;
  (void)SF_tmem;
  (void)clear_accum;
}

} // namespace tl
