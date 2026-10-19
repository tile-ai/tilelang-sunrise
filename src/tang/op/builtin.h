/*!
 * \file tl/tang/op/builtin.h
 * \brief TANG-specific TileLang intrinsic Ops.
 *
 * TANG owns its own intrinsic registry, mirroring src/cuda/op/builtin.h,
 * src/rocm/op/builtin.cc and src/metal/op/builtin.cc. The ops here are
 * registered as `tl.tang_*` (and `tl.tl_tang_gemm`) and are only ever produced
 * by TANG lowering — nothing in the backend-neutral passes, and nothing in
 * another backend, constructs them. Before this file existed those ops lived in
 * src/op/builtin.{h,cc}, i.e. in the backend-neutral registry, which is why a
 * TANG-only op looked like a shared one.
 *
 * This file is compiled only when USE_TANG is on: src/tang/CMakeLists.txt globs
 * src/tang/op/*.cc behind its `if(NOT USE_TANG) return()` guard. That is the
 * correct gate for a TANG-only registry — a build without the TANG backend has
 * no way to construct a TANG-target module in the first place. (CUDA needs its
 * cuda/op/builtin.cc in TILE_LANG_CUDA_ALWAYS_SRCS only because that file also
 * holds backend-neutral ops.)
 */

#ifndef TVM_TL_TANG_OP_BUILTIN_H_
#define TVM_TL_TANG_OP_BUILTIN_H_

#include "op/builtin.h"

namespace tvm {
namespace tl {

TVM_DLL const Op &tl_tang_gemm();
TVM_DLL const Op &tang_fill_fragment();
TVM_DLL const Op &tang_tcgen05_mma_ss();
TVM_DLL const Op &tang_tcgen05_mma_ts();

/*!
 * \brief TANG intrinsic: stage block-scaled GEMM scale factors global -> TMEM
 * (stcuv2). Emits the warp-0 `.32x32b` staging loop (SFA -> cols [0,16),
 * SFB -> cols [16,32)) plus a trailing fence_stt.
 * args: [SFA_ptr, SFB_ptr, SF_tmem_ptr, M, N, k_cells, stride_sfa, stride_sfb].
 */
TVM_DLL const Op &tang_scale_stage();

/*!
 * \brief TANG intrinsic: block-scaled tcgen5 MMA (mxf8f6f4/mxf4/nvfp4, stcuv2).
 * Maps to tang::ptx::mma_scale<enable_input_d> with mma_data_desc<> /
 * mma_scale_desc<> marker intrinsics.
 * args: [C_ptr, A_ptr, B_ptr, SF_tmem_ptr, LBO_A, SBO_A, SW_A, LBO_B, SBO_B,
 *        SW_B, D_FMT, A_FMT, B_FMT, AMaj, BMaj, M, N, K, scale_vec,
 *        scale_format, enable_input_d].
 */
TVM_DLL const Op &tang_tcgen05_mma_scale();
TVM_DLL const Op &tang_init_tensor_memory();
TVM_DLL const Op &tang_deallocate_tensor_memory();
TVM_DLL const Op &tang_tcgen05_mma_arrive();
TVM_DLL const Op &tang_tmem_ld();

/*!
 * \brief TANG intrinsic: tensor memory store 16x256b (stcuv2).
 * Maps to tang::ptx::stt_16x256b_x{N}(in, taddr), the register->TMEM store
 * counterpart of tang_tmem_ld_16x256b. args: [in_ref, taddr, num_chunks];
 * each thread stores 4*num_chunks 32-bit registers.
 */
TVM_DLL const Op &tang_tmem_st_16x256b();

/*!
 * \brief TANG intrinsic: stage an MMA A operand into tensor memory (stcuv2).
 * Maps to tl::tang_tmem_st_a_operand<CELLS>(frag, taddr), a `.32x32b` store in
 * which lane == A row and the row's K elements are bit-packed into 32-bit TMEM
 * cells. This is the layout mma_atmem reads, and it is unrelated to the
 * accumulator's tang_tmem_st_16x256b. args: [in_ref, taddr, num_cells].
 */
TVM_DLL const Op &tang_tmem_st_a_operand();
TVM_DLL const Op &tang_tmem_ld_16x256b();
TVM_DLL const Op &tang_tmem_ld_16x256b_x16();
TVM_DLL const Op &tang_tmem_drain_16x256b_to_global();
TVM_DLL const Op &tang_tmem_st();

/*!
 * \brief TANG intrinsic: tensor memory -> shared memory copy (stcuv2).
 * Maps to tl::tang_cp_tmem_to_shared_sw128a8(smem, taddr, rows), i.e. the
 * cpt2s copy engine path that bypasses the register file entirely.
 * args: [smem_ptr, tmem_addr, rows].
 */
TVM_DLL const Op &tang_cp_tmem_to_shared();

/*!
 * \brief TANG intrinsic: shared memory -> tensor memory copy (stcuv2).
 * Maps to tl::tang_cp_shared_to_tmem(smem, taddr, rows, cols, row_words), the
 * cps2t copy engine path used to stage an MMA A operand without a register
 * round trip. `cols` and `row_words` count 32-bit words, not elements.
 * args: [smem_ptr, tmem_addr, rows, cols, row_words].
 */
TVM_DLL const Op &tang_cp_shared_to_tmem();
TVM_DLL const Op &tang_tmem_fence();
TVM_DLL const Op &tang_cp_async_bulk();

/*!
 * \brief TANG intrinsic: swizzled 3D bulk copy with an explicit swizzle mode
 * and pack mode (global <-> shared, stcuv2). Maps to
 * tl::tang_bulk_{g2s,s2g}<SwizzleMode, PackPaddingMode>.
 *
 * `pack_mode` is a tang::ptx::PackPaddingMode: 0 = no_pack, 1 = b4p4x16, which
 * makes the copy engine spread packed global fp4 (2 e2m1 codes per byte) into
 * the one-code-per-byte layout an mxf8f6f4 operand reads. Only the load
 * direction with a 32-byte atom and a 128-byte shared row supports a non-zero
 * mode; the lowering checks that before emitting one.
 *
 * args: [direction, dst, src, rows, smem_row_bytes, gmem_row_bytes,
 *        num_threads, swizzle_mode, pack_mode].
 */
TVM_DLL const Op &tang_cp_async_bulk_sw();

/*!
 * \brief TANG intrinsic: unswizzled 1D bulk copy (global <-> shared, stcuv2).
 * Maps to tl::tang_bulk_{g2s,s2g}_1d. Used for shared buffers that are pure
 * staging stops rather than MMA operands: the swizzled 3D pair does not
 * round-trip (docs/tang_bulk_copy_gs_layout.md §15.2), the 1D form does.
 * args: [direction, dst, src, rows, smem_row_bytes, gmem_row_bytes,
 *        num_threads].
 */
TVM_DLL const Op &tang_cp_async_bulk_1d();
TVM_DLL const Op &tang_fence_tc();
TVM_DLL const Op &tang_fence_tc_arrive();
TVM_DLL const Op &tang_fence_g2s_arrive();
TVM_DLL const Op &tang_sync_wait();
TVM_DLL const Op &tang_sync_arrive();

/*!
 * \brief TANG intrinsic: order the shared.mbarrier init stores before the
 *        barrier can be used (stcuv2).
 *
 * The CUDA counterpart is ptx_fence_barrier_init, which emits
 * `fence.mbarrier_init.release.cluster`. TANG has no mbarrier-init-specific
 * fence: the whole fence family is fence.mem / fence.alu / fence.tc / fence.fg
 * / fence.tmem / fence.all / fence.arrive*, and the vendor documents fence.mem
 * as the CTA-scope release primitive ("Prefer CTA release via fence.mem; do not
 * use fence.all / sync.wait here", __clang_tang_builtin_vars.h). So this op
 * lowers to a plain `fence.mem`.
 *
 * Why not rely on the fence already inside mbarrier_init: tang::ptx::
 * mbarrier_init does end in `fence.mem`, but that is an implementation detail
 * of one toolchain configuration, not a TANG guarantee — the alternative
 * `-fstpu-sreg-mbarrier` path (__sreg_mbarrier_*, a software publish through a
 * shared-register lock) contains no fence at all. Emitting the fence here makes
 * this op's contract self-contained instead of depending on the device library
 * staying as it is today.
 *
 * This only orders the init; publishing it to the rest of the block is still
 * the `__syncthreads()` the lowering emits right after. No arguments.
 */
TVM_DLL const Op &tang_fence_barrier_init();

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TANG_OP_BUILTIN_H_
