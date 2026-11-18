/*!
 * \file tl/tang/op/builtin.h
 * \brief TANG-specific TileLang intrinsic Ops.
 *
 * TANG owns its own intrinsic registry, mirroring src/cuda/op/builtin.h,
 * src/rocm/op/builtin.cc and src/metal/op/builtin.cc. The ops here are
 * registered as `tl.tang_*` (and `tl.tl_tang_gemm`) and are only produced by
 * TANG lowering; backend-neutral passes and other backends do not construct
 * them.
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

// The three descriptor ops below are pure value functions, but they are
// registered as `TCallEffectKind::kOpaque` like every other TANG builtin: the
// fork's CSE pass excludes any expression containing a `Call`, so `kPure` would
// buy no deduplication, and opaqueness keeps them exactly where the emitter put
// them, next to the MMA whose operands the STCU TCProcess pass inspects.

/*!
 * \brief TANG intrinsic: assemble the shared-memory operand descriptor of one
 * tcgen05 MMA operand (a value, not a side effect).
 * args: [ptr, LBO, SBO, SW] or [ptr, offset, LBO, SBO, SW]. The offset form
 * folds the offset into the printed pointer expression (`ptr + offset`), the
 * 4-argument form prints a bare `ptr`. Emits the
 * `tang::ptx::mma_data_desc<LBO, SBO, (SwizzleMode)SW>(ptr)` marker call that
 * the STCU TCProcess pass reads structurally.
 */
TVM_DLL const Op &tang_mma_data_desc();

/*!
 * \brief TANG intrinsic: assemble the tcgen05 MMA instruction descriptor.
 * args: [D_FMT, A_FMT, B_FMT, AMaj, BMaj, M, N, K]. Emits
 * `tang::ptx::mma_desc<(TmemType)D_FMT, (EleType)A_FMT, (EleType)B_FMT,
 * ...>()`.
 */
TVM_DLL const Op &tang_mma_instr_desc();

/*!
 * \brief TANG intrinsic: instruction descriptor of the block-scaled MMA.
 * args: [D_FMT, A_FMT, B_FMT, AMaj, BMaj, M, N, K, scale_vec, scale_format].
 * Emits `tang::ptx::mma_scale_desc<..., (ScaleVecType)scale_vec,
 * (bool)scale_format, 0, 0>()`.
 */
TVM_DLL const Op &tang_mma_scale_instr_desc();

/*!
 * \brief TANG intrinsic: tcgen5 MMA, A/B from shared descriptors, C in TMEM.
 * Maps to tang::ptx::mma<enable_input_d>; the descriptors arrive as
 * `tang_mma_data_desc` / `tang_mma_instr_desc` values.
 * args: [C_ptr, d_off, a_desc, b_desc, i_desc, enable_input_d].
 */
TVM_DLL const Op &tang_tcgen05_mma_ss();

/*!
 * \brief TANG intrinsic: tcgen5 MMA with A in tensor memory.
 * Maps to tang::ptx::mma_atmem<enable_input_d>; A is a bare TMEM address, so
 * only the B and instruction descriptors are passed.
 * args: [C_ptr, A_ptr, b_desc, i_desc, enable_input_d].
 */
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
 * Maps to tang::ptx::mma_scale<enable_input_d>; the descriptors arrive as
 * `tang_mma_data_desc` / `tang_mma_scale_instr_desc` values. Scale factors are
 * either staged into one combined TMEM workspace by `tang_scale_stage`, or
 * supplied directly as separate SFA/SFB TMEM regions.
 * args: [C_ptr, SF_tmem_ptr, a_desc, b_desc, i_desc, enable_input_d], or
 *       [C_ptr, SFA_tmem_ptr, SFB_tmem_ptr, a_desc, b_desc, i_desc,
 *        enable_input_d].
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
 * \brief TANG intrinsic: exactly one cpt2s tensor-memory -> shared-memory
 *        instruction (stcuv2).
 *
 * The leaf of the cpt2s path, which stages an accumulator tile for a bulk s2g
 * store without routing it through the register file. The warp election, the
 * row loop, the shared offsets, the tmem row stride and the fence-group drain
 * live in the IR *around* it -- lowering emits them, so the sync passes can see
 * them -- and the block-wide `__syncthreads()` the device helper used to hide
 * at its end is not emitted at all: whoever consumes the staged tile reads
 * *shared* memory, so ThreadSync derives that publish from the accesses.
 *
 * One call moves 64 tensor memory rows, i.e. 512 * (swizzle bytes / atom) bytes
 * of shared memory. The swizzle mode is an immarg because the atom width is
 * what makes the staged tile row-major; a wider atom interleaves atom/8 tensor
 * memory rows into each atom instead.
 *
 * args: [smem_access_ptr, smem_off_bytes, tmem_addr, tmem_row_off, swizzle,
 * fg].
 *   tmem_row_off the row this call starts at, in rows: the engine reads the
 *                row from the address high half, so codegen shifts it in
 *                (`+ (row << 16)`). Kept out of the IR expression on purpose:
 *                the tmem buffer still carries its logical element type here
 *                (float32 for an accumulator), so adding an integer offset to
 *                the BufferLoad is a dtype mismatch.
 *   swizzle 0 = sw128a8 (the only mode measured)
 *   fg      2 = fg_t2s_default (same encoding as tang_fence_mem)
 */
TVM_DLL const Op &tang_cpt2s_16x64b_2d();

/*!
 * \brief TANG intrinsic: exactly one cps2t shared-memory -> tensor-memory
 *        instruction (stcuv2).
 *
 * The leaf of the cps2t path, which stages an MMA A operand into tensor memory
 * without routing it through the register file. As with cpt2s, the warp
 * election, the row and word loops, the de-swizzle word index, the engine's
 * shared-memory stride setup and the fence-group drain live in the IR *around*
 * this instruction, so the sync passes can see them.
 *
 * The word offset handed to the engine is where the de-swizzle lives: cps2t
 * cannot undo a swizzle in hardware (it rejects a per-lane offset together with
 * a SwizzleMode), so the map from logical (row, word) to physical word is a
 * PrimExpr the lowering builds.
 *
 * args: [smem_access_ptr, smem_off_words, tmem_addr, tmem_row_off,
 *        tmem_col_off, sw, fg].
 *   smem_access_ptr a *read* access pointer: the engine reads shared memory, so
 *                   ThreadSync needs that read to order it against the fill.
 *   smem_off_words  the per-lane word offset inside the shared tile (physical,
 *                   i.e. after the de-swizzle map).
 *   tmem_addr       the tmem element the tile starts at (a BufferLoad).
 *   tmem_row_off    the row that element is on, in rows: the engine reads the
 *                   row from the address high half, so codegen shifts it in
 *                   (`+ (row << 16)`). Kept out of the IR expression on
 *                   purpose: the tmem buffer still carries its logical element
 *                   type here (float32 for an accumulator), so adding an
 *                   integer offset to the BufferLoad is a dtype mismatch.
 *   tmem_col_off    the 32-bit tensor-memory column, in columns.
 *   sw              0 = SwizzleMode::NONE (the only mode a per-lane offset
 *                   allows)
 *   fg              3 = fg_s2t_default (same encoding as tang_fence_mem)
 */
TVM_DLL const Op &tang_cps2t_1d();

/*!
 * \brief TANG intrinsic: the cps2t engine's shared-memory stride setup
 *        (stcuv2).
 *
 * `__stvm_set_mem_dim0_s2t(tmem_stride, smem_stride, size)` programs the
 * per-lane stride the *next* cps2t walks -- it is what turns `shm_off` from the
 * lane id into a row stride. Emitted before every cps2t in the loop because the
 * instruction carries no state across calls.
 *
 * args: [tmem_stride, smem_stride, size], all immarg; 128/128/1 is the measured
 * configuration (stride in 32-bit words, `size` = 1 row).
 */
TVM_DLL const Op &tang_stvm_set_mem_dim0_s2t();

/*!
 * \brief TANG intrinsic: publish the TMEM traffic staged above this point.
 *
 * A marker, not an instruction. Lowering emits it right after the shared->TMEM
 * staging. `tl.tang.transform.MaterializeTangMbarriers` turns it into an
 * mbarrier arrive/wait handshake. The TMEM domain has no ThreadSync
 * equivalent -- ThreadSync reasons about buffers, and what has to be ordered
 * here is "the copy engine's write landed" against "the MMA reads that tensor
 * memory" -- so nothing else would derive it.
 *
 * No args: the scope is "everything staged into tensor memory above here" and
 * the pass owns the barrier's shape. Keeping the *position* a first-class IR
 * node rather than a sync hidden in lowering lets the synchronization pass
 * replace it with an mbarrier arrive/wait pair.
 */
TVM_DLL const Op &tang_publish_tmem_operand();

TVM_DLL const Op &tang_tmem_fence();

/*!
 * \brief TANG intrinsic: publish the shared->tensor-memory staging above into a
 *        shared mbarrier (stcuv2).
 *
 * Lowers to ``tl::fence_s2t_arrive_mbarrier`` -- ``fence_mem_arrive_mbarrier``
 * on the s2t default fence group, i.e. warp-level, exactly one arrive,
 * non-blocking, and it *is* the fence: the warp's cps2t retires before any
 * waiter may observe the arrival. The counterpart of
 * ``tang_fence_g2s_arrive_mbarrier`` for the copy engine's other direction.
 *
 * This is what lets the publish live where the staging lives: an arrive is
 * legal inside a `T.ws` region guard (and is exactly what the participating
 * warps should do), while a block-wide barrier is not -- it would only be
 * reached by the threads that enter the branch.
 *
 * args: [barrier] -- the mbarrier slot, sized with the number of warps that
 * execute it (one arrive each).
 */
TVM_DLL const Op &tang_fence_s2t_arrive_mbarrier();

/*!
 * \brief TANG intrinsic: exactly one 1D bulk-copy instruction (global ->
 *        shared, stcuv2).
 *
 * The leaf of the 1D path. The warp gate, the row split, the per-row loop and
 * the fence-group drain live in the IR *around* it -- lowering emits them, so
 * the sync passes can see them -- and the block-wide `__syncthreads()` the
 * helper used to hide at its end is emitted by the caller instead, as a plain
 * `tvm_storage_sync`.
 *
 * The 1D form exists for shared buffers that are pure staging stops rather
 * than MMA operands: unlike the swizzled 3D pair, it round-trips.
 *
 * The two pointers are the *whole-tile* (or whole-segment) base addresses, i.e.
 * what the 1D helper was handed, and the byte offsets are applied to them here.
 * Keeping the pipeline-version offset in the base pointer and the stripe offset
 * out of it lets lowering record the whole-version stage while ThreadSync reads
 * the extent.
 *
 * args: [smem, gmem, smem_off_bytes, gmem_off_bytes, nbytes].
 */
TVM_DLL const Op &tang_fcpg2s_1d();

/*!
 * \brief TANG intrinsic: exactly one 1D bulk-copy instruction (shared ->
 *        global, stcuv2). Mirror of ``tang_fcpg2s_1d``.
 *
 * args: [gmem, smem, gmem_off_bytes, smem_off_bytes, nbytes].
 */
TVM_DLL const Op &tang_fcps2g_1d();

/*!
 * \brief TANG intrinsic: exactly one swizzled 3D bulk-copy instruction
 *        (global -> shared, stcuv2).
 *
 * The stripe split, region warp gate, drain and block barrier are IR around
 * this call, so codegen prints one instruction and nothing else.
 *
 * `pack_mode` is a tang::ptx::PackPaddingMode: 0 = no_pack, 1 = b4p4x16, which
 * makes the copy engine spread packed global fp4 (2 e2m1 codes per byte) into
 * the one-code-per-byte layout an mxf8f6f4 operand reads. Only the load
 * direction with a 32-byte atom and a 128-byte shared row supports a non-zero
 * mode; the lowering checks that before emitting one.
 *
 * The two pointers are the *whole-tile* base addresses and the byte offsets are
 * applied to them here, for the same reason as ``tang_fcpg2s_1d``.
 *
 * args: [smem, gmem, smem_off_bytes, gmem_off_bytes, rows, smem_row_bytes,
 *        gmem_row_bytes, swizzle_mode, pack_mode].
 */
TVM_DLL const Op &tang_fcpg2s_3d();

/*!
 * \brief TANG intrinsic: exactly one swizzled 3D bulk-copy instruction
 *        (shared -> global, stcuv2).
 *
 * The 3D store takes only the *global* row pitch: the shared pitch follows from
 * the swizzle mode.
 *
 * args: [gmem, smem, gmem_off_bytes, smem_off_bytes, rows, gmem_row_bytes,
 *        swizzle_mode].
 */
TVM_DLL const Op &tang_fcps2g_3d();

/*!
 * \brief TANG bulk-copy record for a copy carrying a v2 pipeline plan.
 *
 * Not an instruction. The emitter records facts that the async handshake pass
 * should not infer from lowered address expressions. The pass consumes the
 * typed identity and execution counts directly.
 *
 * The record sits in the *same scope* as the copy, outside its warp gate: the
 * gate is the copy's own business (``active_warps`` states the participants),
 * while a plain top-level call is what the downstream attribution walk can
 * still see -- it refuses anything that is not a call.
 *
 * Codegen prints nothing for it: with no consumer it must be invisible in the
 * generated source. If one ever reaches codegen, the handshake pass did not
 * consume it and printing nothing leaves the kernel in its blocking form.
 *
 * Args:
 *   [2, buffer_key, stage, versions, active_warps, region_warps, fg,
 *    loop_uid, buffer_uid, distance, producer_stage, consumer_stage,
 *    ready_owner, barrier?].
 *
 * The record carries the exact plan copied from the tile-copy annotations;
 * ``ready_owner`` is 0 when the S3 handshake pass owns publication and 1 when
 * InjectSoftwarePipeline already owns it through the optional barrier.
 *
 *   buffer_key   the shared side's data variable, as a string (NOT the
 *                access_ptr: a shared pointer argument would make the sync
 *                analysis see this call as a shared access)
 *   stage        the version this copy writes (constant 0 when not rotated)
 *   versions     how many versions the buffer has. Stated rather than derived:
 *                the pipeline may reindex a rotated buffer affinely, and then
 *                the stage expression carries no version count at all
 *   active_warps warps that issue the copy, i.e. the barrier's pcnt
 *   region_warps warps of the copy's own region. Not the same number:
 *                `active_warps` is clamped to the stripes the tile has, while
 *                this is what the free edge's wait uses as its ccnt
 *   fg           0 = g2s, 1 = s2g (same encoding as tang_fence_mem)
 */
TVM_DLL const Op &tang_bulk_async_commit();

/*!
 * \brief TANG intrinsic: drain this warp's outstanding bulk copies in a fence
 *        group (stcuv2).
 *
 * The *wait* form of the mem-domain fence, ``fence_mem(fg)``: it blocks the
 * issuing warp until its own DMAs retire. A bulk copy is asynchronous, so
 * whoever issued one must drain it before another thread may observe the
 * result; that drain is per-warp and it is *not* a barrier, so no pass may move
 * it across a barrier or drop it as redundant.
 *
 * args: [fg] with 0 = fg_g2s_default (after a load), 1 = fg_s2g_default (after
 * a store), 2 = fg_t2s_default (after a tmem -> shared copy), 3 =
 * fg_s2t_default (after a shared -> tmem copy). The last two are the copy
 * engines' groups: fence_mem only covers async work in *its own* group, so a
 * tmem copy that drained the g2s/s2g group would not be ordered at all.
 */
TVM_DLL const Op &tang_fence_mem();
TVM_DLL const Op &tang_fence_tc();

/*!
 * \brief TANG intrinsic: publish a warp's outstanding bulk g2s copies into a
 *        shared mbarrier (stcuv2).
 *
 * Lowers to tl::fence_g2s_arrive_mbarrier: ``fence_mem_arrive_mbarrier`` on the
 * g2s default fence group, i.e. warp-level, exactly one arrive, non-blocking.
 * An mbarrier counts only arrivals, so the waiting and issuing regions may
 * contain different numbers of warps.
 *
 * args: [barrier] where barrier is ``get_mbarrier(bar_buf[i])``.
 */
TVM_DLL const Op &tang_fence_g2s_arrive_mbarrier();

/*!
 * \brief TANG intrinsic: spin until an mbarrier's phase has flipped past
 *        ``parity`` (stcuv2).
 *
 * Lowers to tl::mbarrier_wait, the same parity poll the
 * ``mbarrier_wait_parity`` builtin uses, but on a raw ``__mbarrier_t*`` rather
 * than a ``Barrier`` object: the async-copy pass addresses barriers by buffer
 * index, so there is no typed Barrier to call a method on.
 *
 * args: [barrier, parity].
 */
TVM_DLL const Op &tang_mbar_wait_parity();

/*!
 * \brief TANG intrinsic: publish a warp's outstanding tensor-core group into a
 *        shared mbarrier (stcuv2).
 *
 * Lowers to tl::fence_tc_arrive_mbarrier: warp-level, exactly one arrive,
 * non-blocking. The fence group is the device default (fg_tc_default = fg0,
 * the only group the TC domain exposes here).
 *
 * args: [barrier] where barrier is ``get_mbarrier(bar_buf[i])``.
 */
TVM_DLL const Op &tang_fence_tc_arrive_mbarrier();
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
