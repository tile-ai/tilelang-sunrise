/*!
 * \file tl/op/builtin.cc
 * \brief Registration of backend-neutral TileLang intrinsic Ops.
 */

#include "builtin.h"
#include "cuda/op/builtin.h"

#include <tvm/ir/transform.h>

#include "builtin_registry.h"

namespace tvm {
namespace tirx {
namespace builtin {
const Op &tvm_global_barrier_kinit() {
  static const Op &op = Op::Get("tirx.tvm_global_barrier_kinit");
  return op;
}
TVM_REGISTER_OP("tirx.tvm_global_barrier_kinit")
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
} // namespace builtin
} // namespace tirx

namespace tl {

using namespace tirx;

TVM_REGISTER_PASS_CONFIG_OPTION(kDebugMergeSharedMemoryAllocations, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableSafeMemoryLegalize, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableThreadStorageSync, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kConfigIndexBitwidth, Integer);
TVM_REGISTER_PASS_CONFIG_OPTION(kEnableAggressiveSharedMemoryMerge, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableSharedMemoryReuse, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kForceLetInline, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kEnableFastMath, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kEnableAsyncCopy, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kReducerForceBaseline, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kEnableReducerPlanVerbose, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kLayoutCostModel, ffi::String);
TVM_REGISTER_PASS_CONFIG_OPTION(kEnableVectorizePlannerVerbose, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kStorageRewriteDetectInplace, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kASTPrintEnable, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kLayoutVisualizationEnable, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kLayoutVisualizationFormats, ffi::String);
TVM_REGISTER_PASS_CONFIG_OPTION(kDeviceCompileFlags, ffi::Array<ffi::String>);
TVM_REGISTER_PASS_CONFIG_OPTION(kEmitLineDirectives, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableDataRaceCheck, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableBufferInitCheck, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableLoopUnswitching, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kLoopUnswitchingAllowNonTrivialElse, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kIfStmtBindingInlineReplayableBinds, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableOutOfBoundWarning, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kEnableDumpIR, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDumpIRDir, ffi::String);
TVM_REGISTER_PASS_CONFIG_OPTION(kTangDisableWarpAlu, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kUseAsyncCop4, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableRemoveRedundantSyncs, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableMergeLoop, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableLoopPeeling, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableGemmPad, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kGemmPadM, Integer);
TVM_REGISTER_PASS_CONFIG_OPTION(kGemmPadN, Integer);
TVM_REGISTER_PASS_CONFIG_OPTION(kGemmPadK, Integer);
TVM_REGISTER_PASS_CONFIG_OPTION(kEnableGemmPad, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kEnableHoistCopyAddresses, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kEnableCopyStagingPad, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kPassProfile, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kPassProfileThresholdMs, FloatImm);

DataType cuTensorMapType() { return CuTensorMapType(); }

TIR_DEFINE_TL_BUILTIN(tvm_ffi_call_with_result)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(access_ptr)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(region).set_num_inputs(-1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(add2).set_num_inputs(2).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(sub2).set_num_inputs(2).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(mul2).set_num_inputs(2).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(fma2).set_num_inputs(3).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(max2).set_num_inputs(2).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(min2).set_num_inputs(2).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(abs2).set_num_inputs(1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(mbarrier_wait_parity)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(mbarrier_expect_tx)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ptx_stmatrix)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ptx_cp_async)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(annotate_producer_reg_dealloc)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(annotate_consumer_reg_alloc)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(no_set_max_nreg)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(wait_wgmma)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(pack_b16).set_num_inputs(2).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

TIR_DEFINE_TL_BUILTIN(sync_grid).set_num_inputs(0).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(sync_warp).set_num_inputs(-1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(any_sync).set_num_inputs(2).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(all_sync).set_num_inputs(2).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ballot_sync)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ballot).set_num_inputs(1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(activemask)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(syncthreads_count)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(syncthreads_and)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(syncthreads_or)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(shfl_sync).set_num_inputs(4).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(shfl_xor_sync)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(shfl_down_sync)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(shfl_up_sync)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(match_any_sync)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(match_all_sync)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(loop_break)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(atomic_add_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(atomic_add_ret_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(atomic_addx2_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(atomic_addx2_ret_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(atomic_addx4_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(atomic_addx4_ret_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(atomic_load_elem_op)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(atomic_store_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(atomic_or_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(atomic_max_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(atomic_max_ret_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(atomic_min_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(atomic_min_ret_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(atomic_sub_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(atomic_exch_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(atomic_inc_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(atomic_dec_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(atomic_cas_elem_op)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(atomic_and_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(atomic_xor_elem_op)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(warp_reduce_sum)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(warp_reduce_max)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(warp_reduce_min)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(warp_reduce_bitand)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(warp_reduce_bitor)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(tl_gemm).set_num_inputs(4).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(tl_gemm_sp)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(pts_load_async)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(pts_store_async)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(pts_syncthreads)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(get_mbarrier)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));
TIR_DEFINE_TL_BUILTIN(tl_tang_gemm)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(tang_fill_fragment)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(tang_tcgen05_mma_ss)
    .set_num_inputs(21)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(tang_tcgen05_mma_ts)
    .set_num_inputs(14)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(tang_scale_stage)
    .set_num_inputs(8)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(tang_tcgen05_mma_scale)
    .set_num_inputs(21)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(tang_init_tensor_memory)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(tang_deallocate_tensor_memory)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(tang_tcgen05_mma_arrive)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(tang_tmem_ld)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(tang_tmem_ld_16x256b)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(tang_tmem_ld_16x256b_x16)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(tang_tmem_drain_16x256b_to_global)
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(tang_tmem_st)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(tang_cp_tmem_to_shared)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(tang_cp_shared_to_tmem)
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(tang_tmem_st_16x256b)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(tang_tmem_st_a_operand)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(tang_tmem_fence)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(tang_cp_async_bulk)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(tang_cp_async_bulk_sw)
    .set_num_inputs(9)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(tang_cp_async_bulk_1d)
    .set_num_inputs(7)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(tang_fence_tc)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(tang_fence_tc_arrive)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(tang_fence_g2s_arrive)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(tang_sync_wait)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(tang_sync_arrive)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(__ldg).set_num_inputs(-1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kPure));

} // namespace tl
} // namespace tvm

#undef TIR_DEFINE_TL_BUILTIN
