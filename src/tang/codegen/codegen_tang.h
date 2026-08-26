/*!
 * \file target/codegen.h
 * \brief Utility to generate code
 */
#ifndef TVM_TL_TARGET_CODEGEN_TANG_H_
#define TVM_TL_TARGET_CODEGEN_TANG_H_

#include <tvm/target/codegen.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "target/source/codegen_c.h"

namespace tvm {
namespace codegen {

class CodeGenTileLangTANG final : public CodeGenC {
public:
  CodeGenTileLangTANG();
  std::string Finish();
  // override behavior
  void PrintFuncPrefix(std::ostream &os) final;
  void PrintExtraAttrs(const PrimFunc &f);
  void VisitStmt_(const ForNode *op) final;
  void PrintStorageSync(const CallNode *op) final;
  void PrintStorageScope(const std::string &scope,
                         std::ostream &os) final; // NOLINT(*)
  void PrintVecBinaryOp(const std::string &op, DataType t, PrimExpr lhs,
                        PrimExpr rhs,
                        std::ostream &os) final; // NOLINT(*)
  void PrintVecConstructor(DataType t, std::ostream &os) final;
  void PrintType(DataType t, std::ostream &os) final; // NOLINT(*)
  void PrintVecElemLoad(const std::string &vec, DataType t, int i,
                        std::ostream &os) final; // NOLINT(*)
  void PrintVecElemStore(const std::string &vec, DataType t, int i,
                         const std::string &value) final;
  std::string GetVecLoad(DataType t, const BufferNode *buffer,
                         PrimExpr base) final;
  void PrintVecStore(const BufferNode *buffer, DataType t, PrimExpr base,
                     const std::string &value) final;
  void BindThreadIndex(const IterVar &iv) final; // NOLINT(*)
  void PrintVecElemLoadExpr(DataType t, int i, const std::string &value,
                            std::ostream &os) final;
  std::string CastFromTo(std::string value, DataType from,
                         DataType target) final;
  // overload visitor
  void VisitExpr_(const RampNode *op, std::ostream &os) final;      // NOLINT(*)
  void VisitExpr_(const BroadcastNode *op, std::ostream &os) final; // NOLINT(*)
  void VisitExpr_(const ShuffleNode *op, std::ostream &os) final;   // NOLINT(*)
  void VisitExpr_(const FloatImmNode *op, std::ostream &os) final;
  void VisitExpr_(const CallNode *op, std::ostream &os) final;
  void VisitExpr_(const CastNode *op, std::ostream &os) final;
  void VisitExpr_(const MinNode *op, std::ostream &os) final;
  void VisitExpr_(const MaxNode *op, std::ostream &os) final;
  void VisitStmt_(const EvaluateNode *op) final;
  void VisitStmt_(const AllocBufferNode *op) final;
  void VisitStmt_(const AttrStmtNode *op) final;
  void VisitExpr_(const BufferLoadNode *op, std::ostream &os) final;

  // Override this as a work around for __grid_constant__ parameter
  void AddFunction(const GlobalVar &gvar, const PrimFunc &f);
  void InitFuncState(const PrimFunc &f) final;
  void PrintFunctionSignature(const ffi::String &function_name,
                              const PrimFunc &func, std::ostream &os);

private:
  // Cache for parameters that should NOT have const qualifier
  // (used in async_load or atomic_add operations)
  std::unordered_map<const tirx::PrimFuncNode *,
                     std::unordered_set<const tirx::VarNode *>>
      non_const_param_cache_;

  // Detect parameters used in async_load or atomic_add operations
  // These parameters should NOT be marked as const
  std::unordered_set<const tirx::VarNode *>
  DetectParamsNeedingNonConst(const PrimFunc &func);

  // Helper to emit const qualifier for a parameter
  // Returns true if const was emitted
  bool EmitConstQualifier(
      std::ostream &os, const tirx::Var &v, int param_index,
      const std::unordered_set<int> &ro_param_indices,
      const std::unordered_set<const tirx::VarNode *> &non_const_params);

protected:
  virtual std::string GetBufferRef(DataType t, const BufferNode *buffer,
                                   PrimExpr index) final;
  void PrintCallExtern(Type ret_type, ffi::String global_symbol,
                       const ffi::Array<PrimExpr> &args, bool skip_first_arg,
                       std::ostream &os) final; // NOLINT(*)

private:
  // Handle volatile loads
  void HandleVolatileLoads(const std::string &value, const BufferLoadNode *op,
                           std::ostream &os) final;

  // Whether scope such as "__shared__" or "__constant__"  is part of type.
  bool IsScopePartOfType() const final { return false; }

  friend void PrintConst(const FloatImmNode *op, std::ostream &os,
                         CodeGenTileLangTANG *p);

  // Whether global barrier is needed.
  bool need_global_barrier_{false};
  // Global barrier state
  std::string vid_global_barrier_state_;
  // Global barrier expected node.
  std::string vid_global_barrier_expect_;

  // whether enable fp16
  bool enable_fp16_{false};
  // whether enable bf16
  bool enable_bf16_{false};
  // whether enable fp8
  bool enable_fp8_{false};
  // whether enable fp6
  bool enable_fp6_{false};
  // whether enable fp4
  bool enable_fp4_{false};
  // whether enable int8
  bool enable_int8_{false};
  // whether enable sparse gemm
  bool enable_sparse_gemm_{false};
  // whether enable warp shuffle intrinsics
  bool enable_warp_shuffle_{false};
  // whether need __clang_tang_builtin_vars.h
  bool need___clang_tang_builtin_vars_h{false};
  // whether need need___clang_tang_fp16.h
  bool need___clang_tang_fp16_h{false};
  // whether need need___clang_tang_bf16.h
  bool need___clang_tang_bf16_h{false};
  // whether need mma.h
  bool need_mma_h_{false};
  // whether need tl mma instruction header
  bool need_mma_instruction_h_{false};
  // whether need tl wgmma instruction header
  bool need_wgmma_instruction_h_{false};
  // whether need tl tcgen05mma instruction header
  bool need_tcgen05mma_instruction_h_{false};
  // whether need tl mma_sm70 instruction header
  bool need_mma_sm70_instruction_h_{false};
  // whether need tcgen_05 common header
  bool need_tcgen05_common_h_{false};
  // whether need the TANG swizzled bulk-copy helper header
  // (tang/copy_fcp_g_s.h)
  bool need_cp_async_bulk_h_{false};
  // whether need cast_smem_ptr_to_int helper function
  bool need_cast_smem_ptr_to_int_{false};
  // whether need cooperative_groups.h
  bool need_cooperative_groups_{false};
  // Tang RNG state for tl.rng_* intrinsics, reset per-PrimFunc in
  // InitFuncState. Keep declaration discovery separate from source-order
  // initialization so rng_rand before rng_init is still rejected.
  // _type_ is the algorithm suffix: "philox"/"mrg32k3a"/"xorwow".
  bool rng_state_declared_{false};
  bool rng_initialized_{false};
  std::string tang_random_generator_state_;
  std::string tang_random_generator_state_type_;
  // Op attribute map
  OpAttrMap<bool> op_need_warp_shuffle_ =
      Op::GetAttrMap<bool>("tang.need_warp_shuffle");

  // The name of the barrier array in shared memory
  const std::string barrier_name_ = "barrier";
  // The size of the barrier array in shared memory
  int barrier_count_ = -1;
  // The name of the mbarrier array in shared memory
  const std::string mbarrier_name_ = "mbarrier";
  // The type name of the mbarrier array
  const std::string mbarrier_dtype_ = "Barrier";
  // The alignment of the barrier array in shared memory
  // Set to 16 to maintain minimum alignment requirements for async bulk copy
  const int barrier_alignment_bytes_ = 16;

  std::unordered_map<const VarNode *, std::string> fragment_shapes;
  std::unordered_map<const VarNode *, std::string> fragment_layouts;
  std::unordered_map<const VarNode *, IntImm> unroll_factor;
  // Per-buffer storage alignment (bytes) collected from storage_alignment
  // AttrStmts, used to emit __align__(N) on shared/local allocations.
  std::unordered_map<const VarNode *, int> alloc_storage_alignment_;
  // Map function param VarNode to its index, populated by iterating
  // f->params in AddFunction.  The index matches the PrimFunc's parameter
  // declaration order and is used to map global buffers to PTX base regs.
  std::unordered_map<const VarNode *, int> func_param_index_;
  friend void PrintConst(const FloatImmNode *op, std::ostream &os,
                         CodeGenTileLangTANG *p);
  void PrintWmmaScope(const std::string &scope, DataType t,
                      const VarNode *variable, std::ostream &os);
  int32_t GetWmmaFragmentSize(const std::string &scope, const VarNode *variable,
                              int32_t size);

  std::vector<std::string> eviction_policy_names_ = {
      "EVICT_NORMAL", "EVICT_FIRST", "EVICT_LAST"};
  std::unordered_set<std::string> bf16_supported_ops_ = {
      "bf1622float2", "bf1622int16", "float22bf162", "bf162bf162"};
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_CODEGEN_TANG_H_
