/*!
 * \file verify_s3_mma_descriptors.cc
 * \brief Check the tcgen05 MMA descriptor contract before codegen (stcuv2).
 *
 * Why this exists
 * ---------------
 * The STCU backend consumes the generated source as LLVM IR, where the TANG
 * marker calls are vendor intrinsics:
 *
 *   tang::ptx::mma_data_desc<LBO, SBO, SW>(ptr)  -> __stvm_smem_descriptor
 *   tang::ptx::mma_desc<...>()                   -> __stvm_mma_descriptor
 *   tang::ptx::mma_scale_desc<...>()             -> __stvm_mma_scale_descriptor
 *   tang::ptx::mma<...>(d, a_desc, b_desc, i_desc) -> __stvm_smem_mma
 *
 * ``MMAAnalyzer::analyze`` (llvm/lib/Target/STCU/STCUV2TCProcess.cpp) reads
 * each descriptor *straight off the MMA intrinsic's operand list*, and requires
 * the operand to be that descriptor intrinsic itself:
 *
 *   Value *ADesc = mma->getArgOperand(1);
 *   IntrinsicInst *ADescIntr = dyn_cast<IntrinsicInst>(ADesc);
 *   if (!ADescIntr || ... != Intrinsic::stvm_smem_descriptor)
 *     reportUnsupport("Missing description information for matrix (smem).");
 *
 * ``reportUnsupport`` exits the compiler, and the message names neither the
 * kernel nor the statement. Anything that puts another instruction between the
 * descriptor and the MMA -- a variable that went through a load/phi/select, a
 * cast, a folded constant -- therefore turns into an unattributable backend
 * failure. On the TIR side that violation is still visible, together with the
 * statement that caused it, which is what this pass checks.
 *
 * What it checks / does not check
 * -------------------------------
 * Checked: every tcgen05 MMA operand that TCProcess reads as a descriptor *is*
 * the matching descriptor value op (``tl.tang_mma_data_desc`` /
 * ``tl.tang_mma_instr_desc`` / ``tl.tang_mma_scale_instr_desc``), with the
 * intended dtype, and the MMA op carries the operand count the codegen expects.
 *
 * Not checked: adjacency or sharing. TCProcess matches by def-use, so a
 * descriptor may sit in another basic block, outside a guard, or be consumed by
 * several MMAs -- none of that breaks the backend contract, so a verification
 * pass must not reject it either.
 */
#include "support/check.h"
#include "tang/op/builtin.h"
#include "tang/target_utils.h"

#include <tvm/ir/transform.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;
using tvm::transform::PassContext;

namespace {

/*!
 * \brief Verify that each tcgen05 MMA descriptor operand is the descriptor
 * value op TCProcess will read.
 */
class MmaDescriptorChecker : public StmtExprVisitor {
public:
  void VisitExpr_(const CallNode *op) final {
    const bool is_ss = op->op.same_as(tang_tcgen05_mma_ss());
    const bool is_ts = op->op.same_as(tang_tcgen05_mma_ts());
    const bool is_scale = op->op.same_as(tang_tcgen05_mma_scale());
    if (is_ss || is_ts || is_scale) {
      const char *name = is_ss   ? "tl.tang_tcgen05_mma_ss"
                         : is_ts ? "tl.tang_tcgen05_mma_ts"
                                 : "tl.tang_tcgen05_mma_scale";
      // The TS form takes an optional A column offset (the A operand's K
      // origin in TMEM cells) as its third argument; everything after it shifts
      // by one.
      const size_t expected_args = is_ts ? 5U : 6U;
      const bool ts_with_a_offset =
          is_ts && op->args.size() == expected_args + 1U;
      const bool scale_with_direct_tmem =
          is_scale && op->args.size() == expected_args + 1U;
      ICHECK(op->args.size() == expected_args || ts_with_a_offset ||
             scale_with_direct_tmem)
          << "TANG stcuv2: " << name << " expects " << expected_args
          << " arguments, or one extra direct-TMEM scale/A-offset argument, "
             "but got "
          << op->args.size() << " (see src/tang/op/builtin.h).";
      const size_t desc_base =
          (ts_with_a_offset || scale_with_direct_tmem) ? 3U : 2U;
      if (is_ss) {
        CheckDataDesc(op->args[2], name, "A");
        CheckDataDesc(op->args[3], name, "B");
        CheckInstrDesc(op->args[4], name, "instruction", tang_mma_instr_desc(),
                       "tl.tang_mma_instr_desc");
      } else if (is_ts) {
        CheckDataDesc(op->args[desc_base], name, "B");
        CheckInstrDesc(op->args[desc_base + 1], name, "instruction",
                       tang_mma_instr_desc(), "tl.tang_mma_instr_desc");
      } else {
        CheckDataDesc(op->args[desc_base], name, "A");
        CheckDataDesc(op->args[desc_base + 1], name, "B");
        CheckInstrDesc(op->args[desc_base + 2], name,
                       "block-scaled instruction", tang_mma_scale_instr_desc(),
                       "tl.tang_mma_scale_instr_desc");
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

private:
  static void CheckDescriptor(const PrimExpr &value, const Op &expected,
                              const char *expected_name, DataType dtype,
                              const char *mma_name, const char *role) {
    const auto *call = value.as<CallNode>();
    ICHECK(call != nullptr && call->op.same_as(expected))
        << "TANG stcuv2: the " << role << " descriptor operand of " << mma_name
        << " is not a `" << expected_name
        << "` value. The STCU backend's TCProcess pass reads every descriptor "
           "directly off the MMA call (MMAAnalyzer::analyze), so no lowering "
           "step "
           "may bind, cast, load or select a descriptor in between; such a "
           "step "
           "would make the backend fail with \"Missing description "
           "information\" "
           "and no mention of this kernel.";
    ICHECK_EQ(call->dtype, dtype)
        << "TANG stcuv2: the " << role << " descriptor operand of " << mma_name
        << " is a `" << expected_name << "` value of dtype " << call->dtype
        << ", but the codegen expects " << dtype << ".";
  }

  static void CheckDataDesc(const PrimExpr &value, const char *mma_name,
                            const char *role) {
    CheckDescriptor(value, tang_mma_data_desc(), "tl.tang_mma_data_desc",
                    DataType::UInt(64), mma_name, role);
  }

  static void CheckInstrDesc(const PrimExpr &value, const char *mma_name,
                             const char *role, const Op &expected,
                             const char *expected_name) {
    CheckDescriptor(value, expected, expected_name, DataType::UInt(32),
                    mma_name, role);
  }
};

} // namespace

/*!
 * \brief Verify the tcgen05 MMA descriptor contract (stcuv2 only).
 *
 * A pure verification: the module is returned unchanged, and the pass is a
 * no-op on other targets.
 */
tvm::transform::Pass VerifyMmaDescriptors() {
  auto pass_func = [](PrimFunc f, const IRModule &, const PassContext &) {
    Optional<Target> opt_target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!opt_target.defined() || !TargetTangIsSTCUV2(opt_target.value())) {
      return f;
    }
    MmaDescriptorChecker checker;
    checker(f->body);
    return f;
  };
  return tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.tang.VerifyMmaDescriptors", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.tang.transform.VerifyMmaDescriptors",
                        VerifyMmaDescriptors);
}

} // namespace tl
} // namespace tvm
