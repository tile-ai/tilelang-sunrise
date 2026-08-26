/*!
 * \file tl/tang/op/atomic.cc
 * \brief TANG implementations for TileLang atomic tile operators.
 */

#include "backend/common/op/atomic_reduce.h"
#include "op/atomic_add.h"
#include "op/atomic_generic.h"
#include "tang/target_utils.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

bool MatchTangAtomicTarget(Target target) { return TargetIsTang(target); }

LayoutMap InferTangAtomicAdd(const AtomicAddNode &op,
                             const LayoutInferArgs &layout_args,
                             InferLevel level) {
  return backend::AtomicReduce::InferLayout(op, layout_args, level);
}

Stmt LowerTangAtomicAdd(const AtomicAddNode &op, const LowerArgs &lower_args,
                        arith::Analyzer *analyzer) {
  if (auto use_tma = op.annotations.Get("use_tma")) {
    if (const auto *int_value = use_tma->as<IntImmNode>()) {
      ICHECK_EQ(int_value->value, 0)
          << "TMA atomic_add is not supported by the TANG backend";
    }
  }
  return backend::AtomicReduce::Lower(op, lower_args, analyzer);
}

LayoutMap InferTangAtomicGeneric(const AtomicGenericNode &op,
                                 const LayoutInferArgs &layout_args,
                                 InferLevel level) {
  return backend::AtomicReduce::InferLayout(op, layout_args, level);
}

Stmt LowerTangAtomicGeneric(const AtomicGenericNode &op,
                            const LowerArgs &lower_args,
                            arith::Analyzer *analyzer) {
  return backend::AtomicReduce::Lower(op, lower_args, analyzer);
}

bool RegisterTangAtomicOps() {
  RegisterAtomicAddImpl(AtomicAddImpl{
      "tang.AtomicAdd",
      MatchTangAtomicTarget,
      InferTangAtomicAdd,
      LowerTangAtomicAdd,
  });
  RegisterAtomicReduceImpl(AtomicReduceImpl{
      "tang.AtomicReduce",
      MatchTangAtomicTarget,
      backend::AtomicReduce::InferLayout,
      backend::AtomicReduce::Lower,
  });
  RegisterAtomicGenericImpl(AtomicGenericImpl{
      "tang.AtomicGeneric",
      MatchTangAtomicTarget,
      InferTangAtomicGeneric,
      LowerTangAtomicGeneric,
  });
  return true;
}

const bool tang_atomic_ops_registered = RegisterTangAtomicOps();

} // namespace

} // namespace tl
} // namespace tvm
