/*!
 * \file tl/op/atomic_generic.h
 * \brief Target-dispatched generic atomic tile operations.
 */

#ifndef TVM_TL_OP_ATOMIC_GENERIC_H_
#define TVM_TL_OP_ATOMIC_GENERIC_H_

#include "atomic_reduce.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

class AtomicGenericNode : public AtomicOpBaseNode {
public:
  String elem_op_name;
  PrimExpr src_value2;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.AtomicGeneric", AtomicGenericNode,
                                    TileOperatorNode);

  Stmt Lower(const LowerArgs &lower_args,
             arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &layout_args,
                        InferLevel level) const;
  const Op &GetElemOp() const override;
  Array<PrimExpr> GetExtraValues() const override;
  TileOperator Clone() const;

  static void RegisterReflection();
};

using AtomicGenericTargetPredicate = bool (*)(Target target);

struct AtomicGenericImpl {
  const char *name;
  AtomicGenericTargetPredicate match_target;
  LayoutMap (*infer_layout)(const AtomicGenericNode &op,
                            const LayoutInferArgs &layout_args,
                            InferLevel level);
  Stmt (*lower)(const AtomicGenericNode &op, const LowerArgs &lower_args,
                arith::Analyzer *analyzer);
};

void RegisterAtomicGenericImpl(AtomicGenericImpl impl);

#define TVM_TL_DECLARE_ATOMIC_GENERIC_WRAPPER(ClassName)                       \
  class ClassName : public TileOperator {                                      \
  public:                                                                      \
    TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ClassName, TileOperator,        \
                                               AtomicGenericNode);             \
    TVM_DLL                                                                    \
    ClassName(Array<PrimExpr> args,                                            \
              Map<String, ObjectRef> annotations = Map<String, ObjectRef>());  \
    static const Op &Get();                                                    \
  }

TVM_TL_DECLARE_ATOMIC_GENERIC_WRAPPER(AtomicSub);
TVM_TL_DECLARE_ATOMIC_GENERIC_WRAPPER(AtomicExch);
TVM_TL_DECLARE_ATOMIC_GENERIC_WRAPPER(AtomicInc);
TVM_TL_DECLARE_ATOMIC_GENERIC_WRAPPER(AtomicDec);
TVM_TL_DECLARE_ATOMIC_GENERIC_WRAPPER(AtomicCAS);
TVM_TL_DECLARE_ATOMIC_GENERIC_WRAPPER(AtomicAnd);
TVM_TL_DECLARE_ATOMIC_GENERIC_WRAPPER(AtomicOr);
TVM_TL_DECLARE_ATOMIC_GENERIC_WRAPPER(AtomicXor);

#undef TVM_TL_DECLARE_ATOMIC_GENERIC_WRAPPER

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_ATOMIC_GENERIC_H_
