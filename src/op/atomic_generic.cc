/*!
 * \file tl/op/atomic_generic.cc
 * \brief Generic atomic tile operators with target-specific lowering.
 */

#include "atomic_generic.h"

#include "builtin.h"
#include "utils.h"
#include <tvm/tirx/op_attr_types.h>

#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

std::vector<AtomicGenericImpl> &AtomicGenericImplRegistry() {
  static std::vector<AtomicGenericImpl> registry;
  return registry;
}

const AtomicGenericImpl &ResolveAtomicGenericImpl(Target target) {
  const AtomicGenericImpl *matched_impl = nullptr;
  for (const AtomicGenericImpl &impl : AtomicGenericImplRegistry()) {
    if (!impl.match_target(target)) {
      continue;
    }
    ICHECK(matched_impl == nullptr)
        << "tl.atomic_generic found multiple target-specific implementations "
           "for "
        << target->str() << ": " << matched_impl->name << " and " << impl.name;
    matched_impl = &impl;
  }
  ICHECK(matched_impl != nullptr)
      << "tl.atomic_generic requires a target-specific implementation, but no "
         "implementation is registered for "
      << target->str();
  return *matched_impl;
}

ObjectPtr<AtomicGenericNode>
MakeAtomicGeneric(Array<PrimExpr> args, Map<String, ObjectRef> annotations,
                  const char *elem_op_name, bool is_cas) {
  const size_t expected_args = is_cas ? 3 : 2;
  ICHECK_EQ(args.size(), expected_args)
      << elem_op_name << " expects " << expected_args << " arguments, got "
      << args.size();

  ObjectPtr<AtomicGenericNode> node = make_object<AtomicGenericNode>();
  std::vector<AccessRegion> access_regions;
  size_t dst_index = 1;

  if (is_cas) {
    ICHECK(!IsBufferLikeExpr(args[0]) && !IsBufferLikeExpr(args[1]))
        << "tensor-wise atomic CAS currently requires scalar compare and "
           "replacement values";
    node->src_value = args[0];
    node->src_value2 = args[1];
    dst_index = 2;
  } else if (IsBufferLikeExpr(args[0])) {
    auto src_access = NormalizeToAccessRegion(args[0], kAccessRead);
    node->src = src_access.region->buffer;
    node->src_range = src_access.region->region;
    access_regions.push_back(std::move(src_access));
  } else {
    node->src_value = args[0];
  }

  auto dst_access = NormalizeToAccessRegion(args[dst_index], kAccessReadWrite);
  dst_access.access_mask = kAccessReadWrite;
  node->dst = dst_access.region->buffer;
  node->dst_range = dst_access.region->region;
  access_regions.push_back(std::move(dst_access));
  node->SetAccessRegions(std::move(access_regions));
  node->elem_op_name = elem_op_name;
  node->annotations = std::move(annotations);
  return node;
}

} // namespace

void RegisterAtomicGenericImpl(AtomicGenericImpl impl) {
  ICHECK(impl.name != nullptr);
  ICHECK(impl.match_target != nullptr);
  ICHECK(impl.infer_layout != nullptr);
  ICHECK(impl.lower != nullptr);
  AtomicGenericImplRegistry().push_back(impl);
}

Stmt AtomicGenericNode::Lower(const LowerArgs &lower_args,
                              arith::Analyzer *analyzer) const {
  return ResolveAtomicGenericImpl(lower_args.target)
      .lower(*this, lower_args, analyzer);
}

LayoutMap AtomicGenericNode::InferLayout(const LayoutInferArgs &layout_args,
                                         InferLevel level) const {
  return ResolveAtomicGenericImpl(layout_args.target)
      .infer_layout(*this, layout_args, level);
}

const Op &AtomicGenericNode::GetElemOp() const { return Op::Get(elem_op_name); }

Array<PrimExpr> AtomicGenericNode::GetExtraValues() const {
  if (src_value2.defined()) {
    return {src_value2};
  }
  return {};
}

TileOperator AtomicGenericNode::Clone() const {
  auto op = make_object<AtomicGenericNode>(*this);
  if (par_op_.defined()) {
    op->par_op_ = Downcast<ParallelOp>(par_op_->Clone());
  }
  return TileOperator(op);
}

void AtomicGenericNode::RegisterReflection() {
  namespace refl = reflection;
  refl::ObjectDef<AtomicGenericNode>()
      .def_ro("src", &AtomicGenericNode::src)
      .def_ro("src_value", &AtomicGenericNode::src_value)
      .def_ro("src_value2", &AtomicGenericNode::src_value2)
      .def_ro("dst", &AtomicGenericNode::dst)
      .def_ro("src_range", &AtomicGenericNode::src_range)
      .def_ro("dst_range", &AtomicGenericNode::dst_range)
      .def_ro("elem_op_name", &AtomicGenericNode::elem_op_name)
      .def_ro("annotations", &AtomicGenericNode::annotations);
}

#define TVM_TL_DEFINE_ATOMIC_GENERIC_WRAPPER(ClassName, OpName, ElemOpName,    \
                                             IsCAS)                            \
  ClassName::ClassName(Array<PrimExpr> args,                                   \
                       Map<String, ObjectRef> annotations) {                   \
    data_ = MakeAtomicGeneric(std::move(args), std::move(annotations),         \
                              ElemOpName, IsCAS);                              \
  }                                                                            \
  TIR_REGISTER_TL_TILE_OP(ClassName, OpName)                                   \
      .set_num_inputs(IsCAS ? 3 : 2)                                           \
      .set_attr<TCallEffectKind>("TCallEffectKind",                            \
                                 Integer(CallEffectKind::kOpaque))

TVM_TL_DEFINE_ATOMIC_GENERIC_WRAPPER(AtomicSub, atomicsub,
                                     "tl.atomic_sub_elem_op", false);
TVM_TL_DEFINE_ATOMIC_GENERIC_WRAPPER(AtomicExch, atomicexch,
                                     "tl.atomic_exch_elem_op", false);
TVM_TL_DEFINE_ATOMIC_GENERIC_WRAPPER(AtomicInc, atomicinc,
                                     "tl.atomic_inc_elem_op", false);
TVM_TL_DEFINE_ATOMIC_GENERIC_WRAPPER(AtomicDec, atomicdec,
                                     "tl.atomic_dec_elem_op", false);
TVM_TL_DEFINE_ATOMIC_GENERIC_WRAPPER(AtomicCAS, atomiccas,
                                     "tl.atomic_cas_elem_op", true);
TVM_TL_DEFINE_ATOMIC_GENERIC_WRAPPER(AtomicAnd, atomicand,
                                     "tl.atomic_and_elem_op", false);
TVM_TL_DEFINE_ATOMIC_GENERIC_WRAPPER(AtomicOr, atomicor, "tl.atomic_or_elem_op",
                                     false);
TVM_TL_DEFINE_ATOMIC_GENERIC_WRAPPER(AtomicXor, atomicxor,
                                     "tl.atomic_xor_elem_op", false);

#undef TVM_TL_DEFINE_ATOMIC_GENERIC_WRAPPER

TVM_FFI_STATIC_INIT_BLOCK() { AtomicGenericNode::RegisterReflection(); }

} // namespace tl
} // namespace tvm
