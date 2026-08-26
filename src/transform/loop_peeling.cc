/*!
 * \file loop_peeling.cc
 * \brief Peel the M/N tail of GEMM block copies when a matrix dimension is
 *        not divisible by the block size.
 *
 * For a copy like `T.copy(A[by * block_M, k * block_K], A_shared)`, when the
 * M or N dimension is not divisible by the block size the last grid block
 * reads past the end of A.  This pass rewrites the copy into an if/else over
 * the peeled boundary: the main branch keeps the full block extent (covering
 * (dim / block) * block elements), and the tail branch shrinks the extent to
 * the static remainder `dim % block`.
 *
 * The M/N tail is safe to peel directly: the uninitialized tail of the shared
 * buffer lands in the accumulator tail that the correspondingly-shrunk C store
 * never writes back.  The K tail is not peeled here: it is the accumulation
 * dimension, so a shrunk copy would leave garbage in every accumulator
 * element; PadGemmTail handles K instead (full extent + out-of-bounds
 * predicate).  Both branches keep a *constant* loop extent, so the vectorizer
 * and layout inference stay happy.
 */

#include "support/check.h"
#include <tvm/ir/transform.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_set>
#include <utility>

#include "op/operator.h"
#include "op/region.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

class LoopPeelingMutator : public StmtExprMutator {
  // Loop vars of blockIdx.* thread-extent bindings (the M/N grid dimensions).
  std::unordered_set<const VarNode *> block_idx_vars_;

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      if (const auto *iter_var = op->node.as<IterVarNode>()) {
        std::string tag = iter_var->thread_tag;
        if (tag.rfind("blockIdx.", 0) == 0) {
          block_idx_vars_.insert(iter_var->var.get());
        }
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  bool UsesBlockIdxVar(const PrimExpr &expr) {
    bool found = false;
    UsesVar(expr, [&](const VarNode *v) {
      if (block_idx_vars_.count(v)) {
        found = true;
        return true;
      }
      return false;
    });
    return found;
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    static const Op &copy_op = Op::Get("tl.tileop.copy");
    const CallNode *call = op->value.as<CallNode>();
    if (call == nullptr || !call->op.same_as(copy_op))
      return StmtExprMutator::VisitStmt_(op);
    const auto *src_call = call->args[0].as<CallNode>();
    const auto *dst_call = call->args[1].as<CallNode>();
    if (src_call == nullptr || !src_call->op.same_as(RegionOp::Get()) ||
        dst_call == nullptr || !dst_call->op.same_as(RegionOp::Get()))
      return StmtExprMutator::VisitStmt_(op);

    RegionOp src_region(src_call->args);
    RegionOp dst_region(dst_call->args);
    const Buffer &src_buf = src_region->GetBuffer();
    const Buffer &dst_buf = dst_region->GetBuffer();
    const Array<Range> &src_ranges = src_region->GetRanges();
    const Array<Range> &dst_ranges = dst_region->GetRanges();
    if (src_ranges.size() != dst_ranges.size())
      return StmtExprMutator::VisitStmt_(op);

    Call copy = GetRef<Call>(call);
    return PeelDim(copy, src_buf, dst_buf, src_ranges, dst_ranges, 0);
  }

  // Recursively peel tail dimensions.  A tail dim is one whose buffer shape is
  // not statically divisible by the copy extent.  Returns the (possibly
  // if-branched) statement for the copy.
  Stmt PeelDim(const Call &copy, const Buffer &src_buf, const Buffer &dst_buf,
               const Array<Range> &src_ranges, const Array<Range> &dst_ranges,
               size_t dim) {
    if (dim >= src_ranges.size())
      return Evaluate(copy);

    bool src_mn = UsesBlockIdxVar(src_ranges[dim]->min);
    bool dst_mn = UsesBlockIdxVar(dst_ranges[dim]->min);

    // Only peel M/N dims (min depends on a blockIdx var).  The K dim (min
    // depends on the pipeline loop var) is left to PadGemmTail.
    if (!src_mn && !dst_mn)
      return PeelDim(copy, src_buf, dst_buf, src_ranges, dst_ranges, dim + 1);

    // Tail detection must look at the *global* side (A/B/C), not the shared
    // side (A_shared, ...) whose shape is the block size.  For a global->shared
    // copy the source is global; for shared->global it is the destination.
    bool src_is_global = src_buf.scope() == "global";
    const PrimExpr &shape =
        src_is_global ? src_buf->shape[dim] : dst_buf->shape[dim];
    const PrimExpr &min =
        src_is_global ? src_ranges[dim]->min : dst_ranges[dim]->min;
    const int64_t *shape_c = as_const_int(shape);
    const int64_t *extent_c = as_const_int(src_ranges[dim]->extent);
    if (shape_c == nullptr || extent_c == nullptr || *extent_c == 0 ||
        (*shape_c % *extent_c) == 0)
      return PeelDim(copy, src_buf, dst_buf, src_ranges, dst_ranges, dim + 1);

    // Static remainder: the last block/iteration copies only `remainder` elems.
    int64_t remainder = *shape_c % *extent_c;
    PrimExpr boundary = IntImm(min->dtype, (*shape_c / *extent_c) * *extent_c);
    PrimExpr main_cond = min < boundary;

    Stmt main_stmt =
        PeelDim(copy, src_buf, dst_buf, src_ranges, dst_ranges, dim + 1);
    Call tail_copy = RebuildExtent(copy, dim, IntImm(min->dtype, remainder));
    Stmt tail_stmt =
        PeelDim(tail_copy, src_buf, dst_buf, src_ranges, dst_ranges, dim + 1);

    return IfThenElse(main_cond, main_stmt, tail_stmt);
  }

  // Rebuild the copy call with `new_extent` at the given dim on both sides.
  Call RebuildExtent(const Call &copy, size_t dim, PrimExpr new_extent) {
    Array<PrimExpr> new_args;
    for (const auto &arg : copy->args) {
      const auto *region_call = arg.as<CallNode>();
      if (region_call != nullptr && region_call->op.same_as(RegionOp::Get())) {
        Array<PrimExpr> region_args;
        for (size_t j = 0; j < region_call->args.size(); ++j) {
          if (j == 2 + dim)
            region_args.push_back(new_extent);
          else
            region_args.push_back(region_call->args[j]);
        }
        new_args.push_back(Call(region_call->dtype, region_call->op,
                                region_args, region_call->annotations,
                                region_call->span));
      } else {
        new_args.push_back(arg);
      }
    }
    return Call(copy->dtype, copy->op, new_args, copy->annotations, copy->span);
  }
};

} // namespace

tvm::transform::Pass LoopPeeling() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) -> PrimFunc {
    LoopPeelingMutator mutator;
    func.CopyOnWrite()->body = mutator(func->body);
    return func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LoopPeeling", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LoopPeeling", LoopPeeling);
}

} // namespace tl
} // namespace tvm
