/*!
 * \file pad_gemm_tail.cc
 * \brief Pad the K tail of GEMM block copies when K is not divisible by
 *        block_K.
 *
 * For a copy like `T.copy(A[by * block_M, k * block_K], A_shared)`, when K is
 * not divisible by block_K the last pipeline iteration reads past the end of
 * A.  This pass wraps the copy in an if/else over the padded boundary:
 *
 *   if (k * block_K < boundary) { full-extent copy } else { full-extent copy }
 *
 * Both branches keep the *full* block extent (the copy is not shrunk).  The
 * out-of-bounds loads in the tail branch are legalized by the copy lowering
 * (`CopyNode::MakeSIMTLoop` emits `if_then_else(pred, load, 0)` for OOB
 * global loads), so the K tail reads 0 — equivalent to zero-padding the
 * accumulation dimension.
 *
 * The value of the if/else is that the analyzer now proves `min + iv < K`
 * inside the *main* branch (constrained by `k * block_K < boundary`), so the
 * main branch's copies lose the per-element runtime predicate that otherwise
 * penalizes the common non-tail path for non-divisible K.
 *
 * Only the K dimension (serial-loop-indexed) is padded.  M/N are left to
 * LoopPeeling: a shrunk M/N tail copies less work, whereas a full-extent pad
 * concentrates the OOB predicate in one grid block, and the slowest block sets
 * the whole kernel's latency under lockstep execution.
 *
 * Note: this is the *kernel-level* fallback.  The preferred path for arbitrary
 * M/N/K padding is the data layer — `tl.gemm_pad_m/n/k` (extra padding size)
 * makes tilelang.jit pad the A/B inputs before invoking the kernel, which is
 * faster and covers all three dimensions.
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

#include "op/builtin.h"
#include "op/operator.h"
#include "op/region.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

class PadGemmTailMutator : public StmtExprMutator {
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

  // True when the expression references a var that is NOT a blockIdx var, i.e.
  // a pipeline serial-loop var.  Used to identify the K dimension (min = k *
  // block_K).
  bool UsesNonBlockIdxVar(const PrimExpr &expr) {
    bool found = false;
    UsesVar(expr, [&](const VarNode *v) {
      if (!block_idx_vars_.count(v)) {
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
    return PadDim(copy, src_buf, dst_buf, src_ranges, dst_ranges, 0);
  }

  // Recursively pad the K tail dimension.  A tail dim is one whose global-side
  // buffer shape is not statically divisible by the copy extent.  Returns the
  // (possibly if-branched) statement for the copy.
  Stmt PadDim(const Call &copy, const Buffer &src_buf, const Buffer &dst_buf,
              const Array<Range> &src_ranges, const Array<Range> &dst_ranges,
              size_t dim) {
    if (dim >= src_ranges.size())
      return Evaluate(copy);

    bool src_mn = UsesBlockIdxVar(src_ranges[dim]->min);
    bool dst_mn = UsesBlockIdxVar(dst_ranges[dim]->min);
    bool src_k = !src_mn && UsesNonBlockIdxVar(src_ranges[dim]->min);
    bool dst_k = !dst_mn && UsesNonBlockIdxVar(dst_ranges[dim]->min);

    // Only pad the K dimension (serial-loop-indexed).  M/N (blockIdx-indexed)
    // are left to LoopPeeling.  A dim whose min is constant (e.g. the stage-1
    // prefetch `0 * block_K`) is left alone.
    if (!src_k && !dst_k)
      return PadDim(copy, src_buf, dst_buf, src_ranges, dst_ranges, dim + 1);

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
      return PadDim(copy, src_buf, dst_buf, src_ranges, dst_ranges, dim + 1);

    PrimExpr boundary = IntImm(min->dtype, (*shape_c / *extent_c) * *extent_c);
    PrimExpr main_cond = min < boundary;

    // Both branches run the SAME full-extent copy: the copy is not shrunk.
    // The if/else exists only to give the analyzer the `min < boundary`
    // constraint in the main branch, letting it prove `min + iv < K` and drop
    // the per-element runtime predicate. The tail branch still reads OOB, which
    // CopyNode::MakeSIMTLoop legalizes to 0 (equivalent to zero-padding K).
    Stmt main_stmt =
        PadDim(copy, src_buf, dst_buf, src_ranges, dst_ranges, dim + 1);
    Stmt tail_stmt =
        PadDim(copy, src_buf, dst_buf, src_ranges, dst_ranges, dim + 1);

    return IfThenElse(main_cond, main_stmt, tail_stmt);
  }
};

} // namespace

tvm::transform::Pass PadGemmTail() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) -> PrimFunc {
    PadGemmTailMutator mutator;
    func.CopyOnWrite()->body = mutator(func->body);
    return func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.PadGemmTail", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.PadGemmTail", PadGemmTail);
}

} // namespace tl
} // namespace tvm
