/*!
 * \file merge_loop.cc
 * \brief Merge adjacent For loops with the same iteration domain
 *
 * Adjacent For loops from independent copy operations are merged into a single
 * loop with a SeqStmt body, to reduce loop overhead and improve
 * instruction-level parallelism:
 *
 *   for i in parallel(0, N) { load(A) }      for i in parallel(0, N) {
 *   for j in parallel(0, N) { load(B) }  →     load(A)
 *                                              load(B)
 *                                            }
 *
 * Two loops merge when they agree on shape — same ForKind, min 0, deep-equal
 * extent, same loop_var dtype, trivial step, no thread_binding, no annotations
 * (see CanMergeLoops) — *and* the fusion is dependence-legal (IsFusionLegal)
 * and wrapper-compatible (WrappersCompatible). A shape-legal fusion is also
 * declined when it would cost a shared-memory reuse opportunity worth more than
 * the saved loop overhead (see FusionWouldBlockSharedReuse).
 *
 * The pass is exposed only as the tl.MergeLoop TIR pass; there is no C++ entry
 * point, so it has no header.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_set>

#include "../op/builtin.h"
#include "runtime/thread_storage_scope.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

/*!
 * \brief Per-loop-body summary of which allocations a candidate loop touches.
 *
 * Fusing two loops reorders every iteration of the second loop against every
 * *later* iteration of the first: after fusion, iteration i of loop B runs
 * before iteration i+1 of loop A. So the fused form is only equivalent when no
 * access in B can observe or clobber an access loop A performs at a different
 * index. We answer that at allocation granularity — see IsFusionLegal.
 *
 * `opaque` means "there is, or may be, an access here that cannot be attributed
 * to a specific allocation" — opaque calls, address-of / access-ptr escapes, or
 * statement kinds we do not model. It must alias everything: a disjointness
 * test has to bail out. Getting this polarity wrong silently fuses a racing
 * pair, which is how the pass produced wrong results before.
 */
struct AccessSummary {
  std::unordered_set<const VarNode *> reads;
  std::unordered_set<const VarNode *> writes;
  bool opaque{false};
};

class AccessCollector : public StmtExprVisitor {
public:
  explicit AccessCollector(AccessSummary *out) : out_(out) {}

private:
  void VisitExpr_(const BufferLoadNode *op) final {
    out_->reads.insert(op->buffer->data.get());
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    out_->writes.insert(op->buffer->data.get());
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    // A call that escapes a raw pointer (address_of, tvm_access_ptr) or that
    // updates opaque state may touch memory in ways the buffer-level walk
    // cannot see. Treat it as aliasing everything.
    if (SideEffect(GetRef<Call>(op)) > CallEffectKind::kReadState) {
      out_->opaque = true;
    } else if (op->op.same_as(builtin::address_of()) ||
               op->op.same_as(builtin::tvm_access_ptr())) {
      out_->opaque = true;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  // Statement kinds we do not model: be conservative rather than silent.
  void VisitStmt_(const WhileNode *op) final {
    out_->opaque = true;
    StmtExprVisitor::VisitStmt_(op);
  }

  AccessSummary *out_;
};

void CollectAccess(const Stmt &stmt, AccessSummary *out) {
  AccessCollector collector(out);
  collector(stmt);
}

/*!
 * \brief Is fusing `first` before `second` (sharing one loop var) legal?
 *
 * Legal when the two bodies are *disjoint at allocation granularity*: no
 * allocation is written by one and touched by the other. Read-read sharing is
 * fine, so only write-write, write-read and read-write pairs are rejected.
 *
 * Disjointness is the whole test on purpose — nothing finer is attempted.
 * An earlier draft tried to also admit conflicting allocations whose index
 * expressions were structurally identical, on the theory that identical
 * indices keep the dependence inside one fused iteration. That is wrong: an
 * index that does not involve the fused loop var, e.g.
 *
 *   for i: for u in 4: S[u] = f(i)      // structurally identical indices...
 *   for i: for v in 4: ... = S[v]       // ...but reads f(N-1) before fusion
 *
 * hits the same elements every iteration, so fusion changes which value is
 * observed while passing an identical-index check. Admitting that case
 * soundly needs the index to be injective in the fused loop var, which needs
 * affine analysis; disjointness needs none and is trivially sound.
 *
 * This is deliberately conservative: it rejects legal fusions that touch a
 * shared allocation at provably distinct elements, in exchange for never
 * accepting an illegal one.
 */
bool IsFusionLegal(const Stmt &first, const Stmt &second) {
  AccessSummary a, b;
  CollectAccess(first, &a);
  CollectAccess(second, &b);

  if (a.opaque || b.opaque)
    return false;

  auto intersects = [](const std::unordered_set<const VarNode *> &lhs,
                       const std::unordered_set<const VarNode *> &rhs) {
    for (const VarNode *var : lhs) {
      if (rhs.count(var))
        return true;
    }
    return false;
  };

  if (intersects(a.writes, b.writes))
    return false; // WAW
  if (intersects(a.writes, b.reads))
    return false; // RAW
  if (intersects(a.reads, b.writes))
    return false; // WAR
  return true;
}

static bool IsSharedVar(const VarNode *var) {
  const auto *ptr_type = var->type_annotation.as<PointerTypeNode>();
  if (ptr_type == nullptr)
    return false;
  runtime::StorageScope scope =
      runtime::StorageScope::Create(ptr_type->storage_scope);
  return scope.rank == runtime::StorageRank::kShared;
}

/*!
 * \brief Would fusing `first` before `second` destroy a shared-memory reuse
 * opportunity that MergeSharedMemoryAllocations would otherwise take?
 *
 * MergeSharedMemoryAllocations packs shared buffers into one arena, giving two
 * buffers the same offset when their lifetimes do not overlap. Its liveness is
 * computed per statement, so a buffer drained in one loop and a buffer first
 * written in the *next* loop are non-overlapping and share an offset.
 *
 * Fusion collapses both into one statement, and inside a fused loop the
 * lifetimes genuinely do overlap: iteration i+1 re-reads the first buffer after
 * iteration i has already written the second. The arena packer is right to
 * refuse, so the reuse is lost — trading a whole buffer's worth of shared
 * memory (which caps occupancy) for one loop's launch overhead.
 *
 * The shape that loses reuse is asymmetric, which is what makes this cheap to
 * detect: `first` *reads* a shared buffer that `second` does not touch, while
 * `second` *writes* a different shared buffer. That is the drain-then-refill
 * pattern above. It deliberately does not fire on the write/write case — two
 * loops each filling their own shared tile, as in a GEMM's As/Bs prologue —
 * because those buffers are both live into the consumer regardless, so there
 * was no reuse to lose and the fusion is pure win.
 *
 * Only fusion is declined; nothing here affects correctness, so a false
 * positive costs at most one unfused loop.
 */
bool FusionWouldBlockSharedReuse(const Stmt &first, const Stmt &second) {
  AccessSummary a, b;
  CollectAccess(first, &a);
  CollectAccess(second, &b);

  auto has_shared_not_in = [](const std::unordered_set<const VarNode *> &vars,
                              const AccessSummary &other) {
    for (const VarNode *var : vars) {
      if (!IsSharedVar(var))
        continue;
      if (other.reads.count(var) || other.writes.count(var))
        continue;
      return true;
    }
    return false;
  };

  // `first` drains a shared buffer `second` never touches, and `second` starts
  // filling a shared buffer of its own: exactly the pair the arena packer would
  // have overlapped.
  return has_shared_not_in(a.reads, b) && has_shared_not_in(b.writes, a);
}

} // namespace

class MergeLoopRewriter : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc &f) {
    auto rewriter = MergeLoopRewriter();
    f.CopyOnWrite()->body = rewriter(f->body);
    return f;
  }

private:
  MergeLoopRewriter() = default;

  // A candidate in a mergeable run: the original statement (a For, possibly
  // wrapped in an AttrStmt) together with the ForNode it unwraps to. Carrying
  // both means the unwrap happens exactly once — at the point where UnwrapFor
  // already established the shape — so no later step has to re-derive it and
  // handle a "cannot happen" failure.
  struct Candidate {
    Stmt stmt;
    const ForNode *loop;
  };

  void FlattenAppend(const Stmt &s, Array<Stmt> *out) {
    if (const auto *seq = s.as<SeqStmtNode>()) {
      for (const Stmt &e : seq->seq) {
        FlattenAppend(e, out);
      }
    } else {
      out->push_back(s);
    }
  }

  // Try to unwrap a For node from inside an AttrStmt wrapper.
  // E.g. AttrStmt(pragma_unroll_explicit=0, body=For(...)) → the ForNode.
  // Returns nullptr if stmt is not a For (possibly AttrStmt-wrapped).
  static const ForNode *UnwrapFor(const Stmt &stmt) {
    if (const auto *f = stmt.as<ForNode>()) {
      return f;
    }
    if (const auto *a = stmt.as<AttrStmtNode>()) {
      return a->body.as<ForNode>();
    }
    return nullptr;
  }

  // Does this AttrStmt's `node` refer to the wrapped loop's own loop_var?
  //
  // LowerOpaqueBlock lowers a loop annotation to AttrStmt(op->loop_var, key,
  // value, loop) — the node is the loop's *own* var by construction, as a
  // handle naming which loop the pragma applies to. It carries no meaning that
  // could differ between two otherwise-identical wrappers, and after fusion the
  // surviving wrapper's node is the surviving loop's var, so it stays correct.
  static bool NodeIsOwnLoopVar(const AttrStmtNode *attr) {
    const auto *loop = attr->body.as<ForNode>();
    if (loop == nullptr)
      return false;
    const auto *node_var = attr->node.as<VarNode>();
    return node_var != nullptr && node_var == loop->loop_var.get();
  }

  // An AttrStmt wrapper (e.g. async_scope) marks a per-loop scope, but a merged
  // run keeps only one wrapper for the single fused loop. Merging is therefore
  // only wrapper-safe when both stmts carry the *same* wrapper: then applying
  // that one wrapper to the fused body is semantically identical (the GEMM case
  // of two async_scope copies). Any other combination silently changes scope:
  //   [async, plain] would pull the plain loop into the async scope;
  //   [plain, async] would drop the async scope entirely;
  //   mismatched keys/values would apply the wrong scope.
  // So require the wrappers to agree on attr_key and value, and on node —
  // *except* that a node which is just the loop's own loop_var is exempt from
  // the node comparison, since it names the loop rather than describing scope
  // and therefore necessarily differs between two distinct loops. Comparing it
  // structurally made every pragma-annotated pair incompatible, which silently
  // reduced the whole pass to a no-op (pragma_unroll_explicit wraps essentially
  // every lowered loop). Both sides must be self-referential or neither: a
  // self-referential node paired with a real one is a genuine mismatch.
  // This is checked per candidate, not just against the first loop, since the
  // whole run collapses to one wrapper.
  static bool WrappersCompatible(const Stmt &lhs, const Stmt &rhs) {
    const auto *la = lhs.as<AttrStmtNode>();
    const auto *ra = rhs.as<AttrStmtNode>();
    if (la == nullptr && ra == nullptr)
      return true;
    if (la == nullptr || ra == nullptr)
      return false;
    if (la->attr_key != ra->attr_key)
      return false;
    if (!StructuralEqual()(la->value, ra->value))
      return false;
    bool l_self = NodeIsOwnLoopVar(la);
    bool r_self = NodeIsOwnLoopVar(ra);
    if (l_self != r_self)
      return false;
    if (l_self)
      return true;
    return StructuralEqual()(la->node, ra->node);
  }

  bool CanMergeLoops(const ForNode *a, const ForNode *b) {
    // Must have the same ForKind
    if (a->kind != b->kind)
      return false;

    // Min must be zero for both
    if (!is_zero(a->min) || !is_zero(b->min))
      return false;

    // Extents must be structurally equal
    if (!ExprDeepEqual()(a->extent, b->extent))
      return false;

    // Loop variable types must match
    if (a->loop_var->dtype != b->loop_var->dtype)
      return false;

    // A non-unit step changes the iteration set, and the merged loop keeps only
    // the first loop's step. Merging `step=1` with `step=2` would silently run
    // the second body on iterations it never had.
    if (!a->HasTrivialStep() || !b->HasTrivialStep())
      return false;

    // Skip if either has thread_binding
    if (a->thread_binding.defined() || b->thread_binding.defined())
      return false;

    // Skip if either has annotations (conservative for v1)
    if (!a->annotations.empty() || !b->annotations.empty())
      return false;

    return true;
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    // First, recursively flatten nested SeqStmt and visit children.
    Array<Stmt> flat_seq;
    for (const Stmt &stmt : op->seq) {
      Stmt new_stmt = this->VisitStmt(stmt);
      FlattenAppend(new_stmt, &flat_seq);
    }

    // Merge adjacent For loops (possibly wrapped in AttrStmt) that satisfy
    // CanMergeLoops. Two stmts are "adjacent For" if both unwrap to ForNode,
    // even if one or both are wrapped in AttrStmt (e.g. async_scope).
    Array<Stmt> new_seq;
    size_t i = 0;
    while (i < flat_seq.size()) {
      const ForNode *for_node = UnwrapFor(flat_seq[i]);
      if (for_node != nullptr) {
        // Collect a run of mergeable (possibly AttrStmt-wrapped) For loops.
        std::vector<Candidate> run;
        run.push_back({flat_seq[i], for_node});
        size_t j = i + 1;
        while (j < flat_seq.size()) {
          const ForNode *next_for = UnwrapFor(flat_seq[j]);
          if (next_for == nullptr)
            break;
          if (!CanMergeLoops(for_node, next_for))
            break;
          // Fusing a run of n loops reorders *every* pair in it, not just each
          // loop against the first: after fusion iteration i of loop n runs
          // before iteration i+1 of loop 1. So the new candidate must be
          // independent of, and wrapper-compatible with, every loop already in
          // the run (the whole run collapses to one loop under one wrapper).
          bool legal = true;
          for (const Candidate &prev : run) {
            if (!WrappersCompatible(prev.stmt, flat_seq[j]) ||
                !IsFusionLegal(prev.loop->body, next_for->body) ||
                FusionWouldBlockSharedReuse(prev.loop->body, next_for->body)) {
              legal = false;
              break;
            }
          }
          if (!legal)
            break;
          run.push_back({flat_seq[j], next_for});
          j++;
        }

        if (j == i + 1) {
          // Only one loop, keep as-is
          new_seq.push_back(flat_seq[i]);
        } else {
          // If the first stmt was AttrStmt-wrapped, keep the wrapper for the
          // merged For so that annotations (async_scope) are preserved.
          const auto *wrapper = run[0].stmt.as<AttrStmtNode>();

          // Collect the inner For bodies, rewriting each subsequent loop's var
          // to the first loop's var so one loop var drives the fused body.
          Array<Stmt> bodies;
          bodies.push_back(for_node->body);
          for (size_t k = 1; k < run.size(); k++) {
            Map<Var, PrimExpr> var_map;
            var_map.Set(run[k].loop->loop_var, for_node->loop_var);
            bodies.push_back(tirx::Substitute(run[k].loop->body, var_map));
          }

          // This branch runs only when j > i + 1, i.e. the run holds at least
          // two loops, so `bodies` always needs a SeqStmt wrapper.
          ICHECK_GE(bodies.size(), 2u);
          For merged_for(for_node->loop_var, for_node->min, for_node->extent,
                         for_node->kind, SeqStmt(bodies),
                         for_node->thread_binding, for_node->annotations,
                         for_node->step, for_node->span);

          if (wrapper != nullptr) {
            // Every loop in the run carries the same wrapper (guaranteed by
            // WrappersCompatible), so reapplying the first one to the fused
            // body preserves the shared scope, e.g. async_scope.
            new_seq.push_back(AttrStmt(wrapper->node, wrapper->attr_key,
                                       wrapper->value, merged_for));
          } else {
            new_seq.push_back(merged_for);
          }
        }
        i = j;
      } else {
        // Non-For statement, pass through
        new_seq.push_back(flat_seq[i]);
        i++;
      }
    }

    return new_seq.size() == 1 ? new_seq[0] : SeqStmt(new_seq);
  }
};

using namespace tirx::transform;
tvm::transform::Pass MergeLoop() {
  auto pass_func = [=](PrimFunc f, const IRModule &m,
                       const PassContext &ctx) -> PrimFunc {
    // Escape hatch: fusion changes both the barrier insertion points available
    // to the later ThreadSync and the AttrStmt scopes a run of loops sits
    // under, so a kernel that is wrong with the pass on and right with it off
    // points here. Checked before any mutation so disabling is an exact
    // identity. Default to disabled: the fused SeqStmt body lands in the
    // `#pragma unroll 8` (partial-unroll) branch of codegen_tang.cc, which the
    // STCU/ptcc backend miscompiles (see example_mha_bwd_bhsd.py
    // backward-kernel corruption).
    if (ctx->GetConfig<Bool>(kDisableMergeLoop, Bool(true)).value()) {
      return f;
    }
    return MergeLoopRewriter::Substitute(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.MergeLoop", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.MergeLoop", MergeLoop);
}

} // namespace tl
} // namespace tvm
