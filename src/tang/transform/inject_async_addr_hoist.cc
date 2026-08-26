/*!
 * \brief Hoist loop-invariant sub-expressions from async DMA address offsets.
 * \file inject_async_addr_hoist.cc
 *
 * For each pts_load_async / pts_store_async intrinsic inside a For loop,
 * splits the buffer address offset into:
 *   invariant base (blockIdx/threadIdx terms, independent of the loop var)
 *   loop-variant offset (terms containing the loop var)
 *
 * The invariant base is bound to an AttrStmt("hoisted_addr_base"), which the
 * TANG codegen emits as a `const int` local before the DMA loop.
 *
 * Three sub-passes run in order:
 *   1. AsyncAddrHoister      -- per-loop invariant base extraction
 *   2. HoistedBaseSubtermCSE -- share additive threadIdx sub-terms across bases
 *   3. ThreadIdxSubexprCSE   -- share sub-exprs buried in non-additive
 *                               operators (swizzle bitwise ops), recursively
 *
 * Not wired into any pipeline (no pass-config option or env switch exists);
 * invoke manually via the tl.tang.transform.InjectAsyncAddrHoist FFI entry.
 * The backend compiler already hoists the same address sub-expressions for
 * current GEMM templates, so keep this pass opt-in until a kernel benefits
 * from explicit IR-level hoisting.
 */
#include "support/check.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "op/builtin.h"
#include "tir/ir/buffer_common.h"
#include "tir/transforms/ir_utils.h"
#include "tvm/tirx/stmt.h"

#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

bool IsDMAIntrinsic(const CallNode *call) {
  return call->op.same_as(tl::pts_load_async()) ||
         call->op.same_as(tl::pts_store_async());
}

bool ContainsVar(const PrimExpr &expr, const Var &var) {
  bool found = false;
  PostOrderVisit(expr, [&](const ObjectRef &obj) {
    if (auto *v = obj.as<VarNode>()) {
      if (v == var.get())
        found = true;
    }
  });
  return found;
}

PrimExpr ExtractAddressOffsetIndex(const PrimExpr &addr_expr) {
  if (auto *call = addr_expr.as<CallNode>()) {
    if (call->op.same_as(builtin::address_of()) && call->args.size() == 1) {
      if (auto *bl = call->args[0].as<BufferLoadNode>()) {
        if (bl->indices.size() == 1)
          return bl->indices[0];
      }
    }
  }
  return PrimExpr();
}

PrimExpr RebuildAddressOffset(const PrimExpr &orig_addr,
                              const PrimExpr &new_offset) {
  if (auto *call = orig_addr.as<CallNode>()) {
    if (call->op.same_as(builtin::address_of()) && call->args.size() == 1) {
      if (auto *bl = call->args[0].as<BufferLoadNode>()) {
        return Call(call->dtype, call->op,
                    {BufferLoad(bl->buffer, {new_offset})});
      }
    }
  }
  return orig_addr;
}

std::pair<PrimExpr, PrimExpr> SplitAdditiveByVar(const PrimExpr &expr,
                                                 const Var &var) {
  if (!expr.as<AddNode>()) {
    if (ContainsVar(expr, var))
      return {make_zero(DataType::Int(32)), expr};
    else
      return {expr, make_zero(DataType::Int(32))};
  }
  std::vector<PrimExpr> inv, variant;
  std::function<void(const PrimExpr &)> collect = [&](const PrimExpr &e) {
    if (auto *add = e.as<AddNode>()) {
      collect(add->a);
      collect(add->b);
    } else {
      if (ContainsVar(e, var))
        variant.push_back(e);
      else
        inv.push_back(e);
    }
  };
  collect(expr);
  auto sum = [](const std::vector<PrimExpr> &terms) -> PrimExpr {
    if (terms.empty())
      return make_zero(DataType::Int(32));
    PrimExpr r = terms[0];
    for (size_t i = 1; i < terms.size(); ++i)
      r = Add(r, terms[i]);
    return r;
  };
  return {sum(inv), sum(variant)};
}

std::string ExprKey(const PrimExpr &e) {
  std::ostringstream oss;
  oss << e;
  return oss.str();
}

// Split an expression into its additive terms.
void CollectAddTerms(const PrimExpr &e, std::vector<PrimExpr> *out) {
  if (auto *add = e.as<AddNode>()) {
    CollectAddTerms(add->a, out);
    CollectAddTerms(add->b, out);
  } else {
    out->push_back(e);
  }
}

// A hoistable pure-threadIdx sub-term references threadIdx, is not a bare
// constant, and does not reference blockIdx (blockIdx terms are too few to
// benefit and add register pressure).
bool IsPureThreadIdxTerm(const PrimExpr &term) {
  if (term.as<IntImmNode>())
    return false;
  bool has_thread = false, has_block = false;
  PostOrderVisit(term, [&](const ObjectRef &obj) {
    if (auto *v = obj.as<VarNode>()) {
      const std::string &nm = v->name_hint;
      if (nm.find("threadIdx") != std::string::npos || nm == "tx" ||
          nm == "ty" || nm == "tz")
        has_thread = true;
      if (nm.find("blockIdx") != std::string::npos || nm == "bx" ||
          nm == "by" || nm == "bz")
        has_block = true;
    }
  });
  return has_thread && !has_block;
}

// Wrap `inner` with the given hoisted_addr_base bindings (declared outermost
// first, so a binding that references an earlier one stays in scope).
Stmt WrapHoistBindings(Stmt inner,
                       const std::vector<std::pair<Var, PrimExpr>> &bindings) {
  for (auto it = bindings.rbegin(); it != bindings.rend(); ++it)
    inner = AttrStmt(it->first, "hoisted_addr_base", it->second, inner);
  return inner;
}

// Insert the shared hoisted bindings just inside the INNERMOST threadIdx
// thread_extent, so every threadIdx var they reference is already in scope and
// they still dominate every DMA loop (all of which sit deeper).
//
// Anchoring on threadIdx (not on the swizzle pattern) is what makes this
// correct: IsPureThreadIdxTerm rejects any term containing blockIdx, so the
// bindings never reference blockIdx and need not follow the swizzle attr that
// defines it.
//
// Two structural notes, both of which cost a debugging round:
//   - tirx nests thread_extent AttrStmts (bx wraps by wraps tx ...), but
//     DeclBuffer / AllocBuffer / Bind are LEAF statements sitting as siblings
//     in a SeqStmt -- they have no `body` field at all, and tir's AllocateNode
//     does not exist here. So the walk must descend AttrStmt bodies *and*
//     SeqStmt elements, and must not expect a single nested container chain.
//   - Splicing the bindings as flat SeqStmt siblings at the level where the
//     first thread_extent appears puts them AFTER the nested body that uses
//     them, and codegen fails with "Find undefined Variable addr_base".
//
// Returns true when the bindings were placed. If no threadIdx thread_extent
// exists the caller leaves the body untouched: emitting threadIdx-dependent
// bindings with no threadIdx in scope would be invalid.
bool TryInsertHoistBindings(
    Stmt *stmt, const std::vector<std::pair<Var, PrimExpr>> &bindings) {
  if (auto *attr = stmt->as<AttrStmtNode>()) {
    Stmt sub = attr->body;
    // Prefer a deeper threadIdx: the innermost one dominates the fewest
    // statements while still covering every DMA loop.
    if (TryInsertHoistBindings(&sub, bindings)) {
      AttrStmt node = ffi::GetRef<AttrStmt>(attr);
      node.CopyOnWrite()->body = sub;
      *stmt = std::move(node);
      return true;
    }
    if (attr->attr_key == tirx::attr::thread_extent) {
      if (const auto *iv = attr->node.as<IterVarNode>()) {
        if (iv->thread_tag.find("threadIdx") != std::string::npos) {
          AttrStmt node = ffi::GetRef<AttrStmt>(attr);
          node.CopyOnWrite()->body = WrapHoistBindings(attr->body, bindings);
          *stmt = std::move(node);
          return true;
        }
      }
    }
    return false;
  }
  if (auto *seq = stmt->as<SeqStmtNode>()) {
    // Last-to-first: the launch_thread chain sits at the tail of the prologue,
    // after the decl_buffer leaves.
    for (size_t i = seq->seq.size(); i-- > 0;) {
      Stmt sub = seq->seq[i];
      if (TryInsertHoistBindings(&sub, bindings)) {
        Array<Stmt> out = seq->seq;
        out.Set(i, sub);
        *stmt = SeqStmt(out);
        return true;
      }
    }
    return false;
  }
  if (auto *ite = stmt->as<IfThenElseNode>()) {
    Stmt sub = ite->then_case;
    if (TryInsertHoistBindings(&sub, bindings)) {
      IfThenElse node = ffi::GetRef<IfThenElse>(ite);
      node.CopyOnWrite()->then_case = sub;
      *stmt = std::move(node);
      return true;
    }
    return false;
  }
  return false;
}

void InsertHoistBindings(
    Stmt *body, const std::vector<std::pair<Var, PrimExpr>> &bindings) {
  TryInsertHoistBindings(body, bindings);
}

} // namespace

// ---------------------------------------------------------------------------
// Pass 1: per-loop invariant address base extraction.
// ---------------------------------------------------------------------------
class AsyncAddrHoister : public StmtExprMutator {
public:
  Stmt VisitStmt_(const ForNode *op) final {
    Var prev = current_loop_var_;
    current_loop_var_ = op->loop_var;

    std::map<std::string, Var> *prev_map = hoist_map_;
    std::vector<std::pair<Var, PrimExpr>> *prev_bindings = hoist_bindings_;
    std::map<std::string, Var> loop_map;
    std::vector<std::pair<Var, PrimExpr>> loop_bindings;
    hoist_map_ = &loop_map;
    hoist_bindings_ = &loop_bindings;

    Stmt body = this->VisitStmt(op->body);

    hoist_map_ = prev_map;
    hoist_bindings_ = prev_bindings;
    current_loop_var_ = prev;

    if (!loop_bindings.empty()) {
      for (auto it = loop_bindings.rbegin(); it != loop_bindings.rend(); ++it)
        body = AttrStmt(it->first, "hoisted_addr_base", it->second, body);
    }

    if (!body.same_as(op->body)) {
      For fnode = ffi::GetRef<For>(op);
      fnode.CopyOnWrite()->body = body;
      return std::move(fnode);
    }
    // body already visited above; a second visit here would re-run with the
    // outer loop var and a stale bindings pointer, corrupting the hoist map.
    return ffi::GetRef<Stmt>(op);
  }

private:
  PrimExpr VisitExpr_(const CallNode *op) final {
    if (IsDMAIntrinsic(op) && current_loop_var_.defined() &&
        op->args.size() > 1) {
      PrimExpr off = ExtractAddressOffsetIndex(op->args[1]);
      if (off.defined() && ContainsVar(off, current_loop_var_)) {
        auto split = SplitAdditiveByVar(off, current_loop_var_);
        const PrimExpr &invariant = split.first;
        const PrimExpr &variant = split.second;
        if (!is_zero(invariant)) {
          Var hv = GetOrCreateHoistVar(invariant, hoist_map_, hoist_bindings_);
          PrimExpr new_off =
              is_zero(variant) ? static_cast<PrimExpr>(hv) : Add(hv, variant);
          PrimExpr new_addr = RebuildAddressOffset(op->args[1], new_off);
          Array<PrimExpr> new_args = op->args;
          new_args.Set(1, new_addr);
          return Call(op->dtype, op->op, new_args);
        }
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  static Var
  GetOrCreateHoistVar(const PrimExpr &invariant,
                      std::map<std::string, Var> *map,
                      std::vector<std::pair<Var, PrimExpr>> *bindings) {
    std::string key = ExprKey(invariant);
    auto it = map->find(key);
    if (it != map->end())
      return it->second;
    Var hv("addr_base", DataType::Int(32));
    (*map)[key] = hv;
    if (bindings)
      bindings->push_back({hv, invariant});
    return hv;
  }

  Var current_loop_var_;
  std::map<std::string, Var> *hoist_map_{nullptr};
  std::vector<std::pair<Var, PrimExpr>> *hoist_bindings_{nullptr};
};

// ---------------------------------------------------------------------------
// Pass 2: sub-term CSE across the per-loop hoisted address bases.
//
// Different loops (prologue / main / epilogue) produce bases that share the
// same threadIdx sub-terms (e.g. tx//64*4096, tx%64*2). Left inline, each base
// recomputes them, and under register pressure the backend spills the repeats.
//
// Terms appearing in MORE THAN ONE base are bound once to a shared
// "hoisted_addr_base" emitted right after threadIdx. The per-loop base itself
// stays where it was, so the backend keeps its local scheduling freedom --
// forcing the whole base to function scope measured *worse* upstream (spill
// 32B -> 40-48B), because it keeps the value live across the entire kernel.
// ---------------------------------------------------------------------------
class HoistedBaseSubtermCSE : public StmtExprMutator {
public:
  Stmt Run(Stmt body) {
    CountTerms(body);
    for (auto &kv : term_count_) {
      if (kv.second > 1)
        shared_keys_.insert(kv.first);
    }
    if (shared_keys_.empty())
      return body;
    body = this->VisitStmt(body);
    if (!shared_bindings_.empty()) {
      InsertHoistBindings(&body, shared_bindings_);
    }
    return body;
  }

private:
  void CountTerms(const Stmt &body) {
    PostOrderVisit(body, [&](const ObjectRef &obj) {
      if (auto *attr = obj.as<AttrStmtNode>()) {
        if (attr->attr_key == "hoisted_addr_base") {
          std::vector<PrimExpr> terms;
          CollectAddTerms(attr->value, &terms);
          for (const PrimExpr &t : terms) {
            if (IsPureThreadIdxTerm(t))
              term_count_[ExprKey(t)]++;
          }
        }
      }
    });
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == "hoisted_addr_base") {
      std::vector<PrimExpr> terms;
      CollectAddTerms(op->value, &terms);
      std::vector<PrimExpr> rebuilt;
      bool changed = false;
      for (const PrimExpr &t : terms) {
        if (IsPureThreadIdxTerm(t) && shared_keys_.count(ExprKey(t))) {
          rebuilt.push_back(GetOrCreateSharedVar(t));
          changed = true;
        } else {
          rebuilt.push_back(t);
        }
      }
      Stmt new_body = this->VisitStmt(op->body);
      if (!changed && new_body.same_as(op->body)) {
        return ffi::GetRef<Stmt>(op);
      }
      PrimExpr new_val = rebuilt[0];
      for (size_t i = 1; i < rebuilt.size(); ++i)
        new_val = Add(new_val, rebuilt[i]);
      return AttrStmt(op->node, op->attr_key, new_val, new_body);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Var GetOrCreateSharedVar(const PrimExpr &term) {
    std::string key = ExprKey(term);
    auto it = shared_map_.find(key);
    if (it != shared_map_.end())
      return it->second;
    Var hv("addr_base", DataType::Int(32));
    shared_map_[key] = hv;
    shared_bindings_.push_back({hv, term});
    return hv;
  }

  std::map<std::string, int> term_count_;
  std::set<std::string> shared_keys_;
  std::map<std::string, Var> shared_map_;
  std::vector<std::pair<Var, PrimExpr>> shared_bindings_;
};

// ---------------------------------------------------------------------------
// Pass 3: general loop-invariant threadIdx sub-expression CSE across all async
// DMA address indices (both the shared destination and the global source).
//
// Unlike the additive-term CSE above, this reaches sub-expressions buried
// inside non-additive operators (e.g. the bitwise_xor of a swizzle index),
// which is where the bulk of the repeated threadIdx arithmetic lives.
//
// Sub-expressions are replaced largest-first so a hoisted parent subsumes its
// children, then ExpandBindingsInternally recurses so a binding value like
// (tx%64//4)*64 references the shared var for tx%64//4 rather than recomputing
// it.
// ---------------------------------------------------------------------------
class ThreadIdxSubexprCSE : public StmtExprMutator {
public:
  Stmt Run(Stmt body) {
    CountPass(body);
    for (auto &kv : occ_) {
      if (kv.second > 1)
        shared_keys_.insert(kv.first);
    }
    if (shared_keys_.empty())
      return body;
    body = this->VisitStmt(body);
    ExpandBindingsInternally();
    if (!shared_bindings_.empty()) {
      InsertHoistBindings(&body, shared_bindings_);
    }
    return body;
  }

private:
  Stmt VisitStmt_(const ForNode *op) final {
    active_loop_vars_.push_back(op->loop_var);
    Stmt r = StmtExprMutator::VisitStmt_(op);
    active_loop_vars_.pop_back();
    return r;
  }

  // Rewrite the per-loop hoisted base values too: replace inner shared
  // sub-terms but keep the base's own top node, so the base reuses the shared
  // threadIdx vars.
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == "hoisted_addr_base") {
      PrimExpr new_val = ReplaceInsideValue(op->value);
      Stmt new_body = this->VisitStmt(op->body);
      if (!new_val.same_as(op->value) || !new_body.same_as(op->body)) {
        return AttrStmt(op->node, op->attr_key, new_val, new_body);
      }
      return ffi::GetRef<Stmt>(op);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  bool ContainsAnyLoopVar(const PrimExpr &e) const {
    for (const Var &v : active_loop_vars_)
      if (ContainsVar(e, v))
        return true;
    return false;
  }

  // Compound (non-leaf) node worth CSE-ing. Call covers the bitwise ops.
  static bool IsCompound(const PrimExpr &e) {
    return e.as<MulNode>() || e.as<DivNode>() || e.as<ModNode>() ||
           e.as<FloorDivNode>() || e.as<FloorModNode>() || e.as<AddNode>() ||
           e.as<SubNode>() || e.as<CallNode>();
  }

  bool IsCandidate(const PrimExpr &e) const {
    if (!IsCompound(e))
      return false;
    if (ContainsAnyLoopVar(e))
      return false;
    return IsPureThreadIdxTerm(e);
  }

  // Counting pass: walk statements tracking loop scope, and for each DMA call
  // count the candidate sub-expressions of its address indices.
  void CountPass(const Stmt &stmt) {
    if (auto *f = stmt.as<ForNode>()) {
      active_loop_vars_.push_back(f->loop_var);
      CountPass(f->body);
      active_loop_vars_.pop_back();
      return;
    }
    if (auto *seq = stmt.as<SeqStmtNode>()) {
      for (const Stmt &s : seq->seq)
        CountPass(s);
      return;
    }
    if (auto *attr = stmt.as<AttrStmtNode>()) {
      // Fold the per-loop hoisted base values in too, so e.g. tx//64*4096
      // reuses the shared tx//64 var instead of recomputing it.
      if (attr->attr_key == "hoisted_addr_base")
        CountIndex(attr->value);
      CountPass(attr->body);
      return;
    }
    if (auto *ite = stmt.as<IfThenElseNode>()) {
      CountPass(ite->then_case);
      if (ite->else_case.defined())
        CountPass(ite->else_case.value());
      return;
    }
    // Leaf-ish statement: scan for DMA calls and count their index candidates.
    PostOrderVisit(stmt, [&](const ObjectRef &obj) {
      if (auto *c = obj.as<CallNode>()) {
        if (IsDMAIntrinsic(c)) {
          for (const PrimExpr &arg : c->args) {
            PrimExpr idx = ExtractAddressOffsetIndex(arg);
            if (idx.defined())
              CountIndex(idx);
          }
        }
      }
    });
  }

  // Count candidate sub-expressions in one index, in the current loop scope.
  void CountIndex(const PrimExpr &idx) {
    std::set<std::string> seen_here;
    PostOrderVisit(idx, [&](const ObjectRef &obj) {
      if (auto *e = obj.as<PrimExprNode>()) {
        PrimExpr pe = ffi::GetRef<PrimExpr>(e);
        if (IsCandidate(pe)) {
          std::string k = ExprKey(pe);
          if (seen_here.insert(k).second)
            occ_[k]++;
        }
      }
    });
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (IsDMAIntrinsic(op)) {
      Array<PrimExpr> new_args;
      bool changed = false;
      for (const PrimExpr &arg : op->args) {
        PrimExpr idx = ExtractAddressOffsetIndex(arg);
        if (idx.defined()) {
          PrimExpr new_idx = ReplaceInIndex(idx);
          if (!new_idx.same_as(idx)) {
            new_args.push_back(RebuildAddressOffset(arg, new_idx));
            changed = true;
            continue;
          }
        }
        new_args.push_back(arg);
      }
      if (changed)
        return Call(op->dtype, op->op, new_args);
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  // Replace shared candidate sub-expressions inside one index, largest-first:
  // ExprMutator visits a node before its children, so once a parent matches and
  // becomes a Var its children are never descended.
  PrimExpr ReplaceInIndex(const PrimExpr &idx) {
    struct Local : public ExprMutator {
      ThreadIdxSubexprCSE *self;
      explicit Local(ThreadIdxSubexprCSE *s) : self(s) {}
      PrimExpr VisitExpr(const PrimExpr &e) final {
        if (self->IsCandidate(e) && self->shared_keys_.count(ExprKey(e)))
          return self->GetOrCreateSharedVar(e);
        return ExprMutator::VisitExpr(e);
      }
    };
    Local m(this);
    return m(idx);
  }

  Var GetOrCreateSharedVar(const PrimExpr &e) {
    std::string key = ExprKey(e);
    auto it = shared_map_.find(key);
    if (it != shared_map_.end())
      return it->second;
    Var hv("addr_base", DataType::Int(32));
    shared_map_[key] = hv;
    shared_bindings_.push_back({hv, e});
    return hv;
  }

  // Replace shared candidates strictly *inside* `e` (its children and below),
  // leaving `e`'s own top node intact. New inner candidates get their own
  // shared var, which may append more bindings.
  PrimExpr ReplaceInsideValue(const PrimExpr &e) {
    struct Local : public ExprMutator {
      ThreadIdxSubexprCSE *self;
      bool at_root{true};
      explicit Local(ThreadIdxSubexprCSE *s) : self(s) {}
      PrimExpr VisitExpr(const PrimExpr &e) final {
        if (!at_root && self->IsCandidate(e) &&
            self->shared_keys_.count(ExprKey(e))) {
          return self->GetOrCreateSharedVar(e);
        }
        at_root = false;
        return ExprMutator::VisitExpr(e);
      }
    };
    Local m(this);
    return m(e);
  }

  // Expand each shared binding so its value references other shared vars, then
  // topologically sort so each var is declared after its dependencies.
  void ExpandBindingsInternally() {
    // Index-based loop, not a range-for: expanding a value appends new
    // bindings, so the container grows while being iterated.
    for (size_t i = 0; i < shared_bindings_.size(); ++i) {
      shared_bindings_[i].second =
          ReplaceInsideValue(shared_bindings_[i].second);
    }
    std::map<const VarNode *, size_t> var_to_idx;
    for (size_t i = 0; i < shared_bindings_.size(); ++i)
      var_to_idx[shared_bindings_[i].first.get()] = i;

    std::vector<std::pair<Var, PrimExpr>> ordered;
    // 0 = unvisited, 1 = active (cycle guard), 2 = done
    std::vector<int> state(shared_bindings_.size(), 0);
    std::function<void(size_t)> dfs = [&](size_t i) {
      if (state[i] == 2)
        return;
      state[i] = 1;
      PostOrderVisit(shared_bindings_[i].second, [&](const ObjectRef &obj) {
        if (auto *v = obj.as<VarNode>()) {
          auto it = var_to_idx.find(v);
          if (it != var_to_idx.end() && it->second != i &&
              state[it->second] != 1)
            dfs(it->second);
        }
      });
      state[i] = 2;
      ordered.push_back(shared_bindings_[i]);
    };
    for (size_t i = 0; i < shared_bindings_.size(); ++i)
      dfs(i);
    shared_bindings_ = std::move(ordered);
  }

  std::vector<Var> active_loop_vars_;
  std::map<std::string, int> occ_;
  std::set<std::string> shared_keys_;
  std::map<std::string, Var> shared_map_;
  std::vector<std::pair<Var, PrimExpr>> shared_bindings_;
};

using namespace tirx::transform;

tvm::transform::Pass InjectAsyncAddrHoist() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    auto *n = f.CopyOnWrite();
    n->body = AsyncAddrHoister()(n->body);
    n->body = HoistedBaseSubtermCSE().Run(n->body);
    n->body = ThreadIdxSubexprCSE().Run(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectAsyncAddrHoist", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.tang.transform.InjectAsyncAddrHoist",
                        InjectAsyncAddrHoist);
}

} // namespace tl
} // namespace tvm
