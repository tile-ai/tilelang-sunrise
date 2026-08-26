/*!
 * \file hoist_copy_addresses.cc
 * \brief Convert loop-var-based copy addresses to incremental induction
 * variables.
 *
 * Transform:
 *   for (k, k0, N) {
 *     A_shared[off_a] = A[base_a + k * stride_a]
 *     B_shared[off_b] = B[base_b + k * stride_b]
 *     gemm(...)
 *   }
 *
 * Into:
 *   alloc a_addr[1], b_addr[1]           // local scalars
 *   a_addr[0] = base_a + k0 * stride_a
 *   b_addr[0] = base_b + k0 * stride_b
 *   for (k, k0, N) {
 *     A_shared[off_a] = A[a_addr[0]]
 *     B_shared[off_b] = B[b_addr[0]]
 *     gemm(...)
 *     a_addr[0] = a_addr[0] + stride_a   // bumped once, at loop tail
 *     b_addr[0] = b_addr[0] + stride_b
 *   }
 *
 * The addresses live in loop-carried scalars, so `k * stride` disappears from
 * the body: one add per copy per iteration instead of a shift plus an add.
 *
 * Each summand of the address is classified on its own, so an index only
 * partially hoistable still benefits. For a real GEMM copy address
 *   (k+1)%2*8192 + i*512 + tx*2 + k*64
 * the k-linear term folds into the running scalar, `tx*2` is hoisted into its
 * base, and the double-buffer modulo plus the inner-loop term stay in place:
 *   A_addr_0[0] + ((k+1)%2*8192 + i*512)
 * This holds because the pass runs after VectorizeLoop/UnrollLoop, so a copy's
 * inner loop is vectorized (or gone) by then and `i` is bound inside the `k`
 * loop being rewritten. Were `i` still a serial loop, the pass would claim it
 * first instead — see test_invariant_terms_fold_into_initialiser.
 *
 * Directions handled (see IsSupportedDirection):
 *   global → shared / register   shared → register / global   register → global
 * Both sides of a copy are rewritten when both are memory-resident, so e.g. a
 * shared→global copy gets running addresses for the read and the write.
 *
 * Correctness constraints enforced below:
 *  - `base` must be loop-invariant (it is evaluated once, before the loop).
 *  - The increment happens at the loop *tail*, so every read within one
 *    iteration observes the same address. Rewriting mid-body would give later
 *    reads the next iteration's address.
 *  - Only serial loops qualify: a parallel/vectorized/thread-bound loop has no
 *    well-defined sequential carry, and unrolling would duplicate the update.
 *  - Register-scope indices are never rewritten: registers need constant
 *    indices, and a runtime index would spill the array to scratch memory.
 *
 * Scheduling: replacing an affine index with a runtime scalar defeats affine
 * analysis, so this must run after VectorizeLoop. It runs *before* UnrollLoop
 * so that loops UnrollLoop would expand are still serial loops with a live
 * loop_var; a loop already expanded has no For node left to rewrite, and one
 * marked kUnrolled is skipped outright (see the kSerial check below).
 * See TANGPassPipelineBody.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <functional>
#include <string>
#include <unordered_set>
#include <vector>

#include "../op/builtin.h"
#include "../op/utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

/*! \brief One running-address scalar, shared by every matching index. */
struct Site {
  Buffer buffer;   // buffer whose index is being replaced (read or written)
  DataType dtype;  // dtype of the replaced index expression
  int64_t stride;  // multiplier of loop_var
  PrimExpr base;   // additive part (0 if none)
  Buffer addr_buf; // scalar holding the running address
};

/*! \brief Flatten an Add tree into its summands; non-Add nodes stay opaque. */
void FlattenAddTerms(const PrimExpr &e, std::vector<PrimExpr> *out) {
  if (const auto *add = e.as<AddNode>()) {
    FlattenAddTerms(add->a, out);
    FlattenAddTerms(add->b, out);
  } else {
    out->push_back(e);
  }
}

/*!
 * \brief Coefficient of \p loop_var if \p term is `loop_var`, `loop_var * imm`
 *        or `imm * loop_var`; nullopt for any other shape.
 */
std::optional<int64_t> LinearCoeff(const PrimExpr &term, const Var &loop_var) {
  if (term.same_as(loop_var))
    return 1;
  if (const auto *mul = term.as<MulNode>()) {
    if (mul->a.same_as(loop_var))
      if (const auto *imm = mul->b.as<IntImmNode>())
        return imm->value;
    if (mul->b.same_as(loop_var))
      if (const auto *imm = mul->a.as<IntImmNode>())
        return imm->value;
  }
  return std::nullopt;
}

/*!
 * \brief Split an index into a hoistable running address plus a residual.
 *
 * The whole Add tree is flattened and each summand is classified on its own:
 *  - `loop_var * imm` (or bare `loop_var`) — folds into \p stride;
 *  - anything mentioning loop_var non-linearly, or a variable bound inside the
 *    loop body — stays in \p residual, evaluated in place each iteration;
 *  - everything else — folds into \p base, evaluated once before the loop.
 *
 * So `(k+1)%2*8192 + i*512 + tx*2 + k*64` yields stride 64, base `tx*2`, and
 * residual `(k+1)%2*8192 + i*512`, and the index becomes `addr[0] + residual`.
 * Keeping the residual in place is what lets real kernels match: their copy
 * addresses mix a k-linear term with double-buffer modulo and inner-loop terms.
 * (`i` is bound inside the rewritten loop here — a vectorized lane index by the
 * time this pass runs. A serial `i` loop would instead be rewritten on its
 * own.)
 *
 * \param is_inner_bound Predicate telling whether a Var is bound inside the
 * body.
 * \returns nullopt when there is no usable linear term.
 */
struct AddrSplit {
  PrimExpr base;     // loop-invariant, hoisted out of the loop
  PrimExpr residual; // recomputed in place; null when there is none
  int64_t stride;    // total coefficient of loop_var
  // Set when the index was a Ramp (vectorized copy); the rebuilt index is
  // re-wrapped as Ramp(addr + residual, ramp_stride, lanes).
  PrimExpr ramp_stride;
  PrimExpr lanes;
};

std::optional<AddrSplit>
SplitLoopAddress(const PrimExpr &expr, const Var &loop_var,
                 const std::function<bool(const VarNode *)> &is_inner_bound) {
  // A vectorized copy indexes with Ramp(base, stride, lanes); only its base
  // carries the loop-varying address, so split that and re-wrap afterwards.
  if (const auto *ramp = expr.as<RampNode>()) {
    auto inner = SplitLoopAddress(ramp->base, loop_var, is_inner_bound);
    if (!inner)
      return std::nullopt;
    inner->lanes = ramp->lanes;
    inner->ramp_stride = ramp->stride;
    return inner;
  }

  std::vector<PrimExpr> terms;
  FlattenAddTerms(expr, &terms);

  int64_t stride = 0;
  bool found_linear = false;
  PrimExpr base, residual;
  auto accumulate = [](PrimExpr *acc, const PrimExpr &t) {
    *acc = acc->defined() ? *acc + t : t;
  };

  for (const auto &t : terms) {
    if (auto coeff = LinearCoeff(t, loop_var)) {
      stride += *coeff;
      found_linear = true;
      continue;
    }
    // Cannot be evaluated before the loop: keep it where it is.
    if (UsesVar(t, [&](const VarNode *v) {
          return v == loop_var.get() || is_inner_bound(v);
        })) {
      accumulate(&residual, t);
    } else {
      accumulate(&base, t);
    }
  }

  // No linear term, or the coefficients cancel out: no running sum to build.
  if (!found_linear || stride == 0)
    return std::nullopt;
  // expr is scalar here: the Ramp case above recurses on the Ramp's base.
  if (!base.defined())
    base = make_zero(expr.dtype());
  AddrSplit split;
  split.base = base;
  split.residual = residual;
  split.stride = stride;
  return split;
}

/*! \brief Collects loop and bind Vars that \p stmt binds internally. */
class BoundVarCollector : public StmtExprVisitor {
public:
  static std::unordered_set<const VarNode *> Collect(const Stmt &stmt) {
    BoundVarCollector c;
    c(stmt);
    return std::move(c.bound_);
  }

private:
  void VisitStmt_(const ForNode *op) final {
    bound_.insert(op->loop_var.get());
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const BindNode *op) final {
    bound_.insert(op->var.get());
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const LetNode *op) final {
    bound_.insert(op->var.get());
    StmtExprVisitor::VisitExpr_(op);
  }

  std::unordered_set<const VarNode *> bound_;
};

/*! \brief Storage scope of the running-address scalars this pass allocates. */
constexpr const char *kAddrScope = "local";

/*! \brief Each running address is a one-element array; this is its only index.
 */
inline Array<PrimExpr> AddrIndex() { return {0}; }

/*!
 * \brief Scopes whose addresses are worth turning into running scalars.
 *
 * Deliberately excludes register scopes: a register-resident array must keep
 * constant indices to stay in registers, so making its index a runtime value
 * would force it to scratch memory. For a register↔memory copy only the memory
 * side is rewritten; the register side is left exactly as it was.
 *
 * Shared memory is admitted only via IsSharedBuffer, i.e.
 * "shared"/"shared.dyn": "shared.barrier" holds mbarrier state and
 * "shared.tmem" is an opaque tensor-memory handle, and neither is addressed by
 * a plain running index.
 */
bool IsHoistableScope(const Buffer &buffer) {
  return IsGlobalBuffer(buffer) || IsSharedBuffer(buffer);
}

/*! \brief True if this src→dst copy direction is one we rewrite. */
bool IsSupportedDirection(const Buffer &src, const Buffer &dst) {
  // global → shared / register
  if (IsGlobalBuffer(src) && (IsSharedBuffer(dst) || IsRegisterBuffer(dst)))
    return true;
  // shared → register / global
  if (IsSharedBuffer(src) && (IsRegisterBuffer(dst) || IsGlobalBuffer(dst)))
    return true;
  // register → global
  if (IsRegisterBuffer(src) && IsGlobalBuffer(dst))
    return true;
  return false;
}

/*! \brief True if this store is a copy whose addresses this pass can rewrite.
 */
bool IsHoistableCopy(const BufferStoreNode *store, const BufferLoadNode *load) {
  return IsSupportedDirection(load->buffer, store->buffer);
}

/*!
 * \brief Replace each matching index with `addr[0] + residual`, where addr is
 *        the site's running scalar and residual is the part that must stay.
 * \returns true if any index was rewritten.
 */
bool RewriteIndices(
    const Buffer &buffer, Array<PrimExpr> *indices, const Var &loop_var,
    const std::vector<Site> &sites,
    const std::function<bool(const VarNode *)> &is_inner_bound) {
  if (!IsHoistableScope(buffer))
    return false;
  bool changed = false;
  for (size_t i = 0; i < indices->size(); i++) {
    auto split = SplitLoopAddress((*indices)[i], loop_var, is_inner_bound);
    if (!split)
      continue;

    // Buffer identity is part of the key so that two buffers sharing a
    // stride/base shape never collapse onto one address.
    for (const auto &site : sites) {
      if (site.buffer.same_as(buffer) && split->stride == site.stride &&
          StructuralEqual()(split->base, site.base)) {
        PrimExpr addr = BufferLoad(site.addr_buf, AddrIndex());
        if (split->residual.defined())
          addr = addr + split->residual;
        // Restore the vector shape of a vectorized copy.
        if (split->lanes.defined())
          addr = Ramp(addr, split->ramp_stride, split->lanes);
        indices->Set(i, addr);
        changed = true;
        break;
      }
    }
  }
  return changed;
}

/*!
 * \brief Rewrite a qualifying copy so both its memory-side addresses come from
 *        running-address scalars.
 * \returns the rewritten store, or Stmt() if nothing matched.
 */
Stmt TryReplaceCopyAddress(
    const BufferStoreNode *store, const Var &loop_var,
    const std::vector<Site> &sites,
    const std::function<bool(const VarNode *)> &is_inner_bound) {
  const auto *load = store->value.as<BufferLoadNode>();
  if (!load)
    return Stmt();
  if (!IsHoistableCopy(store, load))
    return Stmt();

  Array<PrimExpr> load_indices = load->indices;
  Array<PrimExpr> store_indices = store->indices;
  bool changed = RewriteIndices(load->buffer, &load_indices, loop_var, sites,
                                is_inner_bound);
  changed |= RewriteIndices(store->buffer, &store_indices, loop_var, sites,
                            is_inner_bound);
  if (!changed)
    return Stmt();

  return BufferStore(store->buffer, BufferLoad(load->buffer, load_indices),
                     store_indices);
}

} // namespace

class CopyAddressHoister : public StmtMutator {
public:
  static PrimFunc Substitute(PrimFunc &f) {
    f.CopyOnWrite()->body = CopyAddressHoister::Apply(f->body);
    return f;
  }

  static Stmt Apply(Stmt stmt) {
    CopyAddressHoister hoister;
    return hoister(stmt);
  }

private:
  CopyAddressHoister() = default;

  Stmt VisitStmt_(const ForNode *op) final {
    // Recurse first so inner loops are handled independently.
    auto n = Downcast<For>(StmtMutator::VisitStmt_(op));

    // Only a serial loop has a well-defined sequential carry. Parallel,
    // vectorized and thread-bound loops have no iteration order to chain
    // through, and unrolling would duplicate the tail update.
    if (n->kind != ForKind::kSerial)
      return n;

    const Var &loop_var = n->loop_var;

    // Vars bound inside the body (inner loop vars, lets) cannot appear in a
    // base that is evaluated once before the loop; terms using them stay in
    // the residual instead.
    auto inner_bound = BoundVarCollector::Collect(n->body);
    auto is_inner_bound = [inner_bound](const VarNode *v) {
      return inner_bound.count(v) > 0;
    };

    // ---- Phase 1: Scan for copies with a hoistable k*stride component ----
    std::vector<Site> sites;

    // Collect one site per distinct (buffer, stride, base) address found on a
    // hoistable side of a qualifying copy.
    auto collect = [&](const Buffer &buffer, const Array<PrimExpr> &indices) {
      if (!IsHoistableScope(buffer))
        return;
      for (const auto &idx : indices) {
        auto split = SplitLoopAddress(idx, loop_var, is_inner_bound);
        if (!split)
          continue;
        // Deduplicate on (buffer, stride, base); the residual stays in place so
        // two indices sharing a base/stride can share one address scalar even
        // when their residuals differ.
        bool dup = false;
        for (const auto &s : sites)
          if (s.buffer.same_as(buffer) && s.stride == split->stride &&
              StructuralEqual()(s.base, split->base)) {
            dup = true;
            break;
          }
        if (dup)
          continue;

        // Derive the name from the buffer itself rather than pattern-matching
        // GEMM's A/B naming convention.
        const std::string &buf_name = buffer->name;
        std::string hint = buf_name.empty() ? "dma_addr" : buf_name + "_addr";
        // Use the scalar dtype of the address itself: for a vectorized copy
        // idx.dtype() is a vector type, but the running address is scalar.
        DataType addr_dtype = split->base.dtype();
        Buffer addr_buf =
            decl_buffer({1}, addr_dtype,
                        hint + "_" + std::to_string(sites.size()), kAddrScope);
        sites.push_back(
            {buffer, addr_dtype, split->stride, split->base, addr_buf});
      }
    };

    PostOrderVisit(n->body, [&](const ObjectRef &obj) {
      const auto *store = obj.as<BufferStoreNode>();
      if (!store)
        return;
      const auto *load = store->value.as<BufferLoadNode>();
      if (!load || !IsHoistableCopy(store, load))
        return;
      collect(load->buffer, load->indices);
      collect(store->buffer, store->indices);
    });

    if (sites.empty())
      return n;

    // ---- Phase 2: Rewrite the body to read the running addresses ----
    // Every read in one iteration must observe the same address, so the
    // rewrite is a plain in-place substitution; the bump happens at the tail.
    AddressRewriter rewriter(loop_var, sites, is_inner_bound);
    Stmt body = rewriter(n->body);
    if (!rewriter.changed)
      return n;

    // ---- Phase 3: Append the tail increments ----
    Array<Stmt> body_seq{body};
    for (const auto &site : sites) {
      PrimExpr cur = BufferLoad(site.addr_buf, AddrIndex());
      body_seq.push_back(BufferStore(site.addr_buf,
                                     cur + make_const(site.dtype, site.stride),
                                     AddrIndex()));
    }
    body = SeqStmt::Flatten(body_seq);

    Stmt result = For(loop_var, n->min, n->extent, n->kind, body,
                      n->thread_binding, n->annotations, std::nullopt, n->span);

    // ---- Phase 4: Allocate and initialize the scalars before the loop ----
    Array<Stmt> pre;
    for (const auto &site : sites) {
      // No annotations: these are plain scalars with no alignment or
      // special-memory requirements.
      pre.push_back(AllocBuffer(site.addr_buf));
    }
    for (const auto &site : sites) {
      // init = base + min * stride, folded on the site's own dtype.
      PrimExpr init = n->min * make_const(site.dtype, site.stride);
      if (!is_zero(site.base))
        init = site.base + init;
      pre.push_back(
          BufferStore(site.addr_buf, cast(site.dtype, init), AddrIndex()));
    }
    pre.push_back(result);
    return SeqStmt::Flatten(pre);
  }

  /*! \brief Replaces matched copy addresses with loads of the running scalar.
   */
  class AddressRewriter : public StmtExprMutator {
  public:
    AddressRewriter(const Var &loop_var, const std::vector<Site> &sites,
                    std::function<bool(const VarNode *)> is_inner_bound)
        : loop_var_(loop_var), sites_(sites),
          is_inner_bound_(std::move(is_inner_bound)) {}

    bool changed = false;

  private:
    Stmt VisitStmt_(const BufferStoreNode *op) final {
      auto s = TryReplaceCopyAddress(op, loop_var_, sites_, is_inner_bound_);
      if (s.defined()) {
        changed = true;
        return s;
      }
      return StmtExprMutator::VisitStmt_(op);
    }

    const Var &loop_var_;
    const std::vector<Site> &sites_;
    std::function<bool(const VarNode *)> is_inner_bound_;
  };
};

using namespace tirx::transform;
tvm::transform::Pass HoistCopyAddresses() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return CopyAddressHoister::Substitute(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.HoistCopyAddresses", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.HoistCopyAddresses", HoistCopyAddresses);
}

} // namespace tl
} // namespace tvm
