/*!
 *  \file promote_tmem_accumulator.cc
 *  \brief stcuv2: promote a local.fragment GEMM accumulator into shared.tmem.
 *
 *  stcuv2 has no MMA form that accumulates into registers.  SelectInst
 *  (src/tang/op/gemm.cc) routes a C in shared.tmem to tcgen5 and a C in shared
 *  to wgmma; everything else falls through to TMMA, which only exists on stcu.
 *  So a `T.gemm` whose C came from `T.alloc_fragment` has no lowering at all on
 *  stcuv2 -- it dies with "No GEMM implementation registered for instruction
 *  tang.tmma".
 *
 *  Rather than make every kernel spell the accumulator out as `T.alloc_tmem`,
 *  this pass rewrites the natural fragment spelling into the tmem one:
 *
 *      C = T.alloc_fragment((BM, BN), "float32")  |  C = T.alloc_tmem(...)
 *      T.clear(C)                                 |  (dropped)
 *      for k in T.Pipelined(...):                 |  for k in T.Pipelined(...):
 *          T.gemm(A_s, B_s, C)                    |      T.gemm(...,
 * clear_accum=k == 0) T.copy(C, C_global)                        |  T.copy(C,
 * C_global)
 *
 *  This is a *buffer promotion*, not a copy insertion: the fragment is replaced
 *  by a shared.tmem buffer for its whole lifetime, so the reduction loop gains
 *  no accumulator traffic.  Staging the tile in and out of tmem around each MMA
 *  instead would push the whole accumulator across the LDT/STT path once per K
 *  step -- for a 128x128 fp32 tile at BK=32 that is 64 transfers where resident
 *  tmem needs none -- and would also depend on the tmem<->fragment copy, which
 *  is restricted to a single warp today (src/tang/op/copy.cc).
 *
 *  `T.clear` has to fold into `clear_accum` rather than survive as a store:
 *  tmem is not directly storable (LowerSharedTmem rejects data stores into it),
 *  so the MMA's own accumulator clear is the only way to initialize it.  The
 *  trailing `T.copy(C, C_global)` needs no rewrite -- with C in shared.tmem the
 *  TANG copy lowering re-resolves it to the TMEM->global drain.
 *
 *  Deliberately NOT re-checked here: the tcgen5 legality that
 *  tilelang/tang/op/gemm/gemm_tcgen5.py already validates during lowering with
 *  precise messages (operand scopes, scale-descriptor consistency, an operand
 *  row over the 128-byte MMA cap).  Duplicating those rules would only let the
 *  two copies drift apart.  The one exception is an operand row *under* 128
 *  bytes, which nothing downstream diagnoses -- see CheckOperandRowBytes.
 *
 *  A fragment accumulator this pass cannot promote raises instead of being left
 *  alone, because leaving it alone only defers the failure to the far more
 *  confusing tang.tmma registry error.  The cases are listed on PromotionError.
 */
#include "op/utils.h"
#include "tang/target_utils.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

const Op &TileOpGemm() {
  static const Op op = Op::Get("tl.tileop.gemm");
  return op;
}
const Op &TileOpFill() {
  static const Op op = Op::Get("tl.tileop.fill");
  return op;
}
const Op &TileOpCopy() {
  static const Op op = Op::Get("tl.tileop.copy");
  return op;
}

// Argument positions in a tl.tileop.gemm call; see the arg layout documented on
// Gemm::Gemm in src/op/gemm.cc.
constexpr size_t kGemmArgA = 0;
constexpr size_t kGemmArgB = 1;
constexpr size_t kGemmArgC = 2;
constexpr size_t kGemmArgK = 7;
constexpr size_t kGemmArgClearAccum = 9;

// Bytes of one operand row a single tcgen5 MMA reads out of shared memory.
constexpr int64_t kMmaOperandRowBytes = 128;

constexpr const char *kFragmentScope = "local.fragment";
constexpr const char *kTmemScope = "shared.tmem";

/*! \brief How one local.fragment buffer is used inside a block. */
struct FragmentUse {
  /*! \brief A `tl.tileop.gemm` that accumulates into this buffer. */
  struct GemmSite {
    const CallNode *call;
    /*! \brief The For loops enclosing the call, outermost first. */
    std::vector<const ForNode *> loops;
  };
  /*! \brief Accumulating gemms, in pre-order (= execution order). */
  std::vector<GemmSite> gemms;
  /*! \brief The zero `T.clear` that initializes the accumulator, if any. */
  const CallNode *fill = nullptr;
  std::vector<const ForNode *> fill_loops;
  /*!
   * \brief Why this buffer cannot be promoted, or empty when it can.
   *
   * Only consulted for buffers that turn out to be gemm accumulators; a plain
   * fragment used for anything else is left alone whatever this says.
   */
  std::string reject;
};

/*!
 * \brief Classify every use of every local.fragment buffer in a block body.
 *
 * The three tile ops that a promotable accumulator may appear in (gemm C, a
 * zero fill, a copy to global) are matched structurally and their buffer-region
 * arguments are not descended into.  Everything else that mentions a fragment
 * -- an element-wise read or write, a reduction, a gemm A/B operand, a copy in
 * the wrong direction -- reaches the generic BufferLoad/BufferStore handlers
 * below and records a rejection.  That way an unforeseen use is rejected by
 * default rather than silently promoted.
 */
class FragmentUseCollector : public StmtExprVisitor {
public:
  std::unordered_map<const VarNode *, FragmentUse> uses;

private:
  std::vector<const ForNode *> loops_;

  static bool IsFragment(const Buffer &buf) {
    return buf.scope() == kFragmentScope;
  }

  /*! \brief The buffer a gemm/fill/copy region argument refers to, if any. */
  static Optional<Buffer> RegionBuffer(const PrimExpr &arg) {
    if (!IsBufferLikeExpr(arg))
      return std::nullopt;
    return NormalizeToAccessRegion(arg).region->buffer;
  }

  FragmentUse &Entry(const Buffer &buf) { return uses[buf->data.get()]; }

  void Reject(const Buffer &buf, const std::string &reason) {
    FragmentUse &use = Entry(buf);
    if (use.reject.empty())
      use.reject = reason;
  }

  void VisitStmt_(const ForNode *op) final {
    loops_.push_back(op);
    StmtExprVisitor::VisitStmt_(op);
    loops_.pop_back();
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(TileOpGemm()) && op->args.size() > kGemmArgClearAccum) {
      VisitGemm(op);
      return;
    }
    if (op->op.same_as(TileOpFill()) && op->args.size() >= 2) {
      VisitFill(op);
      return;
    }
    if (op->op.same_as(TileOpCopy()) && op->args.size() >= 2) {
      VisitCopy(op);
      return;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitGemm(const CallNode *op) {
    // The A and B regions are deliberately neither classified nor descended
    // into. A fragment operand is not this pass's business: gemm_tcgen5.py's
    // _check_operand_scopes already rejects it with a better message ("needs A
    // and B in shared memory ... no register-operand (RS/SR/RR) MMA on this
    // architecture"), and an accumulator that is *also* a later GEMM's A
    // operand is best served by promotion anyway, since A in tensor memory is
    // supported (mma_atmem) while A in a fragment is not.
    if (auto buf = RegionBuffer(op->args[kGemmArgC]); buf && IsFragment(*buf)) {
      Entry(*buf).gemms.push_back({op, loops_});
    }
    // The regions above are classified; the remaining arguments are scalars but
    // may in principle mention a buffer, so keep walking them.
    for (size_t i = kGemmArgC + 1; i < op->args.size(); ++i)
      VisitExpr(op->args[i]);
  }

  void VisitFill(const CallNode *op) {
    auto buf = RegionBuffer(op->args[0]);
    if (!buf || !IsFragment(*buf)) {
      StmtExprVisitor::VisitExpr_(op);
      return;
    }
    FragmentUse &use = Entry(*buf);
    if (!is_zero(op->args[1])) {
      Reject(*buf, "it is initialized to a non-zero value, while the MMA's own "
                   "accumulator clear can only zero it.");
      return;
    }
    BufferRegion region = NormalizeToAccessRegion(op->args[0]).region;
    if (!CoversWholeBuffer(region)) {
      Reject(*buf, "only part of it is cleared, while the MMA's accumulator "
                   "clear always covers the whole tile.");
      return;
    }
    if (use.fill != nullptr) {
      Reject(*buf, "it is cleared more than once, so there is no single point "
                   "to fold the clear into.");
      return;
    }
    use.fill = op;
    use.fill_loops = loops_;
  }

  void VisitCopy(const CallNode *op) {
    auto src = RegionBuffer(op->args[0]);
    auto dst = RegionBuffer(op->args[1]);
    if (dst && IsFragment(*dst)) {
      Reject(*dst, "it is written by a T.copy, while an accumulator in tensor "
                   "memory can only be produced by the MMA itself.");
    }
    if (src && IsFragment(*src) && dst && (*dst).scope() != "global") {
      Reject(*src, "it is copied to '" + std::string((*dst).scope()) +
                       "', and only a copy to global memory has a "
                       "tensor-memory drain (T.copy(C_tmem, C_global)).");
    }
    // Both regions are classified either way; do not descend into them.
  }

  static bool CoversWholeBuffer(const BufferRegion &region) {
    const Buffer &buf = region->buffer;
    if (region->region.size() != buf->shape.size())
      return false;
    arith::Analyzer ana;
    for (size_t i = 0; i < region->region.size(); ++i) {
      if (!is_zero(region->region[i]->min))
        return false;
      if (!ana.CanProveEqual(region->region[i]->extent, buf->shape[i]))
        return false;
    }
    return true;
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    if (IsFragment(op->buffer)) {
      Reject(op->buffer,
             "it is read element-wise (outside a GEMM or a copy to "
             "global), and reading a tensor-memory accumulator back "
             "needs an LDT into a fragment, which is limited to a "
             "single warp today.");
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    if (IsFragment(op->buffer)) {
      Reject(op->buffer, "it is written element-wise, while a tensor-memory "
                         "accumulator can only be written by the MMA.");
    }
    StmtExprVisitor::VisitStmt_(op);
  }
};

} // namespace

/*!
 * \brief Replace local.fragment GEMM accumulators with shared.tmem buffers.
 */
class TmemAccumulatorPromoter : public StmtExprMutator {
public:
  static Stmt Rewrite(Stmt body) {
    TmemAccumulatorPromoter promoter;
    return promoter(std::move(body));
  }

private:
  // data Var of a promoted fragment -> its replacement tmem buffer.
  std::unordered_map<const VarNode *, Buffer> promoted_;
  Map<Var, Var> var_remap_;
  // `T.clear` calls whose initialization moved into a gemm's clear_accum.
  std::unordered_set<const CallNode *> dropped_fills_;
  // gemm call -> the clear_accum predicate to install on it.
  std::unordered_map<const CallNode *, PrimExpr> clear_accum_;

  /*!
   * \brief Remedy for a fragment whose *uses* rule promotion out.
   *
   * Kept separate from the message so that a shape-driven rejection, where
   * restructuring the uses would not help at all, can offer its own advice.
   */
  static constexpr const char *kStructureAdvice =
      " Restructure the kernel so the accumulator is only cleared, accumulated "
      "into by T.gemm and copied to global memory, or allocate it explicitly "
      "with T.alloc_tmem and drive clear_accum yourself.";

  [[noreturn]] static void Fail(const Buffer &buf, const std::string &reason) {
    LOG(FATAL) << "PromoteTmemAccumulator: '" << buf->name
               << "' is the accumulator of a T.gemm and lives in "
               << kFragmentScope
               << ", which stcuv2 cannot accumulate into (it has no "
                  "register-destination MMA). Promoting it to tensor memory "
                  "did not work either: "
               << reason;
  }

  /*!
   * \brief Reject a gemm whose operand row is not exactly one 128-byte MMA row.
   *
   * tcgen5 reads exactly 128 bytes of an operand row per MMA, which is what
   * fixes K per dtype: fp32 32, fp16/bf16 64, fp8 128, packed fp4 256.  A wider
   * row is caught and explained downstream by gemm_tcgen5.py; a narrower one is
   * not -- it has no legal swizzle, and ptcc rejects the kernel with the
   * undirected "Unsupported A matrix swizzle layout" (measured for fp16 at K=32
   * and for fp32 at K=16, both 64-byte rows).  A kernel written for the
   * fragment path had no reason to size its BK to a tcgen5 operand row, so this
   * is the constraint promotion trips most often and the one worth naming
   * precisely.
   */
  static void CheckOperandRowBytes(const Buffer &frag, const CallNode *gemm) {
    const auto *k_imm = gemm->args[kGemmArgK].as<IntImmNode>();
    if (k_imm == nullptr)
      return; // dynamic K: nothing to compare against here
    const int64_t k = k_imm->value;
    for (size_t idx : {kGemmArgA, kGemmArgB}) {
      if (!IsBufferLikeExpr(gemm->args[idx]))
        continue;
      Buffer operand = NormalizeToAccessRegion(gemm->args[idx]).region->buffer;
      // An operand staged in tensor memory never goes through the shared-memory
      // swizzle, so the row width does not constrain it.
      if (operand.scope() != "shared" && operand.scope() != "shared.dyn")
        continue;
      const int bits = operand->dtype.bits();
      if (bits <= 0)
        continue;
      const int64_t row_bytes = (k * bits + 7) / 8;
      if (row_bytes == kMmaOperandRowBytes)
        continue;
      Fail(frag,
           "the GEMM's " + std::string(idx == kGemmArgA ? "A" : "B") +
               " operand spans " + std::to_string(row_bytes) +
               " bytes per MMA row (K=" + std::to_string(k) + " x " +
               std::to_string(bits) + "-bit), while tcgen5 reads exactly " +
               std::to_string(kMmaOperandRowBytes) +
               ". Use K == " + std::to_string(kMmaOperandRowBytes * 8 / bits) +
               " for this dtype, tiling a longer reduction with "
               "T.Pipelined.");
    }
  }

  /*!
   * \brief The `clear_accum` predicate replacing a dropped `T.clear`.
   *
   * True exactly on the first execution of the gemm per initialization, i.e. at
   * the first iteration of every loop that encloses the gemm but not the clear.
   * Loops enclosing both (an outer tile loop that re-clears the accumulator
   * each time round) must not appear, or the accumulator would only be cleared
   * on the first tile.  For the usual single reduction loop this is `k == 0`.
   */
  static PrimExpr
  ClearAccumPredicate(const std::vector<const ForNode *> &gemm_loops,
                      const std::vector<const ForNode *> &fill_loops) {
    std::unordered_set<const ForNode *> encloses_fill(fill_loops.begin(),
                                                      fill_loops.end());
    PrimExpr pred = const_true();
    for (const ForNode *loop : gemm_loops) {
      if (encloses_fill.count(loop))
        continue;
      pred = pred && (loop->loop_var == loop->min);
    }
    arith::Analyzer ana;
    return ana.Simplify(pred);
  }

  void Promote(const Buffer &frag, const FragmentUse &use,
               Map<Buffer, Buffer> *block_remap) {
    if (!use.reject.empty())
      Fail(frag, use.reject + kStructureAdvice);
    // LowerSharedTmem addresses tensor memory as (row, column), so a promoted
    // accumulator has to be exactly 2D. A batched accumulator would need its
    // leading modes folded into columns, which only T.alloc_tmem can express.
    if (frag->shape.size() != 2) {
      Fail(frag, "it is " + std::to_string(frag->shape.size()) +
                     "-dimensional, while a tensor-memory accumulator is "
                     "addressed as a 2D (row, column) tile." +
                     kStructureAdvice);
    }
    for (const FragmentUse::GemmSite &site : use.gemms)
      CheckOperandRowBytes(frag, site.call);

    Var tmem_data(frag->data->name_hint,
                  PointerType(PrimType(frag->dtype), kTmemScope));
    Buffer tmem(tmem_data, frag->dtype, frag->shape, frag->strides,
                frag->elem_offset, frag->name, frag->data_alignment,
                frag->offset_factor, frag->buffer_type);
    block_remap->Set(frag, tmem);
    promoted_[frag->data.get()] = tmem;
    var_remap_.Set(frag->data, tmem_data);

    const FragmentUse::GemmSite &first = use.gemms.front();
    if (use.fill != nullptr) {
      // Only the first gemm takes over the clear; any further gemm in the same
      // iteration keeps accumulating on top of it, which is what its own
      // (already false) clear_accum says.
      dropped_fills_.insert(use.fill);
      clear_accum_[first.call] =
          ClearAccumPredicate(first.loops, use.fill_loops);
    } else if (is_zero(first.call->args[kGemmArgClearAccum])) {
      Fail(frag,
           std::string("it is never initialized -- there is neither a "
                       "T.clear nor a clear_accum on the first T.gemm, so "
                       "the accumulator would start from whatever the "
                       "tile happened to hold.") +
               kStructureAdvice);
    }
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    std::vector<Buffer> fragments;
    for (const Buffer &buf : op->alloc_buffers) {
      if (buf.scope() == kFragmentScope)
        fragments.push_back(buf);
    }
    if (fragments.empty())
      return StmtExprMutator::VisitStmt_(op);

    FragmentUseCollector collector;
    collector(op->body);

    Map<Buffer, Buffer> block_remap;
    for (const Buffer &frag : fragments) {
      auto it = collector.uses.find(frag->data.get());
      if (it == collector.uses.end())
        continue;
      // Only accumulators are our business; every other fragment is left alone.
      if (it->second.gemms.empty())
        continue;
      Promote(frag, it->second, &block_remap);
    }
    if (block_remap.empty())
      return StmtExprMutator::VisitStmt_(op);

    SBlock block = GetRef<SBlock>(op);
    Array<Buffer> alloc_buffers = op->alloc_buffers;
    alloc_buffers.MutateByApply([&block_remap](Buffer buf) {
      auto it = block_remap.find(buf);
      return it == block_remap.end() ? buf : (*it).second;
    });
    block.CopyOnWrite()->alloc_buffers = alloc_buffers;
    return StmtExprMutator::VisitStmt_(block.get());
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      if (dropped_fills_.count(call))
        return Evaluate(0);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    auto it = clear_accum_.find(op);
    PrimExpr updated = StmtExprMutator::VisitExpr_(op);
    if (it == clear_accum_.end())
      return updated;
    Call call = Downcast<Call>(updated);
    // CopyOnWrite rather than a fresh Call: the gemm carries TANG annotations
    // (tang_k_step, tang_scale_*) that a rebuilt node would drop.
    call.CopyOnWrite()->args.Set(kGemmArgClearAccum, it->second);
    return call;
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    auto it = promoted_.find(load->buffer->data.get());
    if (it == promoted_.end())
      return load;
    load.CopyOnWrite()->buffer = it->second;
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    auto it = promoted_.find(store->buffer->data.get());
    if (it == promoted_.end())
      return store;
    store.CopyOnWrite()->buffer = it->second;
    return store;
  }

  PrimExpr VisitExpr_(const VarNode *op) final {
    Var var = GetRef<Var>(op);
    if (var_remap_.count(var))
      return var_remap_[var];
    return var;
  }
};

PrimFunc PromoteTmemAccumulator(PrimFunc f) {
  auto target_opt = f->GetAttr<Target>(tvm::attr::kTarget);
  ICHECK(target_opt.defined())
      << "PromoteTmemAccumulator: Require the target attribute";
  ICHECK(TargetTangIsSTCUV2(target_opt.value()))
      << "PromoteTmemAccumulator only applies to the TANG stcuv2 subtarget, "
         "which is the only one with tensor memory; got "
      << target_opt.value()->str();
  f.CopyOnWrite()->body = TmemAccumulatorPromoter::Rewrite(f->body);
  return f;
}

namespace transform {
using namespace tirx::transform;

tvm::transform::Pass PromoteTmemAccumulator() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return tl::PromoteTmemAccumulator(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.PromoteTmemAccumulator", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.tang.transform.PromoteTmemAccumulator",
                        PromoteTmemAccumulator);
}

} // namespace transform
} // namespace tl
} // namespace tvm
