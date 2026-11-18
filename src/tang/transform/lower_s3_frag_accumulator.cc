/*!
 *  \file lower_s3_frag_accumulator.cc
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
#include "op/builtin.h"
#include "op/utils.h"
#include "tang/target_utils.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/builtin.h>
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
constexpr size_t kGemmArgTransposeA = 3;
constexpr size_t kGemmArgTransposeB = 4;
constexpr size_t kGemmArgK = 7;
constexpr size_t kGemmArgClearAccum = 9;

// Argument positions in a tl.tileop.copy call.
constexpr size_t kCopyArgSrc = 0;
constexpr size_t kCopyArgDst = 1;

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

/*! \brief Whether `region` starts at the buffer origin and spans its shape. */
bool RegionCoversWholeBuffer(const BufferRegion &region) {
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
    if (!RegionCoversWholeBuffer(region)) {
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

/*!
 * \brief One enclosing ``if`` and which branch a statement sits in.
 *
 * The loop nest alone is not enough to say "the same iteration": a staging copy
 * outside a guard and its write-back inside one (or the other way round) are in
 * the same loop nest yet do not run together. Folding such a pair moves the
 * write across the guard -- the producer's copy becomes the global write, so it
 * runs even when the consumer would not have, or stops running when it would.
 */
struct StagingGuard {
  const IfThenElseNode *node;
  /*! \brief 0 for the then-branch, 1 for the else-branch. */
  int branch;

  bool operator==(const StagingGuard &other) const {
    return node == other.node && branch == other.branch;
  }

  bool operator!=(const StagingGuard &other) const { return !(*this == other); }
};

/*!
 * \brief How one shared buffer is used while staging a GEMM accumulator.
 *
 * The shape this pass looks for moves an accumulator through shared memory on
 * its way out:
 *
 *     T.copy(C_frag, S)             // stage the accumulator
 *     T.copy(S[prefix], C_global)   // write it out (possibly a column slice)
 *
 * On stcuv2 the promoted accumulator lives in tensor memory, whose only exits
 * are the fused TMEM->global drain, a single-warp LDT and the cpt2s engine
 * (128-byte rows, 64-row tiles).  A kernel that stages the tile through shared
 * memory purely to slice or reshape it before the writeback therefore has no
 * lowering at all, even though the shared tile never holds anything but the
 * accumulator's own bytes.
 */
struct StagingUses {
  /*! \brief Number of `T.copy(..., S)` statements that write S. */
  int writes = 0;
  /*! \brief Number of `T.copy(S, ...)` statements that read S. */
  int reads = 0;
  /*! \brief Set when S is touched any other way (element-wise, MMA operand...).
   */
  bool poisoned = false;
  /*! \brief Set when either copy sits inside a warp-specialized region. */
  bool in_warp_specialize = false;
  /*! \brief `T.copy(C_frag, S)`, the copy that stages the accumulator. */
  const CallNode *producer = nullptr;
  /*! \brief `T.copy(S[prefix], G)`, the copy that writes the staged tile out.
   */
  const CallNode *consumer = nullptr;
  /*! \brief The accumulator fragment the producer stages from. */
  Buffer accum;
  /*! \brief Loops enclosing the two copies, to require the same iteration. */
  std::vector<const ForNode *> producer_loops;
  std::vector<const ForNode *> consumer_loops;
  /*! \brief Conditionals enclosing the two copies, to require the same guard.
   */
  std::vector<StagingGuard> producer_guards;
  std::vector<StagingGuard> consumer_guards;
};

/*!
 * \brief Classify how every shared buffer is used, to find pure staging pairs.
 *
 * Same structural approach as FragmentUseCollector: a copy's two region
 * arguments are classified and not descended into, while anything else that
 * mentions the buffer -- an element-wise access, a GEMM operand, a copy in
 * another direction -- marks it *poisoned*, so it is never folded.
 */
class AccumulatorStagingCollector : public StmtExprVisitor {
public:
  std::unordered_map<const VarNode *, StagingUses> uses;
  std::unordered_map<const CallNode *, const CallNode *> adjacent_copies;

  void VisitStmt_(const SeqStmtNode *op) final {
    for (size_t i = 0; i + 1 < op->seq.size(); ++i) {
      const auto *first = op->seq[i].as<EvaluateNode>();
      const auto *second = op->seq[i + 1].as<EvaluateNode>();
      const auto *producer = first ? first->value.as<CallNode>() : nullptr;
      const auto *consumer = second ? second->value.as<CallNode>() : nullptr;
      if (producer && consumer && producer->op.same_as(TileOpCopy()) &&
          consumer->op.same_as(TileOpCopy())) {
        adjacent_copies[producer] = consumer;
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode *op) final {
    loops_.push_back(op);
    StmtExprVisitor::VisitStmt_(op);
    loops_.pop_back();
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    // The condition itself is not guarded by the branch, so it is visited with
    // the enclosing guards only; each branch gets its own guard entry.
    VisitExpr(op->condition);
    guards_.push_back({op, 0});
    VisitStmt(op->then_case);
    guards_.pop_back();
    if (op->else_case.defined()) {
      guards_.push_back({op, 1});
      VisitStmt(op->else_case.value());
      guards_.pop_back();
    }
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    const bool warp_specialized =
        op->attr_key.find("warp_specialize") != std::string::npos;
    if (warp_specialized)
      ++warp_specialize_depth_;
    StmtExprVisitor::VisitStmt_(op);
    if (warp_specialized)
      --warp_specialize_depth_;
  }

  void VisitExpr_(const CallNode *op) final {
    if (!op->op.same_as(TileOpCopy()) || op->args.size() < 2) {
      StmtExprVisitor::VisitExpr_(op);
      return;
    }
    const Buffer src = OperandBuffer(op->args[0]);
    const Buffer dst = OperandBuffer(op->args[1]);
    if (IsSharedBuffer(dst))
      NoteWrite(op, src, dst);
    if (IsSharedBuffer(src))
      NoteRead(op, dst, src);
    // The two regions are classified; the remaining arguments are scalars.
    for (size_t i = 2; i < op->args.size(); ++i)
      VisitExpr(op->args[i]);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    if (IsSharedBuffer(op->buffer))
      Poison(op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    if (IsSharedBuffer(op->buffer))
      Poison(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

private:
  std::vector<const ForNode *> loops_;
  std::vector<StagingGuard> guards_;
  int warp_specialize_depth_ = 0;

  static Buffer OperandBuffer(const PrimExpr &arg) {
    if (!IsBufferLikeExpr(arg))
      return Buffer();
    return NormalizeToAccessRegion(arg).region->buffer;
  }

  static bool IsSharedBuffer(const Buffer &buf) {
    return buf.defined() &&
           (buf.scope() == "shared" || buf.scope() == "shared.dyn");
  }

  StagingUses &Entry(const Buffer &buf) { return uses[buf->data.get()]; }

  void Poison(const Buffer &buf) { Entry(buf).poisoned = true; }

  /*! \brief Record `T.copy(src, S)`; only a whole accumulator qualifies. */
  void NoteWrite(const CallNode *call, const Buffer &src, const Buffer &dst) {
    StagingUses &use = Entry(dst);
    ++use.writes;
    use.in_warp_specialize |= warp_specialize_depth_ > 0;
    const bool from_accumulator =
        src.defined() && src.scope() == kFragmentScope;
    if (!from_accumulator || use.producer != nullptr) {
      // A second writer, or a tile filled from anywhere but an accumulator:
      // this shared buffer is not a pure staging step.
      use.producer = nullptr;
      use.accum = Buffer();
      return;
    }
    use.producer = call;
    use.accum = src;
    use.producer_loops = loops_;
    use.producer_guards = guards_;
  }

  /*! \brief Record `T.copy(S, dst)`; only a global write-out qualifies. */
  void NoteRead(const CallNode *call, const Buffer &dst, const Buffer &src) {
    StagingUses &use = Entry(src);
    ++use.reads;
    use.in_warp_specialize |= warp_specialize_depth_ > 0;
    if (!dst.defined() || dst.scope() != "global") {
      // Reading the staged tile anywhere but global is a second use, not the
      // write-out: mark it the same way an element-wise access is marked, so
      // the verdict does not depend on which read was visited first (the
      // consumer pointer alone is order-sensitive) and a later relaxation of
      // the read count cannot resurrect the fold.
      Poison(src);
      use.consumer = nullptr;
      return;
    }
    if (use.consumer != nullptr) {
      use.consumer = nullptr;
      return;
    }
    use.consumer = call;
    use.consumer_loops = loops_;
    use.consumer_guards = guards_;
  }
};

/*!
 * \brief Elide a pure accumulator staging step.
 *
 * `T.copy(C_frag, S); T.copy(S[prefix], C_global)` becomes
 * `T.copy(C_frag[prefix], C_global)` and the staged buffer is dropped, when
 *
 *  - S is written by exactly that one copy and read by exactly that one copy,
 *  - the copies are adjacent and stage the same dtype and shape,
 *  - nothing else touches S (no element-wise access, no MMA operand),
 *  - both copies sit in the same loop nest *and under the same conditionals*,
 *    so the staged tile is consumed exactly where it was produced -- folding a
 *    pair with different guards would either drop a guard (writing the output
 *    when the consumer never ran) or add one,
 *  - neither copy is inside a warp-specialized region (the drain has no
 *    meaning there),
 *  - the staged region is the whole accumulator and the write-out reads a
 *    prefix window of it (a column slice keeps the row origin),
 *  - the staging copy carries no annotations of its own.
 *
 * Every other shape is left untouched, so PromoteTmemAccumulator still reports
 * why such a kernel cannot be promoted.
 */
class AccumulatorStagingFolder : public StmtExprMutator {
public:
  /*!
   * \brief Fold every pure staging pair, reporting why the rest were left.
   *
   * `declined` maps an accumulator's data Var to the reason its staging pair
   * was not folded, so the rejection that follows can name the real obstacle
   * instead of the generic "it is copied to shared".
   */
  static Stmt Fold(Stmt body,
                   std::unordered_map<const VarNode *, std::string> *declined) {
    AccumulatorStagingCollector collector;
    collector(body);
    AccumulatorStagingFolder folder;
    folder.Plan(collector, declined);
    if (folder.rewrites_.empty())
      return body;
    return folder(std::move(body));
  }

private:
  struct Rewrite {
    PrimExpr src;
    PrimExpr dst;
  };
  // staging copy -> the write-back it is folded into
  std::unordered_map<const CallNode *, Rewrite> rewrites_;
  // copies whose statement is elided
  std::unordered_set<const CallNode *> dropped_;
  // staged buffers whose allocation may have become dead
  std::unordered_map<const VarNode *, Buffer> staged_buffers_;

  void Plan(const AccumulatorStagingCollector &collector,
            std::unordered_map<const VarNode *, std::string> *declined) {
    for (const auto &entry : collector.uses) {
      const StagingUses &use = entry.second;
      if (use.writes != 1 || use.reads != 1 || use.poisoned ||
          use.in_warp_specialize || use.producer == nullptr ||
          use.consumer == nullptr || !use.accum.defined())
        continue;
      if (use.producer_loops != use.consumer_loops ||
          use.producer_guards != use.consumer_guards) {
        // Report it rather than silently declining: the caller's fallback
        // message ("it is copied to shared") hides the actual obstruction.
        if (declined != nullptr && use.accum.defined()) {
          (*declined)[use.accum->data.get()] =
              "its staging copy and the write-back that reads the staged tile "
              "do not run in the same loop nest and under the same guards, so "
              "folding them would move the global write across a loop or a "
              "branch.";
        }
        continue;
      }
      const CallNode *producer = use.producer;
      const CallNode *consumer = use.consumer;
      auto adjacent = collector.adjacent_copies.find(producer);
      if (adjacent == collector.adjacent_copies.end() ||
          adjacent->second != consumer) {
        continue;
      }
      if (producer->args.size() != 2 || consumer->args.size() != 2)
        continue;
      if (!producer->annotations.empty() || !consumer->annotations.empty())
        continue; // a hinted staging copy is not a plain move

      const AccessRegion staged = NormalizeToAccessRegion(producer->args[1]);
      const AccessRegion accum = NormalizeToAccessRegion(producer->args[0]);
      const AccessRegion staged_read =
          NormalizeToAccessRegion(consumer->args[0]);
      const AccessRegion out = NormalizeToAccessRegion(consumer->args[1]);
      // Only a whole-tile stage maps onto the accumulator: a partial one would
      // need a tensor-memory origin the fragment spelling cannot name.
      if (!RegionCoversWholeBuffer(staged.region) ||
          !RegionCoversWholeBuffer(accum.region))
        continue;
      if (out.region->buffer.scope() != "global")
        continue;
      if (staged.region->buffer->dtype != accum.region->buffer->dtype ||
          staged.region->buffer->shape.size() !=
              accum.region->buffer->shape.size()) {
        continue;
      }
      arith::Analyzer analyzer;
      bool same_shape = true;
      for (size_t i = 0; i < accum.region->buffer->shape.size(); ++i) {
        same_shape &= analyzer.CanProveEqual(staged.region->buffer->shape[i],
                                             accum.region->buffer->shape[i]);
      }
      if (!same_shape) {
        continue;
      }
      const size_t ndim = accum.region->region.size();
      if (staged_read.region->region.size() != ndim)
        continue;
      bool prefix_window = true;
      for (const Range &range : staged_read.region->region)
        prefix_window &= is_zero(range->min);
      if (!prefix_window)
        continue; // an offset window has no name on the accumulator

      Array<PrimExpr> mins;
      Array<PrimExpr> extents;
      for (const Range &range : staged_read.region->region) {
        mins.push_back(range->min);
        extents.push_back(range->extent);
      }
      Array<PrimExpr> region_args{
          BufferLoad(use.accum, mins),
          IntImm(DataType::Int(32), staged_read.access_mask)};
      for (const PrimExpr &extent : extents)
        region_args.push_back(extent);
      rewrites_[producer] = {Call(DataType::Handle(), region(), region_args),
                             consumer->args[1]};
      dropped_.insert(consumer);
      staged_buffers_[entry.first] = staged.region->buffer;
    }
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      if (dropped_.count(call))
        return Evaluate(0);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    PrimExpr updated = StmtExprMutator::VisitExpr_(op);
    auto it = rewrites_.find(op);
    if (it == rewrites_.end())
      return updated;
    Call call = Downcast<Call>(updated);
    // CopyOnWrite rather than a fresh Call: the copy may carry annotations.
    call.CopyOnWrite()->args.Set(kCopyArgSrc, it->second.src);
    call.CopyOnWrite()->args.Set(kCopyArgDst, it->second.dst);
    return call;
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    SBlock block = Downcast<SBlock>(StmtExprMutator::VisitStmt_(op));
    if (staged_buffers_.empty())
      return block;
    Array<Buffer> kept;
    bool changed = false;
    for (const Buffer &buf : block->alloc_buffers) {
      if (staged_buffers_.count(buf->data.get()) &&
          !ReferencedIn(block->body, buf->data.get())) {
        changed = true;
        continue;
      }
      kept.push_back(buf);
    }
    if (!changed)
      return block;
    block.CopyOnWrite()->alloc_buffers = kept;
    return block;
  }

  /*! \brief Whether `data` still appears in `body` after the fold. */
  static bool ReferencedIn(const Stmt &body, const VarNode *data) {
    bool found = false;
    PostOrderVisit(body, [&](const ObjectRef &node) {
      if (const auto *load = node.as<BufferLoadNode>())
        found |= load->buffer->data.get() == data;
      else if (const auto *store = node.as<BufferStoreNode>())
        found |= store->buffer->data.get() == data;
      else if (const auto *decl = node.as<DeclBufferNode>())
        found |= decl->buffer->data.get() == data;
      else if (const auto *call = node.as<CallNode>())
        if (call->args.size() >= 2 &&
            call->op.same_as(tirx::builtin::tvm_access_ptr()))
          if (const auto *var = call->args[1].as<VarNode>())
            found |= var == data;
    });
    return found;
  }
};

} // namespace

/*!
 * \brief Replace local.fragment GEMM accumulators with shared.tmem buffers.
 */
class TmemAccumulatorPromoter : public StmtExprMutator {
public:
  static Stmt
  Rewrite(Stmt body,
          const std::unordered_map<const VarNode *, std::string> &declined) {
    TmemAccumulatorPromoter promoter;
    promoter.declined_ = declined;
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
  // accumulator data Var -> why its staging pair could not be folded
  std::unordered_map<const VarNode *, std::string> declined_;

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
   * \brief Whether a GEMM operand has K as its contiguous axis.
   *
   * Same convention as the TANG layout generator (``_make_tang_ab_layout``):
   * ``a_major = transpose_A`` and ``b_major = !transpose_B`` name the MN-major
   * spellings, so A is K-major without ``transpose_A`` and B is K-major with
   * ``transpose_B``. A non-constant flag is treated as K-major, matching the
   * layout generator's own default, since that is the spelling the frontend
   * produces.
   */
  static bool IsKMajorOperand(const CallNode *gemm, bool is_a) {
    const size_t pos = is_a ? kGemmArgTransposeA : kGemmArgTransposeB;
    if (gemm->args.size() <= pos)
      return true;
    const auto *imm = gemm->args[pos].as<IntImmNode>();
    if (imm == nullptr)
      return true;
    return is_a ? imm->value == 0 : imm->value != 0;
  }

  /*!
   * \brief Reject a gemm whose operand row is not one or more 128-byte MMA
   * rows.
   *
   * tcgen5 reads at most 128 bytes of an operand row per MMA, which is what
   * fixes K per dtype: fp32 32, fp16/bf16 64, fp8 128, packed fp4 256.  A
   * K-major operand row that is a *whole number* of MMA rows is read one
   * segment at a time (gemm_tcgen5.py splits both the copy and the MMA), so it
   * is fine; an MN-major one runs its row along M or N and cannot be split,
   * which is caught and explained downstream by gemm_tcgen5.py.  A narrower row
   * is not: it has no legal swizzle, and ptcc rejects the kernel with the
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
      const bool k_major = IsKMajorOperand(gemm, idx == kGemmArgA);
      if (k_major && row_bytes > kMmaOperandRowBytes &&
          row_bytes % kMmaOperandRowBytes == 0)
        continue; // segmented operand: read one 128-byte row per MMA
      Fail(frag,
           "the GEMM's " + std::string(idx == kGemmArgA ? "A" : "B") +
               " operand spans " + std::to_string(row_bytes) +
               " bytes per MMA row (K=" + std::to_string(k) + " x " +
               std::to_string(bits) + "-bit), while tcgen5 reads " +
               std::to_string(kMmaOperandRowBytes) + " bytes per MMA" +
               (k_major ? " (a whole multiple of that is split into segments; "
                          "anything else is not)"
                        : " and this operand is MN-major, so its row cannot be "
                          "split") +
               ". Use K == " + std::to_string(kMmaOperandRowBytes * 8 / bits) +
               " for this dtype, a multiple of " +
               std::to_string(kMmaOperandRowBytes * 8 / bits) +
               " if the operand is K-major, or tile a longer reduction with "
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
    if (!use.reject.empty()) {
      auto declined = declined_.find(frag->data.get());
      if (declined != declined_.end())
        Fail(frag, declined->second + kStructureAdvice);
      Fail(frag, use.reject + kStructureAdvice);
    }
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
  // A pure `accumulator -> shared -> global` staging step is elided first, so
  // the promoted accumulator drains straight into the consumer's global region.
  std::unordered_map<const VarNode *, std::string> declined;
  f.CopyOnWrite()->body = AccumulatorStagingFolder::Fold(f->body, &declined);
  f.CopyOnWrite()->body = TmemAccumulatorPromoter::Rewrite(f->body, declined);
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
