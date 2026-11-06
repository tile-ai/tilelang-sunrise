/*!
 * \file src/tang/transform/materialize_s3_mbarriers.cc
 * \brief Publish the shared->tensor-memory staging as an mbarrier handshake
 *        (stcuv2).
 *
 * Tensor memory has no ThreadSync equivalent. ThreadSync reasons about
 * *buffers* -- it inserts a barrier between a write and a conflicting access --
 * and what has to be ordered on this edge is the cps2t copy engine's write of
 * tensor memory against whoever reads that tensor memory next (the MMA, an ldt,
 * a drain). The engine is not a buffer access the analysis can see, the reader
 * is usually a different warp, and `fence_mem` only retires the issuing warp's
 * own group. The edge must therefore be explicit IR that synchronization
 * passes can inspect.
 *
 * Lowering states the publish as a marker op (`tl.tang_publish_tmem_operand`,
 * carrying the staged tile and the number of warps that will arrive) and this
 * pass turns it into the handshake:
 *
 *   writer warps (exactly the ones that staged):
 * mbar[slot].arrive_on_s2t_fence() first statement that reads the tile:
 * mbar[slot].wait(0) block prologue (elected thread, before any region): init +
 * fence + publish
 *
 * Why an mbarrier rather than a block barrier:
 *
 *   * the arrive is legal where the staging is. A shared->tensor-memory staging
 *     can be scoped to a `T.ws` region, and then the marker sits inside that
 *     region's guard -- a block-wide `__syncthreads()` there is reached only by
 *     the threads that enter the branch. An arrive is not a barrier, so the
 *     handshake works where a block-wide barrier cannot.
 *   * it decouples the two sides: one arrive per participating warp, one wait
 *     per reader, and the count is a *warp* count, not a block size.
 *   * it survives every sync-elision pass by construction: it is not a
 *     `tvm_storage_sync`, so `RemoveRedundantSyncs` has nothing to remove.
 * `fence_mem_arrive_mbarrier` is the fence *and* the arrive in one instruction,
 * which is why the writer's `fence_mem(fg_s2t_default)` is subsumed rather than
 * duplicated: the warp's cps2t retires before any waiter can observe the
 * arrival.
 *
 * The consumer is found by *buffer identity*: the marker carries the staged
 * tile, and the pass waits for the first statement after the marker that
 * mentions that tile at all (a `BufferLoad` for the MMA's `A_t[0]`, or the
 * `access_ptr` the ldt/drain ops carry). No op-pattern matching, so a new
 * reader op cannot silently bypass the wait. The wait goes *inside* that
 * statement's guard when a conditional encloses the reader, and in front of it
 * otherwise: `tl::mbarrier_wait` is a poll loop, so a block-wide waiter makes
 * every warp of the block pay for an edge that concerns one region. A marker
 * whose tile is never read afterwards keeps its arrive and gets no wait -- the
 * arrive is still the fence the warp needed, and with no reader there is no
 * cross-warp edge to order.
 *
 * Runs after `RemoveRedundantSyncs` and `ExpandS3TmemDrain`, so the marker's
 * neighbours are final; before `MakePackedAPI`, because the prologue names
 * `threadIdx.x`.
 */
#include <cstdlib>
#include <string>
#include <vector>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "../op/builtin.h"
#include "../target_utils.h"
#include "cuda/op/builtin.h"
#include "s3_mbarrier_utils.h"

namespace tvm {
namespace tl {
namespace tang {
namespace transform {

using namespace tirx;
using namespace ffi;

namespace {

/*!
 * \brief Does `stmt` mention `buf`'s data var at all?
 *
 * Read-side detection only needs "does this statement touch the staged tile":
 * the tile is written by the staging that sits *before* the marker, so anything
 * at or after the marker that names it is a reader (or another writer, in which
 * case waiting first is still the right order -- the write must not race the
 * engine either).
 *
 * Both spellings are matched: a `BufferLoad` (the MMA's `A_t[0]`, the source of
 * a `T.copy(A_t, fragment)`) and the `tvm_access_ptr` the ldt/drain ops carry.
 */
class TouchesTile : public StmtExprMutator {
public:
  static bool Check(const Stmt &stmt, const Var &tile) {
    TouchesTile checker(tile);
    checker(stmt);
    return checker.touched_;
  }

  /*!
   * \brief The same question for an expression -- a branch condition, say.
   */
  static bool CheckExpr(const PrimExpr &expr, const Var &tile) {
    TouchesTile checker(tile);
    checker(expr);
    return checker.touched_;
  }

private:
  explicit TouchesTile(Var tile) : tile_(std::move(tile)) {}

  void Look(const Buffer &buffer) {
    if (buffer.defined() && buffer->data.same_as(tile_)) {
      touched_ = true;
    }
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    Look(op->buffer);
    return StmtExprMutator::VisitExpr_(op);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    Look(op->buffer);
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    // `tvm_access_ptr(TypeAnnotation, data, offset, extent, rw_mask)`: the
    // buffer is named by args[1].
    if (op->op.same_as(tirx::builtin::tvm_access_ptr()) &&
        op->args.size() >= 2) {
      if (const auto *data = op->args[1].as<VarNode>()) {
        if (tile_.same_as(GetRef<Var>(data))) {
          touched_ = true;
        }
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  Var tile_;
  bool touched_{false};
};

/*!
 * \brief Turn every TMEM publish marker into an arrive, and every reader of the
 *        staged tile into a wait in front of it.
 */
class TmemPublishInjector : public StmtExprMutator {
public:
  /*!
   * \brief ``base`` is the first mbarrier slot this pass may use.
   *
   * The kernel has one mbarrier array and the pipeline handshake (an earlier
   * pass) already numbered its own slots from 0, so numbering this pass's
   * handshakes from 0 too would initialise a slot twice with different arrive
   * counts -- see ``FreeMbarrierSlot`` in s3_mbarrier_utils.h.
   */
  explicit TmemPublishInjector(int64_t base) : base_(base) {}

  Stmt Inject(const Stmt &body) { return VisitStmt(body); }

  const std::vector<int64_t> &arrive_counts() const { return arrive_counts_; }
  int64_t handshakes() const { return handshakes_; }
  bool waited() const { return waited_; }

private:
  /*!
   * \brief Walk a statement list in order: the wait a marker sets up is placed
   *        in front of (or inside) the first later statement that reads the
   *        staged tile.
   */
  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> out;
    for (const Stmt &child : op->seq) {
      out.push_back(PlacePendingWait(child));
    }
    return SeqStmt(out);
  }

  /*!
   * \brief Place the pending wait, then rewrite `child`.
   *
   * Two placements, preferred in this order:
   *
   *   * **inside the reader's own guard**, when `child` is a conditional whose
   *     then- or else-branch reads the tile. The wait is a poll loop
   *     (`tl::mbarrier_wait` spins on `mbarrier_try_wait_parity`), so every
   *     warp that runs it spends issue slots and shared-memory bandwidth on
   *     it -- and in the shape this pass exists for (one region stages into
   *     TMEM, another region reads it) the readers are a minority of the
   *     block, so waiting block-wide makes the whole block pay for an edge
   *     that concerns one region.
   *   * **in front of `child`** otherwise: a reader that no conditional
   *     encloses is reached by every thread that reaches the statement, so
   *     there is nothing to narrow.
   *
   * Narrowing is safe by construction, and the reason is the single-slot
   * pending state: a wait installed here can only be driven by a marker
   * *earlier in this list*, never by one inside `child`'s own subtree. A branch
   * that stages *and* then reads the tile gets its wait from the branch's own
   * list, behind its marker, because no marker was pending when the branch was
   * entered. The wait therefore cannot be hoisted above the arrive that must
   * precede it.
   */
  Stmt PlacePendingWait(const Stmt &child) {
    if (!pending_barrier_.defined() || !pending_tile_.defined() ||
        !TouchesTile::Check(child, pending_tile_)) {
      return VisitStmt(child);
    }
    PrimExpr barrier = pending_barrier_;
    if (const auto *branch = child.as<IfThenElseNode>()) {
      Stmt else_case =
          branch->else_case.defined() ? branch->else_case.value() : Stmt();
      const bool then_reads =
          TouchesTile::Check(branch->then_case, pending_tile_);
      const bool else_reads =
          else_case.defined() && TouchesTile::Check(else_case, pending_tile_);
      if ((then_reads || else_reads) &&
          !TouchesTile::CheckExpr(branch->condition, pending_tile_)) {
        ClearPending();
        return VisitStmt(IfThenElse(
            branch->condition,
            then_reads ? PrependWait(barrier, branch->then_case)
                       : branch->then_case,
            else_reads ? PrependWait(barrier, else_case) : else_case));
      }
    }
    ClearPending();
    return VisitStmt(SeqStmt({WaitOn(barrier), child}));
  }

  /*!
   * \brief The wait itself: a spin on the barrier's parity-0 phase.
   */
  Stmt WaitOn(const PrimExpr &barrier) const {
    return Evaluate(Call(DataType::Handle(), tl::mbarrier_wait_parity(),
                         {barrier, IntImm(DataType::Int(32), 0)}));
  }

  /*!
   * \brief Put the wait in front of `body`, which is either a single statement
   *        or the list a region guard wraps.
   */
  Stmt PrependWait(const PrimExpr &barrier, const Stmt &body) const {
    Array<Stmt> out;
    out.push_back(WaitOn(barrier));
    if (const auto *seq = body.as<SeqStmtNode>()) {
      for (const Stmt &stmt : seq->seq) {
        out.push_back(stmt);
      }
    } else {
      out.push_back(body);
    }
    return SeqStmt(out);
  }

  void ClearPending() {
    pending_barrier_ = PrimExpr();
    pending_tile_ = Var();
    waited_ = true;
  }

  /*!
   * \brief A marker inside a loop is refused instead of mis-lowered.
   *
   * The handshake waits on parity 0, which is the whole story for a marker that
   * runs once per kernel. Inside a loop the phase alternates and the arrive
   * count would have to be re-armed per iteration, so this pass must not guess:
   * The partial-sync materializer below refuses in-loop sites for the same
   * reason.
   * No lowering emits the marker there today -- the cps2t staging sits outside
   * its own row/column loops.
   */
  Stmt VisitStmt_(const ForNode *op) final {
    ++loop_depth_;
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    --loop_depth_;
    return stmt;
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    const auto *eval = stmt.as<EvaluateNode>();
    if (eval == nullptr) {
      return stmt;
    }
    const auto *call = eval->value.as<CallNode>();
    // Matched by registered name rather than by Op identity: the emitter
    // (src/tang/op/copy.cc) is another translation unit, and a mismatch here
    // would silently leave the publish to the code generator's fallback.
    if (call == nullptr || !call->op.same_as(tang_publish_tmem_operand())) {
      return stmt;
    }
    ICHECK_EQ(call->args.size(), 2U)
        << "tang_publish_tmem_operand takes [tile, arrive_warps]";
    const auto *tile = call->args[0].as<BufferLoadNode>();
    const auto *warps = call->args[1].as<IntImmNode>();
    ICHECK(tile != nullptr && warps != nullptr && warps->value >= 1)
        << "the TMEM publish marker needs a BufferLoad tile and a positive "
           "constant arrive count";
    ICHECK_EQ(loop_depth_, 0)
        << "a TMEM publish marker inside a loop needs per-iteration mbarrier "
           "phase bookkeeping, which tl.tang.MaterializeTangMbarriers does not "
           "implement";
    ICHECK(!pending_barrier_.defined())
        << "two TMEM publish markers with no reader in between: the first "
           "one's "
           "wait was never placed, which means the staged tile it publishes is "
           "never read";

    const int slot = static_cast<int>(base_ + arrive_counts_.size());
    arrive_counts_.push_back(warps->value);
    PrimExpr barrier = Call(DataType::Handle(), tl::get_mbarrier(),
                            {IntImm(DataType::Int(32), slot)});
    pending_barrier_ = barrier;
    pending_tile_ = tile->buffer->data;
    ++handshakes_;
    return Evaluate(
        Call(DataType::Handle(), tang_fence_s2t_arrive_mbarrier(), {barrier}));
  }

  std::vector<int64_t> arrive_counts_;
  int64_t base_{0};
  PrimExpr pending_barrier_;
  Var pending_tile_;
  int64_t handshakes_{0};
  int loop_depth_{0};
  bool waited_{false};
};

constexpr int64_t kMinPartialThreads = 64;
constexpr int64_t kWarpSize = 32;

bool IsSharedSync(const CallNode *call) {
  return call != nullptr && call->op.same_as(builtin::tvm_storage_sync()) &&
         !call->args.empty() && call->args[0].as<StringImmNode>() != nullptr &&
         call->args[0].as<StringImmNode>()->value == "shared";
}

/*! \brief Materialise surviving, out-of-loop multi-warp partial syncs. */
class PartialSyncRewriter : public StmtExprMutator {
public:
  explicit PartialSyncRewriter(int64_t base) : base_(base) {}

  Stmt Rewrite(const Stmt &body) { return VisitStmt(body); }
  const std::vector<int64_t> &arrive_counts() const { return arrive_counts_; }
  int64_t in_loop_sites() const { return in_loop_sites_; }

private:
  Stmt VisitStmt_(const ForNode *op) final {
    ++loop_depth_;
    Stmt body = VisitStmt(op->body);
    --loop_depth_;
    For node = GetRef<For>(op);
    node.CopyOnWrite()->body = body;
    return Stmt(node);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (call == nullptr || !IsSharedSync(call) || call->args.size() < 3) {
      return StmtExprMutator::VisitStmt_(op);
    }
    const auto *threads = call->args[2].as<IntImmNode>();
    if (threads == nullptr || threads->value < kMinPartialThreads) {
      return StmtExprMutator::VisitStmt_(op);
    }
    if (loop_depth_ != 0) {
      ++in_loop_sites_;
      return StmtExprMutator::VisitStmt_(op);
    }
    ICHECK_EQ(threads->value % kWarpSize, 0)
        << "a partial sync must cover a warp multiple, got " << threads->value;
    const int64_t slot = base_ + arrive_counts_.size();
    arrive_counts_.push_back(threads->value / kWarpSize);
    PrimExpr barrier =
        Call(DataType::Handle(), tl::get_mbarrier(),
             {IntImm(DataType::Int(32), static_cast<int>(slot))});
    return SeqStmt(
        {Evaluate(Call(DataType::Handle(), tl::tang_fence_g2s_arrive_mbarrier(),
                       {barrier})),
         Evaluate(Call(DataType::Handle(), tl::mbarrier_wait_parity(),
                       {barrier, IntImm(DataType::Int(32), 0)}))});
  }

  int loop_depth_{0};
  int64_t in_loop_sites_{0};
  int64_t base_{0};
  std::vector<int64_t> arrive_counts_;
};

} // namespace

tvm::transform::Pass MaterializeTangMbarriers() {
  auto pass_func = [](PrimFunc f, const IRModule &,
                      const tvm::transform::PassContext &ctx) {
    Optional<Target> opt_target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!opt_target.defined() || !TargetTangIsSTCUV2(opt_target.value())) {
      return f;
    }
    const int64_t base = tang::transform::FreeMbarrierSlot(f->body);
    TmemPublishInjector injector(base);
    Stmt body = injector.Inject(f->body);
    const int64_t partial_base = base + injector.arrive_counts().size();
    PartialSyncRewriter partials(partial_base);
    body = partials.Rewrite(body);
    if (std::getenv("TL_DEBUG_TMEM_PUBLISH") != nullptr) {
      std::string name = "<?>";
      if (auto sym = f->GetAttr<String>(tvm::attr::kGlobalSymbol)) {
        name = sym.value();
      }
      LOG(INFO) << "[tmem_publish] " << name << ": " << injector.handshakes()
                << " handshake(s), wait placed: " << injector.waited();
    }
    if (std::getenv("TL_DEBUG_MBAR_PARTIAL") != nullptr) {
      std::string name = "<?>";
      if (auto sym = f->GetAttr<String>(tvm::attr::kGlobalSymbol)) {
        name = sym.value();
      }
      LOG(INFO) << "[mbar_partial] " << name << ": rewrote "
                << partials.arrive_counts().size()
                << " out-of-loop partial sync(s); " << partials.in_loop_sites()
                << " in-loop site(s) left to the code generator's warning";
    }
    std::vector<int64_t> arrive_counts = injector.arrive_counts();
    arrive_counts.insert(arrive_counts.end(), partials.arrive_counts().begin(),
                         partials.arrive_counts().end());
    if (arrive_counts.empty()) {
      return f;
    }
    // Both late protocols share one allocation plan and one publication
    // prologue. Their semantic requests remain separate above.
    ThreadExtentPrologue prologue(
        arrive_counts, "the late TANG mbarrier array", base,
        /*disable_shuffle_elect=*/
        ctx->GetConfig<Bool>(kDisableShuffleElect, Bool(false)).value());
    body = prologue.Insert(body);
    ICHECK(prologue.inserted())
        << "no threadIdx.x binding found for the late TANG mbarrier prologue";
    f.CopyOnWrite()->body = body;
    return f;
  };
  return tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.tang.MaterializeTangMbarriers", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.tang.transform.MaterializeTangMbarriers",
                        MaterializeTangMbarriers);
}

} // namespace transform
} // namespace tang
} // namespace tl
} // namespace tvm
