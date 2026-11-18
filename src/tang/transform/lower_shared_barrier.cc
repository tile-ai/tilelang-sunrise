/*!
 * \file lower_shared_barrier.cc
 * \brief Turn TANG shared.barrier allocations into an mbarrier init prologue.
 *
 * T.alloc_barrier only allocates the buffer and records each barrier's arrive
 * count in a `barrier_init` block annotation. Translating that annotation into
 * ptx_init_barrier_thread_count calls -- run by one elected thread, then
 * published to the rest of the block -- is this pass. Without it arrive/wait
 * run against an uninitialised shared word, which fails silently.
 *
 * TANG carries its own copy rather than reusing the CUDA pass of the same name.
 * src/cuda/transform is compiled only when USE_CUDA is on, so an S3 build would
 * not contain that pass at all. The two rewrites share the `barrier_init`
 * contract, so a change to how T.alloc_barrier records arrive counts has to
 * touch both.
 *
 * A shared.cluster_barrier is initialised here as well, but unlike a
 * block-scoped barrier its init is published cluster-wide
 * (storage_sync("cluster") -> tl::cluster_sync(), i.e. sync.all.bkcls 0) so a
 * peer CTA that arrives on it remotely observes the init first.
 * T.mbarrier_arrive(mbar, cta_id) keeps both forms: cta_id 0 is rewritten to a
 * block-scoped local arrive, a non-zero constant rank is left for codegen to
 * lower to the DSHM remote arrive (tl::cluster_arrive_barrier).
 */

#include "cuda/op/builtin.h"
#include "support/check.h"
#include "tang/op/builtin.h"

#include <tvm/ir/type.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>
#include <utility>

namespace tvm {
namespace tl {

namespace {

using namespace tirx;
using namespace ffi;

// BlockAttr recording the arrive counts for each barrier allocation.
constexpr const char *kBarrierInitAttr = "barrier_init";

class TangSharedBarrierRewriter : public StmtExprMutator {
public:
  static Stmt Rewrite(Stmt body, bool disable_shuffle_elect) {
    TangSharedBarrierRewriter rewriter(disable_shuffle_elect);
    return rewriter(std::move(body));
  }

private:
  explicit TangSharedBarrierRewriter(bool disable_shuffle_elect)
      : disable_shuffle_elect_(disable_shuffle_elect) {}

  Stmt VisitStmt_(const SBlockNode *op) final {
    // Only the buffers allocated by THIS block, not those inherited from a
    // parent: each allocation is initialised once, where it appears.
    Array<Buffer> barrier_buffers;
    // Per-block flag: only THIS block's cluster_barrier allocations publish
    // their init cluster-wide. Must be local, not a member -- otherwise a
    // cluster barrier in one block would wrongly flip later block-level
    // barrier inits to cluster_sync.
    bool has_cluster_barrier = false;
    for (const Buffer &buffer : op->alloc_buffers) {
      const auto *ptr_type =
          buffer->data->type_annotation.as<PointerTypeNode>();
      if (!ptr_type)
        continue;
      const std::string scope = ptr_type->storage_scope;
      // shared.cluster_barrier is initialised like a block mbarrier, but its
      // init is published cluster-wide below so peer CTAs can arrive on it
      // remotely.
      if (scope == "shared.barrier" || scope == "shared.cluster_barrier") {
        barrier_buffers.push_back(buffer);
        if (scope == "shared.cluster_barrier") {
          has_cluster_barrier = true;
        }
      }
    }

    if (barrier_buffers.empty()) {
      return StmtExprMutator::VisitStmt_(op);
    }

    ICHECK(thread_var_.defined())
        << "barrier buffer '" << barrier_buffers[0]->name
        << "' is allocated outside any threadIdx.x extent, so there is no "
           "thread to elect for its init";

    /*
    Transform:
        mbarrier_list = T.alloc_barrier(arrive_counts: list[int], "handle",
                                        scope="shared.barrier")

    into:
        # emitted by the definition of T.alloc_barrier
        mbarrier_list = T.alloc_buffer(len(arrive_counts), "handle",
                                       scope="shared.barrier")

        # emitted by this pass
        if elected_thread:
          for i in range(len(arrive_counts)):
            T.ptx_init_barrier_thread_count(mbarrier_list[i], arrive_counts[i])
        fence; __syncthreads()
    */
    ICHECK(op->annotations.count(kBarrierInitAttr))
        << "barrier buffer '" << barrier_buffers[0]->name
        << "' is allocated without the barrier_init annotation carrying its "
           "arrive counts";
    auto barrier_init_map = op->annotations.Get(kBarrierInitAttr)
                                ->as<Map<Var, Array<PrimExpr>>>()
                                .value();

    Array<Stmt> init_calls;
    for (const Buffer &buffer : barrier_buffers) {
      ICHECK(barrier_init_map.count(buffer->data))
          << "barrier buffer '" << buffer->name
          << "' is missing from the barrier_init annotation";
      Array<PrimExpr> arrive_counts = barrier_init_map.at(buffer->data);
      const auto *extent = buffer->shape[0].as<IntImmNode>();
      ICHECK(extent != nullptr)
          << "barrier buffer '" << buffer->name
          << "' needs a constant length, but got " << buffer->shape[0];
      ICHECK(arrive_counts.size() == static_cast<size_t>(extent->value))
          << "the number of arrive counts (" << arrive_counts.size()
          << ") must match the length of barrier buffer '" << buffer->name
          << "' (" << extent->value << ")";

      for (size_t i = 0; i < arrive_counts.size(); ++i) {
        Call init(DataType::Handle(), builtin::ptx_init_barrier_thread_count(),
                  {BufferLoad(buffer,
                              {IntImm(DataType::Int(32), static_cast<int>(i))}),
                   arrive_counts[i]});
        init_calls.push_back(Evaluate(init));
      }
    }

    PrimExpr elected;
    if (disable_shuffle_elect_) {
      elected = EQ(thread_var_->var, 0);
    } else {
      elected = Call(DataType::Bool(), tl_shuffle_elect(), {0});
    }

    SBlock block = GetRef<SBlock>(op);
    Array<Stmt> new_body;
    new_body.push_back(IfThenElse(elected,
                                  init_calls.size() == 1 ? init_calls.back()
                                                         : SeqStmt(init_calls),
                                  Stmt()));
    // TANG's fence is a real CTA-scope release: fence.mem, the vendor's
    // documented CTA-scope primitive. TANG has no fence.mbarrier_init, so this
    // is the closest instruction to CUDA's fence of the same purpose.
    //
    // mbarrier_init already ends in a fence.mem of its own, but that lives in
    // the compiler's builtin header (__clang_tang_builtin_vars.h) and only on
    // the non-sreg path: under -fstpu-sreg-mbarrier the wrapper is a bare call
    // to __sreg_mbarrier_init with no fence in its body. Rather than depend on
    // the fence being present in both modes, the pass that knows a barrier was
    // just initialised owns the ordering here.
    //
    // This does not stand in for the storage sync below. A fence orders; it
    // does not rendezvous. Publishing the init to the rest of the block is
    // still that sync's job, and no thread may reach an arrive or await before
    // it.
    new_body.push_back(
        Evaluate(Call(DataType::Handle(), tang_fence_barrier_init(), {})));
    // A cluster-scoped barrier must publish its init to the whole cluster, not
    // just this CTA: a peer CTA may arrive on it remotely
    // (ptx_arrive_cluster_barrier) and must observe the init before doing so.
    // storage_sync("cluster") codegens to tl::cluster_sync() (sync.all.bkcls).
    new_body.push_back(Evaluate(
        Call(DataType::Handle(), builtin::tvm_storage_sync(),
             {StringImm(has_cluster_barrier ? "cluster" : "shared")})));
    new_body.push_back(block->body);
    block.CopyOnWrite()->body = SeqStmt(new_body);

    return StmtExprMutator::VisitStmt_(block.get());
  }

  // cta_id == 0 is this CTA's own shared mbarrier and keeps the cheap
  // block-scoped arrive. Anything else is a remote arrive: codegen maps the
  // barrier into the peer CTA's address space (__set_block_rank_mbarrier) and
  // issues a DSHM mbarrier arrive. A runtime rank is a legitimate remote
  // arrive too -- the DSHM mapping takes the rank as a register operand.
  // Returns true when the arrive is local, i.e. the constant 0.
  static bool IsLocalCta(const PrimExpr &cta_id) {
    const DataType dtype = cta_id.dtype();
    if (!dtype.is_scalar() || !(dtype.is_int() || dtype.is_uint())) {
      LOG(FATAL) << "TANG mbarrier arrive cta_id must be a scalar integer. Got "
                 << cta_id << " of dtype " << dtype;
    }
    const auto *imm = cta_id.as<IntImmNode>();
    return imm != nullptr && imm->value == 0;
  }

  // The pred form has no remote counterpart: a DSHM mbarrier arrive is
  // unconditional, so a predicated remote arrive is rejected rather than
  // silently dropping the predicate.
  static void RequireLocalCta(const PrimExpr &cta_id) {
    if (!IsLocalCta(cta_id)) {
      LOG(FATAL) << "TANG mbarrier arrive with a predicate only accepts the "
                    "local cta_id == 0: a DSHM remote arrive is unconditional. "
                    "Got cta_id="
                 << cta_id;
    }
  }

  // Constant pred is only 0 (skip) or 1 (arrive). Any other constant used to
  // be rewritten as an unconditional arrive.
  static bool PredConstant01(const PrimExpr &pred, int64_t *value) {
    const DataType dtype = pred.dtype();
    if (!dtype.is_scalar() ||
        !(dtype.is_int() || dtype.is_uint() || dtype.is_bool())) {
      LOG(FATAL) << "TANG mbarrier arrive pred must be a scalar integer or "
                    "bool. Got "
                 << pred << " of dtype " << dtype;
    }
    const auto *imm = pred.as<IntImmNode>();
    if (imm == nullptr) {
      return false;
    }
    if (imm->value != 0 && imm->value != 1) {
      LOG(FATAL) << "TANG mbarrier arrive pred must be 0 or 1, got "
                 << imm->value;
    }
    *value = imm->value;
    return true;
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (call && call->op.same_as(ptx_arrive_cluster_barrier())) {
      if (call->args.size() != 2) {
        LOG(FATAL) << "TANG mbarrier arrive(mbar, cta_id) expects 2 arguments, "
                      "got "
                   << call->args.size();
      }
      if (IsLocalCta(call->args[1])) {
        return Evaluate(
            Call(call->dtype, builtin::ptx_arrive_barrier(), {call->args[0]}));
      }
      // Remote or runtime-ranked arrive: keep the cluster-scoped call so
      // codegen emits tl::cluster_arrive_barrier (DSHM remote mbarrier
      // arrive).
      return StmtExprMutator::VisitStmt_(op);
    }
    if (call && call->op.same_as(builtin::ptx_arrive_barrier()) &&
        call->args.size() == 3) {
      RequireLocalCta(call->args[1]);
      int64_t pred_value = 0;
      Stmt arrive = Evaluate(
          Call(call->dtype, builtin::ptx_arrive_barrier(), {call->args[0]}));
      if (PredConstant01(call->args[2], &pred_value)) {
        return pred_value == 0 ? Evaluate(IntImm(DataType::Int(32), 0))
                               : arrive;
      }
      return IfThenElse(call->args[2], arrive);
    }
    if (call && call->op.same_as(builtin::ptx_arrive_barrier()) &&
        call->args.size() != 1 && call->args.size() != 3) {
      LOG(FATAL) << "TANG mbarrier arrive expects (mbarrier) or "
                    "(mbarrier, cta_id, pred), got "
                 << call->args.size() << " arguments.";
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        ICHECK(iv->dom->extent.as<IntImmNode>());
        thread_var_ = iv;
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  IterVar thread_var_;
  // Warp-specialized kernels set tl.disable_shuffle_elect and elect by thread
  // id instead of through the shuffle.
  bool disable_shuffle_elect_;
};

tvm::transform::Pass LowerSharedBarrier() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc f, const IRModule &m, PassContext ctx) {
    bool disable_shuffle_elect =
        ctx->GetConfig<Bool>(kDisableShuffleElect, Bool(false)).value();
    f.CopyOnWrite()->body =
        TangSharedBarrierRewriter::Rewrite(f->body, disable_shuffle_elect);
    return f;
  };
  // Named apart from the CUDA pass so the two are distinguishable in a pass
  // trace, since they carry the same frontend name.
  return CreatePrimFuncPass(pass_func, 0, "tl.tang.LowerSharedBarrier", {});
}

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.tang.transform.LowerSharedBarrier",
                        LowerSharedBarrier);
}

} // namespace tl
} // namespace tvm
