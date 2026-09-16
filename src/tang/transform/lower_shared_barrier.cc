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
 * not contain that pass at all; and the CUDA rewrite branches on
 * shared.cluster_barrier, which has no TANG counterpart because TANG has no
 * cluster level. The two rewrites share the `barrier_init` contract, so a
 * change to how T.alloc_barrier records arrive counts has to touch both.
 */

#include "cuda/op/builtin.h"
#include "support/check.h"

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
    for (const Buffer &buffer : op->alloc_buffers) {
      const auto *ptr_type =
          buffer->data->type_annotation.as<PointerTypeNode>();
      if (!ptr_type)
        continue;
      const std::string scope = ptr_type->storage_scope;
      // A cluster barrier arriving here would be dropped without a word: it is
      // not a shared.barrier, so nothing below would emit its init. The TANG
      // pipeline rejects the scope before this pass runs, so reaching this
      // point means that ordering broke.
      ICHECK(scope != "shared.cluster_barrier")
          << "TANG has no cluster level, so shared.cluster_barrier buffer '"
          << buffer->name
          << "' must be rejected by the pipeline before barrier lowering";
      if (scope == "shared.barrier") {
        barrier_buffers.push_back(buffer);
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
    // On TANG the fence is a no-op (mbarrier_init already ends in fence.mem);
    // what actually publishes the init to the other threads is the storage
    // sync, and no thread may reach an arrive or a wait before it.
    new_body.push_back(
        Evaluate(Call(DataType::Handle(), ptx_fence_barrier_init(), {})));
    new_body.push_back(
        Evaluate(Call(DataType::Handle(), builtin::tvm_storage_sync(),
                      {StringImm("shared")})));
    new_body.push_back(block->body);
    block.CopyOnWrite()->body = SeqStmt(new_body);

    return StmtExprMutator::VisitStmt_(block.get());
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
