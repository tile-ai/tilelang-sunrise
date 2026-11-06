/*!
 * \file src/tang/transform/s3_mbarrier_utils.h
 * \brief The code the TANG sync passes share, so that one rule has one home.
 *
 * Section 1: the mbarrier array's prologue.
 *
 * Two materializers own mbarrier slots -- the pipeline handshake
 * (``inject_s3_async_copy.cc``) and the late TMEM/partial-sync rewrite
 * (``materialize_s3_mbarriers.cc``) -- and both have to publish them in
 * exactly the same way: one elected thread initialises every slot, a fence
 * orders the init, and a block-wide shared sync makes it visible before any
 * warp may arrive or wait. The code was identical (29 lines, differing in one
 * diagnostic string and one redundant cast) until it moved here, so a protocol
 * fix now lands once.
 *
 * What the two call sites do *not* share is how they find the binding to insert
 * into: the handshake pass already knows its ``threadIdx.x`` var from its own
 * analysis and matches it by identity, while the partial-sync pass runs after
 * every other sync pass and has to match the first ``threadIdx.x`` binding by
 * tag. ``ThreadExtentPrologue`` therefore takes the optional var, and keeps
 * whichever check it is given (passing it = identity, omitting it = tag).
 */
#ifndef TVM_TL_TANG_TRANSFORM_S3_MBARRIER_UTILS_H_
#define TVM_TL_TANG_TRANSFORM_S3_MBARRIER_UTILS_H_

#include <algorithm>
#include <utility>
#include <vector>

#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include "cuda/op/builtin.h"
#include "tang/op/builtin.h"

namespace tvm {
namespace tl {
namespace tang {
namespace transform {

// The helpers below name a lot of TIR nodes, and the only includers are the
// TANG sync passes, which already pull both of these in (`Array`/`GetRef` come
// from ffi, the node classes from tirx). Confining the directives to this
// header's own namespace keeps `tirx::`/`ffi::` off every Stmt/Call/IntImm
// below without leaking them to a translation unit that did not
// ask for it.
using namespace tirx;
using namespace ffi;

/*!
 * \brief The first mbarrier slot that is still free in ``body``.
 *
 * The whole kernel has ONE mbarrier array: codegen sizes it by the largest
 * ``create_barriers`` request and every producer indexes into it. Two passes
 * that each number their slots from 0 therefore initialise the same slot twice
 * with different arrive counts -- the later init wins, and the handshake whose
 * count was overwritten waits for the wrong number of arrivals (it either spins
 * forever or releases early). Every producer that runs after another one
 * allocates above what is already in the IR instead.
 *
 * Both ``get_mbarrier(i)`` uses and the ``create_barriers(n)`` request count: a
 * prologue may size the array beyond the slots it happens to reference.
 *
 * Gated by
 * ``test_tang_tcgen5_gemm_warp_policy.py::test_the_region_scoped_barrier_owns_its_own_slot``,
 * on the one shape that reaches two producers at once (the pipeline handshake
 * plus a 2-warp region's partial rendezvous): it reads the lowering's
 * ``mbarrier[i].init(count)`` table and requires one init per slot, an array
 * that covers the highest slot, and every region-scoped slot above the
 * handshake's whole range.
 */
inline int64_t FreeMbarrierSlot(const Stmt &body) {
  struct Collector : public StmtExprVisitor {
    int64_t used{0}; // one past the highest slot index seen
    void VisitExpr_(const CallNode *op) final {
      if (op->op.same_as(get_mbarrier()) && op->args.size() == 1) {
        if (const auto *imm = op->args[0].as<IntImmNode>()) {
          used = std::max(used, imm->value + 1);
        }
      } else if (op->op.same_as(tirx::builtin::create_barriers()) &&
                 op->args.size() == 1) {
        if (const auto *imm = op->args[0].as<IntImmNode>()) {
          used = std::max(used, imm->value);
        }
      }
      StmtExprVisitor::VisitExpr_(op);
    }
  };
  Collector collector;
  collector(body);
  return collector.used;
}

// ---------------------------------------------------------------------------
// Roles: which warps execute what, and how many warps a region has.
//
// A barrier count is a *per-region* number: the warps that publish an edge and
// the warps that wait on it are the warps of the regions those statements sit
// in, and those two regions coincide only when nothing restricts the statement.
// The one shape where they do not is the warp-specialised one, where the sink
// statement sits inside a guard that names a thread index (what the frontend's
// `T.ws` leaves behind).
//
// Reading that off the IR is the "role" layer's job, so it lives here and is
// published as a loop annotation by producer_consumer_ws.cc; the handshake pass
// consumes the published number instead of re-deriving it. Both use these
// helpers, so the two can only disagree if one of them is wrong.
// ---------------------------------------------------------------------------

/*! \brief The block's threads, as the IR states them. */
struct BlockThreads {
  int64_t warps{-1}; // -1: no threadIdx.x extent found
  Var thread_x;      // the block's threadIdx.x var, for election
};

inline BlockThreads FindBlockThreads(const Stmt &body) {
  struct Finder : public StmtVisitor {
    void VisitStmt_(const AttrStmtNode *op) final {
      if (op->attr_key == tirx::attr::thread_extent) {
        const auto *iv = op->node.as<IterVarNode>();
        if (iv != nullptr && iv->thread_tag == "threadIdx.x") {
          out.thread_x = iv->var;
          if (const auto *imm = op->value.as<IntImmNode>()) {
            out.warps = imm->value / 32;
          }
        }
      }
      StmtVisitor::VisitStmt_(op);
    }
    BlockThreads out;
  } finder;
  finder(body);
  return finder.out;
}

/*!
 * \brief The condition "this thread initialises the array".
 *
 * ``tl_shuffle_elect<0>()`` is what ``LowerSharedBarrier`` emits and what the
 * generic pipeline emits for its own ``pipeline_mbar`` array, so every mbarrier
 * publication in the kernel names its elected thread the same way. On TANG the
 * elect is `lane 0 of warp 0` (``intrin.h``: a plain index comparison, not
 * CUDA's ``elect.sync``), i.e. exactly ``threadIdx.x == 0`` -- but it says what
 * the thread *is* (one representative) instead of pinning a lane index, and it
 * keeps the three producers' generated code identical.
 *
 * ``disable_shuffle_elect`` mirrors the ``tl.disable_shuffle_elect`` pass
 * config, the same knob ``LowerSharedBarrier`` reads: a caller that opts out
 * gets the thread-id test here too, so one kernel never mixes both forms.
 */
inline PrimExpr ElectBarrierInitThread(const Var &thread_x,
                                       bool disable_shuffle_elect) {
  if (disable_shuffle_elect) {
    return EQ(thread_x, IntImm(thread_x.dtype(), 0));
  }
  return Call(DataType::Bool(), tl_shuffle_elect(),
              {IntImm(DataType::Int(32), 0)});
}

/*!
 * \brief The sequence that publishes an mbarrier array: init, fence, sync.
 *
 * The same sequence ``LowerSharedBarrier`` emits for a ``T.alloc_barrier``
 * buffer. ``what`` names the array in the diagnostic a missing ``threadIdx.x``
 * var raises, so each caller keeps saying which of the two arrays it is.
 * ``base`` is the first slot this prologue owns -- see ``FreeMbarrierSlot``.
 */
inline Stmt BarrierPrologue(const std::vector<int64_t> &arrive_counts,
                            int64_t base, const Var &thread_x, const char *what,
                            bool disable_shuffle_elect = false) {
  ICHECK(thread_x.defined())
      << what << " needs a threadIdx.x var to elect the thread that "
      << "initialises it";
  Array<Stmt> init_calls;
  for (size_t i = 0; i < arrive_counts.size(); ++i) {
    init_calls.push_back(Evaluate(
        Call(DataType::Handle(), tirx::builtin::ptx_init_barrier_thread_count(),
             {Call(DataType::Handle(), get_mbarrier(),
                   {IntImm(DataType::Int(32), static_cast<int>(base + i))}),
              IntImm(DataType::Int(32), static_cast<int>(arrive_counts[i]))})));
  }
  Array<Stmt> seq;
  seq.push_back(
      Evaluate(Call(DataType::Handle(), tirx::builtin::create_barriers(),
                    {IntImm(DataType::Int(32),
                            static_cast<int>(base + arrive_counts.size()))})));
  seq.push_back(IfThenElse(
      ElectBarrierInitThread(thread_x, disable_shuffle_elect),
      init_calls.size() == 1 ? init_calls[0] : SeqStmt(init_calls), Stmt()));
  seq.push_back(
      Evaluate(Call(DataType::Handle(), tang_fence_barrier_init(), {})));
  seq.push_back(
      Evaluate(Call(DataType::Handle(), tirx::builtin::tvm_storage_sync(),
                    {StringImm("shared")})));
  return SeqStmt(seq);
}

//! Overload for the first producer of a kernel, which owns slot 0.
inline Stmt BarrierPrologue(const std::vector<int64_t> &arrive_counts,
                            const Var &thread_x, const char *what,
                            bool disable_shuffle_elect = false) {
  return BarrierPrologue(arrive_counts, /*base=*/0, thread_x, what,
                         disable_shuffle_elect);
}

/*!
 * \brief Insert ``BarrierPrologue`` as the first statement inside the
 *        ``thread_extent`` binding of ``threadIdx.x``.
 *
 * The elected-thread test names that var, and using it outside its binding is
 * rejected later by the packed-API check ("kernel variables (tx,) are used, but
 * are not passed in as API arguments"), which is why the init cannot simply be
 * prepended to the function body. ``LowerSharedBarrier`` lands inside the
 * extent for the same reason.
 *
 * Match by tag when the caller has not looked at the kernel's thread binding,
 * and by identity when it has. The prologue is always built with the var the
 * binding introduces, so the array and the elected-thread test agree by
 * construction; a caller whose kernel has no such binding gets
 * ``inserted() == false`` and decides for itself.
 */
class ThreadExtentPrologue : public StmtMutator {
public:
  /*!
   * \brief Match the first ``threadIdx.x`` binding, by tag.
   *
   * ``base`` is the first slot the array owns (see ``FreeMbarrierSlot``); the
   * first producer in a kernel passes nothing and keeps slot 0.
   */
  ThreadExtentPrologue(std::vector<int64_t> arrive_counts, const char *what,
                       int64_t base = 0, bool disable_shuffle_elect = false)
      : arrive_counts_(std::move(arrive_counts)), what_(what), base_(base),
        by_tag_(true), disable_shuffle_elect_(disable_shuffle_elect) {}

  /*!
   * \brief Match the binding that introduces ``thread_x``, by identity.
   *
   * Two constructors rather than an optional var, because a
   * default-constructed ``tirx::Var`` is *defined*: its first parameter has a
   * default (``Var(String name_hint = "v", ...)``), so a ``Var()`` sentinel
   * would silently match nothing.
   */
  ThreadExtentPrologue(std::vector<int64_t> arrive_counts, const char *what,
                       const Var &thread_x, int64_t base = 0,
                       bool disable_shuffle_elect = false)
      : arrive_counts_(std::move(arrive_counts)), what_(what), base_(base),
        by_tag_(false), thread_x_(thread_x),
        disable_shuffle_elect_(disable_shuffle_elect) {}

  Stmt Insert(const Stmt &body) { return VisitStmt(body); }
  bool inserted() const { return inserted_; }

private:
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (!inserted_ && op->attr_key == tirx::attr::thread_extent) {
      const auto *iv = op->node.as<IterVarNode>();
      const bool hit =
          iv != nullptr && (by_tag_ ? iv->thread_tag == "threadIdx.x"
                                    : iv->var.same_as(thread_x_));
      if (hit) {
        inserted_ = true;
        AttrStmt node = GetRef<AttrStmt>(op);
        node.CopyOnWrite()->body =
            SeqStmt({BarrierPrologue(arrive_counts_, base_, iv->var, what_,
                                     disable_shuffle_elect_),
                     VisitStmt(op->body)});
        return Stmt(node);
      }
    }
    return StmtMutator::VisitStmt_(op);
  }

  std::vector<int64_t> arrive_counts_;
  const char *what_;
  int64_t base_{0};
  bool by_tag_{false};
  Var thread_x_;
  bool inserted_{false};
  /*! \brief Election form: see ``ElectBarrierInitThread``. */
  bool disable_shuffle_elect_{false};
};

} // namespace transform
} // namespace tang
} // namespace tl
} // namespace tvm

#endif // TVM_TL_TANG_TRANSFORM_S3_MBARRIER_UTILS_H_
