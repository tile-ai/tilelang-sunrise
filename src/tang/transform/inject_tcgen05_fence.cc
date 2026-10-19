/*!
 * \file inject_tcgen05_fence.cc
 * \brief Inject ``fence_tc<fg>()`` / ``fence_tmem<fg>()`` at conservative
 *        TCGEN05/TMEM synchronization boundaries on stcuv2 (S3) targets.
 *
 * Counterpart of ``tl.cuda.transform.InjectTcgen05Fence``
 * ------------------------------------------------------
 * The tcgen05 accumulator lives in TMEM, an address space whose writes are not
 * published by ordinary thread synchronization (``__syncthreads()``, mbarrier).
 * CUDA bridges that gap with ``tcgen05.fence::before_thread_sync`` /
 * ``::after_thread_sync``; TANG stcuv2 splits the ordering into independent
 * *fence domains*, each with its own group range:
 *
 *   domain        ops retired                        instruction groups
 *   fence.tc      TC instructions (TMUL/TMAD/TMACC)  fence_tc<fg>()         0~1
 *   fence.tmem    TMEM register traffic (LDT/STT)    fence_tmem<fg>()       0~3
 *   fence.mem     async copies / DMA (g2s, s2g, ...) fence_mem(fg)          0~7
 *   fence.alu     ALU pipelines                      fence_alu<fg>()        0~1
 *   fence.all     every domain                       fence_all()            --
 *
 * A fence only retires its *own* domain: ``fence.tc`` does not order LDT/STT,
 * and ``fence.tmem`` does not order TMUL/TMAD/TMACC. This pass therefore
 * carries the domain of the neighbouring region through and emits the matching
 * instruction, with the TANG default groups (``tang::ptx::fg_*_default``):
 * TC work -> ``fence_tc<fg_tc_default>`` (0), TMEM reads (LDT) ->
 * ``fence_tmem<fg_ldt_default>`` (3), TMEM writes (STT) ->
 * ``fence_tmem<fg_stt_default>`` (2).
 *
 * TANG kernels can spell those fences by hand through
 * ``T.tcgen05_before/after_thread_sync(fence_group)``. This pass inserts them
 * automatically at the same handoff points the CUDA pass handles, so a kernel
 * that merely *uses* TMEM (``T.gemm`` with a tmem accumulator,
 * ``T.tcgen05_ld``, ``T.tcgen05_cp``, ...) cannot silently miss the ordering:
 *
 *   1. a ``tvm_storage_sync("shared")`` / ``("shared.dyn")`` gets a fence on
 *      the side(s) where a TCGEN05/TMEM use is adjacent in the linear region;
 *   2. insert a fence after ``mbarrier_wait_parity`` / ``tang_sync_wait`` when
 * a forward scan reaches a TCGEN05/TMEM use before the next boundary;
 *   3. insert a fence before ``ptx_arrive_barrier`` / ``tang_sync_arrive`` when
 *      a backward scan reaches a TCGEN05/TMEM use after the previous boundary.
 *
 * Rule 1 is deliberately *not* "wrap every shared sync" like the CUDA pass
 * does: a barrier that only orders shared-memory traffic (e.g. between two
 * staging copies) has nothing to do with TMEM, and fencing it buys no ordering.
 * The linear scans still look through nested statements, so a barrier whose
 * TMEM neighbour sits inside a loop/if/sblock is fenced. What they cannot see
 * is a TMEM use hidden behind an opaque device call, so this trade-off is: a
 * fence may be *omitted* only when the neighbouring region is opaque.
 *
 * Like the CUDA pass it deliberately does NOT fence
 * ``tang_tcgen05_mma_arrive``: the underlying ``tcgen05.commit.*.mbarrier``
 * already carries the producer-side ordering.
 *
 * Every rule accounts for the fences already present on the side it rewrites,
 * per domain: a fence the kernel placed by hand (``tang_fence_tc`` /
 * ``tang_tmem_fence``) is never duplicated, but a hand-written ``fence_tc``
 * does not suppress the ``fence_tmem`` that a TMEM neighbour still needs -- the
 * two domains are retired independently. ``tang_fence_tc_arrive`` /
 * ``tang_fence_g2s_arrive`` are treated as arrives that already contain their
 * fence.
 *
 * Out of scope (no fence injected, see docs/tang_tcgen05_fence_injection.md):
 * the async copy engines. The global<->shared bulk helpers retire their own
 * transfer with ``fence_mem(fg_g2s_default / fg_s2g_default)``, and
 * ``copy_tmem_shm.h`` documents that smem<->tmem completion is drained with
 * ``fence_mem`` on ``fg_t2s_default`` / ``fg_s2t_default`` -- the mem domain,
 * which is neither ``fence_tc`` nor ``fence_tmem``.
 *
 * Gates
 * -----
 * * target arch is stcuv2 -- TMEM exists on no other TANG arch;
 * * the function body actually contains a TCGEN05/TMEM operation.
 *
 * Group choice stays inside the defaults above. Other groups remain reachable
 * through the explicit ``T.tcgen05_*_thread_sync(fence_group)`` primitives
 * (``fence_tc`` group 1 splits TC work into two independently ordered
 * pipelines).
 */

#include "op/builtin.h"
#include "support/check.h"
#include "tang/op/builtin.h"
#include "tang/target_utils.h"

#include <tvm/ir/transform.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <utility>

namespace tvm {
namespace tl {

namespace {

using namespace tirx;
using namespace ffi;
using tvm::transform::PassContext;

/*!
 * \brief Pass config that lets the TANG pipeline run this pass.
 *
 * Kept next to the pass instead of in a shared header because no other
 * component reads it, and it is registered below so TVM's config legalizer
 * accepts it (mirrors tl.enable_lower_ldgstg and friends).
 */
constexpr const char *kEnableTangInjectTcgen05Fence =
    "tl.enable_tang_inject_tcgen05_fence";

/*! \brief Fence groups come from the per-domain constants below. */

/*!
 * \brief Check if a call is tvm_storage_sync("shared") or
 *        tvm_storage_sync("shared.dyn").
 */
bool IsSharedStorageSync(const CallNode *call) {
  if (!call || !call->op.same_as(builtin::tvm_storage_sync())) {
    return false;
  }
  if (call->args.empty()) {
    return false;
  }
  const auto *scope = call->args[0].as<StringImmNode>();
  if (!scope) {
    return false;
  }
  return scope->value == "shared" || scope->value == "shared.dyn";
}

bool IsMbarrierWaitParity(const CallNode *call) {
  return call && call->op.same_as(mbarrier_wait_parity());
}

/*! \brief TANG named-barrier wait (consumer side of a TMEM handoff). */
bool IsTangSyncWait(const CallNode *call) {
  return call && call->op.same_as(tang_sync_wait());
}

/*! \brief TANG named-barrier arrive (producer side of a TMEM handoff). */
bool IsTangSyncArrive(const CallNode *call) {
  return call && call->op.same_as(tang_sync_arrive());
}

/*! \brief TANG has no cluster level, so ptx_arrive_cluster_barrier is absent.
 */
bool IsPlainBarrierArrive(const CallNode *call) {
  return call && call->op.same_as(builtin::ptx_arrive_barrier());
}

/*!
 * \brief Arrives that already carry their own TC fence, so rule 3 must not add
 *        another before them.
 */
bool IsFencedArrive(const CallNode *call) {
  return call && (call->op.same_as(tang_fence_tc_arrive()) ||
                  call->op.same_as(tang_fence_g2s_arrive()));
}

bool IsFenceTcCall(const CallNode *call) {
  return call && call->op.same_as(tang_fence_tc());
}

/*! \brief ``tl.tang_tmem_fence(fg)`` -- the TMEM-domain fence. */
bool IsFenceTmemCall(const CallNode *call) {
  return call && call->op.same_as(tang_tmem_fence());
}

const CallNode *GetEvaluateCall(const Stmt &stmt) {
  if (const auto *eval = stmt.as<EvaluateNode>()) {
    return eval->value.as<CallNode>();
  }
  return nullptr;
}

/*! \brief Whether ``buf`` lives in the TANG tensor-memory address space. */
bool IsTmemBuffer(const Buffer &buf) {
  return buf.defined() && buf.scope() == "shared.tmem";
}

/*!
 * \brief Fence domains of an stcuv2 region, as bit flags.
 *
 * ``fence.tc`` retires only TMUL/TMAD/TMACC and ``fence.tmem`` only LDT/STT,
 * so a boundary that hands over TC work needs a different instruction than one
 * that hands over TMEM register traffic. The two TMEM directions map to
 * different groups (fg_ldt_default / fg_stt_default), hence the split.
 */
enum FenceDomain : int {
  kDomainNone = 0,
  kDomainTc = 1 << 0, // TC instructions      -> fence_tc<fg_tc_default>
  kDomainTmemRead = 1
      << 1, // LDT (TMEM -> reg)    -> fence_tmem<fg_ldt_default>
  kDomainTmemWrite = 1
      << 2, // STT (reg -> TMEM)    -> fence_tmem<fg_stt_default>
};

/*! \brief Every domain, used to short-circuit the recursive scans. */
constexpr int kAllDomains = kDomainTc | kDomainTmemRead | kDomainTmemWrite;

/*! \brief Fence groups taken from the TANG header defaults (tang::ptx::fg_*).
 */
constexpr int kTcFenceGroup = 0;  // fg_tc_default
constexpr int kLdtFenceGroup = 3; // fg_ldt_default
constexpr int kSttFenceGroup = 2; // fg_stt_default

/*!
 * \brief Domain(s) that a single call retires, or kDomainNone.
 *
 * Deliberately not classified:
 *   * ``tang_tcgen05_mma_arrive`` -- the commit already carries the producer
 *     side ordering (same rationale as the CUDA pass);
 *   * the global<->shared bulk copies and the smem<->tmem copy engines -- their
 *     helpers drain the transfer with ``fence_mem`` on their own group inside
 *     the template (mem domain, not managed here);
 *   * ``tang_init/deallocate_tensor_memory`` -- allocation is not an
 *     asynchronous domain operation;
 *   * the fences themselves, so an existing fence never counts as a use.
 */
int DomainOfCall(const CallNode *call) {
  if (!call || IsFenceTcCall(call) || IsFenceTmemCall(call) ||
      IsFencedArrive(call)) {
    return kDomainNone;
  }
  if (call->op.same_as(tl_tang_gemm()) ||
      call->op.same_as(tang_tcgen05_mma_ss()) ||
      call->op.same_as(tang_tcgen05_mma_ts()) ||
      call->op.same_as(tang_tcgen05_mma_scale())) {
    return kDomainTc;
  }
  if (call->op.same_as(tang_tmem_ld()) ||
      call->op.same_as(tang_tmem_ld_16x256b()) ||
      call->op.same_as(tang_tmem_ld_16x256b_x16()) ||
      call->op.same_as(tang_tmem_drain_16x256b_to_global())) {
    return kDomainTmemRead;
  }
  if (call->op.same_as(tang_tmem_st()) ||
      call->op.same_as(tang_tmem_st_16x256b()) ||
      call->op.same_as(tang_tmem_st_a_operand())) {
    return kDomainTmemWrite;
  }
  return kDomainNone;
}

/*!
 * \brief Calls whose TMEM buffer argument is an *address*, not TMEM traffic.
 *
 * ``tc_alloc<128>(C_t[0])`` / ``tc_dealloc<128>(C_t[0])`` reference the tensor
 * memory only to name it, so the buffer access there must not be mistaken for
 * an LDT/STT of the accumulator.
 */
bool IsTmemAddressOnlyCall(const CallNode *call) {
  return call && (call->op.same_as(tang_init_tensor_memory()) ||
                  call->op.same_as(tang_deallocate_tensor_memory()));
}

/*! \brief Statements that end a linear scan in both directions. */
bool IsFenceSyncBoundary(const CallNode *call) {
  return IsSharedStorageSync(call) || IsMbarrierWaitParity(call) ||
         IsTangSyncWait(call) || IsPlainBarrierArrive(call) ||
         IsTangSyncArrive(call) || IsFencedArrive(call) ||
         (call && call->op.same_as(tang_tcgen05_mma_arrive()));
}

/*!
 * \brief The first statement actually executed by ``stmt``.
 *
 * A barrier may sit at the *head* of a nested block (``sblock { sync; mma; }``)
 * or at its *tail* (``sblock { mma; sync; }``). The linear scans see such a
 * block as a single unit, so the head/tail leaves say which side of it is a
 * boundary and which side therefore reaches into the enclosing sequence.
 * The walk descends the transparent wrappers the TANG pipeline produces
 * (``SBlockRealize`` -> ``SBlock`` -> body, ``AttrStmt`` -> body).
 */
Stmt HeadLeaf(const Stmt &stmt) {
  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    return seq->seq.empty() ? stmt : HeadLeaf(seq->seq[0]);
  }
  if (const auto *realize = stmt.as<SBlockRealizeNode>()) {
    return HeadLeaf(realize->block->body);
  }
  if (const auto *block = stmt.as<SBlockNode>()) {
    return HeadLeaf(block->body);
  }
  if (const auto *attr = stmt.as<AttrStmtNode>()) {
    return HeadLeaf(attr->body);
  }
  if (const auto *loop = stmt.as<ForNode>()) {
    return HeadLeaf(loop->body);
  }
  return stmt;
}

/*! \brief The last statement actually executed by ``stmt``. */
Stmt TailLeaf(const Stmt &stmt) {
  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    return seq->seq.empty() ? stmt : TailLeaf(seq->seq[seq->seq.size() - 1]);
  }
  if (const auto *realize = stmt.as<SBlockRealizeNode>()) {
    return TailLeaf(realize->block->body);
  }
  if (const auto *block = stmt.as<SBlockNode>()) {
    return TailLeaf(block->body);
  }
  if (const auto *attr = stmt.as<AttrStmtNode>()) {
    return TailLeaf(attr->body);
  }
  if (const auto *loop = stmt.as<ForNode>()) {
    return TailLeaf(loop->body);
  }
  return stmt;
}

/*!
 * \brief Domains a statement touches, looking through nested statements.
 *
 * A TMEM read/write can still be a plain buffer access at this point: the drain
 * (``C[i] = C_t[i]``) only becomes ``tang_tmem_ld`` later, in
 * LowerTangTmemDrain. Without this the barrier in front of a drain would look
 * TMEM-free and lose its fence.
 *
 * With ``stop_at_boundary`` the walk visits statements in program order and
 * stops at the first sync boundary, which is what the region scans need: a use
 * sitting past an inner barrier belongs to a different region and must not pull
 * a fence to the outer one.
 */
class DomainCollector : public StmtExprVisitor {
public:
  explicit DomainCollector(bool stop_at_boundary)
      : stop_at_boundary_(stop_at_boundary) {}

  int domains{kDomainNone};
  /*! \brief Whether the walk stopped at a nested sync boundary. */
  bool hit_boundary{false};

  void VisitStmt_(const SeqStmtNode *op) final {
    for (const Stmt &stmt : op->seq) {
      if (stopped()) {
        return;
      }
      VisitStmt(stmt);
    }
  }

  void VisitExpr_(const CallNode *op) final {
    if (stopped()) {
      return;
    }
    if (stop_at_boundary_ && IsFenceSyncBoundary(op)) {
      hit_boundary = true;
      return;
    }
    if (IsTmemAddressOnlyCall(op)) {
      return;
    }
    domains |= DomainOfCall(op);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    if (stopped()) {
      return;
    }
    if (IsTmemBuffer(op->buffer)) {
      domains |= kDomainTmemRead;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    if (stopped()) {
      return;
    }
    if (IsTmemBuffer(op->buffer)) {
      domains |= kDomainTmemWrite;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

private:
  bool stopped() const { return stop_at_boundary_ && hit_boundary; }

  bool stop_at_boundary_{true};
};

/*! \brief Every domain the statement touches, boundaries included. */
int StmtDomains(const Stmt &stmt) {
  DomainCollector collector(/*stop_at_boundary=*/false);
  collector(stmt);
  return collector.domains;
}

/*! \brief The domains of one region, up to the first nested sync boundary. */
DomainCollector ScanRegion(const Stmt &stmt) {
  DomainCollector collector(/*stop_at_boundary=*/true);
  collector(stmt);
  return collector;
}

/*!
 * \brief Domains an already-present fence retires, or kDomainNone.
 *
 * ``fence_tmem`` retires the whole tmem domain, so it covers both directions.
 * ``tang_fence_g2s_arrive`` belongs to the mem domain, which this pass does not
 * manage, so it contributes no domain.
 */
int FenceDomainsOf(const CallNode *call) {
  if (IsFenceTcCall(call)) {
    return kDomainTc;
  }
  if (IsFenceTmemCall(call)) {
    return kDomainTmemRead | kDomainTmemWrite;
  }
  if (call && call->op.same_as(tang_fence_tc_arrive())) {
    return kDomainTc;
  }
  return kDomainNone;
}

/*!
 * \brief Domains that the region after ``start_index`` still needs a fence for.
 *
 * The scan runs forward until the next sync boundary. A fence found on the way
 * retires the domains it owns for everything *further* from the boundary, so it
 * is accounted for per domain: a hand-written ``fence_tc`` cannot retire the
 * LDT/STT traffic of a TMEM handoff, and vice versa.
 */
int DomainsUpcoming(const Array<Stmt> &seq, int start_index) {
  int covered = kDomainNone;
  int need = kDomainNone;
  for (int i = start_index + 1; i < static_cast<int>(seq.size()); ++i) {
    const CallNode *call = GetEvaluateCall(seq[i]);
    covered |= FenceDomainsOf(call);
    DomainCollector region = ScanRegion(seq[i]);
    need |= region.domains & ~covered;
    if (region.hit_boundary || IsFenceSyncBoundary(call)) {
      break;
    }
  }
  return need;
}

/*! \brief Domains that the region before ``start_index`` still needs fenced. */
int DomainsPrior(const Array<Stmt> &seq, int start_index) {
  int covered = kDomainNone;
  int need = kDomainNone;
  for (int i = start_index - 1; i >= 0; --i) {
    const CallNode *call = GetEvaluateCall(seq[i]);
    covered |= FenceDomainsOf(call);
    DomainCollector region = ScanRegion(seq[i]);
    need |= region.domains & ~covered;
    if (region.hit_boundary || IsFenceSyncBoundary(call)) {
      break;
    }
  }
  return need;
}

/*!
 * \brief Check whether the function body contains any TCGEN05 / TMEM operation
 *        that warrants fence insertion.
 */
bool HasTcgen05Operations(const Stmt &body) {
  return StmtDomains(body) != kDomainNone;
}

/*! \brief ``fence_tc<group>()`` / ``fence_tmem<group>()`` as an evaluated call.
 */
inline Stmt MakeFenceStmt(const Op &op, int group) {
  return Evaluate(
      Call(DataType::Handle(), op, {IntImm(DataType::Int(32), group)}));
}

/*!
 * \brief Append the fence(s) that the requested domains need, in a fixed order.
 *
 * One fence per domain: the domains are retired independently, so a region that
 * mixes TC work with TMEM register traffic needs both instructions.
 */
void AppendFenceStmts(Array<Stmt> *out, int domains) {
  if (domains & kDomainTc) {
    out->push_back(MakeFenceStmt(tang_fence_tc(), kTcFenceGroup));
  }
  if (domains & kDomainTmemRead) {
    out->push_back(MakeFenceStmt(tang_tmem_fence(), kLdtFenceGroup));
  }
  if (domains & kDomainTmemWrite) {
    out->push_back(MakeFenceStmt(tang_tmem_fence(), kSttFenceGroup));
  }
}

inline void AppendFlattened(Array<Stmt> *out, const Stmt &stmt) {
  if (!stmt.defined()) {
    return;
  }
  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    for (const Stmt &inner : seq->seq) {
      out->push_back(inner);
    }
    return;
  }
  out->push_back(stmt);
}

/*!
 * \brief Rewriter for conservative TCGEN05/TMEM handoff points (stcuv2).
 *
 * ``fence`` below is the fence of the domain that the neighbouring region hands
 * over: ``fence_tc<fg_tc_default>`` for TC work, ``fence_tmem<fg_ldt_default>``
 * for TMEM reads (LDT) and ``fence_tmem<fg_stt_default>`` for TMEM writes
 * (STT).
 *
 * Supported rewrites:
 *
 *   fence;                         // domain of the preceding region
 *   __syncthreads();               // tvm_storage_sync("shared")
 *   fence;                         // domain of the following region
 *
 *   mbarrier_wait_parity(...);     // or tang_sync_wait(...)
 *   fence;                         // domain of the following region, when the
 *                                  // forward scan reaches a use before the
 *                                  // next boundary
 *
 *   fence;                         // domain of the preceding region, when the
 *   ptx_arrive_barrier(...);       // backward scan reaches a use after the
 *                                  // previous boundary (or tang_sync_arrive)
 */
class TangTcgen05FenceRewriter : public StmtExprMutator {
public:
  Stmt VisitStmt_(const SeqStmtNode *op) final {
    bool saved_in_seq = in_seq_rewrite_;
    in_seq_rewrite_ = true;

    Array<Stmt> mutated_children;
    for (const Stmt &stmt : op->seq) {
      mutated_children.push_back(VisitStmt(stmt));
    }

    in_seq_rewrite_ = saved_in_seq;

    Array<Stmt> flat_seq;
    for (const Stmt &stmt : mutated_children) {
      AppendFlattened(&flat_seq, stmt);
    }

    Array<Stmt> rewritten;
    for (int i = 0; i < static_cast<int>(flat_seq.size()); ++i) {
      const Stmt &stmt = flat_seq[i];
      const CallNode *call = GetEvaluateCall(stmt);

      if (IsSharedStorageSync(call)) {
        // Fence only the side(s) that actually carry a TCGEN05/TMEM use: a
        // barrier between two staging copies needs no fence. Each side gets the
        // fence of the domain(s) it hands over that no existing fence on that
        // side already retires (see DomainsPrior/DomainsUpcoming).
        const int need_before = DomainsPrior(flat_seq, i);
        const int need_after = DomainsUpcoming(flat_seq, i);
        AppendFenceStmts(&rewritten, need_before);
        rewritten.push_back(stmt);
        AppendFenceStmts(&rewritten, need_after);
        continue;
      }

      if (IsMbarrierWaitParity(call) || IsTangSyncWait(call)) {
        rewritten.push_back(stmt);
        AppendFenceStmts(&rewritten, DomainsUpcoming(flat_seq, i));
        continue;
      }

      if (IsPlainBarrierArrive(call) || IsTangSyncArrive(call)) {
        AppendFenceStmts(&rewritten, DomainsPrior(flat_seq, i));
        rewritten.push_back(stmt);
        continue;
      }

      // The statement itself is not a boundary, but a nested block may start or
      // end with one. The region on the far side of that boundary lies in
      // *this* sequence, which the nested rewrite cannot see, so fence it from
      // here.
      const bool head_is_boundary =
          IsFenceSyncBoundary(GetEvaluateCall(HeadLeaf(stmt)));
      const bool tail_is_boundary =
          IsFenceSyncBoundary(GetEvaluateCall(TailLeaf(stmt)));
      if (head_is_boundary || tail_is_boundary) {
        const int need_before =
            head_is_boundary ? DomainsPrior(flat_seq, i) : kDomainNone;
        const int need_after =
            tail_is_boundary ? DomainsUpcoming(flat_seq, i) : kDomainNone;
        if (need_before != kDomainNone || need_after != kDomainNone) {
          AppendFenceStmts(&rewritten, need_before);
          rewritten.push_back(stmt);
          AppendFenceStmts(&rewritten, need_after);
          continue;
        }
      }

      rewritten.push_back(stmt);
    }

    if (rewritten.size() == 1) {
      return rewritten[0];
    }
    return SeqStmt(std::move(rewritten));
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    if (in_seq_rewrite_) {
      return stmt;
    }
    const CallNode *call = GetEvaluateCall(stmt);
    if (IsSharedStorageSync(call)) {
      // A sync with no visible neighbour (e.g. the sole statement of an
      // if-body) leaves the domain unknown, so keep the conservative
      // before+after wrap of every domain -- the same over-approximation the
      // CUDA pass makes by wrapping unconditionally.
      Array<Stmt> wrapped;
      AppendFenceStmts(&wrapped, kAllDomains);
      wrapped.push_back(std::move(stmt));
      AppendFenceStmts(&wrapped, kAllDomains);
      return SeqStmt(std::move(wrapped));
    }
    return stmt;
  }

private:
  bool in_seq_rewrite_{false};
};

tvm::transform::Pass InjectTcgen05Fence() {
  auto pass_func = [](PrimFunc f, const IRModule &, const PassContext &) {
    // TMEM only exists on stcuv2; stcu and every non-TANG target are untouched.
    Optional<Target> opt_target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!opt_target.defined() || !TargetTangIsSTCUV2(opt_target.value())) {
      return f;
    }
    // Only apply if the function actually uses TCGEN05 / TMEM operations.
    if (!HasTcgen05Operations(f->body)) {
      return f;
    }
    TangTcgen05FenceRewriter rewriter;
    f.CopyOnWrite()->body = rewriter(f->body);
    return f;
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                             "tl.tang.InjectTcgen05Fence", {});
}

} // namespace

TVM_REGISTER_PASS_CONFIG_OPTION(kEnableTangInjectTcgen05Fence, Bool);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.tang.transform.InjectTcgen05Fence",
                        InjectTcgen05Fence);
}

} // namespace tl
} // namespace tvm
