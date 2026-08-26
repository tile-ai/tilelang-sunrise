/*!
 * \file src/transform/remove_redundant_syncs.cc
 * \brief Remove redundant thread synchronization barriers.
 *
 * Adjacent duplicate syncs are always collapsed to a single one. Beyond that,
 * barriers are removed by a single hazard-based rule rather than by matching
 * syntactic shapes: a set of barriers is legal iff every pair of conflicting
 * shared-memory accesses (same allocation, at least one write) has a surviving
 * barrier between them. Removal iterates to a fixpoint, re-verifying that
 * property against the barriers that actually remain, so no removal can rely on
 * a barrier that another removal deletes. See the comment on the fixpoint in
 * VisitStmt_(SeqStmtNode) for why the earlier per-pattern structure was
 * unsound.
 *
 * Every decision asks the same question — which shared allocations does a
 * statement read, and which does it write — and gets its answer from the
 * single classifier CollectSharedAccess. See the comment on that function for
 * why one shared classifier is a correctness requirement here.
 *
 * Runs after ThreadSync passes to clean up syncs between non-conflicting
 * async copy operations that write to non-overlapping shared memory regions.
 */
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../op/builtin.h"
#include "../op/utils.h"

namespace tvm {
namespace tl {
namespace transform {

using namespace tirx;
using namespace ffi;

namespace {

Stmt RemoveAdjacentSyncs(const Stmt &stmt) {

  struct RedundantSyncRemover : public StmtExprMutator {
    static bool IsSync(const Stmt &s) {
      if (const auto *eval = s.as<EvaluateNode>()) {
        if (const auto *call = eval->value.as<CallNode>()) {
          if (call->op.same_as(builtin::tvm_storage_sync()) ||
              call->op.same_as(tl::pts_syncthreads())) {
            return true;
          }
        }
      }
      return false;
    }

    // True when `s` is a barrier, possibly behind wrapper nodes.
    //
    // Also true when the wrapper contains a SeqStmt whose FIRST statement is a
    // barrier: lowering emits the GEMM as
    // `AttrStmt(lexical_alloc_scope){ SeqStmt[sync, tl_tang_gemm] }`, and
    // stopping at the SeqStmt hid that barrier entirely. Because it
    // was missing from the barrier list, the hazard scan treated the whole
    // element as an ordinary statement and no barrier inside a k-loop body was
    // ever a removal candidate.
    //
    // Only the FIRST statement counts. A barrier deeper inside the sequence
    // has preceding statements that this element's position in the outer
    // sequence does not account for, so treating the element as "a barrier"
    // would make the scan reason about an order that isn't there.
    static bool IsSyncDeep(const Stmt &s) {
      Stmt cur = s;
      while (true) {
        if (IsSync(cur))
          return true;
        if (const auto *seq = cur.as<SeqStmtNode>()) {
          if (seq->seq.empty())
            break;
          cur = seq->seq[0];
        } else if (const auto *br = cur.as<SBlockRealizeNode>()) {
          cur = br->block->body;
        } else if (const auto *blk = cur.as<SBlockNode>()) {
          cur = blk->body;
        } else if (const auto *attr = cur.as<AttrStmtNode>()) {
          cur = attr->body;
        } else {
          break;
        }
      }
      return false;
    }

    // Remove the barrier that IsSyncDeep found, keeping the wrapper nodes that
    // surround it. Returns nullopt when the statement is nothing but the
    // barrier (possibly wrapped), i.e. when the caller may drop it outright.
    //
    // This mirrors IsSyncDeep's traversal on purpose: every shape IsSyncDeep
    // reports as "this is a sync" must be strippable here, or a barrier the
    // fixpoint decided to remove would silently survive.
    static Optional<Stmt> StripSync(const Stmt &s) {
      if (IsSync(s))
        return std::nullopt;
      // For a wrapper, an empty body means the barrier was the wrapper's only
      // content. The WRAPPER still has to stay: its attr_key / iter_values /
      // predicate are part of the program, and downstream passes (e.g. the
      // lexical_alloc_scope consumer) look for it. Substitute Evaluate(0) for
      // the removed barrier rather than propagating nullopt, which would
      // delete the wrapper along with the barrier — the very bug this exists
      // to prevent.
      if (const auto *attr = s.as<AttrStmtNode>()) {
        Stmt body = StripSync(attr->body).value_or(Evaluate(0));
        return AttrStmt(attr->node, attr->attr_key, attr->value, body);
      }
      // Mirror IsSyncDeep's SeqStmt arm: strip the leading barrier and keep the
      // remaining statements. Dropping the whole sequence here would delete the
      // GEMM that follows the barrier in the lexical_alloc_scope body.
      if (const auto *seq = s.as<SeqStmtNode>()) {
        if (seq->seq.empty())
          return s;
        Optional<Stmt> head = StripSync(seq->seq[0]);
        Array<Stmt> kept;
        if (head.defined())
          kept.push_back(head.value());
        for (size_t i = 1; i < seq->seq.size(); ++i)
          kept.push_back(seq->seq[i]);
        if (kept.empty())
          return std::nullopt;
        if (kept.size() == 1)
          return kept[0];
        return SeqStmt(kept);
      }
      if (const auto *br = s.as<SBlockRealizeNode>()) {
        Optional<Stmt> block = StripSync(br->block);
        if (!block.defined())
          return std::nullopt;
        return SBlockRealize(br->iter_values, br->predicate,
                             Downcast<SBlock>(block.value()));
      }
      if (const auto *blk = s.as<SBlockNode>()) {
        Stmt body = StripSync(blk->body).value_or(Evaluate(0));
        SBlock updated = GetRef<SBlock>(blk);
        updated.CopyOnWrite()->body = body;
        return updated;
      }
      // Not a barrier-only statement: nothing to strip.
      return s;
    }

    // ---- Shared-memory access classification ------------------------------
    //
    // Deciding whether a barrier can be dropped comes down to one question:
    // which shared allocations does a statement read, and which does it write?
    // CollectSharedAccess is the single answer to that question, and
    // HasSharedRead/HasSharedStore are thin wrappers over it.
    //
    // Keeping ONE classifier is a correctness requirement, not tidiness. An
    // earlier version had a separate hand-written recursive walker per removal
    // pattern, and the walkers disagreed: statements one walker recognized as
    // shared traffic were invisible to another, and the fallbacks for
    // unmodelled statement kinds pointed in opposite directions (one assumed
    // "no access", the other "unknown access"). A statement visible to the
    // guard but invisible to the check it guards is exactly how a barrier
    // fencing a real hazard gets dropped.
    //
    // Allocations are keyed on the backing Var (`Buffer::data`) rather than on
    // BufferNode: lowering re-declares one allocation through several Buffer
    // objects (e.g. `A_shared_1 = T.Buffer(..., data=A_shared.data)`), so
    // BufferNode identity would treat aliases of the same memory as distinct.
    //
    // Var identity alone is NOT sufficient, however. This pass runs after
    // MergeSharedMemoryAllocations, which packs every shared buffer into one
    // `buf_dyn_shmem` arena and rebinds each original buffer to
    // `handle_add_byte_offset(arena, off)`. Two DISTINCT Vars can therefore
    // name the same bytes:
    //
    //   a_shared = handle_add_byte_offset(buf_dyn_shmem, 0)      // [0, 36832)
    //   c_shared = handle_add_byte_offset(buf_dyn_shmem, 0)      // [0, 65536)
    //
    // Keying only on the Var makes such a pair look disjoint, so a barrier
    // between a read of `a_shared` and a write of `c_shared` looks redundant
    // and gets dropped -- a real WAR race on one address. Conversely, buffers
    // packed back-to-back (`b_shared` at 36832) must still be recognized as
    // disjoint, or every barrier survives and the pass stops doing its job.
    //
    // So each shared Var carries the arena byte range it resolves to, and
    // conflicts are decided by interval overlap. Ranges are only known for
    // aliases whose offset and extent are both constant; anything else is
    // recorded as "unresolved" and treated as aliasing the whole arena.
    struct ArenaRange {
      const VarNode *base{nullptr}; // arena Var this alias points into
      int64_t begin{0};             // byte offset within the arena
      int64_t end{0}; // exclusive; == begin means "extent unknown"
      bool has_extent{false};

      bool MayOverlap(const ArenaRange &o) const {
        if (base != o.base)
          return false; // different arenas cannot alias
        // An unknown extent covers the rest of the arena: be conservative.
        if (!has_extent || !o.has_extent)
          return true;
        return begin < o.end && o.begin < end;
      }
    };

    struct SharedAccess {
      std::unordered_set<const VarNode *> vars;
      // "There is, or may be, a shared access here that cannot be attributed
      // to a specific allocation." Set for opaque calls and for statement
      // kinds this pass does not model. Consumers must treat an unknown set as
      // aliasing everything: a disjointness test must bail out, and an "is
      // there any shared traffic here" test must answer yes. Getting this
      // polarity wrong in either direction drops a live barrier.
      bool unknown{false};

      bool HasAny() const { return unknown || !vars.empty(); }
      void MarkUnknown() { unknown = true; }
    };

    // Scope test for a raw pointer Var's storage_scope. Buffer-typed operands
    // go through tl::IsSharedBuffer instead; only PointerType has no shared
    // helper to reuse, so the scope names are spelled out just once, here.
    static bool IsSharedPointerScope(const ffi::String &scope) {
      return scope == "shared" || scope == "shared.dyn";
    }

    // Resolve the shared allocation behind a pointer-typed Var, if any.
    static const VarNode *SharedVarOf(const Var &var) {
      const auto *ptr_type = var->type_annotation.as<PointerTypeNode>();
      if (!ptr_type)
        return nullptr;
      if (!IsSharedPointerScope(ptr_type->storage_scope))
        return nullptr;
      return var.get();
    }

    // Record every shared allocation referenced anywhere inside `e`.
    // Handles the address_of(BufferLoad) / tvm_access_ptr(..., data, ...)
    // wrappers that InjectPTSAsyncCopy and the GEMM lowering emit, as well as
    // plain BufferLoad. A bare pointer Var that reaches here without one of
    // those wrappers is still resolved via its type annotation.
    static void CollectSharedVarsInExpr(const PrimExpr &e, SharedAccess *out) {
      PostOrderVisit(e, [&](const ObjectRef &obj) {
        if (const auto *load = obj.as<BufferLoadNode>()) {
          if (tl::IsSharedBuffer(load->buffer))
            out->vars.insert(load->buffer->data.get());
          return;
        }
        if (const auto *var = obj.as<VarNode>()) {
          if (const VarNode *sv = SharedVarOf(GetRef<Var>(var)))
            out->vars.insert(sv);
        }
      });
    }

    // Classify an opaque call's operands by walking its tvm_access_ptr
    // arguments, which carry the access direction explicitly:
    //   tvm_access_ptr(type, var, offset, extent, rw_mask)
    // with rw_mask bit 1 = read and bit 2 = write. The GEMM lowering emits
    // exactly this form (A and B with mask 1, the accumulator with mask 3), and
    // so does the AllReduce workspace argument, so a call whose every operand
    // is an access_ptr can be classified precisely instead of conservatively.
    //
    // Returns false when any argument is NOT an access_ptr — such a call may
    // reach shared memory the arguments do not name, and the caller must fall
    // back to marking both sides unknown.
    static bool ClassifyAccessPtrArgs(const CallNode *call, SharedAccess *reads,
                                      SharedAccess *writes) {
      bool saw_any = false;
      for (const PrimExpr &arg : call->args) {
        // Non-pointer scalars (the template string, lane counts, offsets) carry
        // no shared access; skip them, but only when they cannot hide one.
        if (arg.as<StringImmNode>() || arg.as<IntImmNode>() ||
            arg.as<FloatImmNode>())
          continue;
        const auto *arg_call = arg.as<CallNode>();
        if (!arg_call || !arg_call->op.same_as(builtin::tvm_access_ptr()) ||
            arg_call->args.size() != 5U)
          return false;
        const auto *mask = arg_call->args[4].as<IntImmNode>();
        if (!mask)
          return false;
        SharedAccess operand;
        CollectSharedVarsInExpr(arg_call->args[1], &operand);
        if (operand.unknown)
          return false;
        if (operand.vars.empty())
          continue; // not shared (e.g. a fragment)
        saw_any = true;
        if (mask->value & 1)
          reads->vars.insert(operand.vars.begin(), operand.vars.end());
        if (mask->value & 2)
          writes->vars.insert(operand.vars.begin(), operand.vars.end());
      }
      // A call that named no shared operand at all is suspicious rather than
      // clean: it may still touch shared memory through a route this function
      // does not see. Let the caller be conservative.
      return saw_any;
    }

    // Split a statement's shared accesses into reads and writes.
    static void CollectSharedAccess(const Stmt &s, SharedAccess *reads,
                                    SharedAccess *writes) {
      // Declarations and allocations name a buffer but perform no access.
      // These must be listed explicitly: falling through to the conservative
      // default at the bottom would report them as shared traffic, which
      // latches seen_shared_before_ and needlessly keeps every leading
      // barrier for no correctness gain.
      if (s.as<DeclBufferNode>() || s.as<AllocBufferNode>())
        return;

      if (const auto *eval = s.as<EvaluateNode>()) {
        const auto *call = eval->value.as<CallNode>();
        if (!call) {
          // e.g. Evaluate(BufferLoad(A_shared[...])) — a bare read.
          CollectSharedVarsInExpr(eval->value, reads);
          return;
        }
        if (call->op.same_as(builtin::tvm_storage_sync()) ||
            call->op.same_as(tl::pts_syncthreads()))
          return;
        // Async copies: shared is the destination for a load and the source
        // for a store, and the two intrinsics place their operands in
        // different argument positions (load: dst, src, bytes; store: src,
        // dst, bytes). Scan every argument and let the storage scope decide
        // which side is shared, instead of assuming a fixed index.
        if (call->op.same_as(tl::pts_load_async())) {
          CollectSharedVarsInExpr(eval->value, writes);
          return;
        }
        if (call->op.same_as(tl::pts_store_async())) {
          CollectSharedVarsInExpr(eval->value, reads);
          return;
        }
        // Any other opaque call (e.g. tl_tang_gemm) may read and write the
        // buffers it is handed, and may reach shared memory that does not
        // appear in its arguments at all. Try the precise route first: when
        // every operand is a tvm_access_ptr the rw_mask states the direction,
        // which is enough to classify the call exactly. Only when that fails
        // record what is visible AND mark both sides unknown. The visible vars
        // are not enough on their own: the classifier this replaced treated
        // every such call as a shared consumer, and quietly relaxing that would
        // let a producer->consumer barrier in front of a GEMM be dropped.
        if (ClassifyAccessPtrArgs(call, reads, writes))
          return;
        CollectSharedVarsInExpr(eval->value, reads);
        CollectSharedVarsInExpr(eval->value, writes);
        reads->MarkUnknown();
        writes->MarkUnknown();
        return;
      }
      if (const auto *store = s.as<BufferStoreNode>()) {
        if (tl::IsSharedBuffer(store->buffer))
          writes->vars.insert(store->buffer->data.get());
        // The RHS and the index expressions are reads — e.g. a shared->register
        // copy `reg[i] = A_shared[i]`, a GEMM fragment load, or a reduction.
        // Missing these would make such consumers invisible and let the
        // fixpoint drop the producer->consumer barrier, causing a RAW race.
        CollectSharedVarsInExpr(store->value, reads);
        for (const auto &idx : store->indices)
          CollectSharedVarsInExpr(idx, reads);
        return;
      }
      if (const auto *for_node = s.as<ForNode>()) {
        CollectSharedAccess(for_node->body, reads, writes);
        return;
      }
      if (const auto *while_node = s.as<WhileNode>()) {
        CollectSharedVarsInExpr(while_node->condition, reads);
        CollectSharedAccess(while_node->body, reads, writes);
        return;
      }
      if (const auto *assert_stmt = s.as<AssertStmtNode>()) {
        // tirx's AssertStmt carries no body, so the condition is the whole
        // statement.
        CollectSharedVarsInExpr(assert_stmt->condition, reads);
        return;
      }
      if (const auto *attr = s.as<AttrStmtNode>()) {
        CollectSharedAccess(attr->body, reads, writes);
        return;
      }
      if (const auto *seq = s.as<SeqStmtNode>()) {
        for (const auto &child : seq->seq)
          CollectSharedAccess(child, reads, writes);
        return;
      }
      if (const auto *if_node = s.as<IfThenElseNode>()) {
        CollectSharedAccess(if_node->then_case, reads, writes);
        if (if_node->else_case.defined())
          CollectSharedAccess(if_node->else_case.value(), reads, writes);
        return;
      }
      if (const auto *bind = s.as<BindNode>()) {
        // BindNode has no body; the bound value is the only access.
        CollectSharedVarsInExpr(bind->value, reads);
        return;
      }
      if (const auto *br = s.as<SBlockRealizeNode>()) {
        CollectSharedAccess(br->block, reads, writes);
        return;
      }
      if (const auto *block = s.as<SBlockNode>()) {
        CollectSharedAccess(block->body, reads, writes);
        return;
      }
      // An unrecognized statement kind may hide arbitrary shared traffic.
      reads->MarkUnknown();
      writes->MarkUnknown();
    }

    static bool HasSharedStore(const Stmt &s) {
      SharedAccess reads, writes;
      CollectSharedAccess(s, &reads, &writes);
      return writes.HasAny();
    }

    static bool HasSharedRead(const Stmt &s) {
      SharedAccess reads, writes;
      CollectSharedAccess(s, &reads, &writes);
      return reads.HasAny();
    }

    bool in_for_loop_{false};
    // Tracks whether any shared-memory producer/consumer has appeared in
    // program order so far. Monotonic: once set, it stays set. A leading sync
    // must not be dropped once this is true, because that sync may be fencing a
    // shared-memory op emitted earlier in an enclosing scope (e.g. async DMA
    // before a lexical_alloc_scope wrapping the gemm).
    bool seen_shared_before_{false};

    // Arena range per shared Var, built from the bindings
    // MergeSharedMemoryAllocations emits. A Var absent from this map is not an
    // arena alias (e.g. a standalone `shared` allocation that never got
    // merged); such Vars fall back to identity comparison, which is exact for
    // them because nothing else shares their storage.
    std::unordered_map<const VarNode *, ArenaRange> arena_range_;

    // Record `alias = handle_add_byte_offset(base, off)`.
    void RecordAlias(const Var &alias, const PrimExpr &value) {
      const auto *call = value.as<CallNode>();
      if (!call || !call->op.same_as(builtin::handle_add_byte_offset()) ||
          call->args.size() != 2U)
        return;
      const auto *base = call->args[0].as<VarNode>();
      const auto *off = call->args[1].as<IntImmNode>();
      if (!base)
        return;
      ArenaRange r;
      r.base = base;
      // A chained alias (alias of an alias) accumulates its parent's offset.
      auto it = arena_range_.find(base);
      int64_t base_off = 0;
      if (it != arena_range_.end()) {
        r.base = it->second.base;
        base_off = it->second.begin;
      }
      if (!off) {
        // Non-constant offset: position unknown, so treat as covering the
        // arena.
        r.begin = 0;
        r.has_extent = false;
        arena_range_[alias.get()] = r;
        return;
      }
      r.begin = base_off + off->value;
      r.end = r.begin; // extent filled in by the DeclBuffer that types it
      r.has_extent = false;
      arena_range_[alias.get()] = r;
    }

    // A DeclBuffer over an arena alias supplies the extent the binding lacks.
    // Widen rather than overwrite: one alias may be re-declared through several
    // Buffers of differing length, and the conservative range is the largest.
    void RecordAliasExtent(const Buffer &buf) {
      auto it = arena_range_.find(buf->data.get());
      if (it == arena_range_.end())
        return;
      if (buf->shape.size() != 1U)
        return;
      const auto *n = buf->shape[0].as<IntImmNode>();
      if (!n)
        return;
      int64_t bytes = n->value * buf->dtype.bytes() * buf->dtype.lanes();
      int64_t end = it->second.begin + bytes;
      if (!it->second.has_extent || end > it->second.end) {
        it->second.end = end;
        it->second.has_extent = true;
      }
    }

    // Do these two shared Vars may-alias? Vars with no arena range compare by
    // identity; ranges decide the rest.
    bool VarsMayAlias(const VarNode *a, const VarNode *b) const {
      if (a == b)
        return true;
      auto ia = arena_range_.find(a);
      auto ib = arena_range_.find(b);
      if (ia == arena_range_.end() || ib == arena_range_.end())
        return false;
      return ia->second.MayOverlap(ib->second);
    }

    Stmt VisitStmt_(const BindNode *op) final {
      RecordAlias(op->var, op->value);
      return StmtExprMutator::VisitStmt_(op);
    }

    Stmt VisitStmt_(const DeclBufferNode *op) final {
      RecordAliasExtent(op->buffer);
      return StmtExprMutator::VisitStmt_(op);
    }

    Stmt VisitStmt_(const ForNode *op) final {
      bool old_in_for_loop = in_for_loop_;
      in_for_loop_ = true;
      Stmt result = StmtExprMutator::VisitStmt_(op);
      in_for_loop_ = old_in_for_loop;
      return result;
    }

    Stmt VisitStmt_(const SeqStmtNode *op) final {
      size_t n = op->seq.size();
      std::vector<size_t> sync_pos;
      for (size_t i = 0; i < n; ++i) {
        if (IsSync(op->seq[i]) || IsSyncDeep(op->seq[i]))
          sync_pos.push_back(i);
      }
      std::set<size_t> skip;

      // Helper: collect shared accesses from the "tail" of an IsSyncDeep node
      // (everything after the inner sync).  AttrStmt{SeqStmt{sync, gemm}} is
      // an IsSyncDeep node; the gemm after the sync is invisible to the
      // hazard scan below (which sees only the element's position, not its
      // internals) and to the adjacent-sync dedup, but its shared accesses
      // still need fencing.
      auto CollectSyncDeepTailAccess = [&](size_t pos, SharedAccess *reads,
                                           SharedAccess *writes) {
        if (!IsSyncDeep(op->seq[pos]) || IsSync(op->seq[pos]))
          return;
        std::function<void(const Stmt &)> collect;
        collect = [&](const Stmt &cur) {
          if (const auto *seq = cur.as<SeqStmtNode>()) {
            if (seq->seq.empty())
              return;
            if (IsSync(seq->seq[0])) {
              for (size_t j = 1; j < seq->seq.size(); ++j)
                CollectSharedAccess(seq->seq[j], reads, writes);
              return;
            }
            collect(seq->seq[0]);
          } else if (const auto *attr = cur.as<AttrStmtNode>()) {
            collect(attr->body);
          } else if (const auto *br = cur.as<SBlockRealizeNode>()) {
            collect(br->block->body);
          } else if (const auto *blk = cur.as<SBlockNode>()) {
            collect(blk->body);
          }
        };
        collect(op->seq[pos]);
      };
      // ---- Unified hazard-based barrier elimination -----------------------
      //
      // Earlier revisions of this pass had four independent "patterns", each
      // matching a syntactic shape and each reasoning locally about why some
      // barrier was unnecessary. That structure was unsound, and not because
      // any single pattern's test was too weak: the patterns *interacted*.
      // Pattern 1 dropped a barrier assuming the next one would still fence
      // its writes; Pattern 3 dropped that next barrier assuming the previous
      // one covered its reads. Each decision was locally defensible and the
      // combination removed both fences around a real read-after-write. In the
      // MLA-decode kernel that deleted the barrier between
      //   `S_shared[...] = acc_s[...]`  and  `tl_tang_gemm(S_shared, ...)`
      // and produced sporadic wrong results.
      //
      // The rewrite drops the pattern taxonomy and decides on hazards instead.
      //
      // A barrier's only job is to separate conflicting shared-memory accesses.
      // Two accesses conflict when they touch the same allocation and at least
      // one is a write (RAW / WAR / WAW). So: a set of barriers is legal iff
      // every conflicting pair of statements has at least one *surviving*
      // barrier between them. That is a property of the whole set, not of any
      // one barrier, which is exactly what the per-pattern structure could not
      // express.
      //
      // Removal therefore runs as a fixpoint over the surviving set:
      //   1. every barrier is a candidate;
      //   2. tentatively remove one, then verify the resulting set still
      //      separates all conflicting pairs;
      //   3. commit the removal only if it does, and repeat until a full sweep
      //      commits nothing.
      // Because each verification runs against the barriers that are actually
      // still there, no decision can rest on a barrier a later decision
      // removes. Order-dependence disappears with it: a removal that is only
      // safe while some other barrier stands simply fails its check.

      // Access sets per sequence element, computed once.
      // `tail_*` holds the accesses of an IsSyncDeep element's tail — the
      // statements after its inner barrier, e.g. the gemm in
      // AttrStmt{SeqStmt{sync, gemm}}. Those accesses sit *after* the element's
      // own barrier, which matters for what the barrier separates.
      std::vector<SharedAccess> elem_reads(n), elem_writes(n);
      std::vector<SharedAccess> tail_reads(n), tail_writes(n);
      std::vector<bool> is_sync(n, false);
      for (size_t i = 0; i < n; ++i) {
        is_sync[i] = IsSync(op->seq[i]) || IsSyncDeep(op->seq[i]);
        if (is_sync[i]) {
          if (!IsSync(op->seq[i]))
            CollectSyncDeepTailAccess(i, &tail_reads[i], &tail_writes[i]);
        } else {
          CollectSharedAccess(op->seq[i], &elem_reads[i], &elem_writes[i]);
        }
      }

      // Do these two access sets conflict? Conservative on `unknown`: an
      // unattributable access aliases every allocation, so any write on one
      // side conflicts with any access on the other.
      //
      // Set membership is NOT enough: after the arena merge two different Vars
      // can name the same bytes, so pairs are compared through VarsMayAlias,
      // which falls back to identity for non-arena Vars and to interval
      // overlap for arena aliases.
      auto Overlaps = [this](const SharedAccess &a, const SharedAccess &b) {
        if (!a.HasAny() || !b.HasAny())
          return false;
        if (a.unknown || b.unknown)
          return true;
        for (const VarNode *va : a.vars)
          for (const VarNode *vb : b.vars)
            if (VarsMayAlias(va, vb))
              return true;
        return false;
      };
      auto Conflicts = [&](const SharedAccess &r1, const SharedAccess &w1,
                           const SharedAccess &r2, const SharedAccess &w2) {
        // write-write, write-read, read-write. Read-read never conflicts.
        return Overlaps(w1, w2) || Overlaps(w1, r2) || Overlaps(r1, w2);
      };

      // What element `i` accesses, split by whether the access is fenced from
      // earlier statements by `i`'s own (surviving) barrier.
      //   before: accesses that precede i's barrier — for a plain statement,
      //           everything; for a barrier element, nothing.
      //   after:  accesses that follow i's barrier — a wrapped barrier's tail.
      // When i's barrier is dropped, the tail joins `before`: there is no
      // longer anything separating it from what came earlier.
      auto AccessesBefore = [&](size_t i, bool sync_survives, SharedAccess *r,
                                SharedAccess *w) {
        if (!is_sync[i]) {
          *r = elem_reads[i];
          *w = elem_writes[i];
        } else if (!sync_survives) {
          *r = tail_reads[i];
          *w = tail_writes[i];
        }
      };
      // `after` does not depend on whether i's barrier survives: a wrapped
      // barrier's tail follows the barrier either way.
      auto AccessesAfter = [&](size_t i, SharedAccess *r, SharedAccess *w) {
        if (!is_sync[i]) {
          *r = elem_reads[i];
          *w = elem_writes[i];
        } else {
          *r = tail_reads[i];
          *w = tail_writes[i];
        }
      };

      // Is every conflicting pair in this sequence separated by a surviving
      // barrier? `dropped` is the set of barrier positions to treat as gone.
      //
      // Only pairs with no barrier between them need checking, so the scan
      // walks forward from each statement and stops at the first surviving
      // barrier — everything past it is separated by construction.
      //
      // Inside a loop body the scan continues past the end of the sequence and
      // wraps around: iteration k's trailing accesses meet iteration k+1's
      // leading accesses with only the barriers between them (in program order,
      // the loop's back edge carries no implicit fence). Without the wrap a
      // trailing producer and a leading consumer look unrelated, and the
      // barrier that fenced them across iterations is dropped.
      auto AllHazardsSeparated = [&](const std::set<size_t> &dropped) {
        size_t span = in_for_loop_ ? 2 * n : n;
        for (size_t i = 0; i < n; ++i) {
          SharedAccess src_r, src_w;
          AccessesAfter(i, &src_r, &src_w);
          if (!src_r.HasAny() && !src_w.HasAny())
            continue;
          for (size_t step = i + 1; step < span; ++step) {
            size_t j = step % n;
            if (j == i)
              break; // full lap: the sequence fences itself
            bool j_survives = is_sync[j] && !dropped.count(j);
            if (is_sync[j] && j_survives)
              break; // separated
            SharedAccess rj, wj;
            AccessesBefore(j, j_survives, &rj, &wj);
            if (!rj.HasAny() && !wj.HasAny())
              continue;
            if (Conflicts(src_r, src_w, rj, wj))
              return false;
          }
        }
        return true;
      };

      // A leading barrier may be fencing a producer from an ENCLOSING scope
      // (e.g. async DMA emitted before an AttrStmt that wraps this sequence),
      // which the scan above cannot see. Keep the first surviving barrier
      // whenever such a producer might exist. Inside a loop body the previous
      // iteration is exactly such an unseen producer, but the wrap-around in
      // AllHazardsSeparated already models that, so the loop case needs no
      // extra guard here.
      auto LeadingBarrierSafe = [&](size_t pos,
                                    const std::set<size_t> &dropped) {
        for (size_t i = 0; i < pos; ++i)
          if (is_sync[i] && !dropped.count(i))
            return true; // not the first
        if (seen_shared_before_)
          return false;
        for (size_t i = 0; i < pos; ++i)
          if (elem_reads[i].HasAny() || elem_writes[i].HasAny())
            return false;
        return true;
      };

      // Mirror image of LeadingBarrierSafe. A trailing barrier with no shared
      // access after it *in this sequence* may still fence a consumer in an
      // ENCLOSING scope — LowerThreadAllreduce emits exactly that shape: the
      // sync ends the inner SeqStmt, its read sits in the parent. This walker
      // cannot see the enclosing scope, so stay conservative and keep it. Loop
      // bodies need no guard: AllHazardsSeparated wraps around to the next
      // iteration.
      auto TrailingBarrierSafe = [&](size_t pos,
                                     const std::set<size_t> &dropped) {
        if (in_for_loop_)
          return true;
        if (!IsSync(op->seq[pos]))
          return true; // IsSyncDeep: tail is visible
        for (size_t i = pos + 1; i < n; ++i) {
          if (is_sync[i] && !dropped.count(i))
            return true; // not the last
          if (elem_reads[i].HasAny() || elem_writes[i].HasAny())
            return true;
        }
        return false;
      };

      // Fixpoint: keep sweeping while some removal commits. Each trial is
      // verified against the barriers that actually survive, so the result does
      // not depend on the order candidates are visited.
      bool changed = true;
      while (changed) {
        changed = false;
        for (size_t i = 0; i < n; ++i) {
          if (!is_sync[i] || skip.count(i))
            continue;
          if (!LeadingBarrierSafe(i, skip))
            continue;
          if (!TrailingBarrierSafe(i, skip))
            continue;
          std::set<size_t> trial = skip;
          trial.insert(i);
          if (!AllHazardsSeparated(trial))
            continue;
          skip.insert(i);
          changed = true;
        }
      }

      Array<Stmt> out;
      bool prev_sync = false;
      bool prev_has_tail_access = false; // prev IsSyncDeep has shared tail
      for (size_t i = 0; i < n; ++i) {
        bool cur_sync = IsSync(op->seq[i]) || IsSyncDeep(op->seq[i]);
        // Adjacent-sync dedup: when two sync-like elements appear back-to-back
        // in the SeqStmt, the second is normally redundant.  BUT when the first
        // is an IsSyncDeep node whose tail (the part AFTER the inner sync) has
        // shared-memory access, the second barrier fences that tail against the
        // next producer/consumer — dropping it would lose ordering.
        bool drop =
            skip.count(i) || (cur_sync && prev_sync && !prev_has_tail_access);
        if (drop) {
          // Removing a barrier must remove the BARRIER, not the statement that
          // happens to contain it. When the sync sits inside wrapper nodes
          // (AttrStmt / SBlockRealize / SBlock), dropping the whole seq element
          // would discard the wrapper too — losing its attr_key, iter_values or
          // predicate, and any sibling statements it carries. StripSync returns
          // the element with just the barrier removed, or nullopt when nothing
          // but the barrier is left, in which case the element really can go.
          Optional<Stmt> stripped = StripSync(op->seq[i]);
          if (stripped.defined())
            out.push_back(VisitStmt(stripped.value()));
          // A dropped barrier leaves no barrier in program order, so the next
          // sync has nothing to be adjacent TO. Leave prev_sync alone (the
          // original code reached `continue` without touching it): setting it
          // here would make the following sync look like a duplicate and drop
          // it as well, removing two barriers where one was redundant.
          continue;
        }
        prev_sync = cur_sync;
        // Track whether the previous (non-dropped) sync is an IsSyncDeep node
        // with a software-visible tail.  That tail needs fencing, so the NEXT
        // barrier should NOT be dropped as an adjacent duplicate; the guard
        // above reads prev_has_tail_access on the following element.
        prev_has_tail_access = false;
        if (cur_sync && !IsSync(op->seq[i])) {
          SharedAccess tail_reads, tail_writes;
          CollectSyncDeepTailAccess(i, &tail_reads, &tail_writes);
          prev_has_tail_access = tail_reads.HasAny() || tail_writes.HasAny();
        }
        // Recurse first, then mark program-order shared activity so that
        // nested SeqStmts visited later (e.g. a lexical_alloc_scope body)
        // observe producers/consumers emitted earlier in this scope and
        // keep their own leading sync (LeadingBarrierSafe guard).
        out.push_back(VisitStmt(op->seq[i]));
        if (!seen_shared_before_ &&
            (HasSharedStore(op->seq[i]) || HasSharedRead(op->seq[i]))) {
          seen_shared_before_ = true;
        }
      }
      if (out.empty())
        return Evaluate(0);
      if (out.size() == 1)
        return out[0];
      return SeqStmt(out);
    }

    Stmt VisitStmt_(const SBlockRealizeNode *op) final {
      SBlock block = Downcast<SBlock>(VisitStmt(op->block));
      return SBlockRealize(op->iter_values, op->predicate, block);
    }
    Stmt VisitStmt_(const SBlockNode *op) final {
      // Defer to the base mutator, which visits op->body with THIS remover
      // (dispatching into VisitStmt_(SeqStmtNode) so the fixpoint runs). Do
      // NOT re-run RemoveAdjacentSyncs here with a fresh remover: that reset
      // seen_shared_before_ / in_for_loop_, defeating the leading-sync guard
      // across the SBlock boundary and re-opening the same RAW race the
      // enclosing-scope guard was added to prevent.
      return StmtExprMutator::VisitStmt_(op);
    }
  };

  auto remover = RedundantSyncRemover();
  return remover(stmt);
}

} // namespace

tirx::transform::Pass RemoveRedundantSyncs() {
  using namespace tirx::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m,
                       const PassContext &ctx) -> PrimFunc {
    // Escape hatch for debugging a suspected data race: with the pass disabled
    // every barrier ThreadSync emitted survives, so a kernel that is wrong with
    // it on and right with it off points at a barrier removed here. Checked
    // before any mutation so disabling is an exact identity transform.
    if (ctx->GetConfig<Bool>(kDisableRemoveRedundantSyncs, Bool(false))
            .value()) {
      return f;
    }
    auto *fptr = f.CopyOnWrite();
    fptr->body = RemoveAdjacentSyncs(fptr->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 1, "tl.RemoveRedundantSyncs", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.RemoveRedundantSyncs",
                        RemoveRedundantSyncs);
}

} // namespace transform
} // namespace tl
} // namespace tvm
