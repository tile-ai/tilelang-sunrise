/*
 * S3-specific redundant synchronization removal for certified async edges.
 */
#include "s3_redundant_sync_remover.h"

#include "../../op/builtin.h"
#include "../../op/utils.h"

#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>

#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {
namespace tang {
namespace transform {

using namespace tirx;
using namespace ffi;

namespace {
namespace s3_redundant_sync {

inline bool IsSync(const Stmt &stmt) {
  if (const auto *eval = stmt.as<EvaluateNode>()) {
    if (const auto *call = eval->value.as<CallNode>()) {
      if (call->op.same_as(tl::pts_syncthreads())) {
        return true;
      }
      if (!call->op.same_as(builtin::tvm_storage_sync()) ||
          call->args.empty()) {
        return false;
      }
      const auto *scope = call->args[0].as<StringImmNode>();
      return scope != nullptr &&
             (scope->value == "shared" || scope->value == "shared.dyn");
    }
  }
  return false;
}

inline bool IsSyncDeep(const Stmt &stmt) {
  Stmt current = stmt;
  while (true) {
    if (IsSync(current)) {
      return true;
    }
    if (const auto *seq = current.as<SeqStmtNode>()) {
      if (seq->seq.empty()) {
        break;
      }
      current = seq->seq[0];
    } else if (const auto *realize = current.as<SBlockRealizeNode>()) {
      current = realize->block->body;
    } else if (const auto *block = current.as<SBlockNode>()) {
      current = block->body;
    } else if (const auto *attr = current.as<AttrStmtNode>()) {
      current = attr->body;
    } else {
      break;
    }
  }
  return false;
}

inline Optional<Stmt> StripSync(const Stmt &stmt) {
  if (IsSync(stmt)) {
    return std::nullopt;
  }
  if (const auto *attr = stmt.as<AttrStmtNode>()) {
    Stmt body = StripSync(attr->body).value_or(Evaluate(0));
    return AttrStmt(attr->node, attr->attr_key, attr->value, body);
  }
  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    if (seq->seq.empty()) {
      return stmt;
    }
    Optional<Stmt> head = StripSync(seq->seq[0]);
    Array<Stmt> kept;
    if (head.defined()) {
      kept.push_back(head.value());
    }
    for (size_t i = 1; i < seq->seq.size(); ++i) {
      kept.push_back(seq->seq[i]);
    }
    if (kept.empty()) {
      return std::nullopt;
    }
    if (kept.size() == 1) {
      return kept[0];
    }
    return SeqStmt(kept);
  }
  if (const auto *realize = stmt.as<SBlockRealizeNode>()) {
    Optional<Stmt> block = StripSync(realize->block);
    if (!block.defined()) {
      return std::nullopt;
    }
    return SBlockRealize(realize->iter_values, realize->predicate,
                         Downcast<SBlock>(block.value()));
  }
  if (const auto *block = stmt.as<SBlockNode>()) {
    Stmt body = StripSync(block->body).value_or(Evaluate(0));
    SBlock updated = GetRef<SBlock>(block);
    updated.CopyOnWrite()->body = body;
    return updated;
  }
  return stmt;
}

struct ArenaRange {
  const VarNode *base{nullptr};
  int64_t begin{0};
  int64_t end{0};
  bool has_extent{false};

  bool MayOverlap(const ArenaRange &other) const {
    if (base != other.base) {
      return false;
    }
    if (!has_extent || !other.has_extent) {
      return true;
    }
    return begin < other.end && other.begin < end;
  }
};

class ArenaAliasAnalysis {
public:
  void RecordAlias(const Var &alias, const PrimExpr &value) {
    const auto *call = value.as<CallNode>();
    if (!call || !call->op.same_as(builtin::handle_add_byte_offset()) ||
        call->args.size() != 2U) {
      return;
    }
    const auto *base = call->args[0].as<VarNode>();
    const auto *offset = call->args[1].as<IntImmNode>();
    if (!base) {
      return;
    }
    ArenaRange range;
    range.base = base;
    int64_t base_offset = 0;
    auto parent = ranges_.find(base);
    if (parent != ranges_.end()) {
      range.base = parent->second.base;
      base_offset = parent->second.begin;
    }
    if (!offset) {
      ranges_[alias.get()] = range;
      return;
    }
    range.begin = base_offset + offset->value;
    range.end = range.begin;
    ranges_[alias.get()] = range;
  }

  void RecordAliasExtent(const Buffer &buffer) {
    auto range = ranges_.find(buffer->data.get());
    if (range == ranges_.end() || buffer->shape.size() != 1U) {
      return;
    }
    const auto *elements = buffer->shape[0].as<IntImmNode>();
    if (!elements) {
      return;
    }
    int64_t bytes =
        elements->value * buffer->dtype.bytes() * buffer->dtype.lanes();
    int64_t end = range->second.begin + bytes;
    if (!range->second.has_extent || end > range->second.end) {
      range->second.end = end;
      range->second.has_extent = true;
    }
  }

  bool VarsMayAlias(const VarNode *lhs, const VarNode *rhs) const {
    if (lhs == rhs) {
      return true;
    }
    auto lhs_range = ranges_.find(lhs);
    auto rhs_range = ranges_.find(rhs);
    if (lhs_range == ranges_.end() || rhs_range == ranges_.end()) {
      return false;
    }
    return lhs_range->second.MayOverlap(rhs_range->second);
  }

private:
  std::unordered_map<const VarNode *, ArenaRange> ranges_;
};

} // namespace s3_redundant_sync
} // namespace

Stmt RemoveCertifiedS3Syncs(const Stmt &stmt,
                            S3RedundantSyncRemovalOptions options) {

  struct S3RedundantSyncRemover : public StmtExprMutator {
    explicit S3RedundantSyncRemover(S3RedundantSyncRemovalOptions options)
        : certified_edges_(std::move(options.certified_edges)),
          enable_async_edge_certificates_(
              options.enable_async_edge_certificates),
          preserve_uncertified_loop_barriers_(
              options.preserve_uncertified_loop_barriers),
          async_edge_anchor_attr_(std::move(options.async_edge_anchor_attr)),
          async_copy_write_role_(options.async_copy_write_role),
          async_consumer_read_role_(options.async_consumer_read_role),
          is_ordering_only_call_(std::move(options.is_ordering_only_call)),
          requires_following_sync_call_(
              std::move(options.requires_following_sync_call)) {}

    // Shared-memory access classification remains backend-specific.

    // ---- Shared-memory access classification ------------------------------
    //
    // Deciding whether a barrier can be dropped comes down to one question:
    // which shared allocations does a statement read, and which does it write?
    // CollectSharedAccess is the single answer to that question, and
    // HasSharedRead/HasSharedStore are thin wrappers over it.
    //
    // Keeping one classifier is a correctness requirement: every removal check
    // must see the same accesses, and unmodelled statements must remain
    // conservative.
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
    struct SharedAccess {
      std::unordered_set<const VarNode *> vars;
      // Stable async-edge keys attached before address rewriting. These are
      // intentionally separate from `vars`: StorageRewrite may rename or pack
      // the shared allocation, while the certificate identity must not change.
      std::unordered_set<int64_t> certified_reads;
      std::unordered_set<int64_t> certified_writes;
      // Copy writes inside a complete versioned loop. Unlike an arbitrary WAW,
      // a prologue write followed by one of these writes is ordered by the
      // loop's free wait/read/free arrive cycle before that slot is reused.
      std::unordered_set<int64_t> certified_reuse_writes;
      // "There is, or may be, a shared access here that cannot be attributed
      // to a specific allocation." Set for opaque calls and for statement
      // kinds this pass does not model. Consumers must treat an unknown set as
      // aliasing everything: a disjointness test must bail out, and an "is
      // there any shared traffic here" test must answer yes. Getting this
      // polarity wrong in either direction drops a live barrier.
      bool unknown{false};

      bool HasAny() const {
        return unknown || !vars.empty() || !certified_reads.empty() ||
               !certified_writes.empty();
      }
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

    static void CollectSharedVarsInStmt(const Stmt &s, SharedAccess *out) {
      PostOrderVisit(s, [&](const ObjectRef &obj) {
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
    void CollectSharedAccess(const Stmt &s, SharedAccess *reads,
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
        // Backend ordering operations do not access the shared payload.
        if (is_ordering_only_call_ && is_ordering_only_call_(call))
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
        if (enable_async_edge_certificates_ &&
            for_node->annotations.count("tl_pipelined_num_stages")) {
          std::unordered_map<int64_t, uint32_t> roles;
          PostOrderVisit(for_node->body, [&](const ObjectRef &obj) {
            const auto *anchor = obj.as<AttrStmtNode>();
            if (anchor == nullptr ||
                anchor->attr_key != async_edge_anchor_attr_)
              return;
            const auto *encoded = anchor->value.as<IntImmNode>();
            if (encoded == nullptr)
              return;
            const int64_t role = encoded->value & int64_t{0xff};
            if (role >= 1 && role <= 6) {
              roles[encoded->value & ~int64_t{0xff}] |= uint32_t{1} << role;
            }
          });
          constexpr uint32_t kCompleteProtocolRoles =
              (uint32_t{1} << 1) | (uint32_t{1} << 2) | (uint32_t{1} << 3) |
              (uint32_t{1} << 4) | (uint32_t{1} << 5) | (uint32_t{1} << 6);
          for (const auto &[key, mask] : roles) {
            if ((mask & kCompleteProtocolRoles) == kCompleteProtocolRoles &&
                writes->certified_writes.count(key) &&
                reads->certified_reads.count(key)) {
              writes->certified_reuse_writes.insert(key);
            }
          }
        }
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
        if (enable_async_edge_certificates_ &&
            attr->attr_key == async_edge_anchor_attr_) {
          Stmt body = s;
          std::vector<std::pair<int64_t, int64_t>> anchors;
          while (const auto *anchor = body.as<AttrStmtNode>()) {
            if (anchor->attr_key != async_edge_anchor_attr_)
              break;
            const auto *encoded = anchor->value.as<IntImmNode>();
            if (!encoded) {
              reads->MarkUnknown();
              writes->MarkUnknown();
              return;
            }
            anchors.emplace_back(encoded->value & ~int64_t{0xff},
                                 encoded->value & int64_t{0xff});
            body = anchor->body;
          }

          bool has_read = false;
          bool has_write = false;
          for (const auto &[key, role] : anchors) {
            if (role == async_consumer_read_role_) {
              reads->certified_reads.insert(key);
              has_read = true;
            } else if (role == async_copy_write_role_) {
              writes->certified_writes.insert(key);
              has_write = true;
            }
          }
          // Protocol-only anchors are ordering operations, not accesses to the
          // payload in shared memory.
          if (!has_read && !has_write)
            return;

          SharedAccess visible;
          CollectSharedVarsInStmt(body, &visible);
          if (has_read)
            reads->vars.insert(visible.vars.begin(), visible.vars.end());
          if (has_write)
            writes->vars.insert(visible.vars.begin(), visible.vars.end());
          return;
        }
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

    bool in_for_loop_{false};
    // True when the SeqStmt currently being rewritten IS the body of the
    // innermost enclosing For (reached through non-branching wrappers such as
    // AttrStmt / SBlock / SBlockRealize). LeadingBarrierSafe uses it: only the
    // loop body is fenced against the previous iteration by the back edge. A
    // sequence nested inside the body (e.g. the
    // `AttrStmt(lexical_alloc_scope){SeqStmt[sync, gemm]}` wrapper a GEMM gets)
    // has its predecessors in the same iteration -- earlier siblings in the
    // enclosing sequence -- which the loop back edge says nothing about. That
    // distinction is what keeps the copy->gemm barrier alive.
    bool is_loop_body_{false};
    // Set while visiting the single child of a For; consumed by the first
    // SeqStmt that sees it.
    bool pending_loop_body_{false};
    // Tracks whether any shared-memory producer/consumer has appeared in
    // program order so far. Monotonic: once set, it stays set. A leading sync
    // must not be dropped once this is true, because that sync may be fencing a
    // shared-memory op emitted earlier in an enclosing scope (e.g. async DMA
    // before a lexical_alloc_scope wrapping the gemm).
    bool seen_shared_before_{false};
    std::unordered_set<int64_t> seen_certified_writes_;

    s3_redundant_sync::ArenaAliasAnalysis arena_alias_;
    std::unordered_set<int64_t> certified_edges_;
    bool enable_async_edge_certificates_{false};
    bool preserve_uncertified_loop_barriers_{false};
    std::string async_edge_anchor_attr_;
    int64_t async_copy_write_role_{0};
    int64_t async_consumer_read_role_{0};
    std::function<bool(const CallNode *)> is_ordering_only_call_;
    std::function<bool(const CallNode *)> requires_following_sync_call_;

    Stmt VisitStmt_(const BindNode *op) final {
      arena_alias_.RecordAlias(op->var, op->value);
      return StmtExprMutator::VisitStmt_(op);
    }

    Stmt VisitStmt_(const DeclBufferNode *op) final {
      arena_alias_.RecordAliasExtent(op->buffer);
      return StmtExprMutator::VisitStmt_(op);
    }

    Stmt VisitStmt_(const ForNode *op) final {
      bool old_in_for_loop = in_for_loop_;
      bool old_pending_loop_body = pending_loop_body_;
      in_for_loop_ = true;
      // The loop's body (the single statement visited below, possibly behind
      // wrapper nodes) is the only sequence at the loop boundary. Anything
      // deeper inside the body is a nested sequence and must not reason about
      // the back edge; see is_loop_body_.
      pending_loop_body_ = true;
      Stmt result = StmtExprMutator::VisitStmt_(op);
      pending_loop_body_ = old_pending_loop_body;
      in_for_loop_ = old_in_for_loop;
      return result;
    }

    // A branch is not the whole loop body: the previous iteration's trailing
    // accesses may have entered through the other arm, so a branch sequence
    // must not model itself as the back-edge boundary.
    Stmt VisitStmt_(const IfThenElseNode *op) final {
      bool old_pending_loop_body = pending_loop_body_;
      pending_loop_body_ = false;
      Stmt result = StmtExprMutator::VisitStmt_(op);
      pending_loop_body_ = old_pending_loop_body;
      return result;
    }

    Stmt VisitStmt_(const WhileNode *op) final {
      bool old_pending_loop_body = pending_loop_body_;
      pending_loop_body_ = false;
      Stmt result = StmtExprMutator::VisitStmt_(op);
      pending_loop_body_ = old_pending_loop_body;
      return result;
    }

    Stmt VisitStmt_(const SeqStmtNode *op) final {
      bool old_is_loop_body = is_loop_body_;
      bool old_pending_loop_body = pending_loop_body_;
      is_loop_body_ = pending_loop_body_;
      pending_loop_body_ = false;
      // Every return below must restore the enclosing sequence's state.
      auto restore = [&]() {
        is_loop_body_ = old_is_loop_body;
        pending_loop_body_ = old_pending_loop_body;
      };
      size_t n = op->seq.size();
      std::vector<size_t> sync_pos;
      for (size_t i = 0; i < n; ++i) {
        if (s3_redundant_sync::IsSync(op->seq[i]) ||
            s3_redundant_sync::IsSyncDeep(op->seq[i]))
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
        if (!s3_redundant_sync::IsSyncDeep(op->seq[pos]) ||
            s3_redundant_sync::IsSync(op->seq[pos]))
          return;
        // IsSyncDeep guarantees that the barrier is the first executable
        // statement down this wrapper chain. Classifying the whole element is
        // therefore exactly classifying its post-barrier tail (the sync itself
        // has no memory access), and unlike manually peeling wrappers it keeps
        // an enclosing async-edge anchor attached to that tail.
        CollectSharedAccess(op->seq[pos], reads, writes);
      };
      // ---- Unified hazard-based barrier elimination -----------------------
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
      std::vector<bool> must_keep_sync(n, false);
      for (size_t i = 0; i < n; ++i) {
        is_sync[i] = s3_redundant_sync::IsSync(op->seq[i]) ||
                     s3_redundant_sync::IsSyncDeep(op->seq[i]);
        if (is_sync[i] && i > 0 && requires_following_sync_call_) {
          PostOrderVisit(op->seq[i - 1], [&](const ObjectRef &obj) {
            if (const auto *call = obj.as<CallNode>()) {
              must_keep_sync[i] =
                  must_keep_sync[i] || requires_following_sync_call_(call);
            }
          });
        }
        if (is_sync[i]) {
          if (!s3_redundant_sync::IsSync(op->seq[i]))
            CollectSyncDeepTailAccess(i, &tail_reads[i], &tail_writes[i]);
        } else {
          CollectSharedAccess(op->seq[i], &elem_reads[i], &elem_writes[i]);
          if (in_for_loop_) {
            for (int64_t key : elem_writes[i].certified_writes) {
              if (certified_edges_.count(key))
                elem_writes[i].certified_reuse_writes.insert(key);
            }
          }
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
            if (arena_alias_.VarsMayAlias(va, vb))
              return true;
        return false;
      };
      auto Conflicts = [&](const SharedAccess &r1, const SharedAccess &w1,
                           const SharedAccess &r2, const SharedAccess &w2) {
        auto Certified = [&](const SharedAccess &a,
                             const std::unordered_set<int64_t> &a_keys,
                             const SharedAccess &b,
                             const std::unordered_set<int64_t> &b_keys) {
          if (a.unknown || b.unknown)
            return false;
          for (int64_t key : a_keys) {
            if (b_keys.count(key) && certified_edges_.count(key))
              return true;
          }
          return false;
        };
        auto CertifiedDistinct =
            [&](const SharedAccess &a,
                const std::unordered_set<int64_t> &a_keys,
                const SharedAccess &b,
                const std::unordered_set<int64_t> &b_keys) {
              if (a.unknown || b.unknown || a_keys.empty() || b_keys.empty())
                return false;
              for (int64_t key : a_keys) {
                if (!certified_edges_.count(key) || b_keys.count(key))
                  return false;
              }
              for (int64_t key : b_keys)
                if (!certified_edges_.count(key))
                  return false;
              return true;
            };
        // General WAW stays conservative. The one certified exception is a
        // prologue copy followed by a complete versioned loop: that loop's
        // free edge orders every reused slot before its next copy write.
        if (Overlaps(w1, w2) &&
            !CertifiedDistinct(w1, w1.certified_writes, w2,
                               w2.certified_writes) &&
            !Certified(w1, w1.certified_writes, w2, w2.certified_reuse_writes))
          return true;
        // ready: the copy publishes its write before the MMA wait/read.
        if (Overlaps(w1, r2) &&
            !Certified(w1, w1.certified_writes, r2, r2.certified_reads))
          return true;
        // free: MMA retirement publishes its read before the next overwrite.
        if (Overlaps(r1, w2) &&
            !Certified(r1, r1.certified_reads, w2, w2.certified_writes))
          return true;
        return false;
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
      // Inside a loop BODY the scan continues past the end of the sequence and
      // wraps around: iteration k's trailing accesses meet iteration k+1's
      // leading accesses with only the barriers between them (in program order,
      // the loop's back edge carries no implicit fence). Without the wrap a
      // trailing producer and a leading consumer look unrelated, and the
      // barrier that fenced them across iterations is dropped.
      //
      // `span` stays doubled for any sequence inside a loop. Wrapping only ADDS
      // candidate pairs, i.e. it can keep a barrier that a non-wrapping scan
      // would drop, never the other way round -- except on the certified
      // boundary path below, where the enclosing loop's ready/free edges are
      // the ordering argument. LeadingBarrierSafe and TrailingBarrierSafe cover
      // predecessors and successors that this self-wrap cannot see.
      auto HazardsLegal = [&](const std::set<size_t> &baseline,
                              const std::set<size_t> &dropped,
                              bool ignore_preexisting) {
        size_t span = in_for_loop_ ? 2 * n : n;
        for (size_t i = 0; i < n; ++i) {
          SharedAccess src_r, src_w;
          AccessesAfter(i, &src_r, &src_w);
          if (!src_r.HasAny() && !src_w.HasAny())
            continue;
          bool crossed_baseline_barrier = false;
          for (size_t step = i + 1; step < span; ++step) {
            size_t j = step % n;
            if (j == i)
              break; // full lap: the sequence fences itself
            bool j_survives = is_sync[j] && !dropped.count(j);
            if (is_sync[j] && j_survives)
              break; // separated
            if (is_sync[j] && !baseline.count(j))
              crossed_baseline_barrier = true;
            SharedAccess rj, wj;
            AccessesBefore(j, j_survives, &rj, &wj);
            if (!rj.HasAny() && !wj.HasAny())
              continue;
            if (Conflicts(src_r, src_w, rj, wj) &&
                (!ignore_preexisting || crossed_baseline_barrier))
              return false;
          }
        }
        return true;
      };

      // The one boundary where a conservative pre-existing alias must not
      // poison the whole-sequence check: prologue copies followed by their
      // complete versioned loop. Every payload access before the candidate
      // must be a certified copy write, and the loop must certify both its
      // ready reads and free-ordered slot reuse for those exact edge keys.
      auto IsCertifiedPipelineBoundary = [&](size_t pos,
                                             const std::set<size_t> &dropped) {
        size_t begin = 0;
        for (size_t i = 0; i < pos; ++i)
          if (is_sync[i] && !dropped.count(i))
            begin = i + 1;

        std::unordered_set<int64_t> prologue_keys;
        for (size_t i = begin; i < pos; ++i) {
          if (elem_reads[i].HasAny() || elem_writes[i].unknown)
            return false;
          if (!elem_writes[i].HasAny())
            continue;
          if (elem_writes[i].certified_writes.empty())
            return false;
          prologue_keys.insert(elem_writes[i].certified_writes.begin(),
                               elem_writes[i].certified_writes.end());
        }
        if (prologue_keys.empty())
          return false;

        size_t consumer = pos + 1;
        while (consumer < n && is_sync[consumer] && dropped.count(consumer))
          ++consumer;
        if (consumer >= n)
          return false;
        const SharedAccess &loop_reads = elem_reads[consumer];
        const SharedAccess &loop_writes = elem_writes[consumer];
        if (loop_reads.unknown || loop_writes.unknown)
          return false;
        for (int64_t key : prologue_keys) {
          if (!certified_edges_.count(key) ||
              !loop_reads.certified_reads.count(key) ||
              !loop_writes.certified_reuse_writes.count(key))
            return false;
        }
        return true;
      };

      // A leading barrier may be fencing a producer from an ENCLOSING scope
      // (e.g. async DMA emitted before an AttrStmt that wraps this sequence),
      // which the scan above cannot see. Keep the first surviving barrier
      // whenever such a producer might exist. In a loop BODY the previous
      // iteration is exactly such an unseen producer, but the wrap-around in
      // HazardsLegal already models that, so the loop body needs no extra guard
      // here. A sequence nested inside the body does need this guard: its
      // preceding siblings (the copies) are outside the wrap's field of view.
      //
      // The distinction only matters for a barrier at position 0. Any other
      // barrier is preceded by statements visible to the scan above. The S3
      // pipeline relies on that distinction for the trailing MMA fence.
      auto LeadingBarrierSafe = [&](size_t pos,
                                    const std::set<size_t> &dropped) {
        for (size_t i = 0; i < pos; ++i)
          if (is_sync[i] && !dropped.count(i))
            return true; // not the first
        if (is_loop_body_ || (in_for_loop_ && pos > 0))
          return true; // loop back-edges are modeled by HazardsLegal
        if (seen_shared_before_) {
          // A nested sequence may begin with ThreadSync's ready barrier while
          // its certified copies live in the enclosing sequence.
          SharedAccess following_reads = tail_reads[pos];
          SharedAccess following_writes = tail_writes[pos];
          if (!following_reads.HasAny() && !following_writes.HasAny()) {
            for (size_t i = pos + 1; i < n; ++i) {
              if (is_sync[i] && !dropped.count(i))
                break;
              following_reads.vars.insert(elem_reads[i].vars.begin(),
                                          elem_reads[i].vars.end());
              following_reads.certified_reads.insert(
                  elem_reads[i].certified_reads.begin(),
                  elem_reads[i].certified_reads.end());
              following_reads.unknown |= elem_reads[i].unknown;
              following_writes.vars.insert(elem_writes[i].vars.begin(),
                                           elem_writes[i].vars.end());
              following_writes.certified_writes.insert(
                  elem_writes[i].certified_writes.begin(),
                  elem_writes[i].certified_writes.end());
              following_writes.unknown |= elem_writes[i].unknown;
            }
          }
          if (following_reads.unknown || following_writes.HasAny() ||
              following_reads.certified_reads.empty())
            return false;
          for (int64_t key : following_reads.certified_reads)
            if (!certified_edges_.count(key) ||
                !seen_certified_writes_.count(key))
              return false;
        }
        for (size_t i = 0; i < pos; ++i) {
          if (!elem_reads[i].HasAny() && !elem_writes[i].HasAny())
            continue;
          // Prologue copies may precede the loop boundary barrier in this same
          // sequence. Their stable write certificates let the global hazard
          // check, rather than this enclosing-scope guard, decide legality.
          if (elem_reads[i].HasAny() || elem_writes[i].unknown ||
              elem_writes[i].certified_writes.empty())
            return false;
          for (int64_t key : elem_writes[i].certified_writes)
            if (!certified_edges_.count(key))
              return false;
        }
        return true;
      };

      // Mirror image of LeadingBarrierSafe. A trailing barrier with no shared
      // access after it *in this sequence* may still fence a consumer in an
      // ENCLOSING scope — LowerThreadAllreduce emits exactly that shape: the
      // sync ends the inner SeqStmt, its read sits in the parent. This walker
      // cannot see the enclosing scope, so stay conservative and keep it --
      // except inside a loop, where the free edge (S3) / the back-edge scan
      // HazardsLegal performs is what orders the trailing read against the next
      // iteration's overwrite. That loop rule predates this pass's hazard
      // rewrite and the S3 pipeline depends on it
      // (test_tang_s3_async_copy_pipeline.py::test_pipeline_loop_has_no_block_wide_sync).
      auto TrailingBarrierSafe = [&](size_t pos,
                                     const std::set<size_t> &dropped) {
        if (in_for_loop_)
          return true;
        if (!s3_redundant_sync::IsSync(op->seq[pos]))
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
          // Some ordering instructions publish initialization only when
          // followed by a block rendezvous. Such barriers are protocol
          // requirements, not shared-payload hazards.
          if (must_keep_sync[i])
            continue;
          // A loop back-edge has no implicit rendezvous. Until the function
          // carries a complete async-edge certificate, keep its in-loop
          // barriers: opaque engine/TMEM accesses may not expose enough buffer
          // identity for the cyclic hazard scan to prove the next iteration
          // safe. Certified software pipelines take the precise path below.
          if (in_for_loop_ && preserve_uncertified_loop_barriers_ &&
              certified_edges_.empty())
            continue;
          if (!LeadingBarrierSafe(i, skip))
            continue;
          if (!TrailingBarrierSafe(i, skip))
            continue;
          std::set<size_t> trial = skip;
          trial.insert(i);
          const bool certified_boundary = IsCertifiedPipelineBoundary(i, skip);
          if (!HazardsLegal(skip, trial, certified_boundary))
            continue;
          skip.insert(i);
          changed = true;
        }
      }

      Array<Stmt> out;
      bool prev_sync = false;
      bool prev_has_tail_access = false; // prev IsSyncDeep has shared tail
      for (size_t i = 0; i < n; ++i) {
        bool cur_sync = s3_redundant_sync::IsSync(op->seq[i]) ||
                        s3_redundant_sync::IsSyncDeep(op->seq[i]);
        // Adjacent-sync dedup: when two sync-like elements appear back-to-back
        // in the SeqStmt, the second is normally redundant.  BUT when the first
        // is an IsSyncDeep node whose tail (the part AFTER the inner sync) has
        // shared-memory access, the second barrier fences that tail against the
        // next producer/consumer — dropping it would lose ordering.
        bool drop =
            !must_keep_sync[i] &&
            (skip.count(i) || (cur_sync && prev_sync && !prev_has_tail_access));
        if (drop) {
          // Removing a barrier must remove the BARRIER, not the statement that
          // happens to contain it. When the sync sits inside wrapper nodes
          // (AttrStmt / SBlockRealize / SBlock), dropping the whole seq element
          // would discard the wrapper too — losing its attr_key, iter_values or
          // predicate, and any sibling statements it carries. StripSync returns
          // the element with just the barrier removed, or nullopt when nothing
          // but the barrier is left, in which case the element really can go.
          Optional<Stmt> stripped = s3_redundant_sync::StripSync(op->seq[i]);
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
        if (cur_sync && !s3_redundant_sync::IsSync(op->seq[i])) {
          SharedAccess tail_reads, tail_writes;
          CollectSyncDeepTailAccess(i, &tail_reads, &tail_writes);
          prev_has_tail_access = tail_reads.HasAny() || tail_writes.HasAny();
        }
        // Recurse first, then mark program-order shared activity so that
        // nested SeqStmts visited later (e.g. a lexical_alloc_scope body)
        // observe producers/consumers emitted earlier in this scope and
        // keep their own leading sync (LeadingBarrierSafe guard).
        out.push_back(VisitStmt(op->seq[i]));
        SharedAccess completed_reads, completed_writes;
        CollectSharedAccess(op->seq[i], &completed_reads, &completed_writes);
        seen_certified_writes_.insert(completed_writes.certified_writes.begin(),
                                      completed_writes.certified_writes.end());
        if (!seen_shared_before_ &&
            (completed_writes.HasAny() || completed_reads.HasAny())) {
          seen_shared_before_ = true;
        }
      }
      if (out.empty()) {
        restore();
        return Evaluate(0);
      }
      if (out.size() == 1) {
        restore();
        return out[0];
      }
      restore();
      return SeqStmt(out);
    }

    Stmt VisitStmt_(const SBlockRealizeNode *op) final {
      SBlock block = Downcast<SBlock>(VisitStmt(op->block));
      return SBlockRealize(op->iter_values, op->predicate, block);
    }
    Stmt VisitStmt_(const SBlockNode *op) final {
      // Defer to the base mutator, which visits op->body with this remover
      // (dispatching into VisitStmt_(SeqStmtNode) so the fixpoint runs). Do not
      // re-run RemoveCertifiedS3Syncs here with a fresh remover: that would
      // reset seen_shared_before_ / in_for_loop_, defeating the leading-sync
      // guard across the SBlock boundary and re-opening the same RAW race the
      // enclosing-scope guard was added to prevent.
      return StmtExprMutator::VisitStmt_(op);
    }
  };

  auto remover = S3RedundantSyncRemover(std::move(options));
  return remover(stmt);
}

} // namespace transform
} // namespace tang
} // namespace tl
} // namespace tvm
