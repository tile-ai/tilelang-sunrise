/*!
 *  \file lower_s3_shared_tmem.cc
 *  \brief TANG TMEM (Tensor Memory) lowering passes.
 *
 *  This file consolidates the two TMEM-related TIR passes that together
 *  manage the TMEM lifecycle on TANG (the tensor-core accumulator memory
 *  used by tcgen5 / MMA GEMM):
 *
 *    1. LowerSharedTmem    - Convert `shared.tmem` buffers to a tiny address
 *                            holder + (de)allocation calls, and translate
 *                            logical tmem[row, col] accesses into
 *                            `base_address + physical_offset`.  TANG-only.
 *
 *    2. LowerTangTmemDrain - Rewrite the TMEM->global drain loop (produced by
 *                            T.copy(C_tmem, C_global)) into the
 *                            tang_tmem_drain_16x256b_to_global intrinsic.
 *                            stcuv2-only.
 *
 *    3. ExpandS3TmemDrain    - Materialise that fused drain as its final TIR
 *                             loop nest after synchronization cleanup.
 *
 *  NOTE: these remain THREE SEPARATE passes (registered independently and
 *  scheduled at different points in tilelang/tang/pipeline.py).  They are
 *  co-located in one translation unit for cohesion; they CANNOT be fused into
 *  a single pass invocation because:
 *    - they run at different pipeline positions (LowerSharedTmem early, before
 *      the target-specific middle pipeline; LowerTangTmemDrain after it);
 *    - LowerTangTmemDrain is additionally gated to stcuv2;
 *    - LowerTangTmemDrain consumes the shrunk (1,)-uint32 address-holder buffer
 *      produced by LowerSharedTmem (a producer->consumer dependency, with the
 *      middle pipeline running in between).
 */
// The `T.deallocate_tmem` frontend marker (tl.deallocate_tmem) is declared by
// its owner, the CUDA registry: TMEM arrived there first and the op is still
// defined by src/cuda/op/builtin.cc. TANG lowering consumes the marker rather
// than producing it, so it does not belong in src/tang/op/builtin.h (whose ops
// are tl.tang_* and only ever produced by TANG lowering), and declaring it in
// the backend-neutral src/op/builtin.h would advertise a primitive that only
// the TMEM-capable subtargets -- sm100 and stcuv2 -- have. Take the declaration
// from its owner, the same way lower_shared_barrier.cc takes
// ptx_fence_barrier_init.
#include "cuda/op/builtin.h"
#include "op/builtin.h"
#include "tang/op/builtin.h"
#include "tang/target_utils.h"
#include "tir/transforms/ir_utils.h"
#include "tvm/ir/type.h"
#include "tvm/tirx/builtin.h"
#include "tvm/tirx/expr.h"
#include "tvm/tirx/stmt.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

// ===========================================================================
// Pass 1: LowerSharedTmem
//   Convert shared.tmem buffers to TANG address holders + tc init, and do
//   coordinate translation (from logical address to physical address).
// ===========================================================================

using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;

/*!
 * \brief Collect TMEM buffers explicitly deallocated on fallthrough paths.
 *
 * A "fallthrough path" is one that reaches the end of the statement without
 * hitting thread_return().  A buffer deallocated on such a path already has an
 * explicit dealloc, so the automatic one at block end must be dropped.
 *
 * Mirrors the CUDA pass's helper of the same name; kept separate because the
 * TANG side acts on the merged single reservation rather than per buffer.
 *
 * \return {buffers deallocated on fallthrough, whether the stmt can
 * fallthrough}
 */
static std::pair<VarSet, bool> CollectFallthroughDeallocs(const Stmt &stmt) {
  if (!stmt.defined())
    return {{}, true};

  // Unwrap transparent wrapper nodes
  if (stmt.as<BindNode>())
    return {{}, true};
  if (auto *n = stmt.as<AttrStmtNode>())
    return CollectFallthroughDeallocs(n->body);
  if (auto *n = stmt.as<SBlockNode>())
    return CollectFallthroughDeallocs(n->body);
  if (auto *n = stmt.as<SBlockRealizeNode>())
    return CollectFallthroughDeallocs(n->block->body);
  if (auto *n = stmt.as<ForNode>())
    return CollectFallthroughDeallocs(n->body);

  // Sequential: accumulate deallocs; stop if any child doesn't fallthrough
  if (auto *seq = stmt.as<SeqStmtNode>()) {
    VarSet deallocs;
    for (const auto &child : seq->seq) {
      auto [d, ft] = CollectFallthroughDeallocs(child);
      if (!ft)
        return {{}, false};
      deallocs.insert(d.begin(), d.end());
    }
    return {std::move(deallocs), true};
  }

  // Branch: collect deallocs only from branches that can fallthrough
  if (auto *iff = stmt.as<IfThenElseNode>()) {
    auto [then_d, then_ft] = CollectFallthroughDeallocs(iff->then_case);
    auto [else_d, else_ft] =
        iff->else_case.defined()
            ? CollectFallthroughDeallocs(iff->else_case.value())
            : std::pair<VarSet, bool>{{}, true};
    VarSet deallocs;
    if (then_ft)
      deallocs.insert(then_d.begin(), then_d.end());
    if (else_ft)
      deallocs.insert(else_d.begin(), else_d.end());
    return {std::move(deallocs), then_ft || else_ft};
  }

  // Leaf: detect deallocate_tmem and thread_return
  if (auto *eval = stmt.as<EvaluateNode>()) {
    if (auto *call = eval->value.as<CallNode>()) {
      if (call->op.same_as(tl::deallocate_tmem())) {
        ICHECK_EQ(call->args.size(), 1U);
        auto *buf = call->args[0].as<VarNode>();
        ICHECK(buf) << "tl.deallocate_tmem expects a buffer data Var";
        return {{tvm::ffi::GetRef<Var>(buf)}, true};
      }
      if (call->op.same_as(builtin::thread_return())) {
        return {{}, false};
      }
    }
  }

  return {{}, true};
}

class SharedTmemRewriter : public StmtExprMutator {
public:
  static Stmt Rewrite(Stmt body) {
    SharedTmemRewriter rewriter;
    return rewriter(body);
  }

private:
  Stmt VisitStmt_(const SBlockNode *op) final {
    SBlock block = tvm::ffi::GetRef<SBlock>(op);
    Array<Buffer> alloc_buffers = op->alloc_buffers;
    if (op->annotations.count(attr::kLayoutMap)) {
      auto layout_map = op->annotations.Get(attr::kLayoutMap);
      ICHECK(layout_map) << "layout map is not defined";
      layout_map_ = layout_map->as<Map<Buffer, Layout>>().value();
    }

    // Record the mapping from buffer data var to buffer for later lookup
    for (auto buffer : alloc_buffers) {
      buffer_map_.insert({buffer->data, buffer});
    }
    for (auto match_buffer : op->match_buffers) {
      buffer_map_.insert({match_buffer->buffer->data, match_buffer->buffer});
    }

    Array<Buffer> tmem_buffers;

    for (const auto &[data, buffer] : buffer_map_) {
      const auto *ptr_type =
          buffer->data->type_annotation.as<PointerTypeNode>();
      ICHECK(ptr_type) << "LowerSharedTmem requires buffer " << buffer->name
                       << "'s data Var to have a PointerType annotation";
      auto storage_scope = ptr_type->storage_scope;
      if (storage_scope == "shared.tmem") {
        tmem_buffers.push_back(buffer);
      }
    }

    if (tmem_buffers.empty()) {
      return StmtExprMutator::VisitStmt_(op);
    }

    for (auto buffer : tmem_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }

    /*
    Transform the tmem buffers to new allocations
    transform:
        tmem_buf0 = T.alloc_buffer((128, 128,), "uint64",
    scope="shared.tmem")
        tmem_buf1 = T.alloc_buffer((128, 128,), "uint64",
    scope="shared.tmem")

    into:
        tmem_buf0 = T.alloc_buffer((1,), "uint64", scope="shared.tmem_addr")
        tmem_buf1 = T.alloc_buffer((1,), "uint64", scope="shared.tmem_addr")

        if tx == 0:
          T.ptx_init_tensor_memory(tmem_buf0[0], 128)
          T.ptx_init_tensor_memory(tmem_buf1[0], 128)
    */
    // 1. create new data vars
    Array<Var> new_data_vars;
    for (auto buffer : tmem_buffers) {
      auto data = buffer->data;
      if (var_remap_.count(data))
        continue;
      auto new_data = Var(data->name_hint,
                          PointerType(PrimType(tmem_dtype_), "shared.tmem"));
      var_remap_.Set(data, new_data);
      new_data_vars.push_back(new_data);
    }

    // 2. create new buffers
    Array<Buffer> new_buffers;
    for (auto buffer : tmem_buffers) {
      auto data = buffer->data;
      ICHECK(var_remap_.find(data) != var_remap_.end())
          << "data not found in var_remap_";
      auto new_data = var_remap_.at(data);
      auto new_buffer = Buffer(new_data, tmem_dtype_, Array<PrimExpr>({1}),
                               Array<PrimExpr>({1}), PrimExpr(0), buffer->name,
                               buffer->data_alignment, buffer->offset_factor,
                               buffer->buffer_type);
      new_buffers.push_back(new_buffer);
      buffer_remap_.Set(buffer, new_buffer);
      buffer_data_to_buffer_.Set(new_data, new_buffer);
    }

    // remove the tmem buffers
    alloc_buffers.MutateByApply([this](Buffer buf) {
      if (buffer_remap_.find(buf) != buffer_remap_.end()) {
        return buffer_remap_.at(buf);
      }
      return buf;
    });
    if (!alloc_buffers.same_as(op->alloc_buffers)) {
      block.CopyOnWrite()->alloc_buffers = alloc_buffers;
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }

    // 3. Compute per-buffer column requirements (rounded up to a 16-column
    //    granularity for tmem subarray alignment).
    auto round_up = [](int v, int mult) {
      return ((v + mult - 1) / mult) * mult;
    };
    auto next_pow2_ge = [](int v) {
      int p = 32;
      for (; p < v; p *= 2)
        ;
      return p;
    };

    struct TmemRegion {
      Buffer old_buf;
      Buffer holder;
      int cols_round; // rounded up to 16 for placement
    };
    std::vector<TmemRegion> regions;
    for (auto buffer : tmem_buffers) {
      auto old_buffer = buffer_data_to_buffer_.at(buffer->data);
      auto new_buffer = buffer_remap_.at(old_buffer);
      ICHECK(old_buffer->shape.size() == 2);
      auto analyzer = std::make_shared<arith::Analyzer>();
      int cols_req = analyzer->const_int_bound(old_buffer->shape[1])->max_value;
      ICHECK(cols_req <= 512)
          << "The number of columns required for tmem buffer "
          << old_buffer->name << " is " << cols_req
          << ", which exceeds the maximum of 512 columns";
      regions.push_back({old_buffer, new_buffer, round_up(cols_req, 16)});
    }

    Array<Stmt> new_body;

    // stcuv2 permits one live TMEM reservation, so merge every shared.tmem
    // buffer into a single allocation and assign each a column offset.
    std::stable_sort(regions.begin(), regions.end(),
                     [](const TmemRegion &a, const TmemRegion &b) {
                       if (a.cols_round != b.cols_round)
                         return a.cols_round > b.cols_round;
                       return a.old_buf->name < b.old_buf->name;
                     });
    int total_cols = 0;
    std::vector<int> col_offsets;
    for (auto &r : regions) {
      col_offsets.push_back(total_cols);
      total_cols += r.cols_round;
    }
    int num_cols_allocated = next_pow2_ge(total_cols);

    auto region0_base = BufferLoad(regions[0].holder, {PrimExpr(0)});

    // Publish the merged reservation so that the frontend marker
    // `T.deallocate_tmem` (tl.deallocate_tmem) can be lowered onto it in
    // VisitExpr_(CallNode*) while walking this block's body.
    tmem_region_base_ = region0_base;
    tmem_num_cols_allocated_ = num_cols_allocated;
    tmem_region_ready_ = true;

    // stcuv2 keeps ONE merged reservation for every shared.tmem buffer in the
    // block, so an explicit dealloc of any single buffer releases the whole
    // region. The automatic tail dealloc below must therefore be dropped
    // entirely once the user has taken manual control, or the reservation
    // would be freed twice. (The CUDA path can suppress per buffer because
    // each buffer owns its own allocation there.)
    auto [explicit_deallocs, _fallthrough] =
        CollectFallthroughDeallocs(block->body);
    const bool has_explicit_dealloc = !explicit_deallocs.empty();
    new_body.push_back(
        Evaluate(Call(DataType::Handle(), tl::tang_init_tensor_memory(),
                      {region0_base, PrimExpr(num_cols_allocated)})));
    for (size_t i = 1; i < regions.size(); ++i) {
      new_body.push_back(
          BufferStore(regions[i].holder,
                      BufferLoad(regions[0].holder, {PrimExpr(0)}) +
                          IntImm(tmem_dtype_, col_offsets[i]),
                      {PrimExpr(0)}));
    }
    // No barrier after the reservation. On stcuv2 ``tc_alloc<N>(holder[0])`` is
    // issued by every thread and each of them writes the same base address into
    // that word; the derived holder stores above copy that value verbatim. A
    // later reader therefore reads a value it wrote itself, so there is no
    // cross-thread dependency here for a block rendezvous to order -- emitting
    // one only serialises the prologue. (The barrier before ``tc_dealloc``
    // below is a different case: it is a real collective-release rendezvous.)
    new_body.push_back(block->body);
    // Barrier before the collective tc_dealloc (correctness). On stcuv2
    // tc_dealloc is issued unconditionally by ALL threads (single-reservation
    // requirement) and is a plain per-thread instruction with no built-in
    // synchronization, but the TMEM drain epilogue (ldt/stt into the fragment)
    // runs on a single warp -- the other warps skip it. Without this sync the
    // non-draining warps could race ahead to tc_dealloc and release the
    // CTA-wide TMEM reservation while the draining warp is still reading it.
    // The golden reference (gver) synchronizes the same way -- it issues a
    // sync_wait(...) on all warps before tc_dealloc. The CUDA path instead
    // gates dealloc to warp 0 (the same warp that drains); on stcuv2 dealloc
    // must be collective, so a barrier is required instead.
    if (!has_explicit_dealloc) {
      new_body.push_back(
          Evaluate(Call(DataType::Handle(), builtin::tvm_storage_sync(),
                        {StringImm("shared")})));
      new_body.push_back(
          Evaluate(Call(DataType::Handle(), tl::tang_deallocate_tensor_memory(),
                        {region0_base, PrimExpr(num_cols_allocated)})));
    }

    auto block_ptr = block.CopyOnWrite();
    block_ptr->annotations.erase(attr::kLayoutMap);
    block_ptr->body = SeqStmt(new_body);

    return StmtExprMutator::VisitStmt_(block.get());
  }

  PrimExpr GetTmemOffset(const Buffer &buffer, const Array<PrimExpr> &indices) {
    ICHECK(buffer->shape.size() == 2);
    ICHECK(indices.size() == 2);
    ICHECK(layout_map_.defined());
    ICHECK(layout_map_.count(buffer))
        << "The layout of tmem buffer " << buffer->name
        << " is not defined in the layout map";
    auto layout = layout_map_[buffer];
    ICHECK(layout.defined());
    Array<PrimExpr> tmem_phy_coords = layout->Forward(indices);
    // Row in the high 16 bits, column in the low 16. This is TANG's own tensor
    // memory address format, not one inherited from CUDA: the cpt2s dim
    // descriptors in cccl/tang/__ptx/instructions/tc_cp.h use TMEM strides
    // 0x100000 for dim1 and 0x400000 for dim2, i.e. 16 << 16 and 64 << 16 --
    // stepping whole rows through the high half.
    PrimExpr result = tmem_phy_coords[0] << 16 | tmem_phy_coords[1];
    return result;
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    // Translate tmem[logical_row, logical_col] to tmem[0] + tmem_offset
    // Where
    // - (logical_row, logical_col) is the logical address in the tmem buffer
    // - tmem[0] is the base address allocated for the tmem buffer
    // - tmem_offset = tmem_phy_coords[0]<<16 | tmem_phy_coords[1]
    //   where tmem_phy_coords = layout.Forward(logical_row, logical_col)
    //   is the physical address in the tmem buffer
    auto load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    auto buffer = load->buffer;
    auto indices = load->indices;

    if (buffer_remap_.count(buffer)) {
      auto new_buffer = buffer_remap_[load->buffer];
      return BufferLoad(new_buffer, {0}) + GetTmemOffset(buffer, indices);
    } else if (var_remap_.count(buffer->data)) {
      auto new_buffer = Buffer(
          var_remap_[buffer->data], tmem_dtype_, buffer->shape, buffer->strides,
          buffer->elem_offset, buffer->name, buffer->data_alignment,
          buffer->offset_factor, buffer->buffer_type);
      return BufferLoad(new_buffer, {0}) + GetTmemOffset(buffer, indices);
    }
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    auto buffer = store->buffer;
    // Allow writes to the 1-element uint32 TMEM address holders that this pass
    // creates to carry (base + column-offset) for merged tmem regions; only
    // real (multi-element) matrix-data stores into tmem are forbidden.
    bool is_addr_holder =
        buffer.scope() == "shared.tmem" && buffer->shape.size() == 1;
    ICHECK(buffer.scope() != "shared.tmem" || is_addr_holder)
        << "We should never directly store data into tmem!";
    return store;
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    // Lower the frontend marker `T.deallocate_tmem(buf)` onto TANG's collective
    // dealloc. On stcuv2 all shared.tmem buffers of the block live in one
    // merged reservation, so there is no per-buffer release to emit: the call
    // frees the whole region (and the block-end auto dealloc was suppressed
    // for exactly that reason). The instruction is per-thread with no built-in
    // synchronization and the reservation is CTA-wide, so every thread must
    // reach this call site -- placing it under divergent control is the
    // caller's responsibility, same as on the CUDA path.
    if (op->op.same_as(tl::deallocate_tmem())) {
      ICHECK_EQ(op->args.size(), 1U);
      Var buffer_data = Downcast<Var>(op->args[0]);
      ICHECK(tmem_region_ready_)
          << "tl.deallocate_tmem requires a TMEM buffer allocated in the same "
             "or an enclosing block, but no shared.tmem allocation is in scope";
      ICHECK(var_remap_.count(buffer_data))
          << "tl.deallocate_tmem expects a TMEM buffer allocated in the same "
             "or an enclosing block; buffer "
          << buffer_data->name_hint << " is not tracked by LowerSharedTmem";
      return Call(DataType::Handle(), tl::tang_deallocate_tensor_memory(),
                  {tmem_region_base_, PrimExpr(tmem_num_cols_allocated_)});
    }
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      ICHECK_EQ(op->args.size(), 5U);
      Var buffer_data = Downcast<Var>(op->args[1]);
      if (!var_remap_.count(buffer_data)) {
        return StmtExprMutator::VisitExpr_(op);
      }
      Var new_data = var_remap_[buffer_data];
      return Call(
          op->dtype, op->op,
          {op->args[0], new_data, op->args[2], op->args[3], op->args[4]});
    }
    auto expr = StmtExprMutator::VisitExpr_(op);
    return expr;
  }
  PrimExpr VisitExpr_(const VarNode *op) final {
    Var var = tvm::ffi::GetRef<Var>(op);
    if (var_remap_.count(var)) {
      return var_remap_[var];
    }
    return var;
  }

  // Datatypes for tmem
  const DataType tmem_dtype_ = DataType::UInt(32);
  // The block's merged TMEM reservation, used to lower the frontend
  // `tl.deallocate_tmem` marker. stcuv2 allows a single live reservation, so
  // one (base, num_cols) pair describes the whole region.
  PrimExpr tmem_region_base_;
  int tmem_num_cols_allocated_ = 0;
  bool tmem_region_ready_ = false;
  Map<Var, Var> var_remap_;
  Map<Var, Buffer> buffer_data_to_buffer_;
  Map<Buffer, Buffer> buffer_remap_;
  // Mapping from data Var of a Buffer to Buffer, for lookup
  std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_map_;
  Map<Buffer, Layout> layout_map_;
};

PrimFunc LowerSharedTmem(PrimFunc f) {
  auto target_opt = f->GetAttr<Target>(tvm::attr::kTarget);
  ICHECK(target_opt.defined())
      << "LowerSharedTmem: Require the target attribute";
  Target target = target_opt.value();
  ICHECK(TargetIsTang(target)) << "LowerSharedTmem requires a TANG target";
  f.CopyOnWrite()->body = SharedTmemRewriter::Rewrite(f->body);
  return f;
}

// ===========================================================================
// Pass 2: LowerTangTmemDrain
//   Rewrite TMEM->global drain loops for TANG stcuv2.
//
//   After copy lowering, a T.copy(C_tmem, C) on TANG produces a normal
//   element-wise loop.  This pass detects such loops and replaces them with a
//   proper step-16 warp-collective drain loop using the
//   tang_tmem_ld_16x256b_x16 intrinsic -- as a single fused intrinsic, whose
//   expansion into the loop nest below is done later in this file by
//   ExpandS3TmemDrain.
//
//   The reference pattern (for BM x BN fp32 TMEM):
//     uint32_t _out[64];
//     for (int i = 0; i < BM; i += 16) {
//       if (threadIdx.x < 32) {
//         ldt_16x256b_x16(_out, C_tmem[0] + ((i << 16) | 0));
//         fence_ldt();
//         for (int _c = 0; _c < 16; ++_c) {
//           C[(i+tid/4)*N + 8*_c+(tid%4)*2]   = _out[4*_c];
//           C[(i+tid/4)*N + 8*_c+(tid%4)*2+1] = _out[4*_c+1];
//           C[(i+tid/4+8)*N + 8*_c+(tid%4)*2] = _out[4*_c+2];
//           C[(i+tid/4+8)*N + 8*_c+(tid%4)*2+1] = _out[4*_c+3];
//         }
//       }
//     }
// ===========================================================================

// Column granularity of the drain, shared with the detection pass above so the
// two cannot disagree about what a drainable width is: one ``ldt_16x256b`` atom
// is 8 fp32 columns (32 bytes) of a row and there is no narrower instruction,
// so a write-back region whose column extent is not a multiple of this cannot
// be drained at all.
constexpr int kLdtColsPerChunk = 8;

// Does this subtree store to global from a shared.tmem load, i.e. hold an
// un-rewritten TMEM->global drain?
static bool ContainsTmemToGlobalStore(const Stmt &stmt) {
  bool found = false;
  PostOrderVisit(stmt, [&](const ObjectRef &node) {
    if (found)
      return;
    const auto *store = node.as<BufferStoreNode>();
    if (store == nullptr || store->buffer.scope() != "global")
      return;
    PostOrderVisit(store->value, [&](const ObjectRef &v) {
      if (const auto *bl = v.as<BufferLoadNode>()) {
        if (GetPtrStorageScope(bl->buffer->data).find("shared.tmem") == 0)
          found = true;
      }
    });
  });
  return found;
}

class TangTmemDrainRewriter : public StmtExprMutator {
public:
  static Stmt Rewrite(Stmt body) {
    TangTmemDrainRewriter rewriter;
    Stmt out = rewriter(body);
    // The rewrite is now keyed off the "tmem_drain" marker that copy lowering
    // attaches. If a drain ever reaches this pass without one, the elementwise
    // loop would survive and read the shrunk (1,)-uint32 address holder as if
    // it were the accumulator -- wrong, and quiet. Fail instead.
    ICHECK(!ContainsTmemToGlobalStore(out))
        << "LowerTangTmemDrain: a TMEM->global store survived the rewrite. It "
           "was not wrapped in a \"tmem_drain\" marker, so it did not come "
           "from the TMEM->global branch of the TANG copy lowering.";
    return out;
  }

private:
  // Whether we are inside the "tmem_drain" marker that copy lowering wraps
  // around a TMEM->global drain. Only loops under it are drain candidates.
  bool in_tmem_drain_ = false;
  // Element type of the TMEM accumulator being drained, carried on that same
  // marker (LowerSharedTmem has already erased it from the buffer by now).
  DataType accum_dtype_ = DataType::Float(32);
  // Cache to avoid re-visiting already-processed nodes
  std::unordered_map<const StmtNode *, Stmt> processed_;
  // threadIdx.* iteration vars seen on the way down (intra-tile coordinates).
  // blockIdx.* are deliberately NOT tracked: they parameterize *which* tile a
  // block writes and must survive into the per-block base offset.
  std::unordered_set<const VarNode *> thread_vars_;
  // Per-thread-var iteration extent (threadIdx.{x,y,z}), for bounding the drain
  // store's row index over the full thread range when recovering BM.
  std::unordered_map<const VarNode *, PrimExpr> thread_extents_;
  // Block threadIdx.x extent (total threads). Used to derive how many warps
  // cooperatively drain the TMEM accumulator. 0 => not yet seen.
  int block_threads_x_ = 0;
  // Nesting depth of `warp_specialize` (T.ws) attr regions. A fused
  // TMEM->global drain (`T.copy(C_tmem, C)`) reconstructs its warp-collective
  // ldt geometry (rows-per-strip, per-lane store pattern) from the loop shape
  // of a *128-thread* copy. Inside a T.ws(...,32) region the copy is instead
  // laid out over 32 threads, so the loop collapses (e.g. 128x128 -> extent
  // 512) and that geometry no longer holds -- the drain cannot be reconstructed
  // there. We detect it and fail loudly (see the For handler), pointing at the
  // tcgen05_ld idiom which drains correctly under warp specialization.
  int in_warp_specialize_ = 0;
  // User `T.copy(..., drain_warps=N)` cap for the enclosing drain (0 => unset,
  // meaning drain across all block warps). Carried via a "tmem_drain_warps"
  // AttrStmt wrapping the drain.
  int drain_warps_attr_ = 0;

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      auto iv = Downcast<IterVar>(op->node);
      if (std::string(iv->thread_tag).compare(0, 9, "threadIdx") == 0) {
        thread_vars_.insert(iv->var.get());
        // Remember each thread var's extent so the drain can bound the store
        // row index over the full [0, extent) thread range when recovering BM.
        thread_extents_[iv->var.get()] = op->value;
      }
      // Record the block's threadIdx.x extent so the drain can be spread across
      // all warps of the block (multi-warp writeback). Blocks are launched with
      // a static thread count on stcuv2, so this is a compile-time constant.
      if (iv->thread_tag == "threadIdx.x") {
        if (const auto *ext = op->value.as<IntImmNode>()) {
          block_threads_x_ = static_cast<int>(ext->value);
        }
      }
    }
    if (op->attr_key == "warp_specialize") {
      in_warp_specialize_++;
      Stmt body = StmtExprMutator::VisitStmt_(op);
      in_warp_specialize_--;
      return body;
    }
    // A user `T.copy(..., drain_warps=N)` cap is carried on a durable AttrStmt
    // wrapping the drain (loop annotations don't survive copy lowering). Scope
    // it to the wrapped subtree.
    if (op->attr_key == "tmem_drain") {
      bool saved = in_tmem_drain_;
      DataType saved_dtype = accum_dtype_;
      in_tmem_drain_ = true;
      accum_dtype_ = op->value.dtype();
      Stmt body = StmtExprMutator::VisitStmt_(op);
      in_tmem_drain_ = saved;
      accum_dtype_ = saved_dtype;
      // The marker has done its job; drop it so it doesn't reach codegen.
      if (const auto *attr = body.as<AttrStmtNode>();
          attr && attr->attr_key == "tmem_drain")
        return attr->body;
      return body;
    }
    if (op->attr_key == "tmem_drain_warps") {
      int saved = drain_warps_attr_;
      if (const auto *imm = op->value.as<IntImmNode>())
        drain_warps_attr_ = static_cast<int>(imm->value);
      Stmt body = StmtExprMutator::VisitStmt_(op);
      drain_warps_attr_ = saved;
      // The attr has done its job (carrying the hint); drop the wrapper so it
      // doesn't leak into codegen.
      if (const auto *attr = body.as<AttrStmtNode>();
          attr && attr->attr_key == "tmem_drain_warps")
        return attr->body;
      return body;
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    // Check cache
    auto it = processed_.find(op);
    if (it != processed_.end())
      return it->second;

    // Only loops under the "tmem_drain" marker are candidates. Without this
    // gate any enclosing user loop also matches the store pattern below, and
    // replacing it would discard its other statements -- see the marker's
    // rationale in the TMEM->global branch of tang/op/copy.cc.
    if (!in_tmem_drain_) {
      Stmt result = StmtExprMutator::VisitStmt_(op);
      processed_[op] = result;
      return result;
    }

    // Look for a for-loop whose body stores from a shared.tmem buffer
    // to a global buffer (the TMEM->global drain pattern).
    bool is_tmem_drain = false;
    const BufferStoreNode *tmem_store = nullptr;
    Buffer global_buf;
    Buffer tmem_buf;

    PostOrderVisit(op->body, [&](const ObjectRef &node) {
      if (is_tmem_drain)
        return;
      if (auto *store = node.as<BufferStoreNode>()) {
        if (store->buffer.scope() != "global")
          return;
        PostOrderVisit(store->value, [&](const ObjectRef &v) {
          if (auto *bl = v.as<BufferLoadNode>()) {
            std::string scope = GetPtrStorageScope(bl->buffer->data);
            if (scope.find("shared.tmem") == 0) {
              is_tmem_drain = true;
              tmem_store = store;
              global_buf = store->buffer;
              tmem_buf = bl->buffer;
            }
          }
        });
      }
    });

    if (!is_tmem_drain || !tmem_store || !global_buf.defined()) {
      Stmt result = StmtExprMutator::VisitStmt_(op);
      processed_[op] = result;
      return result;
    }

    // The fused drain's ldt geometry is derived from a 128-thread copy loop;
    // inside a T.ws(...,32) region the copy collapses to 32 threads and this
    // reconstruction breaks (wrong BM / base offset). Fail loudly instead of
    // emitting a garbage drain, and steer to the warp-specialization-safe
    // idiom.
    ICHECK_EQ(in_warp_specialize_, 0)
        << "A fused TMEM->global drain (`T.copy(C_tmem, C_global)`) cannot be "
           "placed inside a warp-specialized region (`T.ws(...)`): under T.ws "
           "the "
           "epilogue copy is laid out over a single 32-thread warp, so the "
           "fused "
           "ldt drain (whose row-strip / lane geometry assumes a 128-thread "
           "copy) "
           "cannot be reconstructed. For a single/specific-warp writeback use "
           "the "
           "explicit warp-specialized idiom instead:\n"
           "    with T.ws(0, warp_group_size=32):\n"
           "        T.tcgen05_sync_arrive(bid)\n"
           "    with T.ws(drain_warp, warp_group_size=32):\n"
           "        T.tcgen05_sync_wait(bid, 1, 1)\n"
           "        T.tcgen05_ld(C_frag, C_tmem)\n"
           "        T.copy(C_frag, C_global)\n"
           "For an all-warp writeback keep `T.copy(C_tmem, C_global)` at the "
           "top "
           "level (outside any T.ws region).";

    // User cap on the number of writeback warps (`T.copy(..., drain_warps=N)`),
    // carried on the enclosing "tmem_drain_warps" AttrStmt. 0 => unset (drain
    // across all block warps).
    int drain_warps_hint = drain_warps_attr_;

    // Get dimensions from the loop extent (BM = rows) and global buffer.
    // NOTE: BN here is the global row STRIDE (the full output N), used for the
    // per-row address arithmetic in the drain; it may exceed the per-block tile
    // width for a grid that splits N. The number of accumulator columns
    // actually drained per row is fixed by the codegen (the 128-wide TMEM
    // tile), not by BN. (A per-block tile narrower than 128 columns -- e.g.
    // N=64 in a single block -- is a separate, currently-unsupported drain
    // configuration.) Neither the tile row count (BM) nor the per-block tile
    // column count (N_tile) may be read from op->extent: the pre-drain copy is
    // a SIMT elementwise loop whose trip count depends on the block thread
    // count (128 rows over 128 threads -> extent 128; over 256 threads the copy
    // packs 2 rows per step -> extent 64), so op->extent under-counts the rows
    // for any block wider than 128 threads. Instead recover each tile extent
    // from the span of the store index: bind the copy's loop / nested-loop /
    // thread vars to their iteration extents, zero the spatial (block) vars,
    // and take the resulting [0, extent) span of that index coordinate.
    std::vector<std::pair<Var, PrimExpr>> loop_vars;
    loop_vars.emplace_back(op->loop_var, op->extent);
    PostOrderVisit(op->body, [&](const ObjectRef &node) {
      if (const auto *f = node.as<ForNode>())
        loop_vars.emplace_back(f->loop_var, f->extent);
    });
    std::unordered_set<const VarNode *> keep;
    for (const auto &lv : loop_vars)
      keep.insert(lv.first.get());
    for (const auto *tv : thread_vars_)
      keep.insert(tv);
    auto intra_extent = [&](const PrimExpr &index) -> int {
      Map<Var, PrimExpr> zero_spatial;
      PostOrderVisit(index, [&](const ObjectRef &node) {
        if (const auto *v = node.as<VarNode>())
          if (!keep.count(v))
            zero_spatial.Set(tvm::ffi::GetRef<Var>(v), make_zero(v->dtype));
      });
      PrimExpr intra = Substitute(index, zero_spatial);
      arith::Analyzer a;
      for (const auto &lv : loop_vars)
        a.Bind(lv.first,
               Range::FromMinExtent(make_zero(lv.first->dtype), lv.second));
      for (const auto *tv : thread_vars_) {
        auto it = thread_extents_.find(tv);
        if (it != thread_extents_.end())
          a.Bind(tvm::ffi::GetRef<Var>(tv),
                 Range::FromMinExtent(make_zero(tv->dtype), it->second));
      }
      auto bound = a.const_int_bound(intra);
      if (bound->max_value != arith::ConstIntBound::kPosInf &&
          bound->max_value >= 0)
        return static_cast<int>(bound->max_value) + 1;
      return 0;
    };

    // The drain writes a 2D (rows x cols) tile, so its row and column
    // coordinates are the LAST two store indices. A destination of rank > 2 (a
    // batched output indexed as D[b, i, j]) carries leading slice-origin
    // indices; reading indices[0] there picks up the batch index, whose span is
    // 1, which would collapse the drain to a single row and leave the rest of
    // the tile unwritten.
    const size_t nidx = tmem_store->indices.size();
    ICHECK_GE(nidx, 1u) << "LowerTangTmemDrain: global store has no indices";
    const size_t row_pos = nidx >= 2 ? nidx - 2 : 0;

    int BM = intra_extent(tmem_store->indices[row_pos]);
    if (BM <= 0) {
      const auto *ext = op->extent.as<IntImmNode>();
      BM = ext ? static_cast<int>(ext->value) : 128;
      LOG(WARNING)
          << "LowerTangTmemDrain: could not bound the store row index; "
             "falling back to (thread-count-dependent) loop extent BM="
          << BM;
    }
    // Per-block accumulator column count. The drain writes ldt_16x256b chunks
    // of 128 fp32 columns each; a tile wider than 128 (e.g. a single block with
    // N=256) needs the codegen to loop over 128-column blocks, so pass N_tile
    // separately from BN (the global row stride). Defaults to 128 when it
    // cannot be bounded (the historical single-chunk assumption).
    int N_tile = 128;
    if (nidx >= 2) {
      int cols = intra_extent(tmem_store->indices[nidx - 1]);
      if (cols > 0) {
        N_tile = cols;
        // A width the drain has no instruction for is a property of the
        // write-back region the user wrote, so report it here, where the
        // destination still has a name, rather than leaving the expansion's
        // internal check to fire.
        if (N_tile % kLdtColsPerChunk != 0) {
          LOG(FATAL)
              << "TANG STCUV2: the promoted accumulator is written back to '"
              << global_buf->name << "' over " << N_tile << " columns, which "
              << "is not a multiple of " << kLdtColsPerChunk << ".\n"
              << "  The tensor-memory drain reads the accumulator in "
              << kLdtColsPerChunk
              << "-column units (a 32-byte LDT atom per row) and there is no "
                 "narrower instruction to finish a remainder with, so the "
                 "write-back region's column extent must be a multiple of "
              << kLdtColsPerChunk
              << ": [:, 0:96] is drainable, [:, 0:100] is "
                 "not.\n"
              << "  Copy the whole accumulator, or slice on a multiple of "
              << kLdtColsPerChunk << " columns.";
        }
      }
    }
    int BN = 128; // fallback
    if (global_buf->shape.size() >= 2) {
      // Innermost dimension: the flat distance between two tile rows. Reading
      // shape[1] instead would be the batch stride at rank 3.
      BN = Downcast<IntImm>(global_buf->shape[global_buf->shape.size() - 1])
               ->value;
    } else if (global_buf->shape.size() == 1) {
      // Flattened buffer: shape[0] = BM * BN
      BN = Downcast<IntImm>(global_buf->shape[0])->value / BM;
    }
    // Per-block base offset into the (whole) global buffer. The generated drain
    // loop indexes the output as global_ptr[(i+tid/4)*BN + ...], i.e. relative
    // to the start of *this block's* 128xBN tile. For a multi-block grid the
    // tile start is C[by*BM, bx*128], so we must hand the intrinsic a pointer
    // already shifted by that per-block offset; otherwise every block writes
    // the (0,0) tile. Recover it by zeroing every non-blockIdx var in the store
    // index (the loop var / threadIdx / inner-loop vars all parameterize the
    // intra-tile position and vanish at the tile origin).
    PrimExpr flat_index = tmem_store->indices[0];
    for (size_t d = 1; d < tmem_store->indices.size(); ++d) {
      flat_index = flat_index * global_buf->shape[d] + tmem_store->indices[d];
    }
    // Intra-tile coordinate vars vanish at the tile origin: the drain loop var,
    // any loop vars nested in its body, and threadIdx.*. Everything left (the
    // block / spatial vars, whatever their name) defines the per-block base.
    std::unordered_set<const VarNode *> intra_vars = thread_vars_;
    intra_vars.insert(op->loop_var.get());
    PostOrderVisit(op->body, [&](const ObjectRef &node) {
      if (const auto *f = node.as<ForNode>())
        intra_vars.insert(f->loop_var.get());
    });
    Map<Var, PrimExpr> zero_intra;
    PostOrderVisit(flat_index, [&](const ObjectRef &node) {
      if (const auto *v = node.as<VarNode>()) {
        if (intra_vars.count(v)) {
          zero_intra.Set(tvm::ffi::GetRef<Var>(v), make_zero(v->dtype));
        }
      }
    });
    arith::Analyzer ana;
    PrimExpr base_offset = ana.Simplify(Substitute(flat_index, zero_intra));

    // Per-instance progress: one line per drain in a kernel, so it stays off
    // the default log. A debug build with TVM_LOG_DEBUG, or
    // TVM_LOG_DEBUG=lower_s3_shared_tmem.cc:1, turns both this and the
    // expansion's counterpart back on.
    VLOG(1) << "LowerTangTmemDrain: detected TMEM drain loop!"
            << " BM=" << BM << " BN=" << BN << " N_tile=" << N_tile
            << " global_shape=" << global_buf->shape
            << " tmem_shape=" << tmem_buf->shape
            << " base_offset=" << base_offset;

    // -- Replace with the tang_tmem_drain_16x256b_to_global intrinsic --
    // Pass C_tmem[0] (value, not pointer) as the TMEM base address
    PrimExpr tmem_base = BufferLoad(tmem_buf, {IntImm(DataType::Int(32), 0)});
    // The 5th arg carries the ACCUMULATOR's element type (via a typed zero) so
    // codegen knows how to read back each 32-bit word ldt returns: fp32 and
    // int32 accumulators fill the word, while a 16-bit accumulator occupies its
    // low half. Narrowing to the C buffer's own type then happens in the store;
    // using the output type here would corrupt 16-bit accumulators.
    // The BM/16 row-strips are round-robin distributed across `nwarps` warps
    // (matches the golden reference's multi-drain-warp epilogue). The intrinsic
    // emits its own block-wide __syncthreads (all threads reach it -- the drain
    // is at block scope, not inside a T.ws) to publish warp0's async MMA, then
    // only the first `nwarps` warps run the ldt writeback. By default every
    // warp of the block drains; a user `drain_warps=N` caps it (e.g. 1 => a
    // single warp writes back while the others still hit the barrier),
    // decoupling the writeback from the copy/MMA warps. Fall back to 4 warps
    // (standard 128-thread stcuv2 GEMM block) if the extent could not be read.
    int block_warps = (block_threads_x_ > 0) ? (block_threads_x_ / 32) : 4;
    if (block_warps < 1)
      block_warps = 1;
    int nwarps = block_warps;
    if (drain_warps_hint > 0) {
      nwarps = std::min(drain_warps_hint, block_warps);
    }
    if (nwarps < 1)
      nwarps = 1;
    // Rows this block actually owns: the destination's row extent minus the
    // tile's row origin. A dynamic row bound (`num_tokens`) is not a multiple
    // of the tile, so the last tile is partial, and the drain -- unlike the
    // shared -> global copy path, which predicates itself -- writes raw buffer
    // stores. Without the count it writes all BM rows and clobbers whatever
    // follows the tensor (the next batch / split / row tile), which is a silent
    // and racy corruption. Only passed when the tile is not provably inside the
    // destination, so in-bounds kernels keep their exact code.
    Array<PrimExpr> drain_args{
        tmem_base,
        global_buf.access_ptr(2, DataType::Handle(), 1, base_offset),
        IntImm(DataType::Int(32), BM),
        IntImm(DataType::Int(32), BN),
        make_zero(accum_dtype_),
        IntImm(DataType::Int(32), nwarps),
        IntImm(DataType::Int(32), N_tile)};
    if (global_buf->shape.size() >= 2 && nidx >= 2) {
      PrimExpr row_origin =
          ana.Simplify(Substitute(tmem_store->indices[row_pos], zero_intra));
      PrimExpr row_extent = global_buf->shape[global_buf->shape.size() - 2];
      PrimExpr rows_owned = ana.Simplify(row_extent - row_origin);
      if (!ana.CanProve(rows_owned >= BM)) {
        drain_args.push_back(rows_owned);
      }
    }
    Stmt drain_call = Evaluate(Call(
        DataType::Handle(), tang_tmem_drain_16x256b_to_global(), drain_args));

    processed_[op] = drain_call;
    return drain_call;
  }
};

// ===========================================================================
// Pass 3: ExpandS3TmemDrain
//   Materialize the fused drain as ordinary TIR after synchronization cleanup.
// ===========================================================================

namespace {

constexpr int kLdtRows = 16;
// One ``ldt_16x256b_x{N}`` call reads 16 rows x (8*N) fp32 columns into 4*N
// registers per lane. The toolchain only provides N = 1, 2, 4, 8, 16, 32, so a
// tile that is not a power-of-two number of 8-column units is drained with
// several calls rather than one over-wide call.
// Widest LDT this pass emits; also the chunk width used for wide tiles.
constexpr int kLdtMaxChunks = 16;
constexpr int kLdtChunkCols = kLdtColsPerChunk * kLdtMaxChunks; // 128 columns
constexpr int kWarpSize = 32;

inline PrimExpr I32(int64_t value) { return IntImm(DataType::Int(32), value); }

class TangTmemDrainExpander : public StmtExprMutator {
public:
  static Stmt Expand(const Stmt &body) {
    TangTmemDrainExpander expander;
    expander.CollectUsedNames(body);
    return expander(body);
  }

private:
  Stmt VisitStmt_(const ForNode *op) final {
    ++loop_depth_;
    Stmt body = VisitStmt(op->body);
    --loop_depth_;
    For node = GetRef<For>(op);
    node.CopyOnWrite()->body = body;
    return Stmt(node);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        thread_x_.push_back(iv->var);
        Stmt body = StmtExprMutator::VisitStmt_(op);
        thread_x_.pop_back();
        return body;
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    const auto *eval = stmt.as<EvaluateNode>();
    if (eval == nullptr) {
      return stmt;
    }
    const auto *call = eval->value.as<CallNode>();
    if (call == nullptr ||
        !call->op.same_as(tang_tmem_drain_16x256b_to_global())) {
      return stmt;
    }
    return ExpandDrain(call);
  }

  void CollectUsedNames(const Stmt &body) {
    PostOrderVisit(body, [this](const ObjectRef &node) {
      if (const auto *load = node.as<BufferLoadNode>()) {
        used_names_.insert(load->buffer->data->name_hint);
      } else if (const auto *store = node.as<BufferStoreNode>()) {
        used_names_.insert(store->buffer->data->name_hint);
      } else if (const auto *decl = node.as<DeclBufferNode>()) {
        used_names_.insert(decl->buffer->data->name_hint);
      } else if (const auto *call = node.as<CallNode>()) {
        if (call->op.same_as(builtin::tvm_access_ptr()) &&
            call->args.size() >= 2) {
          if (const auto *var = call->args[1].as<VarNode>()) {
            used_names_.insert(var->name_hint);
          }
        }
      }
    });
  }

  std::string FreshOutName() {
    std::string base = "_out";
    if (!used_names_.count(base)) {
      return base;
    }
    for (int i = 1;; ++i) {
      std::string candidate = base + "_" + std::to_string(i);
      if (!used_names_.count(candidate)) {
        return candidate;
      }
    }
  }

  static Buffer GlobalBufferOf(const CallNode *access, PrimExpr *base_offset,
                               DataType *elem_dtype) {
    ICHECK(access->args.size() >= 4U)
        << "tvm_access_ptr expects 5 arguments, got " << access->args.size();
    const auto *annotation = access->args[0].as<CallNode>();
    const auto *annotation_op =
        annotation != nullptr ? annotation->op.as<OpNode>() : nullptr;
    ICHECK(annotation_op != nullptr &&
           annotation_op->name == "tirx.type_annotation")
        << "tvm_access_ptr arg0 must be a TypeAnnotation call";
    *elem_dtype = access->args[0].dtype();
    const Var &data = Downcast<Var>(access->args[1]);
    *base_offset = access->args[2];
    PrimExpr extent = access->args[3];
    return Buffer(data, *elem_dtype, {extent}, /*strides=*/{}, I32(0),
                  data->name_hint, /*data_alignment=*/0, /*offset_factor=*/0,
                  BufferType::kDefault);
  }

  Stmt ExpandDrain(const CallNode *call) {
    ICHECK(call->args.size() == 6U || call->args.size() == 7U ||
           call->args.size() == 8U)
        << "tang_tmem_drain_16x256b_to_global expects 6, 7 or 8 arguments, got "
        << call->args.size();
    ICHECK(!thread_x_.empty())
        << "ExpandS3TmemDrain: a TMEM->global drain outside any threadIdx.x "
           "extent cannot be expanded";
    const Var &thread_x = thread_x_.back();

    const PrimExpr &tmem_base = call->args[0];
    const auto *access = call->args[1].as<CallNode>();
    ICHECK(access != nullptr && access->op.same_as(builtin::tvm_access_ptr()))
        << "ExpandS3TmemDrain: drain arg1 must be a tvm_access_ptr";
    PrimExpr base_offset;
    DataType elem_dtype;
    Buffer global = GlobalBufferOf(access, &base_offset, &elem_dtype);

    const auto *bm_imm = call->args[2].as<IntImmNode>();
    const auto *bn_imm = call->args[3].as<IntImmNode>();
    const auto *nwarps_imm = call->args[5].as<IntImmNode>();
    ICHECK(bm_imm != nullptr && bn_imm != nullptr && nwarps_imm != nullptr)
        << "ExpandS3TmemDrain: the drain's BM/BN/nwarps must be constants";
    const int bm = static_cast<int>(bm_imm->value);
    const int bn = static_cast<int>(bn_imm->value);
    int nwarps = static_cast<int>(nwarps_imm->value);
    if (nwarps < 1) {
      nwarps = 1;
    }
    const DataType accum_dtype = call->args[4].dtype();
    int n_tile = kLdtChunkCols;
    // `>= 7`, not `== 7`: the rows-this-tile-owns argument below appends to the
    // call, and an exact-count test silently fell back to the 128-column
    // default -- one ldt_16x256b_x16 reading past the accumulator's columns.
    if (call->args.size() >= 7U) {
      const auto *n_tile_imm = call->args[6].as<IntImmNode>();
      ICHECK(n_tile_imm != nullptr)
          << "ExpandS3TmemDrain: the drain's N_tile must be a constant";
      n_tile = static_cast<int>(n_tile_imm->value);
    }
    // Rows the tile owns at this block's origin, when the detection pass could
    // not prove the whole tile fits; absent otherwise (an in-bounds drain emits
    // exactly the stores it always did).
    PrimExpr rows_owned;
    if (call->args.size() == 8U) {
      rows_owned = call->args[7];
    }
    // A tile narrower than one 128-column LDT must NOT be widened to it: that
    // call would read columns past the accumulator's tensor-memory allocation
    // and its stores would walk past the tile. Drain the real column span as a
    // sequence of descending power-of-two LDT widths instead.
    if (n_tile <= 0) {
      n_tile = kLdtChunkCols;
    }
    // Backstop: LowerTangTmemDrain rejects an undrainable width with the
    // destination's name, so reaching this means some other path built the
    // intrinsic. Keep the message user-facing anyway.
    if (n_tile % kLdtColsPerChunk != 0) {
      LOG(FATAL)
          << "TANG STCUV2: this block's accumulator tile is drained over "
          << n_tile << " columns, which is not a multiple of "
          << kLdtColsPerChunk << ".\n"
          << "  The drain reads the accumulator in " << kLdtColsPerChunk
          << "-column units (a 32-byte LDT atom per row) and there is no "
             "narrower instruction to finish a remainder with. Pad the "
             "accumulator, or slice the write-back on a multiple of "
          << kLdtColsPerChunk << " columns.";
    }
    std::vector<int> chunk_steps; // 8-column units loaded by each LDT call
    for (int remaining = n_tile / kLdtColsPerChunk; remaining > 0;) {
      int step = 1;
      while (step * 2 <= remaining && step * 2 <= kLdtMaxChunks)
        step *= 2;
      chunk_steps.push_back(step);
      remaining -= step;
    }
    const int max_chunks = chunk_steps.front();
    {
      std::string steps;
      for (size_t i = 0; i < chunk_steps.size(); ++i)
        steps += (i ? "x" : "") + std::to_string(chunk_steps[i]);
      VLOG(1) << "ExpandS3TmemDrain: N_tile=" << n_tile
              << " ldt_steps=" << steps << " (x" << kLdtColsPerChunk
              << " columns each)";
    }

    // The per-step address below is ``row << 16 | column`` (see the note on the
    // TMEM address format above TangTmemDrainRewriter), so the column field is
    // 16 bits wide and must not carry into the row half.
    ICHECK_LT(n_tile, 1 << 16)
        << "ExpandS3TmemDrain: the drained tile is " << n_tile
        << " columns wide, which does not fit the 16-bit column field of a "
           "tensor-memory address.";

    const std::string out_name = FreshOutName();
    Var out_data(out_name, PointerType(PrimType(DataType::UInt(32)), "local"));
    Buffer out_buf(out_data, DataType::UInt(32), {I32(4 * max_chunks)},
                   /*strides=*/{}, I32(0), out_name, /*data_alignment=*/0,
                   /*offset_factor=*/0, BufferType::kDefault);

    PrimExpr warp = floordiv(thread_x, I32(kWarpSize));
    PrimExpr lane = floormod(thread_x, I32(kWarpSize));

    Var k("k", DataType::Int(32));
    PrimExpr r_min = warp * I32(kLdtRows);
    PrimExpr r_step = I32(nwarps * kLdtRows);
    PrimExpr r_trip = floordiv(I32(bm) - r_min + r_step - I32(1), r_step);
    r_trip = max(r_trip, I32(0));
    PrimExpr r = r_min + k * r_step;

    PrimExpr row_lo = r + floordiv(lane, I32(4));
    PrimExpr row_hi = row_lo + I32(8);
    PrimExpr bn_expr = I32(bn);
    PrimExpr col_lo = floormod(lane, I32(4)) * I32(2);
    auto element = [&](const Var &c, int64_t word) {
      PrimExpr value = BufferLoad(out_buf, {c * I32(4) + I32(word)});
      ICHECK(accum_dtype.bits() == 32 || accum_dtype.bits() == 16)
          << "ExpandS3TmemDrain: a " << accum_dtype.bits()
          << "-bit TMEM accumulator is not supported by the drain";
      if (accum_dtype.bits() < 32) {
        value = Cast(DataType::UInt(16), value);
      }
      value = reinterpret(accum_dtype, value);
      if (!(value.dtype() == elem_dtype)) {
        value = Cast(elem_dtype, value);
      }
      return value;
    };
    auto store = [&](const Var &c, const PrimExpr &col_base,
                     const PrimExpr &row, const PrimExpr &col, int64_t word) {
      PrimExpr index =
          base_offset + row * bn_expr + col_base + c * I32(8) + col;
      return BufferStore(global, element(c, word), {index});
    };

    // One LDT plus its store loop per column step. ``c`` counts the 8-column
    // chunks inside the step, so the per-lane word offset stays c*4 + word and
    // the column offset advances by the step's width.
    Array<Stmt> drain_steps;
    PrimExpr col_base = I32(0);
    for (int step : chunk_steps) {
      PrimExpr taddr = tmem_base + ((r << I32(16)) | col_base);
      drain_steps.push_back(
          Evaluate(Call(DataType::Handle(), tang_tmem_ld_16x256b(),
                        {out_buf->data, taddr, I32(step)})));
      // A fresh loop var per step: sibling copies must not share one, or bound
      // inference sees a single var with two extents.
      Var c("c", DataType::Int(32));
      Stmt strip =
          For(c, I32(0), I32(step), ForKind::kSerial,
              SeqStmt({store(c, col_base, row_lo, col_lo, 0),
                       store(c, col_base, row_lo, col_lo + I32(1), 1),
                       store(c, col_base, row_hi, col_lo, 2),
                       store(c, col_base, row_hi, col_lo + I32(1), 3)}));
      if (rows_owned.defined()) {
        // One compare per 16-row strip, not one per store: the upper half is
        // writable exactly when row_hi is, and a boundary strip -- row_hi out
        // but row_lo in -- takes the narrow branch with its two stores.
        Var c_tail("c_tail", DataType::Int(32));
        Stmt lower_half =
            For(c_tail, I32(0), I32(step), ForKind::kSerial,
                SeqStmt({store(c_tail, col_base, row_lo, col_lo, 0),
                         store(c_tail, col_base, row_lo, col_lo + I32(1), 1)}));
        strip = IfThenElse(row_hi < rows_owned, strip,
                           IfThenElse(row_lo < rows_owned, lower_half, Stmt()));
      }
      drain_steps.push_back(strip);
      col_base = col_base + I32(step * kLdtColsPerChunk);
    }
    Stmt strip_loop =
        For(k, I32(0), r_trip, ForKind::kSerial, SeqStmt(drain_steps));
    Stmt warp_guard = IfThenElse(warp < I32(nwarps), strip_loop, Stmt());
    Stmt out_decl = SeqStmt(
        {AllocBuffer(out_buf), DeclBuffer(out_buf), std::move(warp_guard)});

    if (nwarps > 1) {
      Array<Stmt> sequence{
          Evaluate(Call(DataType::Handle(), builtin::tvm_storage_sync(),
                        {StringImm("shared")})),
          std::move(out_decl)};
      if (loop_depth_ > 0) {
        sequence.push_back(
            Evaluate(Call(DataType::Handle(), builtin::tvm_storage_sync(),
                          {StringImm("shared")})));
      }
      return SeqStmt(sequence);
    }
    return out_decl;
  }

  std::vector<Var> thread_x_;
  std::unordered_set<std::string> used_names_;
  int loop_depth_{0};
};

PrimFunc ExpandS3TmemDrainPrimFunc(PrimFunc f) {
  auto target_opt = f->GetAttr<Target>(tvm::attr::kTarget);
  if (!target_opt.defined() || !TargetTangIsSTCUV2(target_opt.value())) {
    return f;
  }
  if (f->body.defined()) {
    f.CopyOnWrite()->body = TangTmemDrainExpander::Expand(f->body);
  }
  return f;
}

} // namespace

// ===========================================================================
// Pass registration
// ===========================================================================

namespace transform {
using namespace tirx::transform;

tvm::transform::Pass LowerSharedTmem() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return tl::LowerSharedTmem(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerSharedTmem", {});
}

/*!
 * \brief Create the LowerTangTmemDrain pass.
 * Runs only on the stcuv2 subtarget (gated via pass_filter in phase.py).
 */
tvm::transform::Pass LowerTangTmemDrain() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    Stmt body = TangTmemDrainRewriter::Rewrite(f->body);
    return PrimFunc(f->params, body, f->ret_type, f->buffer_map, f->attrs);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerTangTmemDrain", {});
}

tvm::transform::Pass ExpandS3TmemDrain() {
  auto pass_func = [](PrimFunc f, const IRModule &, const PassContext &) {
    return ExpandS3TmemDrainPrimFunc(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.tang.transform.ExpandS3TmemDrain",
                            {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.tang.transform.LowerSharedTmem", LowerSharedTmem)
      .def("tl.tang.transform.LowerTangTmemDrain", LowerTangTmemDrain)
      .def("tl.tang.transform.ExpandS3TmemDrain", ExpandS3TmemDrain);
}

} // namespace transform
} // namespace tl
} // namespace tvm
