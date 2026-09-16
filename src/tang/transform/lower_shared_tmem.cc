/*!
 *  \file lower_shared_tmem.cc
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
 *  NOTE: these remain TWO SEPARATE passes (registered independently and
 *  scheduled at different points in tilelang/tang/pipeline.py).  They are
 * co-located in one translation unit for cohesion; they CANNOT be fused into a
 * single pass invocation because:
 *    - they run at different pipeline positions (LowerSharedTmem early, before
 *      the target-specific middle pipeline; LowerTangTmemDrain after it);
 *    - LowerTangTmemDrain is additionally gated to stcuv2;
 *    - LowerTangTmemDrain consumes the shrunk (1,)-uint32 address-holder buffer
 *      produced by LowerSharedTmem (a producer->consumer dependency, with the
 *      middle pipeline running in between).
 */
#include "op/builtin.h"
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

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

// ===========================================================================
// Pass 1: LowerSharedTmem
//   Convert shared.tmem buffers to TANG address holders + tc init, and do
//   coordinate translation (from logical address to physical address).
// ===========================================================================

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
    new_body.push_back(
        Evaluate(Call(DataType::Handle(), builtin::tvm_storage_sync(),
                      {StringImm("shared")})));
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
    new_body.push_back(
        Evaluate(Call(DataType::Handle(), builtin::tvm_storage_sync(),
                      {StringImm("shared")})));
    new_body.push_back(
        Evaluate(Call(DataType::Handle(), tl::tang_deallocate_tensor_memory(),
                      {region0_base, PrimExpr(num_cols_allocated)})));

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
//   tang_tmem_ld_16x256b_x16 intrinsic.
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
      if (cols > 0)
        N_tile = cols;
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

    LOG(INFO) << "LowerTangTmemDrain: detected TMEM drain loop!"
              << " BM=" << BM << " BN=" << BN
              << " global_shape=" << global_buf->shape
              << " tmem_shape=" << tmem_buf->shape
              << " base_offset=" << base_offset;

    // -- Replace with the tang_tmem_drain_16x256b_to_global intrinsic --
    // Pass C_tmem[0] (value, not pointer) as the TMEM base address
    PrimExpr tmem_base = BufferLoad(tmem_buf, {IntImm(DataType::Int(32), 0)});
    // The 5th arg carries the ACCUMULATOR's element type (via a typed zero) so
    // codegen knows how to read back each 32-bit word ldt returns: fp32 and
    // int32 accumulators fill the word, while a 16-bit accumulator occupies its
    // low half. Narrowing to the C buffer's own type then happens in the store.
    // This used to pass the C buffer's type as a stand-in, which is only the
    // same thing when the accumulator is 32-bit: a 16-bit accumulator read as
    // fp32 yields a denormal, and the store flushes the whole tile to zero.
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
    Stmt drain_call = Evaluate(
        Call(DataType::Handle(), tang_tmem_drain_16x256b_to_global(),
             {tmem_base,
              global_buf.access_ptr(2, DataType::Handle(), 1, base_offset),
              IntImm(DataType::Int(32), BM), IntImm(DataType::Int(32), BN),
              make_zero(accum_dtype_), IntImm(DataType::Int(32), nwarps),
              IntImm(DataType::Int(32), N_tile)}));

    processed_[op] = drain_call;
    return drain_call;
  }
};

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

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.tang.transform.LowerSharedTmem", LowerSharedTmem)
      .def("tl.tang.transform.LowerTangTmemDrain", LowerTangTmemDrain);
}

} // namespace transform
} // namespace tl
} // namespace tvm
