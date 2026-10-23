/*!
 * \file expand_tang_tmem_drain.cc
 * \brief Expand the fused TMEM->global drain intrinsic into its loop nest, in
 *        IR, so the TANG code generator only prints statements.
 *
 * Why this pass exists
 * --------------------
 * ``LowerTangTmemDrain`` (src/tang/transform/lower_shared_tmem.cc) detects the
 * elementwise ``C[i, tx] = C_tmem[i, tx]`` epilogue that a ``T.copy(C_tmem,
 * C)`` produces and replaces it with one fused intrinsic,
 * ``tl.tang_tmem_drain_16x256b_to_global``. The intrinsic used to be expanded
 * by the code generator, which meant the drain's whole geometry -- the warp
 * split, the 16-row strip loop, the warp-collective ``ldt_16x256b_x16``, its
 * ``fence_ldt()`` checkpoint and the per-lane store pattern -- was codegen
 * knowledge, invisible to every IR-level pass and untestable without compiling.
 *
 * This pass materialises that geometry as ordinary TIR:
 *
 *   [if nwarps > 1] __syncthreads();          // publish warp 0's async MMA
 *   uint _out[64];                            // ldt destination registers
 *   if (threadIdx.x / 32 < nwarps) {
 *     for (r = warp*16; r < BM; r += nwarps*16) {
 *       for (cb = 0; cb < N_tile; cb += 128) {
 *         tang_tmem_ld_16x256b_x16(_out, tmem_base + ((r << 16) | cb), 64);
 *         for (c = 0; c < 16; ++c) {
 *           C[base + (r + lane/4    )*BN + cb + 8*c + (lane%4)*2    ] =
 * acc(_out[4*c + 0]); C[base + (r + lane/4    )*BN + cb + 8*c + (lane%4)*2 + 1]
 * = acc(_out[4*c + 1]); C[base + (r + lane/4 + 8)*BN + cb + 8*c + (lane%4)*2 ]
 * = acc(_out[4*c + 2]); C[base + (r + lane/4 + 8)*BN + cb + 8*c + (lane%4)*2 +
 * 1] = acc(_out[4*c + 3]);
 *         }
 *       }
 *     }
 *   }
 *
 * with ``acc(x)`` = ``reinterpret<accum_dtype>(x)``: each 32-bit word the ldt
 * returns is read back as the accumulator's element type. A 16-bit accumulator
 * (tFP16/tBF16) keeps one element per word in the word's low half, so the
 * reinterpret is a bit-level read, not a numeric conversion -- reading such a
 * word as ``float`` instead yields a denormal that flushes to zero on the way
 * into a 16-bit ``C``, silently emptying the whole tile.
 *
 * Two invariants the expansion inherits from the fused intrinsic and must keep:
 *
 *   * the multi-warp writeback needs the block barrier first: the async
 *     tcgen05 MMA is issued (and fenced) by warp 0 only, so warps 1..nwarps-1
 *     never observe the accumulator's completion and would read stale TMEM.
 *     A single-warp drain (nwarps == 1, warp 0) needs none;
 *   * the ldt destination registers are reused by every strip, so each
 *     ``ldt_16x256b_x16`` must be followed by its ``fence_ldt()`` checkpoint.
 *     That checkpoint is currently emitted by the ``tang_tmem_ld_16x256b_x16``
 *     codegen (the op is 1:1 with the generated instruction), so this pass
 *     emits the op and not a fence of its own; the drain no longer needs the
 *     generator to know anything about its shape.
 *
 * The pass runs late (after ``RemoveRedundantSyncs``, before ``MakePackedAPI``)
 * on purpose: the loops, the barrier and the scratch array it introduces must
 * not be rewritten, vectorized or sync-elided by the general lowering passes,
 * and by then the global buffer is already flattened -- the store index is the
 * plain element offset the fused intrinsic carried.
 */
#include "support/check.h"
#include "tang/op/builtin.h"
#include "tang/target_utils.h"

#include <tvm/ir/transform.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tl {

namespace {

using namespace tirx;
using namespace ffi;
using tvm::transform::PassContext;

/*! \brief Rows one ``ldt_16x256b_x16`` covers (a warp-collective 16-row strip).
 */
constexpr int kLdtRows = 16;
/*! \brief 32-bit words one ``ldt_16x256b_x16`` returns (16 chunks x 4 words).
 */
constexpr int kLdtWords = 64;
/*! \brief 8 column chunks per 16-row strip, 4 words per chunk. */
constexpr int kLdtChunks = 16;
/*! \brief TMEM columns a single ldt chunk spans (one 128-column accumulator).
 */
constexpr int kLdtChunkCols = 128;
/*! \brief Lanes per warp. */
constexpr int kWarpSize = 32;

inline PrimExpr I32(int64_t value) { return IntImm(DataType::Int(32), value); }

/*!
 * \brief Expand every fused TMEM->global drain in ``body`` into its loop nest.
 */
class TangTmemDrainExpander : public StmtExprMutator {
public:
  static Stmt Expand(const Stmt &body) {
    TangTmemDrainExpander expander;
    expander.CollectUsedNames(body);
    return expander(body);
  }

private:
  /*!
   * \brief Track the ``threadIdx.x`` binding so the drain can spell the warp
   * and lane ids.
   *
   * The drain op sits inside the thread extent, so the innermost
   * ``threadIdx.x`` seen on the way down is the one that covers it.
   */
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

  /*! \brief Remember every buffer data-var name so the scratch array is fresh.
   */
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

  /*!
   * \brief The global buffer an access pointer addresses.
   *
   * ``LowerAccessPtr`` turns the drain's ``tl.access_ptr`` into
   * ``tvm_access_ptr(TypeAnnotation(elem_dtype), data, offset, extent, rw)``,
   * so the element type and the (already flattened) base offset both come from
   * there. The buffer itself is rebuilt from the data var with ``elem_offset``
   * pinned to zero -- the offset in the access pointer already includes it, and
   * folding it in twice would shift every store.
   */
  static Buffer GlobalBufferOf(const CallNode *access, PrimExpr *base_offset,
                               DataType *elem_dtype) {
    ICHECK(access->args.size() >= 4U)
        << "tvm_access_ptr expects 5 arguments, got " << access->args.size();
    // arg0 is `TypeAnnotation(elem_dtype)`: the element type lives in the
    // annotation call's *dtype* field and its argument list is empty
    // (`tirx::Call(dtype, Op::Get("tirx.type_annotation"), {})`, see
    // 3rdparty/tvm/src/tirx/ir/stmt.cc), so `args[0].dtype()` -- not
    // `args[0]->args[0]` -- is the element type.
    const auto *annotation = access->args[0].as<CallNode>();
    // `CallNode::op` is a `RelaxExpr` without a name accessor, so the name is
    // read through `OpNode`.
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
    ICHECK(call->args.size() == 6U || call->args.size() == 7U)
        << "tang_tmem_drain_16x256b_to_global expects 6 or 7 arguments, got "
        << call->args.size();
    ICHECK(!thread_x_.empty())
        << "ExpandTangTmemDrain: a TMEM->global drain outside any threadIdx.x "
           "extent cannot be expanded";
    const Var &thread_x = thread_x_.back();

    const PrimExpr &tmem_base = call->args[0];
    const auto *access = call->args[1].as<CallNode>();
    ICHECK(access != nullptr && access->op.same_as(builtin::tvm_access_ptr()))
        << "ExpandTangTmemDrain: drain arg1 must be a tvm_access_ptr";
    PrimExpr base_offset;
    DataType elem_dtype;
    Buffer global = GlobalBufferOf(access, &base_offset, &elem_dtype);

    const auto *bm_imm = call->args[2].as<IntImmNode>();
    const auto *bn_imm = call->args[3].as<IntImmNode>();
    const auto *nwarps_imm = call->args[5].as<IntImmNode>();
    ICHECK(bm_imm != nullptr && bn_imm != nullptr && nwarps_imm != nullptr)
        << "ExpandTangTmemDrain: the drain's BM/BN/nwarps must be constants";
    const int bm = static_cast<int>(bm_imm->value);
    const int bn = static_cast<int>(bn_imm->value);
    int nwarps = static_cast<int>(nwarps_imm->value);
    if (nwarps < 1) {
      nwarps = 1;
    }
    // arg[4] is a typed zero carrying the TMEM accumulator's element type.
    const DataType accum_dtype = call->args[4].dtype();
    // arg[6] is the optional N_tile: the per-block accumulator column count.
    int n_tile = kLdtChunkCols;
    if (call->args.size() == 7U) {
      const auto *n_tile_imm = call->args[6].as<IntImmNode>();
      ICHECK(n_tile_imm != nullptr)
          << "ExpandTangTmemDrain: the drain's N_tile must be a constant";
      n_tile = static_cast<int>(n_tile_imm->value);
    }
    if (n_tile < kLdtChunkCols) {
      n_tile = kLdtChunkCols;
    }

    // `_out`: the ldt destination, kLdtWords 32-bit words per lane. Declared as
    // words rather than accumulator elements: a 16-bit accumulator still lands
    // one element per word (in its low half), and the stores reinterpret the
    // word -- so one word-sized array serves every accumulator type.
    const std::string out_name = FreshOutName();
    Var out_data(out_name, PointerType(PrimType(DataType::UInt(32)), "local"));
    Buffer out_buf(out_data, DataType::UInt(32), {I32(kLdtWords)},
                   /*strides=*/{}, I32(0), out_name, /*data_alignment=*/0,
                   /*offset_factor=*/0, BufferType::kDefault);

    // Warp / lane ids. The generator used to print `_w = threadIdx.x/32` and
    // `_tid = __laneid()`; both are plain expressions of the thread index.
    PrimExpr warp = floordiv(thread_x, I32(kWarpSize));
    PrimExpr lane = floormod(thread_x, I32(kWarpSize));

    // Strip loop: warp `w` handles the row strips w*16, w*16 + nwarps*16, ...
    //
    // Counted loop with the strip index recomputed in the body rather than a
    // strided loop: TIR loops count iterations, `r`/`cb` end up as inline
    // expressions of the loop counters, and the emitted code needs no reliance
    // on the optional `step` field of `tir.For`.
    Var k("k", DataType::Int(32));
    PrimExpr r_min = warp * I32(kLdtRows);
    PrimExpr r_step = I32(nwarps * kLdtRows);
    PrimExpr r_trip = floordiv(I32(bm) - r_min + r_step - I32(1), r_step);
    // A warp whose strips are already covered by the earlier warps runs zero
    // iterations (BM <= warp*16), so the count is clamped at zero.
    r_trip = max(r_trip, I32(0));
    PrimExpr r = r_min + k * r_step;

    // Column-block loop: one ldt drains kLdtChunkCols accumulator columns.
    Var kb("kb", DataType::Int(32));
    PrimExpr cb = kb * I32(kLdtChunkCols);

    // The warp-collective load: `tang_tmem_ld_16x256b_x16` is 1:1 with the
    // generated `ldt_16x256b_x16` and ends with its own fence_ldt checkpoint.
    PrimExpr taddr = tmem_base + ((r << I32(16)) | cb);
    Stmt ldt = Evaluate(Call(DataType::Handle(), tang_tmem_ld_16x256b_x16(),
                             {out_buf->data, taddr, I32(kLdtWords)}));

    // Store loop: 4 words per chunk -- two rows (r + lane/4, +8) by two
    // adjacent columns (lane%4)*2 and +1.
    Var c("c", DataType::Int(32));
    PrimExpr row_lo = r + floordiv(lane, I32(4));
    PrimExpr row_hi = row_lo + I32(8);
    PrimExpr bn_expr = I32(bn);
    PrimExpr col_lo = floormod(lane, I32(4)) * I32(2);
    auto element = [&](int64_t word) {
      // Chunk `c` owns words 4c..4c+3: the four words one ldt chunk returns.
      PrimExpr value = BufferLoad(out_buf, {c * I32(4) + I32(word)});
      // A 16-bit accumulator keeps one element in the word's low half, so the
      // word narrows to 16 bits before the bit-level read. `reinterpret` must
      // preserve the bit width (a 32-bit word read as fp16/bf16 would be a size
      // mismatch), and reading the word as `float` instead would yield a
      // denormal that flushes to zero on the way into a 16-bit destination.
      ICHECK(accum_dtype.bits() == 32 || accum_dtype.bits() == 16)
          << "ExpandTangTmemDrain: a " << accum_dtype.bits()
          << "-bit TMEM accumulator is not supported by the drain";
      if (accum_dtype.bits() < 32) {
        value = Cast(DataType::UInt(16), value);
      }
      value = reinterpret(accum_dtype, value);
      // The store converts (numerically, not bit-wise) into whatever C holds,
      // matching the implicit C++ conversion the generated assignment used to
      // rely on.
      if (!(value.dtype() == elem_dtype)) {
        value = Cast(elem_dtype, value);
      }
      return value;
    };
    auto store = [&](const PrimExpr &row, const PrimExpr &col, int64_t word) {
      PrimExpr index = base_offset + row * bn_expr + cb + c * I32(8) + col;
      return BufferStore(global, element(word), {index});
    };

    Stmt store_loop = For(
        c, I32(0), I32(kLdtChunks), ForKind::kSerial,
        SeqStmt({store(row_lo, col_lo, 0), store(row_lo, col_lo + I32(1), 1),
                 store(row_hi, col_lo, 2), store(row_hi, col_lo + I32(1), 3)}));
    Stmt column_loop =
        For(kb, I32(0), I32((n_tile + kLdtChunkCols - 1) / kLdtChunkCols),
            ForKind::kSerial, SeqStmt({ldt, store_loop}));
    Stmt strip_loop = For(k, I32(0), r_trip, ForKind::kSerial, column_loop);
    Stmt warp_guard = IfThenElse(warp < I32(nwarps), strip_loop, Stmt());

    // The scratch array declaration wraps the loops, so it is in scope for both
    // the ldt and the stores.
    Stmt out_decl = SeqStmt(
        {AllocBuffer(out_buf), DeclBuffer(out_buf), std::move(warp_guard)});

    // Publish warp 0's async MMA before any other warp reads the accumulator.
    if (nwarps > 1) {
      return SeqStmt(
          {Evaluate(Call(DataType::Handle(), builtin::tvm_storage_sync(),
                         {StringImm("shared")})),
           std::move(out_decl)});
    }
    return out_decl;
  }

  std::vector<Var> thread_x_;
  std::unordered_set<std::string> used_names_;
};

PrimFunc ExpandTangTmemDrainPrimFunc(PrimFunc f) {
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

tvm::transform::Pass ExpandTangTmemDrain() {
  auto pass_func = [](PrimFunc f, const IRModule &, const PassContext &) {
    return ExpandTangTmemDrainPrimFunc(std::move(f));
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                             "tl.tang.ExpandTangTmemDrain", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.tang.transform.ExpandTangTmemDrain",
                        ExpandTangTmemDrain);
}

} // namespace tl
} // namespace tvm
