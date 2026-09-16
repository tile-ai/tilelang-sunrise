/*!
 * \file tl/tang/op/copy.cc
 * \brief TANG implementation for ordinary tile copies.
 */

#include "op/copy.h"
#include "layout/cute_layout.h"
#include "layout/layout.h"
#include "layout/tcgen05_layout.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "tang/target_utils.h"

#include <optional>
#include <tvm/ir/transform.h>
#include <vector>

namespace tvm {
namespace tl {
namespace tang {
namespace {

bool IsFragmentToSharedStage(const CopyNode &op) {
  if (!IsFragmentBuffer(op.src))
    return false;
  if (!IsSharedBuffer(op.dst))
    return false;
  if (op.dst->shape.size() != 2)
    return false;
  for (const PrimExpr &dim : op.dst->shape) {
    if (!dim.as<IntImmNode>())
      return false;
  }
  return true;
}

// TANG stcuv2 register <-> TMEM (tcgen05.ld/st): true iff this copy moves data
// between a shared.tmem buffer and a local.fragment buffer (either direction).
bool IsTangTmemFragmentCopy(const CopyNode &op) {
  bool src_tmem = op.src.scope() == "shared.tmem";
  bool dst_tmem = op.dst.scope() == "shared.tmem";
  bool src_frag = op.src.scope() == "local.fragment";
  bool dst_frag = op.dst.scope() == "local.fragment";
  return (src_tmem && dst_frag) || (src_frag && dst_tmem);
}

// A store into a sub-32-bit tensor memory buffer is staging an MMA **A
// operand**, not restoring an accumulator: mma_atmem reads A bit-packed into
// 32-bit TMEM cells (2 x f16/bf16, 4 x s8/u8), which is a different register
// layout and a different store shape than the `.16x256b` accumulator path.
// This mirrors the upstream tcgen05 TS spelling, where the chained-GEMM P
// operand is `T.alloc_tmem([M, N], <16-bit>)` written with a packed
// `T.copy(P_local, P_tmem)`.
//
// tcgen5 accumulators on this target are fp32/int32, so for a sub-32-bit
// destination the element width is already an unambiguous discriminator. A
// 32-bit A operand (tf32) is indistinguishable from an accumulator store by
// shape and type alone, so it would have to be marked explicitly. No frontend
// primitive writes that marker yet (T9), so a 32-bit A goes through shared
// instead: T.tcgen05_cp is unambiguous by direction. The load direction has no
// A-operand meaning and keeps the accumulator layout.
bool IsTangTmemAOperandCopy(const CopyNode &op) {
  if (op.dst.scope() != "shared.tmem" || op.src.scope() != "local.fragment")
    return false;
  if (auto value = op.annotations.Get("tang_tmem_a_operand")) {
    const auto *imm = value.value().as<IntImmNode>();
    ICHECK(imm) << "tang_tmem_a_operand must be an IntImm";
    return imm->value != 0;
  }
  return op.dst->dtype.bits() < 32;
}

// TANG stcuv2 shared -> TMEM (cps2t): staging an MMA A operand straight from
// shared memory. Unlike the fragment direction this needs no marker to pick the
// A-operand *meaning*, since an accumulator is never restored from shared.
bool IsTangSharedToTmemCopy(const CopyNode &op) {
  if (op.dst.scope() != "shared.tmem")
    return false;
  return op.src.scope() == "shared" || op.src.scope() == "shared.dyn";
}

// Set by T.tcgen05_cp / T.tang_cp_tmem_to_shared. Both shared <-> TMEM copy
// engine directions are reachable only through those named primitives: the
// contracts they impose (source swizzle mode, row/column granularity,
// single-warp execution, a full barrier) are invisible at a T.copy call site,
// and the most likely misuse -- a source filled with the wrong swizzle atom --
// produces wrong data rather than an error. The CUDA backend draws the same
// line: src/op/copy.cc has no shared.tmem lowering at all, so there
// T.tcgen05_cp_warpx4 is likewise the only entry point.
bool HasTangTcgen05CpMarker(const CopyNode &op) {
  if (auto value = op.annotations.Get("tang_tcgen05_cp")) {
    const auto *imm = value.value().as<IntImmNode>();
    ICHECK(imm) << "tang_tcgen05_cp must be an IntImm";
    return imm->value != 0;
  }
  return false;
}

// Assign the fixed `.16x256b` ldt/stt register layout to the fragment buffer,
// plus an identity physical layout to the shared.tmem buffer so LowerSharedTmem
// can form the tmem address.
LayoutMap InferTangTmemFragmentLayout(const CopyNode &op,
                                      const LayoutInferArgs &layout_args) {
  bool src_tmem = op.src.scope() == "shared.tmem";
  Buffer tmem_buf = src_tmem ? op.src : op.dst;
  Buffer frag_buf = src_tmem ? op.dst : op.src;
  LayoutMap results;
  if (!layout_args.layout_map.count(frag_buf)) {
    Array<IterVar> ivs = op.MakeIterVars();
    ICHECK_EQ(ivs.size(), 2U) << "TANG tmem copy only supports 2D tiles";
    const auto *rimm = ivs[0]->dom->extent.as<IntImmNode>();
    const auto *cimm = ivs[1]->dom->extent.as<IntImmNode>();
    ICHECK(rimm && cimm) << "TANG tmem copy requires constant tile extents";
    Fragment frag =
        IsTangTmemAOperandCopy(op)
            ? makeTangTmemAOperandFragment(static_cast<int>(rimm->value),
                                           static_cast<int>(cimm->value))
            : makeTangTmem16x256bFragment(static_cast<int>(rimm->value),
                                          static_cast<int>(cimm->value));
    // Single-warp `.16x256b` path (warp 0). TMEM is warp-local, so only the
    // allocating warp can ld/st it today; warp specialization is added later.
    constexpr int kWarp = 0;
    constexpr int kWarpSize = 32;
    Range warp_range =
        Range::FromMinExtent(layout_args.thread_bounds->min +
                                 IntImm(DataType::Int(32), kWarp * kWarpSize),
                             IntImm(DataType::Int(32), kWarpSize));
    results.Set(frag_buf, frag->BindThreadRange(warp_range));
  }
  if (!layout_args.layout_map.count(tmem_buf)) {
    Var vi("i", DataType::Int(32));
    Var vj("j", DataType::Int(32));
    IterVar i(Range(0, tmem_buf->shape[0]), vi, IterVarType::kDataPar);
    IterVar j(Range(0, tmem_buf->shape[1]), vj, IterVarType::kDataPar);
    results.Set(tmem_buf,
                Layout(Array<IterVar>{i, j}, Array<PrimExpr>{vi, vj}));
  }
  return results;
}

// Identity physical layout for a buffer of any rank: logical index == physical
// index, so claiming it neither pads nor permutes anything. Claiming it is
// still meaningful: it records "this buffer carries no swizzle" as a fact other
// operators can read, rather than something they have to infer from the ABSENCE
// of a layout_map entry.
Layout MakeTangIdentityLayout(const Buffer &buffer) {
  Array<IterVar> iters;
  Array<PrimExpr> forward;
  for (size_t i = 0; i < buffer->shape.size(); ++i) {
    Var v("i" + std::to_string(i), DataType::Int(32));
    iters.push_back(
        IterVar(Range(0, buffer->shape[i]), v, IterVarType::kDataPar));
    forward.push_back(v);
  }
  return Layout(iters, forward);
}

// Whether `layout` is the identity claimed above, i.e. the buffer is plain and
// unswizzled. A structural comparison is reliable here because both sides come
// out of MakeTangIdentityLayout for the same buffer.
bool IsTangIdentityLayout(const Layout &layout, const Buffer &buffer) {
  return StructuralEqual()(layout, MakeTangIdentityLayout(buffer));
}

// The cps2t path drives the copy engine directly, so no thread mapping is
// needed -- only the identity physical layout that lets LowerSharedTmem turn a
// BufferLoad on the tmem buffer into a (row << 16) | col address.
LayoutMap InferTangSharedToTmemLayout(const CopyNode &op,
                                      const LayoutInferArgs &layout_args) {
  LayoutMap results;
  const Buffer &tmem_buf = op.dst;
  if (!layout_args.layout_map.count(tmem_buf)) {
    Var vi("i", DataType::Int(32));
    Var vj("j", DataType::Int(32));
    IterVar i(Range(0, tmem_buf->shape[0]), vi, IterVarType::kDataPar);
    IterVar j(Range(0, tmem_buf->shape[1]), vj, IterVarType::kDataPar);
    results.Set(tmem_buf,
                Layout(Array<IterVar>{i, j}, Array<PrimExpr>{vi, vj}));
  }
  return results;
}

bool IsTangBulkCopyDisabled(const CopyNode &op) {
  bool disabled = tvm::transform::PassContext::Current()
                      ->GetConfig<Bool>("tl.disable_tma_lower", Bool(false))
                      .value();
  if (auto value = op.annotations.Get("disable_tma")) {
    if (const auto *flag = value->as<IntImmNode>()) {
      disabled |= flag->value != 0;
    }
  }
  return disabled;
}

LayoutMap InferLayout(const CopyNode &op, const LayoutInferArgs &layout_args,
                      InferLevel level) {
  if (TargetTangIsSTCUV2(layout_args.target) && IsTangTmemFragmentCopy(op)) {
    return InferTangTmemFragmentLayout(op, layout_args);
  }
  if (TargetTangIsSTCUV2(layout_args.target) && IsTangSharedToTmemCopy(op)) {
    return InferTangSharedToTmemLayout(op, layout_args);
  }
  const bool bulk_enabled =
      TargetTangIsSTCUV2(layout_args.target) && !IsTangBulkCopyDisabled(op);
  const bool fp4_unpack =
      TargetTangIsSTCUV2(layout_args.target) && IsFP4UnpackLoad(op.src, op.dst);
  ICHECK(!fp4_unpack || bulk_enabled)
      << "TANG stcuv2 packed FP4 unpack requires bulk copy; disable_tma "
         "and tl.disable_tma_lower have no SIMT fallback for this conversion";
  // Packed FP4 changes element density and cannot form a SIMT copy loop.
  LayoutMap result =
      fp4_unpack ? LayoutMap() : op.InferSIMTLayout(layout_args, level);

  // STCUV2 global<->shared bulk copies: the shared side is what the hardware
  // swizzle applies to, so let it take part in layout inference the way the
  // CUDA bulk copy does (Copy::InferBulkLayout, cuda/op/copy.cc) instead of
  // leaving lowering to infer "plain staging buffer" from the ABSENCE of a
  // layout_map entry. kFree is the last round, so every mandatory claim -- an
  // MMA operand's tiled layout above all -- has already landed and wins by way
  // of the count() guard.
  if (bulk_enabled && level == InferLevel::kFree) {
    // Spelled out rather than via IsSharedBuffer to stay exactly in step with
    // the scopes Lower() routes to LowerSTCUV2BulkCopy; shared.tmem is handled
    // by the cps2t/cpt2s paths and must not be claimed here.
    auto is_smem = [](const Buffer &b) {
      return b.scope() == "shared" || b.scope() == "shared.dyn";
    };
    bool g2s = op.src.scope() == "global" && is_smem(op.dst);
    bool s2g = is_smem(op.src) && op.dst.scope() == "global";
    if (g2s || s2g) {
      const Buffer &shared_tensor = g2s ? op.dst : op.src;
      if (!layout_args.layout_map.count(shared_tensor) &&
          !result.count(shared_tensor)) {
        result.Set(shared_tensor, MakeTangIdentityLayout(shared_tensor));
      }
    }
  }

  if (!IsFragmentToSharedStage(op) || result.count(op.dst))
    return result;

  // Row-padding the staging buffer removes a bank conflict on the
  // fragment→shared write.  Set tl.enable_copy_staging_pad=True in
  // pass_configs to activate it.
  bool enabled = tvm::transform::PassContext::Current()
                     ->GetConfig<Bool>(kEnableCopyStagingPad, Bool(false))
                     .value();
  if (!enabled)
    return result;

  // The staging pad is an optional bank-conflict optimization, so it must
  // defer to layouts that other operators impose on the destination.  A
  // fragment→shared staging buffer that is ALSO a GEMM A/B operand (e.g. the
  // qkT_cast written by ``T.copy(qkT, qkT_cast)`` and then consumed by
  // ``T.gemm(qkT_cast, do, dv)``) already receives a per-tile swizzle layout
  // from the GEMM's infer_layout.  That swizzle is established during the
  // strict phase, so skip the strict phase here (the optional pad must not
  // race the mandatory swizzle for enqueue order), and in the common phase
  // back off whenever the destination already carries a layout.  Pure staging
  // buffers (dk_shared/dv_shared, consumed only by atomic_add which imposes no
  // layout) have no such entry and still get row-padded.
  if (level == InferLevel::kStrict)
    return result;
  if (layout_args.layout_map.count(op.dst))
    return result;

  const int rows = op.dst->shape[0].as<IntImmNode>()->value;
  const int cols = op.dst->shape[1].as<IntImmNode>()->value;

  if (!IsRowPadded(op.dst->dtype.bits(), cols))
    return result;

  Layout padded = MakeTangRowPaddedLayout(rows, cols, op.dst->dtype.bits());
  result.Set(op.dst, padded);
  return result;
}

// Read the shared-operand swizzle atom off a structural decomposition of the
// assigned layout: cute::ComposedLayoutFromTileLang recovers it as
// `Swizzle o offset o affine`, with the analyzer proving the recovered form
// equivalent to the TileLang layout, so this does not depend on how the index
// expression happens to be spelled.
//
// The tile shape falls straight out of the decomposition: the leading (fastest)
// extent of each input axis is that axis's tile extent. _make_tang_ab_layout
// (tilelang/tang/op/gemm/gemm_tmma.py:64) builds an MN-major operand with a
// (128/bits x 8) tile and a K-major one with (8 x 128/bits), e.g. a tf32
// MN-major operand decomposes to shape ((4,32),(8,8)) -> tile (4,8).
//
// Returns nullopt when the layout does not decompose into that form, so the
// caller can report the failure rather than substitute a guess.
std::optional<int> TangTcgen5SwizzleAtomStructural(const Layout &layout,
                                                   int bits) {
  auto composed = cute::ComposedLayoutFromTileLang(layout);
  if (!composed.has_value())
    return std::nullopt;
  const cute::ComposedLayout &c = composed.value();
  // TANG operand layouts are pure tilings; an XOR swizzle or a base offset
  // would mean the generator produced something this mapping does not cover.
  if (c->swizzle->IsSwizzled() || c->offset != 0)
    return std::nullopt;
  cute::IntTuple shape = c->layout->shape;
  if (cute::Rank(shape) != 2)
    return std::nullopt;
  auto tile_extent = [](const cute::IntTuple &axis) -> std::optional<int64_t> {
    if (!cute::IsTuple(axis))
      return std::nullopt;
    cute::IntTuple lead = axis[0];
    if (!cute::IsConst(lead))
      return std::nullopt;
    return cute::AsConst(lead);
  };
  std::optional<int64_t> tile_rows = tile_extent(shape[0]);
  std::optional<int64_t> tile_cols = tile_extent(shape[1]);
  if (!tile_rows.has_value() || !tile_cols.has_value())
    return std::nullopt;
  const int64_t val = 128 / bits;
  if (*tile_rows == val && *tile_cols == 8)
    return 64;
  if (*tile_rows == 8 && *tile_cols == val)
    return 32;
  return std::nullopt;
}

// Legacy point-sampling recovery, kept only to cross-check the structural
// decode above until every shape in the test suite has been seen to agree.
// It evaluates the forward map at two points and compares against the
// plain-tiling formula; unlike the structural form it cannot tell "not the
// layout I expected" from "atom 32", and several guards below answer 32 for a
// layout they simply failed to classify.
int TangTcgen5SwizzleAtomBytesSampled(const Layout &layout, int stride,
                                      int continuous, int bits) {
  if (bits != 32 && bits != 8)
    return 32; // fp16/bf16 always atom32
  const int val = 128 / bits;
  if (val <= 0 || continuous % 8 != 0 || continuous % val != 0)
    return 32;
  // Linear index of logical (r,c) for a (rows x cols) element tile; equals
  // mapped_row*continuous + mapped_col for the offset==0/pad==0 plain tiling.
  auto plain_linear = [&](int rows, int cols, int r, int c) -> long long {
    long long tile = static_cast<long long>(rows) * cols;
    long long idx_in_tile =
        static_cast<long long>(r % rows) * cols + (c % cols);
    long long tile_idx =
        static_cast<long long>(r / rows) * (continuous / cols) + (c / cols);
    return tile_idx * tile + idx_in_tile;
  };
  arith::Analyzer a;
  auto eval_linear = [&](int r, int c) -> long long {
    Array<PrimExpr> out = layout->Forward(
        {IntImm(DataType::Int(32), r), IntImm(DataType::Int(32), c)});
    if (out.size() == 2) {
      const auto *mr = as_const_int(a.Simplify(out[0]));
      const auto *mc = as_const_int(a.Simplify(out[1]));
      if (!mr || !mc)
        return -1;
      return static_cast<long long>(*mr) * continuous + (*mc);
    }
    if (out.size() == 1) {
      const auto *m = as_const_int(a.Simplify(out[0]));
      return m ? static_cast<long long>(*m) : -1;
    }
    return -1;
  };
  // (1,0) and (3,0) each map to distinct linear indices for the MN-major
  // (val x 8) vs K-major (8 x val) tiles when val != 8, i.e. tf32 (val=4) and
  // int8 (val=16) -- the only dtypes reaching here.
  const std::pair<int, int> pts[] = {{1, 0}, {3, 0}};
  bool mn_ok = true;
  for (const auto &pt : pts) {
    if (pt.first >= stride)
      return 32; // too few rows to disambiguate; stay on the safe default
    long long got = eval_linear(pt.first, pt.second);
    if (got < 0 || got != plain_linear(val, 8, pt.first, pt.second))
      mn_ok = false;
  }
  return mn_ok ? 64 : 32;
}

// Recover the tcgen5 shared-operand swizzle atom (32 or 64 bytes) from the
// GEMM-assigned shared layout. The bulk copy that WRITES a shared operand must
// use the same hardware swizzle the MMA descriptor reads it back with
// (gemm_tcgen05.h): an MN-major tf32/int8 operand (the NN GEMM B, or the
// transpose_A A of NT/TT) needs atom64, everything else atom32. fp16/bf16
// always use atom32 regardless of major-ness, so their (8x8) tile -- which is
// both (128/bits x 8) and (8 x 128/bits) -- never has to be disambiguated.
int TangTcgen5SwizzleAtomBytes(const Layout &layout, int stride, int continuous,
                               int bits) {
  if (bits != 32 && bits != 8)
    return 32; // fp16/bf16 always atom32
  std::optional<int> structural = TangTcgen5SwizzleAtomStructural(layout, bits);
  int sampled =
      TangTcgen5SwizzleAtomBytesSampled(layout, stride, continuous, bits);
  if (!structural.has_value()) {
    LOG(WARNING) << "TANG stcuv2: shared operand layout did not decompose into "
                    "a recognised operand tiling; falling back to point "
                    "sampling, which answers "
                 << sampled << "-byte atom. Layout: " << layout->DebugOutput();
    return sampled;
  }
  ICHECK_EQ(*structural, sampled)
      << "TANG stcuv2: the structural and sampled swizzle-atom recoveries "
         "disagree for this shared operand layout, so one of them is wrong "
         "about how the MMA descriptor will read the tile back. Layout: "
      << layout->DebugOutput();
  return *structural;
}

Stmt LowerSTCUV2BulkCopy(const CopyNode &op, const LowerArgs &lower_args,
                         arith::Analyzer *analyzer) {
  bool is_load = op.src.scope() == "global" &&
                 (op.dst.scope() == "shared" || op.dst.scope() == "shared.dyn");
  bool is_store =
      (op.src.scope() == "shared" || op.src.scope() == "shared.dyn") &&
      op.dst.scope() == "global";
  ICHECK(is_load || is_store);

  const auto &shared_range = is_load ? op.dst_range : op.src_range;
  const auto &global_range = is_load ? op.src_range : op.dst_range;
  const Buffer &shared_tensor = is_load ? op.dst : op.src;
  const Buffer &global_tensor = is_load ? op.src : op.dst;

  PrimExpr total_elements = 1;
  for (const Range &range : shared_range) {
    total_elements *= range->extent;
  }

  auto compute_offset_and_strides = [](const Buffer &buffer,
                                       const Array<Range> &ranges) {
    // strides is sized by the buffer rank but indexed by range position, both
    // here and by row_dim below, so the two must agree.
    ICHECK_EQ(ranges.size(), buffer->shape.size())
        << "TANG bulk copy: " << buffer->name << " has " << ranges.size()
        << " access ranges for a " << buffer->shape.size()
        << "-D buffer; expected one range per dimension";
    std::vector<PrimExpr> strides;
    PrimExpr stride = 1;
    for (size_t i = 0; i < buffer->shape.size(); ++i) {
      strides.insert(strides.begin(), stride);
      stride *= buffer->shape[buffer->shape.size() - i - 1];
    }
    PrimExpr offset = 0;
    for (size_t i = 0; i < ranges.size(); ++i) {
      offset += ranges[i]->min * strides[i];
    }
    return std::make_pair(offset, strides);
  };

  auto [shared_offset, shared_strides] =
      compute_offset_and_strides(shared_tensor, shared_range);
  auto [global_offset, global_strides] =
      compute_offset_and_strides(global_tensor, global_range);
  PrimExpr elements = analyzer->Simplify(total_elements);
  PrimExpr shared_addr = shared_tensor.access_ptr(
      is_load ? 2 : 1, DataType::Handle(), 1, shared_offset, elements);
  PrimExpr global_addr = global_tensor.access_ptr(
      is_load ? 1 : 2, DataType::Handle(), 1, global_offset, elements);

  // The bulk copy moves a 2D (rows x cols) tile. When the shared side is
  // multi-buffered by the software pipeline (num_stages >= 2), the pipeline
  // prepends a stage dimension (a single-index slice of extent 1) whose
  // position is already folded into shared_offset above. Skip such leading
  // unit dims before deriving rows/cols; otherwise rows would pick up the stage
  // dim (extent 1) and cols would absorb the real row count, collapsing the
  // copy to rows=1 / col_bytes=whole-tile and overrunning shared memory.
  auto is_unit = [](const PrimExpr &e) {
    const auto *imm = e.as<IntImmNode>();
    return imm != nullptr && imm->value == 1;
  };
  // Index of the dimension that indexes tile rows. Leading unit dims are slice
  // origins already folded into the offsets above, so they are not part of the
  // 2D tile.
  auto row_dim = [&](const Array<Range> &ranges) {
    size_t d = 0;
    while (ranges.size() - d > 2 && is_unit(ranges[d]->extent)) {
      ++d;
    }
    return d;
  };

  size_t rbeg = row_dim(shared_range);
  PrimExpr rows =
      shared_range.empty() ? PrimExpr(1) : shared_range[rbeg]->extent;
  PrimExpr cols = 1;
  for (size_t i = rbeg + 1; i < shared_range.size(); ++i) {
    cols *= shared_range[i]->extent;
  }
  PrimExpr col_bytes =
      analyzer->Simplify(FloorDiv(cols * shared_tensor->dtype.bits(), 8));
  // The global side has to skip the same leading unit dims: gmem_row_bytes is
  // the distance between two rows of the tile, which is the stride of whichever
  // dimension indexes rows. global_strides[0] only coincides with that at rank
  // 2; at rank 3 it is the batch stride, so every row past the first would be
  // fetched a whole matrix too far ahead.
  PrimExpr gmem_row_stride = global_strides.empty()
                                 ? PrimExpr(1)
                                 : global_strides[row_dim(global_range)];
  PrimExpr gmem_row_bytes = analyzer->Simplify(
      FloorDiv(gmem_row_stride * global_tensor->dtype.bits(), 8));

  // Which of the three bulk-copy regimes this copy belongs to. The swizzled 3D
  // pair does NOT round-trip: a tile loaded by fcpg2s.3d and stored back by
  // fcps2g.3d under the same SwizzleMode comes out permuted at atom
  // granularity, and that is an ISA-level asymmetry rather than a
  // parameterisation mistake on our side -- driving the vendor's own public
  // wrappers with the round-trip recipe documented verbatim in cp_async_bulk.h
  // reproduces it exactly (gver/data_copy/fcpg2s_fcps2g_path;
  // docs/tang_bulk_copy_gs_layout.md §15.2). So the swizzled form is only
  // usable where the two ends are known to agree:
  //
  //   * `tang_swizzle_atom_bytes` present  -> the caller states the layout, so
  //     the validated cpt2s pairing (atom 8) and the deliberate swizzled
  //     staging escape hatch both go swizzled.
  //   * a GEMM-assigned shared layout      -> an MMA operand. The load must be
  //     swizzled to match the MMA descriptor; a store has no validated pairing
  //     and is rejected.
  //   * neither                            -> a pure staging buffer, which
  //     takes the unswizzled 1D path (flat byte copy, trivially its own
  //     inverse) in BOTH directions so the two ends stay consistent.
  const bool has_atom_annotation =
      op.annotations.Get("tang_swizzle_atom_bytes").has_value();
  // Every shared buffer a bulk copy touches now carries SOME layout --
  // InferLayout claims the identity for plain staging buffers -- so what marks
  // an MMA operand is a layout that actually permutes, not the mere presence of
  // an entry.
  const bool is_mma_operand =
      lower_args.layout_map.count(shared_tensor) > 0 &&
      !IsTangIdentityLayout(lower_args.layout_map.at(shared_tensor),
                            shared_tensor);

  if (is_store && !has_atom_annotation && is_mma_operand) {
    LOG(FATAL)
        << "TANG STCUV2: shared->global bulk copy of buffer '"
        << shared_tensor->name
        << "', which the GEMM assigned a shared operand layout.\n"
        << "  An MMA operand is laid out swizzled for the MMA descriptor, and "
           "the swizzled shared->global store does not invert the swizzled "
           "load (docs/tang_bulk_copy_gs_layout.md §15.2), so streaming it "
           "back to global would silently scramble it.\n"
        << "  The only validated swizzled store is the cpt2s drain (tensor "
           "memory -> shared -> global); annotate both copies with "
           "`tang_swizzle_atom_bytes: 8` if that is what this is.\n"
        << "  To stage data through shared purely to get it back to global, "
           "use a buffer that is not also a GEMM operand: it then takes the "
           "unswizzled 1D path.";
  }

  // Swizzle atom (32 or 64 bytes). An explicit annotation wins; otherwise
  // recover it from the GEMM-assigned shared-operand layout so the copy's
  // hardware swizzle matches the MMA descriptor (MN-major tf32/int8 -> atom64).
  int atom_bytes = 32;
  if (auto value = op.annotations.Get("tang_swizzle_atom_bytes")) {
    const auto *imm = value->as<IntImmNode>();
    ICHECK(imm) << "tang_swizzle_atom_bytes must be an IntImm";
    atom_bytes = static_cast<int>(imm->value);
  } else if (is_mma_operand) {
    int ndim = static_cast<int>(shared_tensor->shape.size());
    if (ndim >= 2) {
      const auto *p_stride = as_const_int(shared_tensor->shape[ndim - 2]);
      const auto *p_cont = as_const_int(shared_tensor->shape[ndim - 1]);
      if (p_stride && p_cont) {
        atom_bytes = TangTcgen5SwizzleAtomBytes(
            lower_args.layout_map.at(shared_tensor),
            static_cast<int>(*p_stride), static_cast<int>(*p_cont),
            shared_tensor->dtype.bits());
      }
    }
  }

  // fp4 lives in global packed 2 e2m1 codes per byte, but an mxf8f6f4 operand
  // is read unpacked -- one code in the low nibble of its own byte. A dtype
  // pair that crosses those two layouts is asking for that spread, and the
  // copy engine can do it itself with PackPaddingMode b4p4x16 (one warp turns
  // 64 global bytes into 128 shared ones). The alternative, which is what
  // every vendor golden does, is for the host to pre-unpack the whole matrix.
  //
  // The mode is only usable in a narrow window, established by ISS probe
  // rather than inferred from the headers: the swizzled 3D form, one swizzle
  // unit per row, load direction. Outside it the hardware does not fail, it
  // reads the source at the wrong stride and returns a plausible wrong answer
  // -- or, for a too-narrow row, faults the simulator. So each condition is
  // rejected here by name instead of being left to run.
  int pack_mode = 0;
  if (IsFP4PackedToUnpackedStorageCopy(global_tensor->dtype,
                                       shared_tensor->dtype)) {
    const char *kWhy =
        "  Packed global fp4 -> unpacked shared fp4 is done by the copy "
        "engine's b4p4x16 pack mode, which only exists on the swizzled 3D "
        "load path.\n";
    if (is_store) {
      LOG(FATAL) << "TANG STCUV2: shared->global bulk copy of buffer '"
                 << shared_tensor->name << "' would repack unpacked fp4 ("
                 << shared_tensor->dtype << ") into packed global fp4 ("
                 << global_tensor->dtype << ").\n"
                 << kWhy
                 << "  The reverse (repacking) direction has no validated "
                    "pairing, so it is not enabled. Write the result through a "
                    "buffer whose dtype matches global.";
    }
    if (!has_atom_annotation && !is_mma_operand) {
      LOG(FATAL) << "TANG STCUV2: global->shared bulk copy of buffer '"
                 << shared_tensor->name << "' spreads packed global fp4 ("
                 << global_tensor->dtype << ") into unpacked shared fp4 ("
                 << shared_tensor->dtype
                 << "), but the buffer is a plain staging buffer.\n"
                 << kWhy
                 << "  A staging buffer takes the unswizzled 1D path, whose "
                    "global step is hardcoded equal to its shared step; under "
                    "b4p4x16 it would consume half of each step and read every "
                    "other source block. Use the buffer as a GEMM operand so "
                    "it gets a swizzled layout, or annotate "
                    "`tang_swizzle_atom_bytes: 32`.";
    }
    ICHECK_EQ(atom_bytes, 32)
        << "TANG STCUV2: unpacked fp4 operand '" << shared_tensor->name
        << "' has a " << atom_bytes
        << "-byte swizzle atom; only the 32-byte atom (sw128a32) has a "
           "validated b4p4x16 path.";
    // One swizzle unit per row. A wider row makes dim1 iterate, and dim1's
    // global step is hardcoded to the swizzle unit rather than halved; a
    // narrower one rounds dim1's count to zero. Neither is diagnosed by the
    // hardware. 128 unpacked bytes is K=128, which is also the most the
    // operand row-byte cap allows for an 8-bit element, so this is the only
    // shape a well-formed mxf8f6f4 GEMM asks for.
    const auto *c_col_bytes = as_const_int(col_bytes);
    if (!c_col_bytes || *c_col_bytes != 128) {
      LOG(FATAL) << "TANG STCUV2: unpacked fp4 operand '" << shared_tensor->name
                 << "' has a shared row of " << col_bytes
                 << " bytes; b4p4x16 requires exactly 128 (K=128)."
                 << "\n"
                 << kWhy
                 << "  The pack mode reads one 128-byte swizzle unit per row "
                    "and the row stride it applies to the packed source is not "
                    "scaled for any other width. Tile K at 128, or pre-unpack "
                    "in global and use a matching unpacked dtype.";
    }
    pack_mode = 1; // tang::ptx::b4p4x16
  }

  int direction = is_load ? 0 : 1;
  PrimExpr dst_addr = is_load ? shared_addr : global_addr;
  PrimExpr src_addr = is_load ? global_addr : shared_addr;
  // The block's thread count is passed as a codegen-time constant: blockDim is
  // NOT reliable on the stcuv2 ISS, so the helper cannot compute the warp count
  // at runtime. thread_bounds->extent is a compile-time constant here, so emit
  // it directly; the helper partitions the tile's swizzle stripes across all
  // warps of the block.
  ICHECK(is_const_int(lower_args.thread_bounds->extent))
      << "TANG bulk copy requires a constant thread count (thread_bounds), got "
      << lower_args.thread_bounds;
  int num_threads = *as_const_int(lower_args.thread_bounds->extent);

  if (pack_mode != 0) {
    // Swizzle mode 6 = sw128a32, the only atom the guards above let through.
    return Evaluate(Call(
        DataType::Handle(), tang_cp_async_bulk_sw(),
        {IntImm(DataType::Int(32), direction), dst_addr, src_addr, rows,
         col_bytes, gmem_row_bytes, IntImm(DataType::Int(32), num_threads),
         IntImm(DataType::Int(32), 6), IntImm(DataType::Int(32), pack_mode)}));
  }

  if (!has_atom_annotation && !is_mma_operand) {
    // Pure staging buffer: unswizzled 1D. The 1D overloads round `size` UP to
    // the 128-byte transfer chunk, which would write past the destination, so
    // reject shapes that would rely on that instead of silently overrunning.
    // Which size matters depends on how the helper transfers the tile: one flat
    // call when the pitches match, one call per row otherwise.
    const auto *c_rows = as_const_int(rows);
    const auto *c_col_bytes = as_const_int(col_bytes);
    const auto *c_gmem_row_bytes = as_const_int(gmem_row_bytes);
    if (c_rows && c_col_bytes && c_gmem_row_bytes) {
      const bool contiguous = *c_col_bytes == *c_gmem_row_bytes;
      const int64_t unit =
          contiguous ? (*c_rows) * (*c_col_bytes) : *c_col_bytes;
      if (unit % 128 != 0) {
        LOG(FATAL)
            << "TANG STCUV2: global<->shared staging copy of buffer '"
            << shared_tensor->name << "' has a "
            << (contiguous ? "total size" : "shared row width") << " of "
            << unit
            << " bytes, which is not a multiple of the 128-byte 1D transfer "
               "chunk.\n"
            << "  Tile is " << *c_rows << " rows x " << *c_col_bytes
            << " B, global row pitch " << *c_gmem_row_bytes << " B.\n"
            << "  A staging buffer cannot use the swizzled path (it does not "
               "round-trip, docs/tang_bulk_copy_gs_layout.md §15.2), and the "
               "1D path would round the size up and overrun the destination.\n"
            << "  Pad the shared tile so "
            << (contiguous ? "rows * row_bytes" : "row_bytes")
            << " is a multiple of 128.";
      }
    }
    return Evaluate(Call(DataType::Handle(), tang_cp_async_bulk_1d(),
                         {IntImm(DataType::Int(32), direction), dst_addr,
                          src_addr, rows, col_bytes, gmem_row_bytes,
                          IntImm(DataType::Int(32), num_threads)}));
  }

  if (atom_bytes == 64 || atom_bytes == 8) {
    // Swizzle mode codes consumed by codegen: 4 = sw128a8, 7 = sw128a64.
    // sw128a8 is the mode a cpt2s-staged tile is written with, so a tile
    // coming out of tensor memory has to be read back with it too.
    int swizzle_mode = atom_bytes == 64 ? 7 : 4;
    return Evaluate(
        Call(DataType::Handle(), tang_cp_async_bulk_sw(),
             {IntImm(DataType::Int(32), direction), dst_addr, src_addr, rows,
              col_bytes, gmem_row_bytes, IntImm(DataType::Int(32), num_threads),
              IntImm(DataType::Int(32), swizzle_mode),
              IntImm(DataType::Int(32), 0)}));
  }
  ICHECK_EQ(atom_bytes, 32)
      << "TANG STCUV2 bulk copy supports swizzle atom sizes 8, 32 or 64 bytes";
  return Evaluate(Call(DataType::Handle(), tang_cp_async_bulk(),
                       {IntImm(DataType::Int(32), direction), dst_addr,
                        src_addr, rows, col_bytes, gmem_row_bytes,
                        IntImm(DataType::Int(32), num_threads)}));
}

// Emit the TANG stcuv2 register <-> TMEM movement as a single-warp `.16x256b`
// ldt/stt loop. A single warp drives every 16-row sub-block; the tmem row
// offset (16*sub) is encoded into the taddr high bits by LowerSharedTmem via
// the BufferLoad(tmem, {row, col}) index. Codegen for tang_tmem_ld/st_16x256b
// already emits ldt/stt_16x256b + the ld/st fence.
Stmt LowerTangTmemFragmentCopy(const CopyNode &op, const LowerArgs &lower_args,
                               arith::Analyzer *analyzer) {
  bool is_ld = op.src.scope() == "shared.tmem"; // tmem -> fragment
  Buffer tmem_buf = is_ld ? op.src : op.dst;
  Buffer frag_buf = is_ld ? op.dst : op.src;

  Array<IterVar> ivs = op.MakeIterVars();
  ICHECK_EQ(ivs.size(), 2U) << "TANG tmem copy only supports 2D tiles";
  const auto *rimm = ivs[0]->dom->extent.as<IntImmNode>();
  const auto *cimm = ivs[1]->dom->extent.as<IntImmNode>();
  ICHECK(rimm && cimm) << "TANG tmem copy requires constant tile extents";
  int rows = static_cast<int>(rimm->value);
  int cols = static_cast<int>(cimm->value);

  const Array<Range> &tmem_range = is_ld ? op.src_range : op.dst_range;
  // Keep these symbolic: a chunked ld/st indexes tensor memory with the
  // surrounding loop var, and folding a non-constant origin to 0 would silently
  // re-read the first tile on every iteration.
  PrimExpr row_min = IntImm(DataType::Int(32), 0);
  PrimExpr col_min = IntImm(DataType::Int(32), 0);
  if (tmem_range.size() >= 1)
    row_min = tmem_range[0]->min;
  if (tmem_range.size() >= 2)
    col_min = tmem_range[1]->min;

  ICHECK(rows % 32 == 0) << "TANG 16x256b tmem copy needs rows % 32 == 0, got "
                         << rows;

  // A-operand staging: `.32x32b`, lane == row, the row's elements bit-packed
  // into 32-bit cells. One call per 32-row block.
  if (IsTangTmemAOperandCopy(op)) {
    int bits = static_cast<int>(tmem_buf->dtype.bits());
    ICHECK((cols * bits) % 32 == 0)
        << "TANG A-operand tmem staging needs the row to fill whole 32-bit "
           "cells, got "
        << cols << " x " << bits << "-bit";
    int num_cells = cols * bits / 32;
    ICHECK(num_cells == 1 || num_cells == 2 || num_cells == 4 ||
           num_cells == 8 || num_cells == 16 || num_cells == 32 ||
           num_cells == 64)
        << "TANG A-operand tmem staging supports 1/2/4/8/16/32/64 cells per "
           "row, got "
        << num_cells << " (from " << cols << " x " << bits << "-bit)";
    int sub_elems = 32 * cols;
    Array<Stmt> a_calls;
    for (int blk = 0; blk < rows / 32; ++blk) {
      PrimExpr row_for_blk =
          analyzer->Simplify(IntImm(DataType::Int(32), blk * 32) + row_min);
      Array<PrimExpr> args = {
          frag_buf.access_ptr(1, DataType::Handle(), 1,
                              IntImm(DataType::Int(32), blk * sub_elems),
                              IntImm(DataType::Int(32), sub_elems)),
          BufferLoad(tmem_buf, {row_for_blk, analyzer->Simplify(col_min)}),
          IntImm(DataType::Int(32), num_cells)};
      a_calls.push_back(
          Evaluate(Call(DataType::Handle(), tang_tmem_st_a_operand(), args)));
    }
    Stmt a_body = a_calls.size() == 1 ? a_calls[0] : SeqStmt(a_calls);
    PrimExpr lo = lower_args.thread_bounds->min;
    PrimExpr hi = lower_args.thread_bounds->min + IntImm(DataType::Int(32), 32);
    return IfThenElse((lower_args.thread_index >= lo) &&
                          (lower_args.thread_index < hi),
                      a_body, Stmt());
  }

  // Reading a sub-32-bit tensor memory buffer back into registers would need
  // the unpacking counterpart of the A-operand store, which has no use yet.
  // Falling through to the accumulator layout here would unpack nothing and be
  // silently wrong, so refuse instead.
  ICHECK(!(is_ld && tmem_buf->dtype.bits() < 32))
      << "TANG tmem load of a " << tmem_buf->dtype.bits()
      << "-bit tensor memory buffer is not supported: sub-32-bit tensor memory "
         "holds a bit-packed MMA A operand, and the unpacking load is not "
         "implemented. Read the fp32 accumulator instead.";

  ICHECK(cols % 8 == 0) << "TANG 16x256b tmem copy needs cols % 8 == 0, got "
                        << cols;
  int num_chunks = cols / 8;
  int num_subs = rows / 16;
  // access_ptr() takes a *logical* row-major element offset into the fragment
  // and applies the fragment layout to reach the per-lane register; each 16-row
  // sub-block's logical span is 16*cols elements (mapped to 4*num_chunks
  // contiguous per-lane registers by makeTangTmem16x256bFragment).
  int sub_elems = 16 * cols;

  constexpr int kWarp = 0;
  constexpr int kWarpSize = 32;
  const Op &ld_st_op = is_ld ? tang_tmem_ld_16x256b() : tang_tmem_st_16x256b();
  Array<Stmt> calls;
  for (int sub = 0; sub < num_subs; ++sub) {
    PrimExpr row_for_sub =
        analyzer->Simplify(IntImm(DataType::Int(32), sub * 16) + row_min);
    Array<PrimExpr> tmem_args = {
        frag_buf.access_ptr(is_ld ? 2 : 1, DataType::Handle(), 1,
                            IntImm(DataType::Int(32), sub * sub_elems),
                            IntImm(DataType::Int(32), sub_elems)),
        BufferLoad(tmem_buf, {row_for_sub, analyzer->Simplify(col_min)}),
        IntImm(DataType::Int(32), num_chunks)};
    calls.push_back(Evaluate(Call(DataType::Handle(), ld_st_op, tmem_args)));
  }
  Stmt body = calls.size() == 1 ? calls[0] : SeqStmt(calls);
  // Restrict to the selected warp's 32 lanes.
  PrimExpr lane_lo = lower_args.thread_bounds->min +
                     IntImm(DataType::Int(32), kWarp * kWarpSize);
  PrimExpr lane_hi = lower_args.thread_bounds->min +
                     IntImm(DataType::Int(32), (kWarp + 1) * kWarpSize);
  return IfThenElse((lower_args.thread_index >= lane_lo) &&
                        (lower_args.thread_index < lane_hi),
                    body, Stmt());
}

// Stage a tensor memory tile into shared memory with the cpt2s copy engine,
// which moves the data without routing it through the register file.
//
// The engine writes 8-byte atoms holding one tensor memory row each (the
// sw128a8 configuration), so the staged tile is row-major only when the shared
// row is exactly one 128-byte swizzle unit wide. A single call drains 64 rows;
// the device helper loops for taller tiles.
Stmt LowerTangTmemToSharedCopy(const CopyNode &op, const LowerArgs &lower_args,
                               arith::Analyzer *analyzer) {
  const Buffer &tmem_buf = op.src;
  const Buffer &smem_buf = op.dst;

  Array<IterVar> ivs = op.MakeIterVars();
  ICHECK_EQ(ivs.size(), 2U)
      << "TANG stcuv2 tmem->shared copy only supports 2D tiles";
  const auto *rimm = ivs[0]->dom->extent.as<IntImmNode>();
  const auto *cimm = ivs[1]->dom->extent.as<IntImmNode>();
  ICHECK(rimm && cimm)
      << "TANG stcuv2 tmem->shared copy requires constant tile extents";
  int rows = static_cast<int>(rimm->value);
  int cols = static_cast<int>(cimm->value);

  int elem_bytes = static_cast<int>(smem_buf->dtype.bytes());
  ICHECK_EQ(elem_bytes, 4)
      << "TANG stcuv2 tmem->shared copy only supports 32-bit element types, "
         "got "
      << smem_buf->dtype
      << ". Narrowing to 16 bits would need pack_16b, which keeps the low 16 "
         "bits of each word (a bit-level pack, not a numeric conversion).";
  ICHECK_EQ(tmem_buf->dtype.bytes(), 4)
      << "TANG stcuv2 tmem->shared copy needs a 32-bit tensor memory buffer, "
         "got "
      << tmem_buf->dtype;
  ICHECK_EQ(cols * elem_bytes, 128)
      << "TANG stcuv2 tmem->shared copy needs a 128-byte shared row (the "
         "sw128a8 swizzle unit), got "
      << cols << " x " << elem_bytes << " bytes";
  ICHECK(rows % 64 == 0)
      << "TANG stcuv2 tmem->shared copy drains 64 rows per cpt2s, needs rows % "
         "64 == 0, got "
      << rows;

  // Keep these symbolic: a chunked drain indexes tensor memory with the
  // surrounding loop var, and folding a non-constant origin to 0 would silently
  // re-copy the first tile.
  PrimExpr row_min = IntImm(DataType::Int(32), 0);
  PrimExpr col_min = IntImm(DataType::Int(32), 0);
  if (op.src_range.size() >= 1)
    row_min = op.src_range[0]->min;
  if (op.src_range.size() >= 2)
    col_min = op.src_range[1]->min;

  PrimExpr smem_offset = 0;
  {
    // One range per buffer axis; RegionOp builds the ranges from the
    // BufferLoad indices, so a partial region narrows the extents rather than
    // dropping axes. Stated here because the row-major strides below are
    // indexed by axis.
    ICHECK_EQ(op.dst_range.size(), smem_buf->shape.size())
        << "TANG stcuv2 tmem->shared copy expects one range per shared buffer "
           "axis, got "
        << op.dst_range.size() << " ranges for a " << smem_buf->shape.size()
        << "-D buffer";
    std::vector<PrimExpr> strides;
    PrimExpr stride = 1;
    for (size_t i = 0; i < smem_buf->shape.size(); ++i) {
      strides.insert(strides.begin(), stride);
      stride *= smem_buf->shape[smem_buf->shape.size() - i - 1];
    }
    for (size_t i = 0; i < op.dst_range.size(); ++i) {
      smem_offset += op.dst_range[i]->min * strides[i];
    }
  }
  PrimExpr elements = IntImm(DataType::Int(32), rows * cols);
  PrimExpr smem_addr = smem_buf.access_ptr(
      2, DataType::Handle(), 1, analyzer->Simplify(smem_offset), elements);

  // Unguarded on purpose: the helper elects warp 0 internally and ends with a
  // __syncthreads(), so every thread of the block has to reach it.
  return Evaluate(Call(DataType::Handle(), tang_cp_tmem_to_shared(),
                       {smem_addr,
                        BufferLoad(tmem_buf, {analyzer->Simplify(row_min),
                                              analyzer->Simplify(col_min)}),
                        IntImm(DataType::Int(32), rows)}));
}

// Stage a shared-memory tile into tensor memory with the cps2t copy engine,
// which moves the data without routing it through the register file.
//
// This is the A-operand staging path for the TS GEMM variant. Note that
// `shared -> shared.tmem` needs no annotation to say so: an accumulator is
// restored `fragment -> shared.tmem`, so this direction unambiguously means an
// operand. That is exactly the discriminator the 32-bit fragment path lacks
// (see IsTangTmemAOperandCopy), which is why a tf32 A goes global -> shared ->
// tmem rather than through registers.
//
// Each lane reads its own row, so the word index it is handed depends on how
// the shared tile was filled. A staging buffer takes the unswizzled 1D bulk
// copy, and then the index is plainly row-major. A buffer that is also a GEMM
// shared operand (or one whose load was explicitly annotated) was filled
// swizzled, and then the sw128a32 permutation has to be undone in software --
// cps2t cannot do it in hardware, because it rejects a per-lane offset together
// with a SwizzleMode. The guards below pin every assumption either map makes.
Stmt LowerTangSharedToTmemCopy(const CopyNode &op, const LowerArgs &lower_args,
                               arith::Analyzer *analyzer) {
  const Buffer &smem_buf = op.src;
  const Buffer &tmem_buf = op.dst;

  Array<IterVar> ivs = op.MakeIterVars();
  ICHECK_EQ(ivs.size(), 2U)
      << "TANG stcuv2 shared->tmem copy only supports 2D tiles";
  const auto *rimm = ivs[0]->dom->extent.as<IntImmNode>();
  const auto *cimm = ivs[1]->dom->extent.as<IntImmNode>();
  ICHECK(rimm && cimm)
      << "TANG stcuv2 shared->tmem copy requires constant tile extents";
  int rows = static_cast<int>(rimm->value);
  int cols = static_cast<int>(cimm->value);

  int elem_bits = static_cast<int>(smem_buf->dtype.bits());
  ICHECK_EQ(elem_bits, static_cast<int>(tmem_buf->dtype.bits()))
      << "TANG stcuv2 shared->tmem copy needs matching element widths, got "
      << smem_buf->dtype << " -> " << tmem_buf->dtype;
  ICHECK(rows % 32 == 0)
      << "TANG stcuv2 shared->tmem copy stages 32 rows per cps2t, needs rows % "
         "32 == 0, got "
      << rows;
  ICHECK((cols * elem_bits) % 32 == 0)
      << "TANG stcuv2 shared->tmem copy needs the staged row to fill whole "
         "32-bit tensor memory columns, got "
      << cols << " x " << elem_bits << "-bit";
  int col_words = cols * elem_bits / 32;

  // The permutation is anchored at the shared buffer's base, so a tile origin
  // would shift it. Staging the whole buffer keeps that honest.
  for (size_t i = 0; i < op.src_range.size(); ++i) {
    const auto *m = as_const_int(op.src_range[i]->min);
    ICHECK(m && *m == 0)
        << "TANG stcuv2 shared->tmem copy stages the whole shared buffer; the "
           "swizzle permutation is anchored at its base, so a non-zero tile "
           "origin (axis "
        << i << ") would read the wrong words";
  }
  // GetTmemOffset packs the *logical* column into the low half of the address,
  // while the copy engine steps 32-bit columns. Those agree only at column 0
  // unless elements are already word-sized.
  if (elem_bits != 32) {
    for (size_t i = 1; i < op.dst_range.size(); ++i) {
      const auto *m = as_const_int(op.dst_range[i]->min);
      ICHECK(m && *m == 0)
          << "TANG stcuv2 shared->tmem copy of a " << elem_bits
          << "-bit operand must start at tensor memory column 0: the address "
             "encoding counts elements but the copy engine counts 32-bit "
             "columns";
    }
  }

  int ndim = static_cast<int>(smem_buf->shape.size());
  ICHECK_GE(ndim, 2) << "TANG stcuv2 shared->tmem copy needs a 2D shared tile";
  const auto *cont = as_const_int(smem_buf->shape[ndim - 1]);
  ICHECK(cont) << "TANG stcuv2 shared->tmem copy requires a constant shared "
                  "row length";
  long long row_bits = static_cast<long long>(*cont) * elem_bits;
  ICHECK(row_bits % 1024 == 0)
      << "TANG stcuv2 shared->tmem copy needs the shared row to be a whole "
         "number of 128-byte swizzle units, got "
      << (row_bits / 8) << " bytes";
  int row_words = static_cast<int>(row_bits / 32);

  // How the shared tile was filled decides which word each lane has to read.
  // The signals are the same two LowerSTCUV2BulkCopy discriminates on, so the
  // two decisions cannot drift apart: a permuting shared layout or an explicit
  // swizzle annotation means the bulk load was swizzled, and anything else is a
  // staging buffer that took the unswizzled 1D path. A plain identity layout
  // (what InferLayout claims for staging buffers) is NOT a swizzle.
  const bool smem_layout_permutes =
      lower_args.layout_map.count(smem_buf) > 0 &&
      !IsTangIdentityLayout(lower_args.layout_map.at(smem_buf), smem_buf);
  const bool source_is_swizzled =
      smem_layout_permutes ||
      op.annotations.Get("tang_swizzle_atom_bytes").has_value();
  if (source_is_swizzled) {
    // The de-swizzle map is specific to the 32-byte atom. atom8/atom64 permute
    // differently and were never measured, so refuse rather than miscompute.
    int atom_bytes = 32;
    if (smem_layout_permutes) {
      const auto *p_stride = as_const_int(smem_buf->shape[ndim - 2]);
      if (p_stride && cont) {
        atom_bytes = TangTcgen5SwizzleAtomBytes(
            lower_args.layout_map.at(smem_buf), static_cast<int>(*p_stride),
            static_cast<int>(*cont), elem_bits);
      }
    }
    if (auto value = op.annotations.Get("tang_swizzle_atom_bytes")) {
      const auto *imm = value->as<IntImmNode>();
      ICHECK(imm) << "tang_swizzle_atom_bytes must be an IntImm";
      atom_bytes = static_cast<int>(imm->value);
    }
    ICHECK_EQ(atom_bytes, 32)
        << "TANG stcuv2 shared->tmem copy only supports a sw128a32 source tile "
           "(the de-swizzle map was measured for a 32-byte atom), but this "
           "shared buffer's layout implies a "
        << atom_bytes << "-byte atom";
  }

  PrimExpr row_min = IntImm(DataType::Int(32), 0);
  PrimExpr col_min = IntImm(DataType::Int(32), 0);
  if (op.dst_range.size() >= 1)
    row_min = op.dst_range[0]->min;
  if (op.dst_range.size() >= 2)
    col_min = op.dst_range[1]->min;

  PrimExpr elements = IntImm(DataType::Int(32), rows * cols);
  PrimExpr smem_addr = smem_buf.access_ptr(
      1, DataType::Handle(), 1, IntImm(DataType::Int(32), 0), elements);

  // Unguarded on purpose: the helper elects warp 0 internally and ends with a
  // __syncthreads(), so every thread of the block has to reach it.
  return Evaluate(Call(
      DataType::Handle(), tang_cp_shared_to_tmem(),
      {smem_addr,
       BufferLoad(tmem_buf,
                  {analyzer->Simplify(row_min), analyzer->Simplify(col_min)}),
       IntImm(DataType::Int(32), rows), IntImm(DataType::Int(32), col_words),
       IntImm(DataType::Int(32), row_words),
       IntImm(DataType::Int(32), source_is_swizzled ? 1 : 0)}));
}

Stmt Lower(const CopyNode &op, const LowerArgs &lower_args,
           arith::Analyzer *analyzer) {
  if (TargetTangIsSTCUV2(lower_args.target)) {
    if (IsTangTmemFragmentCopy(op)) {
      return LowerTangTmemFragmentCopy(op, lower_args, analyzer);
    }
    if (op.src.scope() == "shared.tmem") {
      if (op.dst.scope() == "global") {
        // LowerTangTmemDrain rewrites this normal loop after TMEM lowering.
        Stmt tmem_copy = LowerNormalCopy(op, lower_args, analyzer);
        // Mark where the drain begins. The rewriter replaces a whole loop with
        // the drain intrinsic, so it must know exactly which loop is the
        // drain's own outermost one. It cannot tell from the loop shape alone:
        // the drain nest has inner loops of its own, so "innermost loop holding
        // the store" is wrong, while "any loop whose subtree holds the store"
        // matches an enclosing user loop too -- and replacing that one silently
        // drops everything else in its body (the operand copies and the MMA of
        // a batched GEMM whose drain sits in the batch loop).
        //
        // The marker's value carries the accumulator's element type, as a typed
        // zero. LowerSharedTmem shrinks every TMEM buffer to a 1-element uint32
        // address holder before the drain rewrite runs, so by then the element
        // type is gone -- and the drain needs it to know how to reinterpret the
        // words it loads.
        tmem_copy = AttrStmt(Integer(0), "tmem_drain", make_zero(op.src->dtype),
                             tmem_copy);
        // Carry a user `drain_warps=N` cap to the post-lowering
        // LowerTangTmemDrain pass. Loop annotations do not survive the
        // fuse/partition in LowerNormalCopy, so wrap the drain in a durable
        // AttrStmt instead.
        if (auto v = op.annotations.Get("drain_warps")) {
          if (const auto *imm = v.value().as<IntImmNode>()) {
            tmem_copy =
                AttrStmt(Integer(0), "tmem_drain_warps",
                         IntImm(DataType::Int(32), imm->value), tmem_copy);
          }
        }
        return tmem_copy;
      }
      if (op.dst.scope() == "shared" || op.dst.scope() == "shared.dyn") {
        if (!HasTangTcgen05CpMarker(op)) {
          LOG(FATAL) << "TANG stcuv2: T.copy() does not move tensor memory "
                        "(shared.tmem) into shared memory. Use "
                        "T.tang_cp_tmem_to_shared(dst, src), the named "
                        "primitive for the cpt2s copy engine -- it validates "
                        "the shared tile shape this path requires. The fused "
                        "TMEM -> global drain, T.copy(tmem, global), is "
                        "unaffected.";
        }
        return LowerTangTmemToSharedCopy(op, lower_args, analyzer);
      }
      LOG(FATAL)
          << "TANG stcuv2: copying directly from tensor memory "
             "(shared.tmem) to a '"
          << op.dst.scope()
          << "' buffer is not supported. Drain tensor memory to global memory, "
             "or stage it through shared memory.";
    }
    if (op.dst.scope() == "shared.tmem") {
      if (IsTangSharedToTmemCopy(op)) {
        if (!HasTangTcgen05CpMarker(op)) {
          LOG(FATAL) << "TANG stcuv2: T.copy() does not move shared memory "
                        "into tensor memory (shared.tmem). Use "
                        "T.tcgen05_cp(dst, src), the named primitive for the "
                        "cps2t copy engine -- it validates the source swizzle "
                        "and tile granularity this path requires.";
        }
        return LowerTangSharedToTmemCopy(op, lower_args, analyzer);
      }
      LOG(FATAL) << "TANG stcuv2: copying from a '" << op.src.scope()
                 << "' buffer into tensor memory (shared.tmem) is not "
                    "supported. Stage the data in shared memory "
                    "(T.tcgen05_cp) or in a fragment (T.tcgen05_st) first.";
    }
    bool global_to_shared =
        op.src.scope() == "global" &&
        (op.dst.scope() == "shared" || op.dst.scope() == "shared.dyn");
    bool shared_to_global =
        (op.src.scope() == "shared" || op.src.scope() == "shared.dyn") &&
        op.dst.scope() == "global";
    if ((global_to_shared || shared_to_global) && !IsTangBulkCopyDisabled(op)) {
      return LowerSTCUV2BulkCopy(op, lower_args, analyzer);
    }
  }
  return LowerNormalCopy(op, lower_args, analyzer);
}

bool RegisterTangCopy() {
  RegisterCopyImpl(CopyImpl{
      "tang.Copy",
      TargetIsTang,
      100,
      InferLayout,
      Lower,
  });
  return true;
}

const bool tang_copy_registered = RegisterTangCopy();

} // namespace
} // namespace tang
} // namespace tl
} // namespace tvm
