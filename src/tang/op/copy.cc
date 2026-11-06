/*!
 * \file tl/tang/op/copy.cc
 * \brief TANG implementation for ordinary tile copies.
 */

#include "op/copy.h"
#include "cuda/op/builtin.h"
#include "layout/cute_layout.h"
#include "layout/layout.h"
#include "layout/tcgen05_layout.h"
#include "op/builtin.h"
#include "op/parallel.h"
#include "op/utils.h"
#include "tang/op/builtin.h"
#include "tang/target_utils.h"
#include "tang/transform/s3_async_pipeline_sync_schema.h"
#include "transform/common/loop_fusion_utils.h"
#include "transform/loop_partition.h"
#include "transform/loop_vectorize.h"

#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>

#include <cstdlib>
#include <optional>
#include <sstream>
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

// Whether `layout` describes a permutation of the shared tile itself -- the
// only thing a bulk copy can fill -- rather than a reshape of it.
//
// A bulk copy lands one element per position: it applies the copy engine's
// swizzle to the tile and reads the declared layout only for the tiling
// geometry that swizzle is parameterised by, so the declared map has to be a
// permutation of the same tile. Every layout the tang lowering assigns is one
// (``_make_tcgen5_segment_layout`` and ``_make_tang_ab_layout`` in Python both
// keep the region's shape). A CUDA ``make_swizzled_layout`` is not: it declares
// the buffer as a stack of swizzle units -- (32, 128) bf16 becomes
// (2, 4, 512) -- because its consumer is a tensor-map descriptor that reads
// that reshaped view. Filling such a buffer with a bulk copy leaves the
// engine's swizzle and the declared map disagreeing, and nothing reports it:
// the loaded elements come back permuted.
//
// Stated on the shapes rather than on the index expressions: the families that
// are legal here keep the region's geometry, and comparing shapes cannot
// misfire on an expression form the copy path has not seen.
bool TangLayoutKeepsTileShape(const Layout &layout) {
  Array<PrimExpr> in_shape = layout->InputShape();
  Array<PrimExpr> out_shape = layout->OutputShape();
  if (in_shape.size() != out_shape.size() || in_shape.empty()) {
    return false;
  }
  for (size_t i = 0; i < in_shape.size(); ++i) {
    if (!StructuralEqual()(in_shape[i], out_shape[i])) {
      return false;
    }
  }
  return true;
}

// Bytes of one operand row a single MMA reads (gemm_tcgen05_descriptors.py
// MMA_OPERAND_ROW_BYTES). A K-major operand row wider than this is stored as
// this many-byte segments (see below).
constexpr int64_t kMmaOperandRowBytes = 128;

// Whether the GEMM declared this shared operand segment-major, and in how many
// 128-byte segments.
//
// A K-major operand whose row exceeds one MMA read is stored as a stack of
// standalone 128-byte-row operands (gemm_tmma.py
// _make_tang_ab_layout_segmented), and the bulk copy that fills it has to land
// the same shape: one swizzled copy per segment. A single copy of the wide row
// would place a swizzled pattern that no descriptor can address -- the row's
// swizzle unit spans both halves of the row, which is exactly what makes the
// wide row unreadable in the first place.
//
// The count is read off the *declared layout* rather than off the tile
// geometry, because the geometry alone cannot tell a K-major wide row (which
// must be split) from an MN-major axis of the same width (which must not be:
// there the wide extent is M or N, so cutting it would split the operand along
// the wrong axis and every value would land in the wrong place). Only the
// segmented declaration has the shape signature this looks for -- output
// columns are one segment, output rows are the input rows times the segment
// count.
//
// Returns 1 for everything else, including MN-major operands and plain layouts.
int64_t TangSegmentedOperandSegments(const Layout &layout, int bits) {
  if (bits <= 0)
    return 1;
  auto in_shape = layout->InputShape();
  auto out_shape = layout->OutputShape();
  if (in_shape.size() != 2 || out_shape.size() != 2)
    return 1;
  const int64_t *in_rows = as_const_int(in_shape[0]);
  const int64_t *in_cols = as_const_int(in_shape[1]);
  const int64_t *out_rows = as_const_int(out_shape[0]);
  const int64_t *out_cols = as_const_int(out_shape[1]);
  if (!in_rows || !in_cols || !out_rows || !out_cols || *out_cols <= 0)
    return 1;
  if (*in_cols % *out_cols != 0)
    return 1;
  const int64_t segments = *in_cols / *out_cols;
  if (segments <= 1)
    return 1;
  if (*out_rows != *in_rows * segments)
    return 1;
  if (*out_cols * bits != kMmaOperandRowBytes * 8)
    return 1;
  return segments;
}

// ---------------------------------------------------------------------------
// IR-emitted 1D bulk copies
// ---------------------------------------------------------------------------
// Emit the warp gate, row split, drain and synchronization as IR so later
// synchronization passes can inspect them. Geometry must be compile-time
// constant because the runtime branch form is not supported by stcuv2.

// The statement a copy emitter returns: its pieces, which are sometimes just
// one (a store that states nothing but the transfer), and a length-1 SeqStmt is
// rejected outright.
Stmt AsBody(const Array<Stmt> &body) {
  ICHECK(!body.empty());
  return body.size() == 1 ? body[0] : SeqStmt(body);
}

// How many versions a pipelined shared buffer has, i.e. how many stages the
// software pipeline gave it. The version dim is a leading dim the copy slices
// to a single index, and one whole tile is its stride -- the same tile size the
// stage division uses. A buffer without such a dim is not rotated (1).
//
// The handshake pass needs this number and cannot read it off the stage: the
// pipeline may reindex a rotated buffer affinely, leaving no version count in
// the stage expression.
int64_t PipelinedVersions(const Buffer &buffer, const Array<Range> &ranges,
                          const std::vector<PrimExpr> &strides,
                          const PrimExpr &tile_elements) {
  const auto *tile = as_const_int(tile_elements);
  if (tile == nullptr) {
    return 1;
  }
  for (size_t d = 0; d < ranges.size() && d < buffer->shape.size(); ++d) {
    const auto *extent = as_const_int(ranges[d]->extent);
    if (extent == nullptr || *extent != 1) {
      break; // leading single-index slices only; past them are the tile dims
    }
    const auto *stride = as_const_int(strides[d]);
    if (stride == nullptr || *stride != *tile) {
      continue; // a sliced batch dim, not the version dim
    }
    const auto *dim = as_const_int(buffer->shape[d]);
    if (dim != nullptr && *dim > 1) {
      return *dim;
    }
  }
  return 1;
}

// How the handshake pass names the buffer a copy fills: the *data variable* of
// the copy's shared side, spelled the same way inject_s3_async_copy.cc spells
// it. Deliberately a string and not the access_ptr itself -- a call carrying a
// shared pointer argument looks like a shared access to the sync analysis.
String KeyOfSharedSide(const PrimExpr &access_ptr) {
  const auto *call = access_ptr.as<CallNode>();
  if (call == nullptr || call->args.size() < 2) {
    return String("");
  }
  std::ostringstream os;
  os << call->args[1];
  return String(os.str());
}

Array<PrimExpr> MakeCopyRecord(const PrimExpr &shared_addr,
                               const PrimExpr &stage, int64_t versions,
                               int64_t arrive, int64_t num_warps,
                               int64_t direction,
                               const transform::PipelinePlan &plan,
                               bool pipeline_ready) {
  transform::PipelinePlan record_plan = plan;
  if (pipeline_ready) {
    record_plan.ready_owner =
        transform::PipelineReadyOwner::kInjectSoftwarePipeline;
  }
  return {
      IntImm(DataType::Int(32), transform::kPipelinePlanSchemaVersion),
      StringImm(KeyOfSharedSide(shared_addr)),
      stage,
      IntImm(DataType::Int(32), versions),
      IntImm(DataType::Int(32), arrive),
      IntImm(DataType::Int(32), num_warps),
      IntImm(DataType::Int(32), direction),
      IntImm(DataType::Int(64), record_plan.loop_uid),
      IntImm(DataType::Int(64), record_plan.buffer_uid),
      IntImm(DataType::Int(32), record_plan.distance),
      IntImm(DataType::Int(32), record_plan.producer_stage),
      IntImm(DataType::Int(32), record_plan.consumer_stage),
      IntImm(DataType::Int(32), static_cast<int64_t>(record_plan.ready_owner))};
}

// The pipeline-version expression of a whole-version access_ptr: the pipeline
// writes a rotated buffer as ``offset == stage * extent``, so the stage is that
// offset over the extent (a constant 0 for a copy that is not rotated). Keep
// the stripe offset out of this pointer so the division stays exact.
PrimExpr WholeVersionStage(const PrimExpr &access_ptr,
                           arith::Analyzer *analyzer) {
  const auto *call = access_ptr.as<CallNode>();
  if (call == nullptr || call->args.size() < 4) {
    return IntImm(DataType::Int(32), 0);
  }
  const size_t n = call->args.size();
  const PrimExpr &offset = call->args[n - 3];
  const auto *extent = call->args[n - 2].as<IntImmNode>();
  if (extent == nullptr || extent->value <= 0) {
    return IntImm(DataType::Int(32), 0);
  }
  return analyzer->Simplify(
      floordiv(offset, IntImm(offset.dtype(), extent->value)));
}

// One ``fcpg2s`` / ``fcps2g`` and nothing else, guarded by the warp gate.
//
// dst_addr / src_addr are the whole-tile (or whole-segment) base pointers, i.e.
// exactly what the helper is handed today, and the stripe offsets are applied
// on top of them here. Keeping the pipeline-version offset in the base pointer
// and the stripe offset out of it is deliberate: the lowering records the
// whole-version stage next to the copy while ThreadSync reads the extent.
// Geometry that does not fold is *refused*, not emitted.
//
// A runtime pitch branch does not compile on stcuv2 because ptcc requires the
// set_mem preamble and its bulk-copy consumer to remain adjacent. Refuse it
// here so the diagnostic names the copy that caused it.
void RefuseUnfoldableBulkCopy(const std::string &buffer, const PrimExpr &rows,
                              const PrimExpr &smem_row_bytes,
                              const PrimExpr &gmem_row_bytes) {
  LOG(FATAL) << "TANG STCUV2: bulk copy of buffer '" << buffer
             << "' has geometry that does not fold -- rows = " << rows
             << ", shared row pitch = " << smem_row_bytes
             << " B, global row pitch = " << gmem_row_bytes << " B.\n"
             << "  The runtime-geometry form of this copy does not compile on "
                "stcuv2 (ptcc: \"set_mem.* lost adjacency to its "
                "cp_async_bulk consumer\"), so lowering refuses it rather "
                "than emitting it.\n"
             << "  Make the geometry constant: static tile shapes, and a "
                "global buffer whose row pitch is a constant (a runtime "
                "extent in the global tensor makes it symbolic).";
}

Stmt EmitStcuv21DCopy(const std::string &buffer, int direction,
                      const PrimExpr &dst_addr, const PrimExpr &src_addr,
                      const PrimExpr &thread_index, const PrimExpr &total_rows,
                      const PrimExpr &row_bytes, const PrimExpr &gmem_bytes,
                      int num_threads, int warp_base, int64_t versions,
                      const std::optional<transform::PipelinePlan> &plan,
                      const PrimExpr &pipeline_barrier,
                      arith::Analyzer *analyzer) {
  LOG(FATAL) << "STCUV2 bulk copy is not supported";
  return Stmt();
}

// ---------------------------------------------------------------------------
// IR-emitted swizzled 3D bulk copies
// ---------------------------------------------------------------------------
// The swizzled 3D pair used to hide the same kind of structure the 1D helpers
// did: stripe geometry, a region-relative warp gate, a fence-group drain --
// and, for the store, a block-wide __syncthreads() on *each* side of the copy.
// None of that is visible to the sync passes, which is why the
// redundant-barrier work had to match kernels by shape. This emitter states the
// structure as IR instead, so the gate, the addresses, the drain and the
// barriers are all readable.
//
// The split itself is the ISA's own geometry: 512 bytes per stripe divided by
// the swizzle atom. The row count folds (a runtime one is refused), so every
// quantity in the balanced split (`active`, `base`, `rem`) is a constant and
// what is left at run time is the region warp id and two predicates.
//
// The pointers handed in here are the whole-tile (or whole-segment) base
// addresses and the stripe offsets are applied on top, exactly as in
// EmitStcuv21DCopy: keep the version offset in the base address so the lowering
// can record the whole-version stage.

// One fcpg2s/fcps2g and nothing else, guarded by the warp gate. `row_off` is
// the stripe origin in *rows*; both sides advance by their own pitch from it.
Stmt EmitStcuv23DCopy(const std::string &buffer, int direction,
                      const PrimExpr &dst_addr, const PrimExpr &src_addr,
                      const PrimExpr &thread_index, const PrimExpr &total_rows,
                      const PrimExpr &smem_row_bytes,
                      const PrimExpr &gmem_row_bytes, int num_threads,
                      int warp_base, int swizzle_mode, int pack_mode,
                      int64_t atom_bytes, int64_t versions,
                      const std::optional<transform::PipelinePlan> &plan,
                      const PrimExpr &pipeline_barrier,
                      arith::Analyzer *analyzer) {
  LOG(FATAL) << "STCUV2 bulk copy is not supported";
  return Stmt();
}

Stmt LowerSTCUV2BulkCopy(const CopyNode &op, const LowerArgs &lower_args,
                         arith::Analyzer *analyzer) {
  std::optional<transform::PipelinePlan> pipeline_plan;
  {
    Array<Integer> encoded{Integer(transform::kPipelinePlanSchemaVersion)};
    bool complete = true;
    for (const char *key :
         {transform::kPipelineLoopUidAttr, transform::kPipelineBufferUidAttr,
          transform::kPipelineVersionsAttr, transform::kPipelineDistanceAttr,
          transform::kPipelineProducerStageAttr,
          transform::kPipelineConsumerStageAttr,
          transform::kPipelineReadyOwnerAttr}) {
      auto value = op.annotations.Get(key);
      const auto *imm =
          value.has_value() ? value.value().as<IntImmNode>() : nullptr;
      if (imm == nullptr) {
        complete = false;
        break;
      }
      encoded.push_back(Integer(imm->value));
    }
    transform::PipelinePlan plan;
    if (complete && transform::ParsePipelinePlan(encoded, &plan)) {
      pipeline_plan = plan;
    }
  }
  // The pipeline's barrier, when the generic expansion owns this copy's edge:
  // the copy then posts its own arrival on it (see EmitStcuv21DCopy /
  // EmitStcuv23DCopy) and the handshake pass no longer has to. Absent for every
  // copy the handshake pass still wires.
  PrimExpr pipeline_barrier;
  int64_t pipeline_arrive_count = 0;
  if (auto barrier = op.annotations.Get("barrier");
      barrier.has_value() && pipeline_plan.has_value()) {
    pipeline_plan->ready_owner =
        transform::PipelineReadyOwner::kInjectSoftwarePipeline;
    pipeline_barrier = Downcast<PrimExpr>(barrier.value());
    auto declared = op.annotations.Get("tang_pipeline_arrive_count");
    ICHECK(declared.has_value())
        << "a stcuv2 pipeline-owned copy is missing ISP's "
           "tang_pipeline_arrive_count declaration";
    const auto *imm = declared.value().as<IntImmNode>();
    ICHECK(imm != nullptr && imm->value > 0)
        << "tang_pipeline_arrive_count must be a positive IntImm";
    pipeline_arrive_count = imm->value;
  }
  if (pipeline_plan.has_value() &&
      pipeline_plan->ready_owner ==
          transform::PipelineReadyOwner::kInjectSoftwarePipeline &&
      !pipeline_barrier.defined()) {
    pipeline_plan.reset();
  }
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
  const int64_t versions =
      PipelinedVersions(shared_tensor, shared_range, shared_strides, elements);
  if (pipeline_plan.has_value() && pipeline_plan->versions != versions) {
    pipeline_plan.reset();
    pipeline_barrier = PrimExpr();
    pipeline_arrive_count = 0;
  }
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

  // A one-dimensional shared tile carries no row structure of its own: the tile
  // is exactly the run of elements the global side hands over. Deriving rows
  // from that flat extent splits a contiguous fill (`X[i, h, 0:n] -> S[0:n]`)
  // into n rows of a single element and then rejects it as an unaligned row
  // copy, although the transfer is one contiguous 2n-byte chunk. Take the tile
  // from the global side whenever the shared side is flat and the global side
  // is a single contiguous run (every leading extent unit); a strided gather
  // into a flat tile keeps the row decomposition, which is what makes its
  // unaligned geometry get reported.
  if (shared_range.size() == 1) {
    bool contiguous_run = !global_range.empty();
    for (size_t d = 0; d + 1 < global_range.size(); ++d) {
      contiguous_run &= is_unit(global_range[d]->extent);
    }
    contiguous_run &= !global_range.empty() &&
                      analyzer->CanProveEqual(global_range.back()->extent,
                                              shared_range[0]->extent);
    if (contiguous_run) {
      rows = PrimExpr(1);
      cols = shared_range[0]->extent;
      col_bytes =
          analyzer->Simplify(FloorDiv(cols * shared_tensor->dtype.bits(), 8));
    }
  }
  PrimExpr gmem_row_bytes = analyzer->Simplify(
      FloorDiv(gmem_row_stride * global_tensor->dtype.bits(), 8));

  // Which of the three bulk-copy regimes this copy belongs to. The swizzled 3D
  // pair does NOT round-trip: a tile loaded by fcpg2s.3d and stored back by
  // fcps2g.3d under the same SwizzleMode comes out permuted at atom
  // granularity, and that is an ISA-level asymmetry rather than a
  // parameterisation mistake on our side -- driving the vendor's own public
  // wrappers with the round-trip recipe documented in cp_async_bulk.h
  // reproduces it exactly. So the swizzled form is only
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
           "load, so streaming it "
           "back to global would silently scramble it.\n"
        << "  The only validated swizzled store is the cpt2s drain (tensor "
           "memory -> shared -> global); annotate both copies with "
           "`tang_swizzle_atom_bytes: 8` if that is what this is.\n"
        << "  To stage data through shared purely to get it back to global, "
           "use a buffer that is not also a GEMM operand: it then takes the "
           "unswizzled 1D path.";
  }

  // Segment-major MMA operand (K-major row over one 128-byte MMA read): the
  // layout says how many segments the row is stored in, and the copy has to
  // fill exactly that shape -- one swizzled copy per segment (see
  // TangSegmentedOperandSegments).
  const int64_t segments = is_mma_operand
                               ? TangSegmentedOperandSegments(
                                     lower_args.layout_map.at(shared_tensor),
                                     shared_tensor->dtype.bits())
                               : 1;

  // Swizzle atom (32 or 64 bytes). An explicit annotation wins; otherwise
  // recover it from the GEMM-assigned shared-operand layout so the copy's
  // hardware swizzle matches the MMA descriptor (MN-major tf32/int8 -> atom64).
  int atom_bytes = 32;
  if (auto value = op.annotations.Get("tang_swizzle_atom_bytes")) {
    const auto *imm = value->as<IntImmNode>();
    ICHECK(imm) << "tang_swizzle_atom_bytes must be an IntImm";
    atom_bytes = static_cast<int>(imm->value);
    ICHECK(segments == 1 || atom_bytes == 32)
        << "TANG STCUV2: segmented MMA operand '" << shared_tensor->name
        << "' is annotated with a " << atom_bytes
        << "-byte swizzle atom, but a segment-major K-major operand is always "
           "sw128a32 (32-byte atom). Drop the annotation.";
  } else if (is_mma_operand && segments == 1) {
    // A segmented operand is skipped here on purpose: its declared layout is a
    // stack of operands, not one operand tiling, so the atom recovery below
    // would have to guess. It is fixed at 32 bytes (sw128a32) instead -- the
    // only swizzle a K-major operand uses, and the one the MMA reads the
    // segment with.
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

  // A swizzle *derived from the declared layout* is only reproducible if that
  // layout is something the copy engine can land. Refuse the rest, because the
  // alternative is silent corruption: the copy writes the hardware swizzle,
  // every other access follows the declared map, and nothing reports the
  // difference. Reached only when the layout is what chose the swizzle -- an
  // explicit `tang_swizzle_atom_bytes` states the atom itself, and a
  // segment-major operand declares a stack of segments whose geometry the copy
  // lands one segment at a time.
  if (is_mma_operand && !has_atom_annotation && !is_store && segments == 1) {
    const Layout &declared = lower_args.layout_map.at(shared_tensor);
    ICHECK(TangLayoutKeepsTileShape(declared))
        << "TANG STCUV2: shared buffer '" << shared_tensor->name
        << "', which a bulk copy fills, declares a layout the copy cannot "
           "land: it reshapes the tile instead of permuting it.\n"
        << "  A bulk copy writes one element per position and applies the copy "
           "engine's own swizzle, reading the declared layout only for the "
           "tiling that swizzle is parameterised by. A layout that reshapes "
           "the buffer (a different output scope, e.g. a stack of swizzle "
           "units) describes how a descriptor reads it, which the copy cannot "
           "reproduce: the fill and the buffer's declared mapping then "
           "disagree and the data comes back permuted with no error.\n"
        << "  On stcuv2 that is what a CUDA "
           "`T.annotate_layout(make_swizzled_layout(...))` produces. Drop the "
           "annotation: the buffer is plain staging, and bulk copies of it "
           "then stay unswizzled on both legs. To state a swizzle on purpose, "
           "annotate the copy with `tang_swizzle_atom_bytes` instead.\n"
        << "  Layout: " << declared->DebugOutput();
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
    if (segments > 1) {
      LOG(FATAL) << "TANG STCUV2: unpacked fp4 operand '" << shared_tensor->name
                 << "' is a segment-major operand (its " << col_bytes
                 << "-byte shared row is " << segments
                 << " MMA rows), and b4p4x16 has only been validated for a "
                    "single 128-byte row.\n"
                 << kWhy
                 << "  Tile K at 128 so the operand is one segment, or "
                    "pre-unpack in global and use a matching unpacked dtype.";
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
  if (pipeline_barrier.defined()) {
    ICHECK_EQ(pipeline_arrive_count, num_threads / 32)
        << "InjectSoftwarePipeline declared " << pipeline_arrive_count
        << " stcuv2 arrivals for one engine copy, but its lowering region has "
        << num_threads / 32
        << " warps. Conditional/role-split copies must not be engine copies.";
  }
  // Warp base of the *region* this copy runs in. Inside a T.ws region
  // (`with T.ws(w, warp_group_size=32)`) the copy is executed by a subset of
  // the block's warps, and thread_bounds is that subset -- so the helper's "one
  // warp does the tile" gate and its stripe split have to be indexed from the
  // region. Handing it the block-relative warp id is what made a copy outside
  // warp 0 move no bytes at all and post no barrier arrival, which deadlocked
  // every consumer waiting on its barrier.
  ICHECK(is_const_int(lower_args.thread_bounds->min))
      << "TANG bulk copy requires a constant thread lower bound "
         "(thread_bounds), got "
      << lower_args.thread_bounds;
  const int warp_base =
      static_cast<int>(*as_const_int(lower_args.thread_bounds->min) / 32);

  // Per-segment addresses for a segmented MMA operand. Segment s occupies its
  // own tile in shared memory (the operand's rows of one 128-byte MMA row, with
  // the segments stacked) and its own column range in the global tile, so both
  // ends of the copy are the whole-tile addresses plus one segment's step. With
  // one segment these are exactly the addresses this copy used before, and the
  // per-segment bodies below collapse back to a single statement.
  PrimExpr col_bytes_one = col_bytes;
  PrimExpr seg_shared_step = rows;
  if (segments > 1) {
    const int64_t *c_col_bytes = as_const_int(col_bytes);
    ICHECK(c_col_bytes && *c_col_bytes % segments == 0)
        << "TANG STCUV2: segmented MMA operand '" << shared_tensor->name
        << "' has a " << col_bytes << "-byte shared row, which is not "
        << segments << " whole " << kMmaOperandRowBytes << "-byte MMA rows.";
    // One segment row: its bytes, and the elements it holds (which is also the
    // step between two segments in the global tile, since the segments are
    // consecutive column ranges of one row).
    const int64_t one_bytes = *c_col_bytes / segments;
    const int64_t one_elems = one_bytes * 8 / shared_tensor->dtype.bits();
    col_bytes_one = IntImm(col_bytes.dtype(), one_bytes);
    seg_shared_step = IntImm(shared_offset.dtype(), one_elems);
  }
  auto segment_offset = [&](const PrimExpr &base, const PrimExpr &step,
                            int64_t s) {
    return analyzer->Simplify(base + IntImm(step.dtype(), s) * step);
  };
  auto shared_part = [&](int64_t s) -> PrimExpr {
    if (segments == 1)
      return shared_addr;
    // The offset is a *logical* element offset: the declared layout maps it
    // into the physical segment (which is why a segment origin is a column
    // offset in row 0 -- `s` segments along K -- rather than the segment's
    // physical tile stride). The extent is one segment's element count.
    PrimExpr off = segment_offset(shared_offset, seg_shared_step, s);
    return shared_tensor.access_ptr(is_load ? 2 : 1, DataType::Handle(), 1, off,
                                    analyzer->Simplify(seg_shared_step * rows));
  };
  auto global_part = [&](int64_t s) -> PrimExpr {
    if (segments == 1)
      return global_addr;
    PrimExpr off = segment_offset(global_offset, seg_shared_step, s);
    return global_tensor.access_ptr(is_load ? 1 : 2, DataType::Handle(), 1, off,
                                    analyzer->Simplify(seg_shared_step * rows));
  };
  auto dst_part = [&](int64_t s) -> PrimExpr {
    return is_load ? shared_part(s) : global_part(s);
  };
  auto src_part = [&](int64_t s) -> PrimExpr {
    return is_load ? global_part(s) : shared_part(s);
  };
  auto emit_per_segment =
      [&](const std::function<Stmt(int64_t)> &build) -> Stmt {
    if (segments == 1)
      return build(0);
    std::vector<Stmt> stmts;
    stmts.reserve(static_cast<size_t>(segments));
    for (int64_t s = 0; s < segments; ++s)
      stmts.push_back(build(s));
    return SeqStmt(std::move(stmts));
  };
  // Fulfil ISP's ready-edge contract mechanically: one publication per
  // logical copy, reached by every warp in this (unconditional) block region.
  // Warps with no stripe bind an empty fence group and arrive immediately;
  // issuing warps arrive only after their own DMA group retires. In both cases
  // the count is exactly ``pipeline_arrive_count`` and does not depend on copy
  // geometry or on how many MMA segments the logical copy lowers into.
  const auto publish_pipeline = [&](Stmt issued) -> Stmt {
    if (!pipeline_barrier.defined()) {
      return issued;
    }
    return SeqStmt({issued, Evaluate(Call(DataType::Handle(),
                                          tang_fence_g2s_arrive_mbarrier(),
                                          {pipeline_barrier}))});
  };

  // A pack-mode copy needs no separate path: the swizzled 3D emitter below
  // passes the pack mode straight to the fcpg2s template argument, and the
  // stripe geometry is the same one the wrapper used. The guards above are what
  // make the mode usable at all (they reject every shape the hardware
  // mis-reads), so pack_mode is simply carried into EmitStcuv23DCopy.

  if (!has_atom_annotation && !is_mma_operand) {
    // Pure staging buffer: unswizzled 1D. The 1D overloads round `size` UP to
    // the 128-byte transfer chunk, which would write past the destination, so
    // reject shapes that would rely on that instead of silently overrunning.
    // Which size matters depends on how the tile is transferred: one flat call
    // when the pitches match, one call per row otherwise. The geometry folds by
    // the time the emitter runs (it would refuse otherwise), so these are
    // constants here and the check below is unconditional.
    const auto *c_rows = as_const_int(rows);
    const auto *c_col_bytes = as_const_int(col_bytes);
    const auto *c_gmem_row_bytes = as_const_int(gmem_row_bytes);
    if (c_rows != nullptr && c_col_bytes != nullptr &&
        c_gmem_row_bytes != nullptr) {
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
            << "  A staging buffer cannot use the swizzled path because it "
               "does not round-trip, and the "
               "1D path would round the size up and overrun the destination.\n"
            << "  Pad the shared tile so "
            << (contiguous ? "rows * row_bytes" : "row_bytes")
            << " is a multiple of 128.";
      }
    }
    // Emit the copy structure as IR (see EmitStcuv21DCopy). There is no
    // monolithic call or runtime fallback:
    // geometry that does not fold is refused by the emitter itself. A planned
    // shape also carries a `tl.tang_bulk_async_commit` record so the sync
    // passes can materialize its declared ordering.
    //
    // A 1D staging copy consumed by a register copy has no tensor-core
    // retirement event, so it keeps the blocking path.
    return publish_pipeline(emit_per_segment([&](int64_t s) {
      return EmitStcuv21DCopy(shared_tensor->name, direction, dst_part(s),
                              src_part(s), lower_args.thread_index, rows,
                              col_bytes_one, gmem_row_bytes, num_threads,
                              warp_base, versions, pipeline_plan,
                              pipeline_barrier, analyzer);
    }));
  }

  // ---- swizzled 3D forms ----
  // Reached only by copies that are neither a pure staging buffer (the 1D
  // block above) nor a pack-mode copy (the branch above, which is load-only).
  //
  // Swizzle mode codes consumed by codegen: 4 = sw128a8, 6 = sw128a32 (the
  // hard-coded default), 7 = sw128a64. The mode is also what sizes the 3D
  // stripe split (512 / atom_bytes), so it is derived once here.
  const int swizzle_mode = atom_bytes == 64 ? 7 : (atom_bytes == 8 ? 4 : 6);

  // Emit the copy's structure as IR. The emitter rejects row counts that are
  // not compile-time constants. Loads also publish a record consumed by
  // InjectS3AsyncCopy; stores do not participate in that handshake.
  //
  // This sits after the 1D block on purpose: a staging buffer's store is a 1D
  // copy and must stay one. Placing it earlier made the staging shapes emit a
  // *swizzled* store instead -- same byte count, different instruction, and for
  // a sub-tile (shared row != global row) the wrong bytes.
  //
  // The version count is stated to the pass rather than recovered from the
  // addresses: a pipeline may reindex a rotated buffer affinely, and then the
  // offset may carry no `k % versions` term from which to recover it.
  ICHECK(atom_bytes == 8 || atom_bytes == 32 || atom_bytes == 64)
      << "TANG STCUV2 bulk copy supports swizzle atom sizes 8, 32 or 64 bytes";
  return publish_pipeline(emit_per_segment([&](int64_t s) {
    return EmitStcuv23DCopy(
        shared_tensor->name, direction, dst_part(s), src_part(s),
        lower_args.thread_index, rows, col_bytes_one, gmem_row_bytes,
        num_threads, warp_base, swizzle_mode, pack_mode, atom_bytes, versions,
        pipeline_plan, pipeline_barrier, analyzer);
  }));
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
// row is exactly one 128-byte swizzle unit wide. A single call drains 64 rows
// (``kCpt2s``), so a taller tile is a counted loop around the leaf op -- the
// loop, strides and fence are IR so the synchronization passes can see them.
constexpr int64_t kCpt2sRowsPerCall = 64;
constexpr int64_t kCpt2sBytesPerCall =
    512 * (128 / 8); // sw128a8: 512 * (swizzle bytes / atom bytes)
constexpr int64_t kFenceGroupT2s = 2; // fg_t2s_default
constexpr int64_t kFenceGroupS2t = 3; // fg_s2t_default
// Rows one cps2t call stages (one 32x32b group; the engine walks the 32 rows
// itself from its per-lane addressing mode).
constexpr int64_t kCps2tRowsPerCall = 32;

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

  const auto c = [](int64_t v) { return IntImm(DataType::Int(32), v); };
  // Region-relative warp id: the *region's* first warp, not the block's. Inside
  // a T.ws region the block's warp 0 need not be one of the warps that execute
  // this copy at all, which would make a block-relative election copy nothing.
  ICHECK(is_const_int(lower_args.thread_bounds->min) &&
         is_const_int(lower_args.thread_bounds->extent))
      << "TANG tmem->shared copy requires constant thread bounds, got "
      << lower_args.thread_bounds;
  const int64_t warp_base =
      static_cast<int64_t>(*as_const_int(lower_args.thread_bounds->min) / 32);
  const int64_t region_warps = static_cast<int64_t>(
      *as_const_int(lower_args.thread_bounds->extent) / 32);
  PrimExpr warp_id = analyzer->Simplify(
      floordiv(lower_args.thread_index, c(32)) - c(warp_base));

  // Kept as the BufferLoad (the tmem buffer still carries its logical element
  // type here) with the starting row passed separately: the engine reads the
  // row out of the address high half, so codegen shifts it in (`(row << 16)`).
  PrimExpr taddr = BufferLoad(
      tmem_buf, {analyzer->Simplify(row_min), analyzer->Simplify(col_min)});

  // Split the row blocks across the region's warps, one contiguous run each --
  // the rotation ExpandS3TmemDrain uses for the multi-warp writeback
  // (`r = warp*16; r += nwarps*16`). Two reasons it beats the single-warp form
  // the helper had: the participating warps then match whoever consumes the
  // staged tile (a bulk s2g store is issued by the leading warps), and when the
  // tile is tall enough for every warp to get a run the guard folds away
  // entirely -- the common case, and one the single-warp form could not reach.
  const int64_t chunks = rows / kCpt2sRowsPerCall;
  int64_t active = std::min<int64_t>(region_warps, chunks);
  if (active < 1 || chunks % active != 0) {
    active = 1; // keep every warp's run whole instead of splitting unevenly
  }
  const int64_t per_warp = chunks / active;
  auto copy_at = [&](const PrimExpr &blk) {
    return Evaluate(
        Call(DataType::Handle(), tang_cpt2s_16x64b_2d(),
             {smem_addr, analyzer->Simplify(blk * c(kCpt2sBytesPerCall)), taddr,
              analyzer->Simplify(blk * c(kCpt2sRowsPerCall)), c(0),
              c(kFenceGroupT2s)}));
  };
  const PrimExpr first = analyzer->Simplify(warp_id * c(per_warp));
  Stmt copy;
  if (per_warp == 1) {
    copy = copy_at(first); // one block per warp: no loop to emit at all
  } else {
    // Counted loop with the index recomputed in the body: a For's `step` does
    // not survive to codegen (it degenerates to ++i).
    Var step("step", DataType::Int(32));
    copy = tirx::For(step, c(0), c(per_warp), tirx::ForKind::kSerial,
                     copy_at(analyzer->Simplify(first + step)));
  }
  // Per-warp drain, not a barrier: it retires the issuing warp's cpt2s before
  // anything reads the staged tile, and fence_mem only covers its own group.
  Stmt drain =
      Evaluate(Call(DataType::Handle(), tang_fence_mem(), {c(kFenceGroupT2s)}));

  // No `__syncthreads()` is emitted. Whoever consumes the staged tile reads
  // *shared* memory -- a bulk s2g store, or an epilogue -- so the publish is a
  // shared hazard the sync passes derive from the accesses themselves. The
  // barrier the device helper hid at its end was invisible to every sync pass,
  // which is why it could not be reasoned about; it was never load-bearing.
  return IfThenElse(warp_id < c(active), SeqStmt({copy, drain}), Stmt());
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

  LOG(FATAL) << "STCUV2 GEMM operand staging is not supported";
  return Stmt();
}

// Element-wise SM-to-SM cluster copy. T.copy_cluster(dst_block=...) writes the
// source shared buffer into the peer CTA's shared memory, element by element,
// via ptx_cluster_store (mapa.shared.cluster + st.shared). This is the SIMT
// fallback; the bulk cp.async.bulk.shared::cluster path is not wired yet.
class ClusterCopyReplacer : public StmtExprMutator {
public:
  ClusterCopyReplacer(Buffer dst, PrimExpr dst_block, Buffer target_dst)
      : dst_(dst), dst_block_(dst_block), target_dst_(target_dst) {}

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    if (!op->buffer.same_as(dst_))
      return StmtExprMutator::VisitStmt_(op);

    // Linearize op->indices against the (possibly remapped) destination shape.
    PrimExpr linearized_index = 0;
    PrimExpr multiplier = 1;
    for (int i = static_cast<int>(op->indices.size()) - 1; i >= 0; --i) {
      linearized_index = linearized_index + op->indices[i] * multiplier;
      if (i > 0)
        multiplier = multiplier * target_dst_->shape[i];
    }

    Buffer target_buffer = target_dst_.same_as(dst_) ? op->buffer : target_dst_;
    PrimExpr total_elems = 1;
    for (const PrimExpr &s : target_buffer->shape)
      total_elems = total_elems * s;

    Stmt remote_store = Evaluate(Call(DataType::Handle(), ptx_cluster_store(),
                                      {target_buffer.access_ptr(2), op->value,
                                       dst_block_, linearized_index}));
    return IfThenElse(linearized_index < total_elems, remote_store, Stmt());
  }

private:
  Buffer dst_;
  PrimExpr dst_block_;
  Buffer target_dst_;
};

Stmt LowerClusterCopyElementWise(const CopyNode &op,
                                 const LowerArgs &lower_args,
                                 arith::Analyzer *analyzer) {
  auto simt_loop = op.MakeSIMTLoop(analyzer);
  auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));

  auto par_op = ParallelOp(fused_loop);
  std::vector<InferLevel> levels = {InferLevel::kCommon, InferLevel::kStrict,
                                    InferLevel::kFree};
  for (auto level : levels) {
    par_op->InferLayout({lower_args.target,
                         lower_args.thread_bounds,
                         lower_args.layout_map,
                         analyzer,
                         lower_args.buffer_remap,
                         {}},
                        level);
  }
  auto loop_layout = par_op->GetLoopLayout();
  auto thread_loop = PartitionLoop(par_op->GetRoot(), lower_args.thread_index,
                                   analyzer, loop_layout);
  auto vectorized_thread_loop =
      VectorizeLoop(thread_loop, lower_args.layout_map, /*vectorize_hint=*/1);

  Buffer target_dst = op.dst;
  if (lower_args.buffer_remap.count(op.dst)) {
    target_dst = lower_args.buffer_remap[op.dst];
  }
  Stmt simt_copy = ClusterCopyReplacer(op.dst, op.dst_block.value(),
                                       target_dst)(vectorized_thread_loop);

  // Remote mbarrier arrive: signal the destination CTA that the element-wise
  // stores are visible. A local shared sync first ensures every thread finished
  // its store, then a single thread arrives on the peer's barrier. The arrive
  // lowers to tl::cluster_arrive_barrier, which maps the barrier into the peer
  // CTA's address space and arrives once (requires the LLVM dev_dshm branch).
  if (auto barrier_opt = op.annotations.Get("barrier")) {
    auto barrier = Downcast<PrimExpr>(barrier_opt.value());
    Stmt sync = Evaluate(Call(DataType::Int(32), builtin::tvm_storage_sync(),
                              {StringImm("shared")}));
    Stmt arrive =
        Evaluate(Call(DataType::Handle(), ptx_arrive_cluster_barrier(),
                      {barrier, op.dst_block.value()}));
    Stmt guarded_arrive = IfThenElse(
        EQ(lower_args.thread_index, lower_args.thread_bounds->min), arrive);
    return SeqStmt({simt_copy, sync, guarded_arrive});
  }
  return simt_copy;
}

Stmt Lower(const CopyNode &op, const LowerArgs &lower_args,
           arith::Analyzer *analyzer) {
  if (TargetTangIsSTCUV2(lower_args.target)) {
    if (IsTangTmemFragmentCopy(op)) {
      return LowerTangTmemFragmentCopy(op, lower_args, analyzer);
    }
    // Cluster DSM copy (T.copy_cluster).
    //   - dst_block     -> SM-to-SM copy between CTA shared memories, lowered
    //                     element-wise via ptx_cluster_store
    //                     (mapa.shared.cluster
    //                     + st.shared). Requires the LLVM dev_dshm branch.
    //   - cluster_mask  -> TMA multicast load broadcast across the cluster, not
    //                     wired yet (cp.async.bulk.shared::cluster).
    if (op.dst_block.defined()) {
      ICHECK(op.src.scope() == "shared" || op.src.scope() == "shared.dyn")
          << "copy_cluster(dst_block=...) requires a shared source buffer";
      ICHECK(op.dst.scope() == "shared" || op.dst.scope() == "shared.dyn")
          << "copy_cluster(dst_block=...) requires a shared destination buffer";
      return LowerClusterCopyElementWise(op, lower_args, analyzer);
    }
    if (op.annotations.Get("cluster_mask")) {
      LOG(FATAL) << "TANG/STCUV2 does not support TMA multicast cluster copy "
                    "yet: copy_cluster(cluster_mask=...) requires "
                    "cp.async.bulk.shared::cluster...multicast::cluster, which "
                    "the STCU LLVM backend has not implemented.";
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
