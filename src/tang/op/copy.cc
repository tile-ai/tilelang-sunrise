/*!
 * \file tl/tang/op/copy.cc
 * \brief TANG implementation for ordinary tile copies.
 */

#include "op/copy.h"
#include "layout/layout.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "tang/target_utils.h"

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

LayoutMap InferLayout(const CopyNode &op, const LayoutInferArgs &layout_args,
                      InferLevel level) {
  LayoutMap result = op.InferSIMTLayout(layout_args, level);
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

  PrimExpr rows = shared_range.empty() ? PrimExpr(1) : shared_range[0]->extent;
  PrimExpr cols = 1;
  for (size_t i = 1; i < shared_range.size(); ++i) {
    cols *= shared_range[i]->extent;
  }
  PrimExpr col_bytes =
      analyzer->Simplify(FloorDiv(cols * shared_tensor->dtype.bits(), 8));
  PrimExpr gmem_row_stride =
      global_strides.empty() ? PrimExpr(1) : global_strides[0];
  PrimExpr gmem_row_bytes = analyzer->Simplify(
      FloorDiv(gmem_row_stride * global_tensor->dtype.bits(), 8));

  int atom_bytes = 32;
  if (auto value = op.annotations.Get("tang_swizzle_atom_bytes")) {
    const auto *imm = value->as<IntImmNode>();
    ICHECK(imm) << "tang_swizzle_atom_bytes must be an IntImm";
    atom_bytes = static_cast<int>(imm->value);
  }

  int direction = is_load ? 0 : 1;
  PrimExpr dst_addr = is_load ? shared_addr : global_addr;
  PrimExpr src_addr = is_load ? global_addr : shared_addr;
  if (atom_bytes == 64) {
    return Evaluate(
        Call(DataType::Handle(), tang_cp_async_bulk_sw(),
             {IntImm(DataType::Int(32), direction), dst_addr, src_addr, rows,
              col_bytes, gmem_row_bytes, IntImm(DataType::Int(32), 0),
              IntImm(DataType::Int(32), 7)}));
  }
  ICHECK_EQ(atom_bytes, 32)
      << "TANG STCUV2 bulk copy supports swizzle atom sizes 32 or 64 bytes";
  return Evaluate(Call(DataType::Handle(), tang_cp_async_bulk(),
                       {IntImm(DataType::Int(32), direction), dst_addr,
                        src_addr, rows, col_bytes, gmem_row_bytes}));
}

Stmt Lower(const CopyNode &op, const LowerArgs &lower_args,
           arith::Analyzer *analyzer) {
  if (TargetTangIsSTCUV2(lower_args.target)) {
    if (op.src.scope() == "shared.tmem") {
      if (op.dst.scope() == "global") {
        // LowerTangTmemDrain rewrites this normal loop after TMEM lowering.
        return LowerNormalCopy(op, lower_args, analyzer);
      }
      LOG(FATAL)
          << "TANG stcuv2: copying directly from tensor memory "
             "(shared.tmem) to a '"
          << op.dst.scope()
          << "' buffer is not supported. Drain tensor memory directly to "
             "global memory.";
    }
    bool global_to_shared =
        op.src.scope() == "global" &&
        (op.dst.scope() == "shared" || op.dst.scope() == "shared.dyn");
    bool shared_to_global =
        (op.src.scope() == "shared" || op.src.scope() == "shared.dyn") &&
        op.dst.scope() == "global";
    if (global_to_shared || shared_to_global) {
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
