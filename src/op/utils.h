/*!
 * \file tl/op/utils.h
 * \brief Common utilities for TL ops.
 */

#ifndef TVM_TL_OP_UTILS_H_
#define TVM_TL_OP_UTILS_H_

#include "./operator.h"
#include "cuda/stubs/cuda.h"
#include "region.h"
#include "support/check.h"
#include "tvm/runtime/base.h"
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/op.h>

namespace tvm {
namespace tl {

using namespace tirx;

// Maps TVM DataType to CUDA's CUtensorMapDataType enum value.
TVM_DLL int to_CUtensorMapDataType(DataType dtype);

// Reverses an array (used for row-major/column-major layout conversion).
template <typename T> ffi::Array<T> ReverseArray(ffi::Array<T> array) {
  return ffi::Array<T>{array.rbegin(), array.rend()};
}

// Check if an PrimExpr is a buffer-like (BufferRegion/BufferLoad/tl.region)
// expression.
TVM_DLL bool IsBufferLikeExpr(const PrimExpr &expr);

// Normalize an argument (BufferRegion/BufferLoad/tl.region)
// to BufferRegion so ops can uniformly consume regions.
// Note: tvm_access_ptr is no longer supported here.
TVM_DLL BufferRegion NormalizeToBufferRegion(const PrimExpr &arg);

// Normalize an argument to BufferRegion together with an access mask.
// If the argument is a tl.region(...) bridge, preserve its encoded mask;
// otherwise fall back to the provided default mask.
TVM_DLL AccessRegion NormalizeToAccessRegion(
    const PrimExpr &arg, int default_access_mask = kAccessReadWrite);

// Build a tvm_access_ptr(handle) from a BufferRegion.
// - If `require_2d` is true, checks buffer ndim >= 2.
// - For 1D regions (when allowed), offset=min, extent=extent.
// - For ndim >= 2, offset sums all but last two dims using row-major strides,
//   extent is product of the last two extents.
TVM_DLL PrimExpr MakeAccessPtrFromRegion(const BufferRegion &region,
                                         int rw_mask, bool require_2d = false);

// Build a tvm_access_ptr(handle) from a BufferLoad.
TVM_DLL PrimExpr MakeAccessPtrFromBufferLoad(const BufferLoad &load,
                                             int rw_mask);

inline bool IsFragmentBuffer(const Buffer &buffer) {
  return buffer.defined() && buffer.scope() == "local.fragment";
}

// Expand a lower-rank layout by prepending the leading dimensions of `buffer`
// so that the resulting layout input shape matches `buffer->shape`.
//
// This is useful when we infer a 2D swizzle layout from the trailing matrix
// dimensions of a higher-rank buffer (e.g. batched GEMM shared-memory buffers).
inline Layout ExpandLayoutToMatchBuffer(const Layout &layout,
                                        const Buffer &buffer) {
  if (!layout.defined() || !buffer.defined()) {
    return layout;
  }
  const size_t buffer_ndim = buffer->shape.size();
  const size_t layout_ndim = layout->InputDim();
  if (buffer_ndim <= layout_ndim) {
    return layout;
  }

  ffi::Array<PrimExpr> leading_shape;
  leading_shape.reserve(buffer_ndim - layout_ndim);
  for (size_t i = 0; i < buffer_ndim - layout_ndim; ++i) {
    leading_shape.push_back(buffer->shape[i]);
  }
  return layout->Expand(leading_shape);
}

// Fit the result of Layout::Forward to a remapped buffer's rank.
//
// Layout::Forward passes through any indices beyond its InputDim() as leading
// dimensions (see LayoutNode::Forward), so its result can have higher rank than
// the buffer it will index. This happens when a layout maps the trailing dims
// to a single linear index while the buffer carries extra leading dims -- e.g.
// a pipelined shared buffer, whose stage dim lower_tile_op.cc prepends as a
// replicate extent.
//
// The excess leading indices step over whole copies of the mapped tail, so
// their element stride is the product of the remaining extents. Fold them into
// the first dimension of `new_buffer` so the index rank matches
// new_buffer->shape.
inline ffi::Array<PrimExpr>
FitForwardIndicesToBuffer(const ffi::Array<PrimExpr> &indices,
                          const Buffer &new_buffer) {
  const size_t rank = new_buffer->shape.size();
  if (rank == 0 || indices.size() <= rank) {
    return indices;
  }
  const size_t excess = indices.size() - rank;
  PrimExpr leading = indices[0];
  for (size_t i = 1; i < excess; ++i) {
    leading = leading * new_buffer->shape[0] + indices[i];
  }
  ffi::Array<PrimExpr> fitted;
  fitted.reserve(rank);
  fitted.push_back(leading * new_buffer->shape[0] + indices[excess]);
  for (size_t i = excess + 1; i < indices.size(); ++i) {
    fitted.push_back(indices[i]);
  }
  return fitted;
}

inline bool IsSharedBuffer(const Buffer &buffer, bool allow_dynamic = true) {
  if (!buffer.defined()) {
    return false;
  }
  if (allow_dynamic) {
    return buffer.scope() == "shared" || buffer.scope() == "shared.dyn";
  }
  return buffer.scope() == "shared";
}

inline bool IsGlobalBuffer(const Buffer &buffer) {
  return buffer.defined() && buffer.scope() == "global";
}

inline bool IsValidCPAsyncTransferBytes(int bytes) {
  return bytes == 4 || bytes == 8 || bytes == 16;
}

inline bool IsLocalBuffer(const Buffer &buffer, bool allow_var = false) {
  if (!buffer.defined()) {
    return false;
  }
  if (allow_var) {
    return buffer.scope() == "local" || buffer.scope() == "local.var";
  }
  return buffer.scope() == "local";
}

inline bool IsLocalVarBuffer(const Buffer &buffer) {
  return buffer.defined() && buffer.scope() == "local.var";
}

// Convenience alias: register-resident buffers have "local", "local.var" or
// "local.fragment" scope. A fragment is register-resident too — its indices
// must stay affine to keep it out of scratch memory — so it counts here.
inline bool IsRegisterBuffer(const Buffer &buffer) {
  return IsLocalBuffer(buffer, /*allow_var=*/true) || IsFragmentBuffer(buffer);
}

// True when global packed FP4 is copied into f8f6f4/mxf8f6f4 unpacked FP4 SMEM.
inline bool IsFP4PackedToUnpackedStorageCopy(DataType global_dtype,
                                             DataType shared_dtype) {
  return global_dtype.is_float4_e2m1fn() &&
         shared_dtype.is_float4_e2m1_unpacked();
}

inline bool IsValidTMALoadDtypePair(DataType global_dtype,
                                    DataType shared_dtype) {
  if (global_dtype.is_float4_e2m1_unpacked() ||
      shared_dtype.is_float4_e2m1_unpacked()) {
    return IsFP4PackedToUnpackedStorageCopy(global_dtype, shared_dtype);
  }
  if (global_dtype == shared_dtype) {
    return true;
  }
  return false;
}

inline bool IsValidTMAStoreDtypePair(DataType global_dtype,
                                     DataType shared_dtype) {
  return global_dtype == shared_dtype &&
         !shared_dtype.is_float4_e2m1_unpacked();
}

inline bool IsValidTMADtypePair(bool is_load, DataType global_dtype,
                                DataType shared_dtype) {
  if (is_load) {
    return IsValidTMALoadDtypePair(global_dtype, shared_dtype);
  }
  return IsValidTMAStoreDtypePair(global_dtype, shared_dtype);
}

// Valid dtype pairs for TMA global->shared copies.
inline bool IsValidTMACopyDtypePair(DataType global_dtype,
                                    DataType shared_dtype) {
  return IsValidTMALoadDtypePair(global_dtype, shared_dtype);
}

// True only for the supported TMA load transition from packed global FP4 to
// unpacked shared FP4. The reverse store direction is not a valid TMA path.
inline bool IsFP4UnpackLoad(const Buffer &src, const Buffer &dst) {
  return IsGlobalBuffer(src) && IsSharedBuffer(dst) &&
         IsFP4PackedToUnpackedStorageCopy(src->dtype, dst->dtype);
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_UTILS_H_
