/*!
 * \file tl/tang/op/reduce.cc
 * \brief TANG implementation for tl.reduce AllReduce lowering.
 */

#include "backend/common/op/reduce.h"
#include "tang/target_utils.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tirx;

namespace tang {

struct Reduce : backend::ReduceLowerer<Reduce> {
  static DataType GetReductionDataType(const ReduceOpNode &op, Target) {
    DataType dtype = op.dst->dtype;
    if (((!op.nan_propagate && (op.type->IsMax() || op.type->IsMin())) ||
         op.type->IsAbsMax()) &&
        (dtype.is_float16() || dtype.is_bfloat16())) {
      return DataType::Float(32, dtype.lanes());
    }
    return dtype;
  }

  static bool SupportsFp16Bf16NanReduce(Target target) {
    return TargetIsTang(target);
  }

  static int GetPreferedVectorizedSize(DataType, Target) { return 1; }

  static std::string MakeBatchAllReduce(std::string reducer,
                                        int reducing_threads, int scale,
                                        PrimExpr thread_offset,
                                        PrimExpr all_threads, int batch,
                                        int workspace_stride, Target) {
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset << ", " << all_threads << ", " << batch
       << ", " << workspace_stride << ">::run_batch";
    return ss.str();
  }

  static std::string MakeScalarAllReduce(std::string reducer,
                                         int reducing_threads, int scale,
                                         PrimExpr thread_offset,
                                         PrimExpr all_threads, Target) {
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset << ", " << all_threads << ">::run";
    return ss.str();
  }
};

namespace {

bool RegisterTangReduce() {
  RegisterReduceImpl(ReduceImpl{
      "tang.Reduce",
      TargetIsTang,
      Reduce::Lower,
  });
  return true;
}

const bool tang_reduce_registered = RegisterTangReduce();

} // namespace
} // namespace tang
} // namespace tl
} // namespace tvm
