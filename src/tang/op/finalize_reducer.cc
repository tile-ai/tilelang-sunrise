/*!
 * \file tl/tang/op/finalize_reducer.cc
 * \brief TANG implementation for tl.finalize_reducer AllReduce lowering.
 */

#include "backend/common/op/finalize_reducer.h"
#include "tang/target_utils.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tirx;

namespace tang {

struct FinalizeReducer : backend::FinalizeReducerLowerer<FinalizeReducer> {
  static int WarpSize(Target) { return 32; }

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

bool RegisterTangFinalizeReducer() {
  RegisterFinalizeReducerImpl(FinalizeReducerImpl{
      "tang.FinalizeReducer",
      TargetIsTang,
      FinalizeReducer::Lower,
  });
  return true;
}

const bool tang_finalize_reducer_registered = RegisterTangFinalizeReducer();

} // namespace
} // namespace tang
} // namespace tl
} // namespace tvm
