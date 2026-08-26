/*!
 * \file tl/tang/op/scan.cc
 * \brief TANG implementation registration for cumulative-sum lowering.
 */

#include "backend/common/op/scan.h"
#include "tang/target_utils.h"

namespace tvm {
namespace tl {
namespace tang {
namespace {

bool RegisterTangScan() {
  RegisterCumSumImpl(CumSumImpl{
      "tang.CumSum",
      TargetIsTang,
      backend::scan::LowerCumSum,
  });
  RegisterCumMaxImpl(CumMaxImpl{
      "tang.CumMax",
      TargetIsTang,
      backend::scan::LowerCumMax,
  });
  return true;
}

const bool tang_scan_registered = RegisterTangScan();

} // namespace
} // namespace tang
} // namespace tl
} // namespace tvm
