/*!
 * \file tl/tang/op/fill.cc
 * \brief TANG implementation for tile fill operations.
 */

#include "backend/common/op/fill.h"
#include "op/fill.h"
#include "tang/target_utils.h"

namespace tvm {
namespace tl {
namespace tang {
namespace {

bool RegisterTangFill() {
  RegisterFillImpl(FillImpl{
      "tang.Fill",
      TargetIsTang,
      backend::Fill::Lower,
  });
  return true;
}

const bool tang_fill_registered = RegisterTangFill();

} // namespace
} // namespace tang
} // namespace tl
} // namespace tvm
