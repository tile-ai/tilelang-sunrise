/*!
 * \file tl/tang/op/transpose.cc
 * \brief TANG implementation for tl.transpose lowering.
 */

#include "backend/common/op/transpose.h"
#include "tang/target_utils.h"

namespace tvm {
namespace tl {
namespace tang {
namespace {

bool RegisterTangTranspose() {
  RegisterTransposeImpl(TransposeImpl{
      "tang.Transpose",
      TargetIsTang,
      backend::Transpose::Lower,
  });
  return true;
}

const bool tang_transpose_registered = RegisterTangTranspose();

} // namespace
} // namespace tang
} // namespace tl
} // namespace tvm
