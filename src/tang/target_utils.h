#ifndef TVM_TL_TANG_TARGET_UTILS_H_
#define TVM_TL_TANG_TARGET_UTILS_H_

#include <tvm/target/target.h>

namespace tvm {
namespace tl {

bool TargetIsTang(Target target);
bool TargetTangIsSTCU(Target target);
bool TargetTangIsSTCUV2(Target target);
bool TargetTangHasTmem(Target target);
bool TargetTangHasBulkCopy(Target target);
bool TargetTangSupportVectorize256(Target target);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TANG_TARGET_UTILS_H_
