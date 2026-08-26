#include "tang/target_utils.h"

#include "dlpack/dlpack.h"
#include <tvm/ffi/reflection/registry.h>

#include <string>

namespace tvm {
namespace tl {
namespace {

std::string GetTangArch(Target target) {
  if (!TargetIsTang(target)) {
    return "";
  }
  auto arch = target->GetAttr<ffi::String>("arch");
  if (!arch.has_value()) {
    return "";
  }
  return arch.value();
}

} // namespace

bool TargetIsTang(Target target) {
  return target->GetTargetDeviceType() == kDLTANG;
}

bool TargetTangIsSTCU(Target target) { return GetTangArch(target) == "stcu"; }

bool TargetTangIsSTCUV2(Target target) {
  return GetTangArch(target) == "stcuv2";
}

bool TargetTangHasTmem(Target target) {
  return TargetIsTang(target) && TargetTangIsSTCUV2(target);
}

bool TargetTangHasBulkCopy(Target target) {
  return TargetIsTang(target) && TargetTangIsSTCUV2(target);
}

bool TargetTangSupportVectorize256(Target target) {
  return TargetIsTang(target) && TargetTangIsSTCUV2(target);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.TargetIsTang",
           [](Target target) { return TargetIsTang(target); })
      .def("tl.TargetTangIsSTCU",
           [](Target target) { return TargetTangIsSTCU(target); })
      .def("tl.TargetTangIsSTCUV2",
           [](Target target) { return TargetTangIsSTCUV2(target); })
      .def("tl.TargetTangHasTmem",
           [](Target target) { return TargetTangHasTmem(target); })
      .def("tl.TargetTangHasBulkCopy",
           [](Target target) { return TargetTangHasBulkCopy(target); })
      .def("tl.TargetTangSupportVectorize256",
           [](Target target) { return TargetTangSupportVectorize256(target); });
}

} // namespace tl
} // namespace tvm
