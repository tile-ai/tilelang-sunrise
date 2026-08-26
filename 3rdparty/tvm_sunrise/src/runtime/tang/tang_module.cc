/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*! \file tang_module.cc \brief Runtime module for compiled TANG kernels. */
#include "tang_module.h"

#include <tang.h>
#include <tang_runtime.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <array>
#include <cstdint>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include "../../support/bytes_io.h"
#include "../pack_args.h"
#include "../thread_storage_scope.h"
#include "tang_common.h"
#include "tang_launch_utils.h"

namespace tvm {
namespace runtime {
namespace {

constexpr int kMaxNumTANGDevices = 32;

void CheckDriverCall(TAresult result, const char* call) {
  if (result == TANG_SUCCESS) return;
  const char* message = nullptr;
  taGetErrorName(result, &message);
  TVM_FFI_THROW(InternalError) << "TANG driver call " << call
                               << " failed: " << (message != nullptr ? message : "unknown error");
}

void CheckDeviceId(int device_id) {
  TVM_FFI_CHECK(device_id >= 0 && device_id < kMaxNumTANGDevices, InternalError)
      << "TANG device id " << device_id << " is outside the supported range [0, "
      << kMaxNumTANGDevices << ")";
}

}  // namespace

class TANGModuleNode final : public ffi::ModuleObj {
 public:
  TANGModuleNode(ffi::Bytes code, ffi::String fmt, ffi::Map<ffi::String, FunctionInfo> fmap,
                 ffi::Map<ffi::String, ffi::String> source)
      : code_(std::move(code)),
        fmt_(std::move(fmt)),
        fmap_(std::move(fmap)),
        source_(std::move(source)) {
    modules_.fill(nullptr);
  }

  ~TANGModuleNode() {
    for (size_t i = 0; i < modules_.size(); ++i) {
      if (modules_[i] == nullptr) continue;
      tangError_t set_result = tangSetDevice(static_cast<int>(i));
      if (set_result != tangSuccess) continue;
      (void)taModuleUnload(modules_[i]);
    }
  }

  const char* kind() const final { return "tang"; }

  int GetPropertyMask() const final {
    return ffi::Module::kBinarySerializable | ffi::Module::kRunnable;
  }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final;

  ffi::Bytes SaveToBytes() const final {
    std::string buffer;
    support::BytesOutStream stream(&buffer);
    stream.Write(fmt_);
    stream.Write(fmap_);
    stream.Write(code_);
    return ffi::Bytes(std::move(buffer));
  }

  ffi::String InspectSource(const ffi::String& format) const final {
    if (format == fmt_) return ffi::String(code_.data(), code_.size());
    if (auto it = source_.find(format); it != source_.end()) return (*it).second;
    if (format.empty()) {
      if (auto it = source_.find("tang"); it != source_.end()) return (*it).second;
      // "t" is a compiled ELF code object produced by the TANG linker, not
      // textual source.  Treating it as text makes launch-error reporting
      // append arbitrary binary bytes and can hide the real driver error
      // behind a Python UnicodeDecodeError after module serialization drops
      // the in-memory source map.
      if (fmt_ == "llir" || fmt_ == "tang") {
        return ffi::String(code_.data(), code_.size());
      }
    }
    return ffi::String();
  }

  TAfunction GetFunc(int device_id, const std::string& func_name) {
    CheckDeviceId(device_id);
    std::lock_guard<std::mutex> lock(mutex_);
    TANG_CALL(tangSetDevice(device_id));
    if (modules_[device_id] == nullptr) {
      CheckDriverCall(taModuleLoadData(&modules_[device_id], code_.data(), code_.size()),
                      "taModuleLoadData");
    }
    TAfunction function = nullptr;
    CheckDriverCall(taModuleGetFunction(&function, modules_[device_id], func_name.c_str()),
                    "taModuleGetFunction");
    return function;
  }

  TAdeviceptr GetGlobal(int device_id, const std::string& global_name, size_t expected_nbytes) {
    CheckDeviceId(device_id);
    std::lock_guard<std::mutex> lock(mutex_);
    if (auto it = globals_[device_id].find(global_name); it != globals_[device_id].end()) {
      TVM_FFI_ICHECK_EQ(it->second.nbytes, expected_nbytes);
      return it->second.address;
    }
    TANG_CALL(tangSetDevice(device_id));
    if (modules_[device_id] == nullptr) {
      CheckDriverCall(taModuleLoadData(&modules_[device_id], code_.data(), code_.size()),
                      "taModuleLoadData");
    }
    TAdeviceptr address = 0;
    size_t nbytes = 0;
    TAresult result =
        taModuleGetGlobal(&address, &nbytes, modules_[device_id], global_name.c_str());
    if (result != TANG_SUCCESS) {
      const char* message = nullptr;
      taGetErrorName(result, &message);
      TVM_FFI_THROW(InternalError)
          << "TANG module global `" << global_name << "` was required by launch metadata but "
          << "could not be resolved: " << (message != nullptr ? message : "unknown error");
    }
    TVM_FFI_CHECK(address != 0, InternalError)
        << "TANG module global `" << global_name << "` resolved to a null device address";
    TVM_FFI_CHECK(nbytes == expected_nbytes, InternalError)
        << "TANG module global `" << global_name << "` has size " << nbytes << " bytes; expected "
        << expected_nbytes << " bytes";
    globals_[device_id].emplace(global_name, GlobalInfo{address, nbytes});
    return address;
  }

 private:
  struct GlobalInfo {
    TAdeviceptr address;
    size_t nbytes;
  };

  ffi::Bytes code_;
  ffi::String fmt_;
  ffi::Map<ffi::String, FunctionInfo> fmap_;
  ffi::Map<ffi::String, ffi::String> source_;
  std::array<TAmodule, kMaxNumTANGDevices> modules_;
  std::array<std::unordered_map<std::string, GlobalInfo>, kMaxNumTANGDevices> globals_;
  std::mutex mutex_;
};

class TANGWrappedFunc {
 public:
  void Init(TANGModuleNode* module, ffi::ObjectPtr<ffi::Object> self, std::string func_name,
            size_t num_void_args, const ffi::Array<ffi::String>& launch_param_tags) {
    module_ = module;
    self_ = std::move(self);
    func_name_ = std::move(func_name);
    functions_.fill(nullptr);
    launch_metadata_ = ParseTANGLaunchMetadata(launch_param_tags);
    launch_param_config_.Init(num_void_args, launch_param_tags);
  }

  void operator()(ffi::PackedArgs args, ffi::Any* rv, void** void_args) const {
    int device_id = 0;
    TANG_CALL(tangGetDevice(&device_id));
    CheckDeviceId(device_id);
    if (functions_[device_id] == nullptr) {
      functions_[device_id] = module_->GetFunc(device_id, func_name_);
    }

    ThreadWorkLoad workload = launch_param_config_.Extract(args);
    TVM_FFI_ICHECK(workload.grid_dim(0) > 0 && workload.grid_dim(1) > 0 && workload.grid_dim(2) > 0)
        << "TANGLaunch Error: grid dimension must be positive, but got grid=("
        << workload.grid_dim(0) << "," << workload.grid_dim(1) << "," << workload.grid_dim(2)
        << ") in kernel " << func_name_;

    TAstream stream = static_cast<TAstream>(TVMFFIEnvGetStream(kDLTANG, device_id));
    size_t dynamic_shared_memory_bytes =
        TANGDynamicSharedMemoryBytes(launch_metadata_, workload.dyn_shmem_size);
    TAresult result;
    if (launch_metadata_.use_cooperative_launch) {
      constexpr size_t kBarrierBytes = sizeof(uint32_t);
      TAdeviceptr barrier = module_->GetGlobal(device_id, "bar0", kBarrierBytes);
      CheckDriverCall(taMemsetAsync(barrier, 0, kBarrierBytes, stream), "taMemsetAsync(bar0)");
      result = taLaunchCooperativeKernel(
          functions_[device_id], workload.grid_dim(0), workload.grid_dim(1), workload.grid_dim(2),
          workload.block_dim(0), workload.block_dim(1), workload.block_dim(2),
          dynamic_shared_memory_bytes, stream, void_args);
    } else {
      result = taLaunchKernel(functions_[device_id], workload.grid_dim(0), workload.grid_dim(1),
                              workload.grid_dim(2), workload.block_dim(0), workload.block_dim(1),
                              workload.block_dim(2), dynamic_shared_memory_bytes, stream, void_args,
                              nullptr);
    }
    if (result != TANG_SUCCESS && result != TANG_ERROR_DEINITIALIZED) {
      const char* message = nullptr;
      taGetErrorName(result, &message);
      std::ostringstream os;
      os << "TANGLaunch Error: " << (message != nullptr ? message : "unknown error") << "\n"
         << " grid=(" << workload.grid_dim(0) << "," << workload.grid_dim(1) << ","
         << workload.grid_dim(2) << "), block=(" << workload.block_dim(0) << ","
         << workload.block_dim(1) << "," << workload.block_dim(2) << ")"
         << " dyn_smem_bytes=" << dynamic_shared_memory_bytes << "\n";
      ffi::String source = module_->InspectSource("");
      if (!source.empty()) {
        os << "// func_name=" << func_name_ << "\n"
           << "// TANG Source\n"
           << "// -----------\n"
           << source;
      }
      TVM_FFI_THROW(InternalError) << os.str();
    }
  }

 private:
  TANGModuleNode* module_{nullptr};
  ffi::ObjectPtr<ffi::Object> self_;
  std::string func_name_;
  mutable std::array<TAfunction, kMaxNumTANGDevices> functions_;
  TANGLaunchMetadata launch_metadata_;
  LaunchParamConfig launch_param_config_;
};

ffi::Optional<ffi::Function> TANGModuleNode::GetFunction(const ffi::String& name) {
  ffi::ObjectPtr<ffi::Object> self = ffi::GetObjectPtr<ffi::Object>(this);
  TVM_FFI_ICHECK_EQ(self.get(), this);
  auto opt_info = fmap_.Get(name);
  if (!opt_info.has_value()) return ffi::Function();
  FunctionInfo info = opt_info.value();
  TANGWrappedFunc function;
  function.Init(this, self, name, info->arg_types.size(), info->launch_param_tags);
  return PackFuncVoidAddr(function, info->arg_types, info->arg_extra_tags);
}

ffi::Module TANGModuleCreate(ffi::Bytes code, ffi::String fmt,
                             ffi::Map<ffi::String, FunctionInfo> fmap,
                             ffi::Map<ffi::String, ffi::String> source) {
  return ffi::Module(ffi::make_object<TANGModuleNode>(std::move(code), std::move(fmt),
                                                      std::move(fmap), std::move(source)));
}

static ffi::Module TANGModuleLoadFromBytes(const ffi::Bytes& bytes) {
  support::BytesInStream stream(bytes);
  ffi::String fmt;
  ffi::Map<ffi::String, FunctionInfo> fmap;
  ffi::Bytes code;
  stream.Read(&fmt);
  TVM_FFI_ICHECK(stream.Read(&fmap));
  stream.Read(&code);
  return TANGModuleCreate(std::move(code), std::move(fmt), std::move(fmap),
                          ffi::Map<ffi::String, ffi::String>());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("ffi.Module.load_from_bytes.tang", TANGModuleLoadFromBytes)
      .def("ffi.Module.create.tang",
           [](ffi::Bytes code, ffi::String fmt, ffi::Map<ffi::String, FunctionInfo> fmap,
              ffi::Map<ffi::String, ffi::String> source) {
             return TANGModuleCreate(std::move(code), std::move(fmt), std::move(fmap),
                                     std::move(source));
           });
}

}  // namespace runtime
}  // namespace tvm
