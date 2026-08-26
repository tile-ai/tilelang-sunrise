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
/*! \file tang_device_api.cc \brief TANG implementation of the TVM runtime DeviceAPI. */
#include <tang.h>
#include <tang_runtime.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/timer.h>

#include <cstring>
#include <exception>
#include <sstream>
#include <string>

#include "tang_common.h"

namespace tvm {
namespace runtime {

class TANGDeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(Device dev) final { TANG_CALL(tangSetDevice(dev.device_id)); }
  void GetAttr(Device dev, DeviceAttrKind kind, ffi::Any* rv) final {
    int value = 0;
    switch (kind) {
      case kExist: {
        int count = 0;
        tangError_t error = tangGetDeviceCount(&count);
        *rv = static_cast<int>(error == tangSuccess && dev.device_id >= 0 && dev.device_id < count);
        return;
      }
      case kMaxThreadsPerBlock:
        TANG_CALL(tangDeviceGetAttribute(&value, tangDevAttrMaxThreadsPerBlock, dev.device_id));
        break;
      case kWarpSize:
        TANG_CALL(tangDeviceGetAttribute(&value, tangDevAttrWarpSize, dev.device_id));
        break;
      case kMaxSharedMemoryPerBlock:
        TANG_CALL(tangDeviceGetAttribute(&value, tangDevAttrMaxSharedMemPerBlock, dev.device_id));
        break;
      case kComputeVersion: {
        int major = 0, minor = 0;
        TANG_CALL(tangDeviceGetAttribute(&major, tangDevAttrComputeCapabilityMajor, dev.device_id));
        TANG_CALL(tangDeviceGetAttribute(&minor, tangDevAttrComputeCapabilityMinor, dev.device_id));
        *rv = std::to_string(major) + "." + std::to_string(minor);
        return;
      }
      case kDeviceName: {
        tangDeviceProp prop;
        TANG_CALL(tangGetDeviceProperties(&prop, dev.device_id));
        *rv = std::string(prop.name);
        return;
      }
      case kMaxClockRate:
        TANG_CALL(tangDeviceGetAttribute(&value, tangDevAttrClockRate, dev.device_id));
        break;
      case kMultiProcessorCount:
        TANG_CALL(tangDeviceGetAttribute(&value, tangDevAttrMultiProcessorCount, dev.device_id));
        break;
      case kMaxThreadDimensions: {
        int dims[3];
        TANG_CALL(tangDeviceGetAttribute(&dims[0], tangDevAttrMaxBlockDimX, dev.device_id));
        TANG_CALL(tangDeviceGetAttribute(&dims[1], tangDevAttrMaxBlockDimY, dev.device_id));
        TANG_CALL(tangDeviceGetAttribute(&dims[2], tangDevAttrMaxBlockDimZ, dev.device_id));
        std::ostringstream os;
        os << "[" << dims[0] << ", " << dims[1] << ", " << dims[2] << "]";
        *rv = os.str();
        return;
      }
      case kMaxRegistersPerBlock:
        TANG_CALL(tangDeviceGetAttribute(&value, tangDevAttrMaxRegsPerBlock, dev.device_id));
        break;
      case kApiVersion:
        *rv = TA_VERSION;
        return;
      case kL2CacheSizeBytes:
        TANG_CALL(tangDeviceGetAttribute(&value, tangDevAttrL2CacheSize, dev.device_id));
        break;
      case kTotalGlobalMemory: {
        tangDeviceProp prop;
        TANG_CALL(tangGetDeviceProperties(&prop, dev.device_id));
        *rv = static_cast<int64_t>(prop.totalGlobalMem);
        return;
      }
      case kAvailableGlobalMemory: {
        SetDevice(dev);
        size_t free_mem = 0, total_mem = 0;
        TANG_CALL(tangMemGetInfo(&free_mem, &total_mem));
        *rv = static_cast<int64_t>(free_mem);
        return;
      }
      case kGcnArch:
      case kDriverVersion:
      case kImagePitchAlignment:
        return;
    }
    *rv = value;
  }

  void* AllocDataSpace(Device dev, size_t nbytes, size_t alignment, DLDataType type_hint) final {
    TVM_FFI_ICHECK_EQ(256 % alignment, 0U) << "TANG space is aligned at 256 bytes";
    void* ptr = nullptr;
    if (dev.device_type == kDLTANGHost) {
      TANG_CALL(tangMallocHost(&ptr, nbytes));
    } else {
      SetDevice(dev);
      TANG_CALL(tangMalloc(&ptr, nbytes));
    }
    return ptr;
  }
  void FreeDataSpace(Device dev, void* ptr) final {
    if (std::uncaught_exceptions() && tangPeekAtLastError() == tangErrorIllegalAddress) return;
    if (dev.device_type == kDLTANGHost) {
      TANG_CALL(tangFreeHost(ptr));
    } else {
      SetDevice(dev);
      TANG_CALL(tangFree(ptr));
    }
  }

 protected:
  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
                      Device dev_from, Device dev_to, DLDataType type_hint,
                      TVMStreamHandle stream) final {
    from = static_cast<const char*>(from) + from_offset;
    to = static_cast<char*>(to) + to_offset;
    if (dev_from.device_type == kDLTANGHost) dev_from.device_type = kDLCPU;
    if (dev_to.device_type == kDLTANGHost) dev_to.device_type = kDLCPU;
    if (dev_from.device_type == kDLCPU && dev_to.device_type == kDLCPU) {
      std::memcpy(to, from, size);
      return;
    }
    tangStream_t tang_stream = static_cast<tangStream_t>(stream);
    if (dev_from.device_type == kDLTANG && dev_to.device_type == kDLTANG) {
      SetDevice(dev_from);
      if (dev_from.device_id == dev_to.device_id) {
        TANG_CALL(tangMemcpyAsync(to, from, size, tangMemcpyDeviceToDevice, tang_stream));
      } else {
        TANG_CALL(
            tangMemcpyPeerAsync(to, dev_to.device_id, from, dev_from.device_id, size, tang_stream));
      }
    } else if (dev_from.device_type == kDLTANG && dev_to.device_type == kDLCPU) {
      SetDevice(dev_from);
      TANG_CALL(tangMemcpyAsync(to, from, size, tangMemcpyDeviceToHost, tang_stream));
    } else if (dev_from.device_type == kDLCPU && dev_to.device_type == kDLTANG) {
      SetDevice(dev_to);
      TANG_CALL(tangMemcpyAsync(to, from, size, tangMemcpyHostToDevice, tang_stream));
    } else {
      TVM_FFI_THROW(InternalError) << "Expected a copy from/to TANG or between TANG devices";
    }
  }

 public:
  TVMStreamHandle CreateStream(Device dev) final {
    SetDevice(dev);
    tangStream_t stream;
    TANG_CALL(tangStreamCreateWithFlags(&stream, tangStreamNonBlocking));
    return static_cast<TVMStreamHandle>(stream);
  }
  void FreeStream(Device dev, TVMStreamHandle stream) final {
    SetDevice(dev);
    TANG_CALL(tangStreamDestroy(static_cast<tangStream_t>(stream)));
  }
  void SyncStreamFromTo(Device dev, TVMStreamHandle src, TVMStreamHandle dst) final {
    SetDevice(dev);
    tangEvent_t event;
    TANG_CALL(tangEventCreate(&event));
    TANG_CALL(tangEventRecord(event, static_cast<tangStream_t>(src)));
    TANG_CALL(tangStreamWaitEvent(static_cast<tangStream_t>(dst), event, 0));
    TANG_CALL(tangEventDestroy(event));
  }
  void StreamSync(Device dev, TVMStreamHandle stream) final {
    SetDevice(dev);
    TANG_CALL(tangStreamSynchronize(static_cast<tangStream_t>(stream)));
  }
  void* AllocWorkspace(Device dev, size_t size, DLDataType type_hint) final {
    return TANGThreadEntry::ThreadLocal()->pool.AllocWorkspace(dev, size);
  }
  void FreeWorkspace(Device dev, void* data) final {
    TANGThreadEntry::ThreadLocal()->pool.FreeWorkspace(dev, data);
  }
  bool SupportsDevicePointerArithmeticsOnHost() final { return true; }
  static TANGDeviceAPI* Global() {
    static auto* instance = new TANGDeviceAPI();
    return instance;
  }
};

TANGThreadEntry::TANGThreadEntry() : pool(kDLTANG, TANGDeviceAPI::Global()) {}
TANGThreadEntry* TANGThreadEntry::ThreadLocal() {
  static thread_local TANGThreadEntry instance;
  return &instance;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed(
          "device_api.tang",
          [](ffi::PackedArgs, ffi::Any* rv) { *rv = static_cast<void*>(TANGDeviceAPI::Global()); })
      .def_packed("device_api.tang_host", [](ffi::PackedArgs, ffi::Any* rv) {
        *rv = static_cast<void*>(TANGDeviceAPI::Global());
      });
}

class TANGTimerNode final : public TimerNode {
 public:
  TANGTimerNode() {
    TANG_CALL(tangEventCreate(&start_));
    TANG_CALL(tangEventCreate(&stop_));
  }
  ~TANGTimerNode() final {
    TANG_CALL(tangEventDestroy(start_));
    TANG_CALL(tangEventDestroy(stop_));
  }
  void Start() final {
    int device_id = 0;
    TANG_CALL(tangGetDevice(&device_id));
    stream_ = TVMFFIEnvGetStream(kDLTANG, device_id);
    TANG_CALL(tangEventRecord(start_, static_cast<tangStream_t>(stream_)));
  }
  void Stop() final { TANG_CALL(tangEventRecord(stop_, static_cast<tangStream_t>(stream_))); }
  int64_t SyncAndGetElapsedNanos() final {
    TANG_CALL(tangEventSynchronize(stop_));
    float milliseconds = 0;
    TANG_CALL(tangEventElapsedTime(&milliseconds, start_, stop_));
    return static_cast<int64_t>(milliseconds * 1e6);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("runtime.tang.TANGTimerNode", TANGTimerNode, TimerNode);

 private:
  tangEvent_t start_, stop_;
  TVMStreamHandle stream_{nullptr};
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("runtime.timer.tang",
                        [](Device) { return Timer(ffi::make_object<TANGTimerNode>()); });
}

TVM_RUNTIME_DLL ffi::String GetTangFreeMemory() {
  size_t free_mem = 0, total_mem = 0;
  TANG_CALL(tangMemGetInfo(&free_mem, &total_mem));
  std::ostringstream os;
  os << "Current TANG memory is " << free_mem << " bytes free out of " << total_mem
     << " bytes on device";
  return os.str();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.GetTangFreeMemory", GetTangFreeMemory)
      .def("runtime.GetTangDeviceCount",
           []() {
             int count = 0;
             TANG_CALL(tangGetDeviceCount(&count));
             return count;
           })
      .def("runtime.get_tang_stream", []() {
        int device_id = 0;
        TANG_CALL(tangGetDevice(&device_id));
        return static_cast<void*>(TVMFFIEnvGetStream(kDLTANG, device_id));
      });
}

}  // namespace runtime
}  // namespace tvm
