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

/*! \file tang_launch_utils.h \brief Pure TANG launch metadata decisions. */
#ifndef TVM_RUNTIME_TANG_TANG_LAUNCH_UTILS_H_
#define TVM_RUNTIME_TANG_TANG_LAUNCH_UTILS_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/string.h>

#include <cstddef>

#include "../metadata.h"

namespace tvm {
namespace runtime {

struct TANGLaunchMetadata {
  bool use_dynamic_shared_memory{false};
  bool use_cooperative_launch{false};
};

inline TANGLaunchMetadata ParseTANGLaunchMetadata(
    const ffi::Array<ffi::String>& launch_param_tags) {
  TANGLaunchMetadata metadata;
  for (const ffi::String& tag : launch_param_tags) {
    if (tag == launch_param::kUseDynamicSharedMemoryTag) {
      metadata.use_dynamic_shared_memory = true;
    } else if (tag == launch_param::kUseCooperativeLaunch) {
      metadata.use_cooperative_launch = true;
    }
  }
  return metadata;
}

inline size_t TANGDynamicSharedMemoryBytes(const TANGLaunchMetadata& metadata,
                                           size_t extracted_dynamic_bytes) {
  // stcu kernels retain the dynamic-shared tag and pass the extracted launch
  // size.  stcuv2 lowering emits the merged shared buffer statically and omits
  // the tag, so passing zero avoids reserving the same buffer a second time.
  return metadata.use_dynamic_shared_memory ? extracted_dynamic_bytes : 0;
}

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_TANG_TANG_LAUNCH_UTILS_H_
