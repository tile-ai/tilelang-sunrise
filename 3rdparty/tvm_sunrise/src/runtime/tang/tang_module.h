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

/*! \file tang_module.h \brief Runtime module for compiled TANG kernels. */
#ifndef TVM_RUNTIME_TANG_TANG_MODULE_H_
#define TVM_RUNTIME_TANG_TANG_MODULE_H_

#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/string.h>

#include "../metadata.h"

namespace tvm {
namespace runtime {

ffi::Module TANGModuleCreate(ffi::Bytes code, ffi::String fmt,
                             ffi::Map<ffi::String, FunctionInfo> fmap,
                             ffi::Map<ffi::String, ffi::String> source);

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_TANG_TANG_MODULE_H_
