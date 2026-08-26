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

/*
 * Pure C test functions using the TVMFFISafeCallType ABI directly.
 * No C++ features: no exceptions, RTTI, static variables, or std library.
 * This isolates the JIT infrastructure from MSVC C++ runtime issues.
 */
#include <tvm/ffi/c_api.h>

/* test_add: add two integers */
TVM_FFI_DLL_EXPORT int __tvm_ffi_test_add(void* self, const TVMFFIAny* args, int32_t num_args,
                                          TVMFFIAny* result) {
  result->type_index = kTVMFFIInt;
  result->zero_padding = 0;
  result->v_int64 = args[0].v_int64 + args[1].v_int64;
  return 0;
}

/* test_multiply: multiply two integers */
TVM_FFI_DLL_EXPORT int __tvm_ffi_test_multiply(void* self, const TVMFFIAny* args, int32_t num_args,
                                               TVMFFIAny* result) {
  result->type_index = kTVMFFIInt;
  result->zero_padding = 0;
  result->v_int64 = args[0].v_int64 * args[1].v_int64;
  return 0;
}
