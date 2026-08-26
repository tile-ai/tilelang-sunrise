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
 * Quick Start Example - Simple Math Functions (Pure C)
 *
 * This file demonstrates how to export C functions using TVM-FFI's
 * C ABI (TVMFFISafeCallType) so they can be loaded dynamically at
 * runtime with tvm_ffi_orcjit.  No C++ runtime dependencies.
 */
#include <tvm/ffi/c_api.h>

/* add: add two integers */
TVM_FFI_DLL_EXPORT int __tvm_ffi_add(void* self, const TVMFFIAny* args, int32_t num_args,
                                     TVMFFIAny* result) {
  result->type_index = kTVMFFIInt;
  result->zero_padding = 0;
  result->v_int64 = args[0].v_int64 + args[1].v_int64;
  return 0;
}

/* multiply: multiply two integers */
TVM_FFI_DLL_EXPORT int __tvm_ffi_multiply(void* self, const TVMFFIAny* args, int32_t num_args,
                                          TVMFFIAny* result) {
  result->type_index = kTVMFFIInt;
  result->zero_padding = 0;
  result->v_int64 = args[0].v_int64 * args[1].v_int64;
  return 0;
}

/* fibonacci: recursive fibonacci */
static int64_t fib(int64_t n) {
  if (n <= 1) return n;
  return fib(n - 1) + fib(n - 2);
}

TVM_FFI_DLL_EXPORT int __tvm_ffi_fibonacci(void* self, const TVMFFIAny* args, int32_t num_args,
                                           TVMFFIAny* result) {
  result->type_index = kTVMFFIInt;
  result->zero_padding = 0;
  result->v_int64 = fib(args[0].v_int64);
  return 0;
}
