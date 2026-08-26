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
 * Pure C test functions exercising type variety: zero-arg, four-arg,
 * float, and void return types via the TVMFFISafeCallType ABI.
 */
#include <tvm/ffi/c_api.h>

/* test_zero_arg: ignores arguments, returns integer 42 */
TVM_FFI_DLL_EXPORT int __tvm_ffi_test_zero_arg(void* self, const TVMFFIAny* args, int32_t num_args,
                                               TVMFFIAny* result) {
  result->type_index = kTVMFFIInt;
  result->zero_padding = 0;
  result->v_int64 = 42;
  return 0;
}

/* test_four_args: sum of four integer arguments */
TVM_FFI_DLL_EXPORT int __tvm_ffi_test_four_args(void* self, const TVMFFIAny* args, int32_t num_args,
                                                TVMFFIAny* result) {
  result->type_index = kTVMFFIInt;
  result->zero_padding = 0;
  result->v_int64 = args[0].v_int64 + args[1].v_int64 + args[2].v_int64 + args[3].v_int64;
  return 0;
}

/* test_float_multiply: multiply two doubles */
TVM_FFI_DLL_EXPORT int __tvm_ffi_test_float_multiply(void* self, const TVMFFIAny* args,
                                                     int32_t num_args, TVMFFIAny* result) {
  result->type_index = kTVMFFIFloat;
  result->zero_padding = 0;
  result->v_float64 = args[0].v_float64 * args[1].v_float64;
  return 0;
}

/* test_void_function: does nothing, returns None */
TVM_FFI_DLL_EXPORT int __tvm_ffi_test_void_function(void* self, const TVMFFIAny* args,
                                                    int32_t num_args, TVMFFIAny* result) {
  result->type_index = kTVMFFINone;
  result->zero_padding = 0;
  result->v_int64 = 0;
  return 0;
}
