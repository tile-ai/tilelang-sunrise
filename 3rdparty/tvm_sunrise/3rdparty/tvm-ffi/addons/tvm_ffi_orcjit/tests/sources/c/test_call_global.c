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
 * Pure C test: look up a host-registered global function by name and call it.
 * Demonstrates JIT code calling back into the host process via the TVM FFI C API.
 */
#include <tvm/ffi/c_api.h>

/*
 * test_call_global_add: look up "test_host_add" global function and call it
 * with two integer arguments. Returns the result from the host function.
 */
TVM_FFI_DLL_EXPORT int __tvm_ffi_test_call_global_add(void* self, const TVMFFIAny* args,
                                                      int32_t num_args, TVMFFIAny* result) {
  int ret_code = 0;
  TVMFFIByteArray func_name;
  TVMFFIObjectHandle func_handle = NULL;
  TVMFFIAny call_args[2];
  TVMFFIAny call_result;

  /* Look up the host-registered global function */
  func_name.data = "test_host_add";
  func_name.size = 13;
  ret_code = TVMFFIFunctionGetGlobal(&func_name, &func_handle);
  if (ret_code != 0) return ret_code;

  /* Prepare arguments: pass through the two integer args */
  call_args[0].type_index = kTVMFFIInt;
  call_args[0].zero_padding = 0;
  call_args[0].v_int64 = args[0].v_int64;
  call_args[1].type_index = kTVMFFIInt;
  call_args[1].zero_padding = 0;
  call_args[1].v_int64 = args[1].v_int64;

  /* Call the global function */
  call_result.type_index = kTVMFFINone;
  call_result.zero_padding = 0;
  call_result.v_int64 = 0;
  ret_code = TVMFFIFunctionCall(func_handle, call_args, 2, &call_result);

  /* Release the function handle */
  TVMFFIObjectDecRef((TVMFFIObject*)func_handle);

  if (ret_code != 0) return ret_code;

  /* Forward the result */
  result->type_index = call_result.type_index;
  result->zero_padding = call_result.zero_padding;
  result->v_int64 = call_result.v_int64;
  return 0;
}

/*
 * test_call_global_mul: look up "test_host_multiply" global function and call it.
 */
TVM_FFI_DLL_EXPORT int __tvm_ffi_test_call_global_mul(void* self, const TVMFFIAny* args,
                                                      int32_t num_args, TVMFFIAny* result) {
  int ret_code = 0;
  TVMFFIByteArray func_name;
  TVMFFIObjectHandle func_handle = NULL;
  TVMFFIAny call_args[2];
  TVMFFIAny call_result;

  func_name.data = "test_host_multiply";
  func_name.size = 18;
  ret_code = TVMFFIFunctionGetGlobal(&func_name, &func_handle);
  if (ret_code != 0) return ret_code;

  call_args[0].type_index = kTVMFFIInt;
  call_args[0].zero_padding = 0;
  call_args[0].v_int64 = args[0].v_int64;
  call_args[1].type_index = kTVMFFIInt;
  call_args[1].zero_padding = 0;
  call_args[1].v_int64 = args[1].v_int64;

  call_result.type_index = kTVMFFINone;
  call_result.zero_padding = 0;
  call_result.v_int64 = 0;
  ret_code = TVMFFIFunctionCall(func_handle, call_args, 2, &call_result);

  TVMFFIObjectDecRef((TVMFFIObject*)func_handle);

  if (ret_code != 0) return ret_code;

  result->type_index = call_result.type_index;
  result->zero_padding = call_result.zero_padding;
  result->v_int64 = call_result.v_int64;
  return 0;
}
