// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

// C++ test: look up a host-registered global function by name and call it.
// Demonstrates JIT code calling back into the host process via the TVM FFI C++ API.

#include <tvm/ffi/function.h>

using namespace tvm::ffi;

// Look up "test_host_add" global function and call it with two integer arguments.
int test_call_global_add_impl(int a, int b) {
  Function host_add = Function::GetGlobalRequired("test_host_add");
  return host_add(a, b).cast<int>();
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(test_call_global_add, test_call_global_add_impl);

// Look up "test_host_multiply" global function and call it.
int test_call_global_mul_impl(int a, int b) {
  Function host_mul = Function::GetGlobalRequired("test_host_multiply");
  return host_mul(a, b).cast<int>();
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(test_call_global_mul, test_call_global_mul_impl);
