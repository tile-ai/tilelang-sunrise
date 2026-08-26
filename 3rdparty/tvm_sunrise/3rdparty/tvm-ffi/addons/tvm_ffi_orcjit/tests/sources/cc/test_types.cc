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

// C++ test functions exercising type variety: zero-arg, four-arg,
// float, and void return types.

#include <tvm/ffi/function.h>

int test_zero_arg_impl() { return 42; }
TVM_FFI_DLL_EXPORT_TYPED_FUNC(test_zero_arg, test_zero_arg_impl);

int test_four_args_impl(int a, int b, int c, int d) { return a + b + c + d; }
TVM_FFI_DLL_EXPORT_TYPED_FUNC(test_four_args, test_four_args_impl);

double test_float_multiply_impl(double a, double b) { return a * b; }
TVM_FFI_DLL_EXPORT_TYPED_FUNC(test_float_multiply, test_float_multiply_impl);

void test_void_function_impl() {}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(test_void_function, test_void_function_impl);
