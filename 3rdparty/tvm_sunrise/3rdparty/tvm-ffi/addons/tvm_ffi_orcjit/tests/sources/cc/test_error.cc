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

// Error propagation test (C++ version): throws std::runtime_error
// which TVM_FFI_SAFE_CALL_END catches and converts to an InternalError.

#include <tvm/ffi/function.h>

#include <stdexcept>

int test_error_impl() { throw std::runtime_error("test error"); }
TVM_FFI_DLL_EXPORT_TYPED_FUNC(test_error, test_error_impl);
