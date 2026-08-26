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
#include <gtest/gtest.h>
#include <tvm/ffi/c_api.h>

namespace {

TEST(Base, Version) {
  TVMFFIVersion version;
  TVMFFIGetVersion(&version);
  EXPECT_EQ(version.major, TVM_FFI_VERSION_MAJOR);
  EXPECT_EQ(version.minor, TVM_FFI_VERSION_MINOR);
  EXPECT_EQ(version.patch, TVM_FFI_VERSION_PATCH);
}

}  // namespace
