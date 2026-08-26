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

#include "../../../src/runtime/tang/tang_launch_utils.h"

#include <gtest/gtest.h>

namespace tvm {
namespace runtime {

TEST(TANGLaunchUtils, DynamicSharedMemoryFollowsMetadata) {
  TANGLaunchMetadata stcu = ParseTANGLaunchMetadata({launch_param::kUseDynamicSharedMemoryTag});
  EXPECT_TRUE(stcu.use_dynamic_shared_memory);
  EXPECT_EQ(TANGDynamicSharedMemoryBytes(stcu, 8192), 8192U);

  TANGLaunchMetadata stcuv2 = ParseTANGLaunchMetadata({});
  EXPECT_FALSE(stcuv2.use_dynamic_shared_memory);
  EXPECT_EQ(TANGDynamicSharedMemoryBytes(stcuv2, 8192), 0U);
}

TEST(TANGLaunchUtils, CooperativeLaunchRequiresExplicitMetadata) {
  TANGLaunchMetadata ordinary = ParseTANGLaunchMetadata({"blockIdx.x"});
  EXPECT_FALSE(ordinary.use_cooperative_launch);

  TANGLaunchMetadata cooperative = ParseTANGLaunchMetadata(
      {launch_param::kUseCooperativeLaunch, launch_param::kUseDynamicSharedMemoryTag});
  EXPECT_TRUE(cooperative.use_cooperative_launch);
  EXPECT_TRUE(cooperative.use_dynamic_shared_memory);
}

}  // namespace runtime
}  // namespace tvm
