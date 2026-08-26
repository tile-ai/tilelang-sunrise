# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

if(USE_TANG)
  find_tang()
  enable_language(TANG)

  message(STATUS "Build with TANG support")
  get_target_property(TANGRT_INCLUDE_DIRS TANGRT::tangrt_shared INTERFACE_INCLUDE_DIRECTORIES)
  include_directories(SYSTEM ${TANGRT_INCLUDE_DIRS})
  tvm_file_glob(GLOB RUNTIME_TANG_SRCS src/runtime/tang/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_TANG_SRCS})
  list(APPEND COMPILER_SRCS src/target/tang/intrin_rule_tang.cc)
  list(APPEND TVM_RUNTIME_LINKER_LIBS TANGRT::tangrt_shared TANG::tang)
else()
  list(APPEND COMPILER_SRCS src/target/opt/build_tang_off.cc)
endif()
