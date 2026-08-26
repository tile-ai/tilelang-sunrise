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

# Discover the imported targets provided by a TANG toolkit installation.
# TANG_DIR/TANGRT_DIR or CMAKE_PREFIX_PATH may be used to select a toolkit.
macro(find_tang)
  find_package(TANG CONFIG REQUIRED)
  if(NOT TARGET TANG::tang)
    message(FATAL_ERROR "The TANG package does not provide TANG::tang")
  endif()

  get_target_property(_tang_include_dirs TANG::tang INTERFACE_INCLUDE_DIRECTORIES)
  list(GET _tang_include_dirs 0 _tang_include_dir)
  get_filename_component(TANG_TOOLKIT_ROOT_DIR "${_tang_include_dir}" DIRECTORY)
  if(NOT EXISTS "${TANG_TOOLKIT_ROOT_DIR}/cmake/CMakeDetermineTANGCompiler.cmake")
    message(FATAL_ERROR "The TANG package does not provide TANG CMake language modules")
  endif()
  list(APPEND CMAKE_MODULE_PATH "${TANG_TOOLKIT_ROOT_DIR}/cmake")

  find_package(TANGRT CONFIG REQUIRED)
  if(NOT TARGET TANGRT::tangrt_shared)
    message(FATAL_ERROR "The TANGRT package does not provide TANGRT::tangrt_shared")
  endif()
endmacro()
