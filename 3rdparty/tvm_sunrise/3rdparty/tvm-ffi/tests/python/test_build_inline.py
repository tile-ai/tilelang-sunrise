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
import tempfile
from pathlib import Path

import numpy
import pytest
import tvm_ffi.cpp
from tvm_ffi.module import Module


def test_build_inline_cpp() -> None:
    output_lib_path = tvm_ffi.cpp.build_inline(
        name="hello",
        cpp_sources=r"""
            void add_one_cpu(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
              // implementation of a library function
              TVM_FFI_ICHECK(x.ndim() == 1) << "x must be a 1D tensor";
              DLDataType f32_dtype{kDLFloat, 32, 1};
              TVM_FFI_ICHECK(x.dtype() == f32_dtype) << "x must be a float tensor";
              TVM_FFI_ICHECK(y.ndim() == 1) << "y must be a 1D tensor";
              TVM_FFI_ICHECK(y.dtype() == f32_dtype) << "y must be a float tensor";
              TVM_FFI_ICHECK(x.size(0) == y.size(0)) << "x and y must have the same shape";
              for (int i = 0; i < x.size(0); ++i) {
                static_cast<float*>(y.data_ptr())[i] = static_cast<float*>(x.data_ptr())[i] + 1;
              }
            }
        """,
        functions=["add_one_cpu"],
    )

    mod: Module = tvm_ffi.load_module(output_lib_path)

    x = numpy.array([1, 2, 3, 4, 5], dtype=numpy.float32)
    y = numpy.empty_like(x)
    mod.add_one_cpu(x, y)
    numpy.testing.assert_equal(x + 1, y)


def test_build_c_file() -> None:
    """Test building a C source file to .o and linking into .so."""
    c_source = r"""
#include <tvm/ffi/c_api.h>

TVM_FFI_DLL_EXPORT int __tvm_ffi_c_add(
    void* self, const TVMFFIAny* args, int32_t num_args, TVMFFIAny* result) {
  result->type_index = kTVMFFIInt;
  result->zero_padding = 0;
  result->v_int64 = args[0].v_int64 + args[1].v_int64;
  return 0;
}
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        c_file = str(Path(tmpdir) / "test_c.c")
        Path(c_file).write_text(c_source)

        obj_path = tvm_ffi.cpp.build(
            name="test_c",
            sources=[c_file],
            output="test_c.o",
        )
        assert obj_path.endswith(".o")

        lib_path = tvm_ffi.cpp.build(
            name="test_c_lib",
            sources=[obj_path],
        )

        mod: Module = tvm_ffi.load_module(lib_path)
        assert mod.c_add(10, 20) == 30


def test_build_inline_object_output() -> None:
    """Test building a .o then linking it into a .so and calling the function."""
    obj_path = tvm_ffi.cpp.build_inline(
        name="hello_obj",
        cpp_sources=r"""
            void add_one_cpu(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
              TVM_FFI_ICHECK(x.ndim() == 1) << "x must be a 1D tensor";
              DLDataType f32_dtype{kDLFloat, 32, 1};
              TVM_FFI_ICHECK(x.dtype() == f32_dtype) << "x must be a float tensor";
              TVM_FFI_ICHECK(y.ndim() == 1) << "y must be a 1D tensor";
              TVM_FFI_ICHECK(y.dtype() == f32_dtype) << "y must be a float tensor";
              TVM_FFI_ICHECK(x.size(0) == y.size(0)) << "x and y must have the same shape";
              for (int i = 0; i < x.size(0); ++i) {
                static_cast<float*>(y.data_ptr())[i] = static_cast<float*>(x.data_ptr())[i] + 1;
              }
            }
        """,
        functions=["add_one_cpu"],
        output="hello_obj.o",
    )
    assert obj_path.endswith(".o")

    # Link the object file into a shared library
    lib_path = tvm_ffi.cpp.build(
        name="hello_from_obj",
        sources=[obj_path],
    )

    mod: Module = tvm_ffi.load_module(lib_path)
    x = numpy.array([1, 2, 3, 4, 5], dtype=numpy.float32)
    y = numpy.empty_like(x)
    mod.add_one_cpu(x, y)
    numpy.testing.assert_equal(x + 1, y)


if __name__ == "__main__":
    pytest.main([__file__])
