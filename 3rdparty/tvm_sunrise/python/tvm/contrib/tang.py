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
"""Utilities for linking TANG device objects."""

import os
import subprocess

import tvm_ffi

from tvm.base import py_str

from . import utils


def _toolchain_bin_dirs():
    """Return explicitly configured TANG toolchain binary directories."""
    roots = [os.environ.get(name) for name in ("TANG_PATH", "TANG_HOME")]
    return [os.path.join(root, "bin") for root in roots if root]


def find_lld(required=True):
    """Find an LLVM linker suitable for TANG device objects.

    Explicit ``TANG_LLD`` takes precedence.  ``TANG_PATH`` and ``TANG_HOME``
    are then searched before the process ``PATH``.  No installation path is
    assumed so the callback follows the toolkit selected by the environment.
    """
    configured_lld = os.environ.get("TANG_LLD")
    if configured_lld:
        resolved = utils.which(configured_lld)
        if resolved:
            return [resolved]
        if required:
            raise RuntimeError(f"TANG_LLD does not name an executable: {configured_lld}")
        return []

    names = ["ld.lld"]
    candidates = [
        os.path.join(directory, name) for directory in _toolchain_bin_dirs() for name in names
    ]
    candidates += names
    valid = [resolved for candidate in candidates if (resolved := utils.which(candidate))]
    if not valid and required:
        raise RuntimeError("cannot find ld.lld; set TANG_LLD or add the TANG toolchain to PATH")
    return valid


def tang_link(in_file, out_file, lld=None):
    """Link a relocatable TANG object into a shared device code object."""
    linker = lld if lld is not None else find_lld()[0]
    args = [linker, "--no-undefined", "-shared", in_file, "-o", out_file]
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        raise RuntimeError("TANG linking failed using ld.lld:\n" + py_str(proc.stdout))


@tvm_ffi.register_global_func("tvm_callback_tang_link")
def callback_tang_link(obj_bin):
    """Link an LLVM object and return the resulting TANG code object bytes."""
    temp = utils.tempdir()
    object_path = temp.relpath("tang_kernel.o")
    code_object_path = temp.relpath("tang_kernel.co")
    with open(object_path, "wb") as output:
        output.write(bytes(obj_bin))
    tang_link(object_path, code_object_path)
    with open(code_object_path, "rb") as output:
        return bytearray(output.read())


@tvm_ffi.register_global_func("tvm_callback_tang_get_arch")
def get_tang_arch(_tang_path=None):
    """Return the configured TANG architecture, defaulting to ``stcu``."""
    return os.environ.get("TANG_ARCH", "stcu")
