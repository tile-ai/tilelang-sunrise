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

from pathlib import Path
from types import SimpleNamespace

import pytest
import tvm_ffi

from tvm.contrib import tang


def test_tang_binary_format_is_not_exposed_as_text_source():
    create_module = tvm_ffi.get_global_func("ffi.Module.create.tang")
    binary = b"\x7fELF\x02\x01\xe6"

    module_without_source = create_module(binary, "t", {}, {})
    assert module_without_source.inspect_source() == ""

    module_with_source = create_module(binary, "t", {}, {"tang": "kernel source"})
    assert module_with_source.inspect_source() == "kernel source"


def test_tang_link_invokes_lld(monkeypatch, tmp_path):
    input_path = tmp_path / "kernel.o"
    output_path = tmp_path / "kernel.co"
    input_path.write_bytes(b"object")

    def run(args, **kwargs):
        assert args == [
            "/toolchain/bin/ld.lld",
            "--no-undefined",
            "-shared",
            str(input_path),
            "-o",
            str(output_path),
        ]
        assert kwargs == {
            "stdout": tang.subprocess.PIPE,
            "stderr": tang.subprocess.STDOUT,
            "check": False,
        }
        output_path.write_bytes(b"code-object")
        return SimpleNamespace(returncode=0, stdout=b"")

    monkeypatch.setattr(tang.subprocess, "run", run)
    tang.tang_link(str(input_path), str(output_path), lld="/toolchain/bin/ld.lld")
    assert output_path.read_bytes() == b"code-object"


def test_tang_link_reports_linker_output(monkeypatch, tmp_path):
    monkeypatch.setattr(
        tang.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=b"undefined symbol: foo"),
    )
    with pytest.raises(RuntimeError, match="undefined symbol: foo"):
        tang.tang_link(str(tmp_path / "kernel.o"), str(tmp_path / "kernel.co"), lld="ld.lld")


def test_callback_tang_link_preserves_bytes(monkeypatch):
    def link(input_path, output_path, lld=None):
        assert lld is None
        assert Path(input_path).read_bytes() == b"input-object"
        Path(output_path).write_bytes(b"output-code-object")

    monkeypatch.setattr(tang, "tang_link", link)
    assert tang.callback_tang_link(bytearray(b"input-object")) == bytearray(b"output-code-object")


def test_get_tang_arch(monkeypatch):
    monkeypatch.delenv("TANG_ARCH", raising=False)
    assert tang.get_tang_arch() == "stcu"
    monkeypatch.setenv("TANG_ARCH", "stcuv2")
    assert tang.get_tang_arch() == "stcuv2"
