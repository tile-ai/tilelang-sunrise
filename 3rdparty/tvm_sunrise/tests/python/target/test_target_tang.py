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

import pytest

import tvm
from tvm.target import Target


def test_tang_target_defaults(monkeypatch):
    monkeypatch.delenv("TANG_ARCH", raising=False)
    target = Target("tang")

    assert target.kind.name == "tang"
    assert target.get_target_device_type() == 20
    assert list(target.keys) == ["tang", "gpu"]
    assert target.attrs["arch"] == "stcu"
    assert target.attrs["thread_warp_size"] == 32
    assert target.attrs["max_num_threads"] == 1024


def test_tang_device_string_roundtrip():
    assert str(tvm.runtime.Device(20, 0)) == "tang:0"
    assert str(tvm.runtime.Device(21, 0)) == "tang_host:0"
    assert tvm.runtime.enabled("tang")


@pytest.mark.parametrize("arch", ["stcu", "stcuv2"])
def test_tang_target_explicit_arch_roundtrip(monkeypatch, arch):
    monkeypatch.setenv("TANG_ARCH", "stcu" if arch == "stcuv2" else "stcuv2")
    target = Target({"kind": "tang", "arch": arch})
    roundtrip = Target(str(target))

    assert target.attrs["arch"] == arch
    assert roundtrip.kind.name == "tang"
    assert roundtrip.attrs["arch"] == arch


def test_tang_target_rejects_invalid_explicit_arch(monkeypatch):
    monkeypatch.setenv("TANG_ARCH", "stcuv2")
    with pytest.raises(ValueError, match="invalid arch.*not-a-tang-arch"):
        Target({"kind": "tang", "arch": "not-a-tang-arch"})


def test_tang_target_uses_environment_arch(monkeypatch):
    monkeypatch.setenv("TANG_ARCH", "stcuv2")
    assert Target("tang").attrs["arch"] == "stcuv2"


def test_tang_target_invalid_environment_arch_falls_back(monkeypatch):
    monkeypatch.setenv("TANG_ARCH", "not-a-tang-arch")
    assert Target("tang").attrs["arch"] == "stcu"


if __name__ == "__main__":
    tvm.testing.main()
