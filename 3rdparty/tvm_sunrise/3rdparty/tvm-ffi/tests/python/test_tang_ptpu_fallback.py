"""CPU-only regression coverage for the torch_ptpu legacy DLPack fallback."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


def test_ptpu_torch_setter_repairs_cpu_legacy_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    tvm_ffi = pytest.importorskip("tvm_ffi")

    # Model the affected torch_ptpu release: the tensor provenance says PTPU,
    # while its legacy DLPack capsule incorrectly reports kDLCPU.
    monkeypatch.setattr(torch.Tensor, "is_ptpu", property(lambda _self: True), raising=False)
    monkeypatch.setattr(
        torch,
        "ptpu",
        SimpleNamespace(current_stream=lambda _device_id: SimpleNamespace(ptpu_stream=0)),
        raising=False,
    )
    monkeypatch.setitem(sys.modules, "torch_ptpu", SimpleNamespace())
    monkeypatch.delattr(torch.Tensor, "__dlpack_c_exchange_api__", raising=False)

    source = torch.arange(4, dtype=torch.float32)
    imported_direct = tvm_ffi.from_dlpack(source)
    assert imported_direct.device.dlpack_device_type() == tvm_ffi.DLDeviceType.kDLTANG
    assert imported_direct.device.index == 0

    calls = 0

    def check_imported(imported: tvm_ffi.Tensor) -> bool:
        nonlocal calls
        calls += 1
        assert imported.device.dlpack_device_type() == tvm_ffi.DLDeviceType.kDLTANG
        assert imported.device.index == 0
        return True

    check = tvm_ffi.convert_func(check_imported)
    assert check(source)
    assert calls == 1
