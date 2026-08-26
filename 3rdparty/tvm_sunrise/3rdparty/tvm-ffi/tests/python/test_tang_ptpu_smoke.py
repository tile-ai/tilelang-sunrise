"""Serial PTPU smoke tests for the internal GitLab runner."""

from __future__ import annotations

import os

import pytest
import torch
import torch_ptpu  # noqa: F401
import tvm_ffi


def _require_ptpu() -> None:
    assert os.environ.get("TANG_VISIBLE_DEVICES"), "preserve an explicit device selection"
    assert hasattr(torch, "ptpu") and torch.ptpu.is_available()


def _assert_torch_exchange_roundtrip(imported: tvm_ffi.Tensor, source: torch.Tensor) -> None:
    """Restore kDLTANG through the optional exchange API's PrivateUse1 mapping."""
    assert hasattr(torch.Tensor, "__dlpack_c_exchange_api__")
    calls = 0

    def check_restored(restored: torch.Tensor) -> bool:
        nonlocal calls
        calls += 1
        assert isinstance(restored, torch.Tensor)
        assert restored.device.type == "ptpu"
        assert restored.device.index == source.device.index
        assert torch.equal(restored.cpu(), source.cpu())
        return True

    check = tvm_ffi.convert_func(check_restored, tensor_cls=torch.Tensor)
    assert check(imported)
    assert calls == 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.int8])
def test_ptpu_legacy_dlpack_roundtrip(dtype: torch.dtype) -> None:
    _require_ptpu()
    source = torch.arange(16, dtype=dtype, device="ptpu")
    exchange_api = torch.Tensor.__dlpack_c_exchange_api__
    del torch.Tensor.__dlpack_c_exchange_api__
    try:
        imported = tvm_ffi.from_dlpack(source)
    finally:
        torch.Tensor.__dlpack_c_exchange_api__ = exchange_api
    assert imported.device.dlpack_device_type() == tvm_ffi.DLDeviceType.kDLTANG
    assert imported.device.index == source.device.index
    _assert_torch_exchange_roundtrip(imported, source)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int8])
def test_ptpu_torch_argument_fallback(dtype: torch.dtype) -> None:
    """The Python Torch setter must preserve PTPU when the C exchange is absent."""
    _require_ptpu()
    source = torch.arange(16, dtype=dtype, device="ptpu")
    exchange_api = torch.Tensor.__dlpack_c_exchange_api__
    del torch.Tensor.__dlpack_c_exchange_api__
    try:
        calls = 0

        def check_imported(imported: tvm_ffi.Tensor) -> bool:
            nonlocal calls
            calls += 1
            assert isinstance(imported, tvm_ffi.Tensor)
            assert imported.device.dlpack_device_type() == tvm_ffi.DLDeviceType.kDLTANG
            assert imported.device.index == source.device.index
            return True

        check = tvm_ffi.convert_func(check_imported)
        assert check(source)
        assert calls == 1
    finally:
        torch.Tensor.__dlpack_c_exchange_api__ = exchange_api


@pytest.mark.parametrize("dtype", [torch.float32, torch.int8])
def test_ptpu_versioned_exchange_roundtrip(dtype: torch.dtype) -> None:
    """The C exchange API supplies versioned DLPack even when torch 2.6 does not."""
    _require_ptpu()
    source = torch.arange(16, dtype=dtype, device="ptpu")
    imported = tvm_ffi.from_dlpack(source)
    assert imported.device.dlpack_device_type() == tvm_ffi.DLDeviceType.kDLTANG
    assert imported.device.index == source.device.index
    _assert_torch_exchange_roundtrip(imported, source)


def test_ptpu_default_and_nondefault_stream_roundtrip() -> None:
    _require_ptpu()
    default_stream = torch.ptpu.current_stream()
    source = torch.arange(8, dtype=torch.float32, device="ptpu")
    assert default_stream.ptpu_stream == torch.ptpu.current_stream().ptpu_stream
    _assert_torch_exchange_roundtrip(tvm_ffi.from_dlpack(source), source)

    nondefault_stream = torch.ptpu.Stream()
    with torch.ptpu.stream(nondefault_stream):
        shifted = source + 1
        imported = tvm_ffi.from_dlpack(shifted)
        assert torch.ptpu.current_stream().ptpu_stream == nondefault_stream.ptpu_stream
        _assert_torch_exchange_roundtrip(imported, shifted)
    default_stream.wait_stream(nondefault_stream)
