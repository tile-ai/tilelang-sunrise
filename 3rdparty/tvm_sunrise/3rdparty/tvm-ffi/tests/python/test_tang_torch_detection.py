"""CPU-only tests for optional torch_ptpu backend detection."""

from __future__ import annotations

import builtins
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_MODULE_PATH = Path(__file__).parents[2] / "python/tvm_ffi/_optional_torch_c_dlpack.py"
os.environ["TVM_FFI_DISABLE_TORCH_C_DLPACK"] = "1"
_SPEC = importlib.util.spec_from_file_location("tvm_ffi_tang_detection", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_detect_torch_device = _MODULE._detect_torch_device
_is_ptpu_available = _MODULE._is_ptpu_available


class _Availability:
    def __init__(self, available: bool) -> None:
        self.available = available

    def is_available(self) -> bool:
        return self.available


def _fake_torch(*, cuda: bool = False, ptpu: bool | None = None) -> SimpleNamespace:
    module = SimpleNamespace(
        cuda=_Availability(cuda),
        version=SimpleNamespace(cuda="12.0" if cuda else None, hip=None),
    )
    if ptpu is not None:
        module.ptpu = _Availability(ptpu)
    return module


def test_cpu_detection_does_not_require_ptpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch_ptpu", None)
    torch_module = _fake_torch()
    with pytest.warns(RuntimeWarning, match="Failed to initialize optional torch_ptpu"):
        assert not _is_ptpu_available(torch_module)
    with pytest.warns(RuntimeWarning, match="Failed to initialize optional torch_ptpu"):
        assert _detect_torch_device(torch_module) == "cpu"


def test_ptpu_initialization_error_warns_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def broken_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch_ptpu":
            raise RuntimeError("driver initialization failed")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "torch_ptpu", raising=False)
    monkeypatch.setattr(builtins, "__import__", broken_import)
    torch_module = _fake_torch()
    with pytest.warns(RuntimeWarning, match="driver initialization failed"):
        assert not _is_ptpu_available(torch_module)


def test_ptpu_detection_takes_precedence_over_cuda() -> None:
    torch_module = _fake_torch(cuda=True, ptpu=True)
    assert _is_ptpu_available(torch_module)
    assert _detect_torch_device(torch_module) == "ptpu"


def test_unavailable_ptpu_falls_back_to_cuda() -> None:
    assert _detect_torch_device(_fake_torch(cuda=True, ptpu=False)) == "cuda"
