import sys
import types

import torch

from tilelang.jit.adapter.base import BaseKernelAdapter
from tilelang.utils import device as device_utils
from tilelang.utils.tensor import _synchronize_ptpu_for_host_readback


class _FakePTPU:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def current_device():
        return 3

    @staticmethod
    def current_stream():
        return types.SimpleNamespace(ptpu_stream=12345)


def test_ptpu_device_and_stream_functors(monkeypatch):
    # Build a mock torch_ptpu module with the _C.rt sub-chain that
    # BaseKernelAdapter.get_current_stream_functor accesses directly.
    _mock_rt = types.SimpleNamespace(
        get_raw_stream=lambda s: 12345,
        current_stream=lambda d: None,
    )
    _mock_C = types.SimpleNamespace(rt=_mock_rt)
    _mock_ptpu = types.ModuleType("torch_ptpu")
    _mock_ptpu._C = _mock_C

    monkeypatch.setitem(sys.modules, "torch_ptpu", _mock_ptpu)
    monkeypatch.setattr(torch, "ptpu", _FakePTPU(), raising=False)

    assert device_utils.is_ptpu_available()
    assert device_utils.get_current_device() == torch.device("ptpu", 3)
    assert BaseKernelAdapter.get_current_device_functor()() == torch.device("ptpu", 3)
    assert BaseKernelAdapter.get_current_stream_functor()() == 12345


def test_ptpu_detection_is_optional(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch_ptpu", raising=False)
    monkeypatch.setitem(sys.modules, "torch_ptpu", None)

    assert not device_utils.is_ptpu_available()


def test_ptpu_host_readback_synchronizes_each_device_once(monkeypatch):
    synchronized = []
    fake_ptpu = types.SimpleNamespace(synchronize=synchronized.append)
    monkeypatch.setattr(torch, "ptpu", fake_ptpu, raising=False)
    ptpu0 = types.SimpleNamespace(device=torch.device("ptpu:0"))
    ptpu1 = types.SimpleNamespace(device=torch.device("ptpu:1"))

    _synchronize_ptpu_for_host_readback(ptpu0, ptpu0, ptpu1)

    assert synchronized == [torch.device("ptpu:0"), torch.device("ptpu:1")]
