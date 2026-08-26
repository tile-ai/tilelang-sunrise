import types

import pytest
import torch

from tilelang import profiler
from tilelang.profiler import bench
from tilelang.utils.tensor import TensorSupplyType


class _FakePTPU:
    @staticmethod
    def current_device():
        return 2

    @staticmethod
    def Event(enable_timing=False):
        return ("ptpu-event", enable_timing)

    @staticmethod
    def synchronize(device=None):
        return device


def test_ptpu_profiler_device_helpers(monkeypatch):
    monkeypatch.setattr(bench, "IS_PTPU", True)
    monkeypatch.setattr(torch, "ptpu", _FakePTPU(), raising=False)

    assert bench._accelerator_kind() == "ptpu"
    assert bench._normalize_accelerator_device(torch.device("ptpu")) == 2
    assert bench._normalize_accelerator_device(torch.device("ptpu:3")) == 3
    assert bench._accelerator_event() == ("ptpu-event", True)


def test_ptpu_profiler_rejects_cuda_device(monkeypatch):
    monkeypatch.setattr(bench, "IS_PTPU", True)
    monkeypatch.setattr(torch, "ptpu", types.SimpleNamespace(current_device=lambda: 0), raising=False)

    with pytest.raises(ValueError, match="must be a ptpu device"):
        bench._normalize_accelerator_device(torch.device("cuda:0"))


def test_cuda_profiler_device_selection_is_preserved(monkeypatch):
    monkeypatch.setattr(bench, "IS_PTPU", False)

    assert bench._accelerator_kind() == "cuda"
    assert bench._normalize_accelerator_device(torch.device("cuda:3")) == 3


def test_ptpu_rejects_cuda_graph_backend(monkeypatch):
    monkeypatch.setattr(bench, "IS_PTPU", True)

    with pytest.raises(RuntimeError, match="requires CUDA"):
        bench._bench_with_cudagraph(lambda: None, None, 1, None, "mean", 0)


def test_profiler_comparison_synchronizes_nested_values(monkeypatch):
    cuda_synchronize_calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: cuda_synchronize_calls.append(None))
    tensor0 = torch.empty(0)
    tensor1 = torch.empty(0)

    profiler._synchronize_values_for_host_readback(
        [tensor0, (None, tensor1)],
        "ignored",
    )

    assert cuda_synchronize_calls == [None]


def test_profiler_comparison_entrypoints_synchronize(monkeypatch):
    synchronize_calls = []
    monkeypatch.setattr(
        profiler,
        "_synchronize_values_for_host_readback",
        lambda *values: synchronize_calls.append(values),
    )
    value = torch.ones(1)
    profile = profiler.Profiler(
        [],
        [],
        TensorSupplyType.Auto,
        adapter=lambda: [value],
    )

    profile.assert_allclose(lambda: [value], input_tensors=[])
    profile.manual_assert_close(
        lambda: [value],
        input_tensors=[],
        manual_check_prog=lambda actual, expected: None,
    )
    profile.assert_consistent(repeat=2)

    assert len(synchronize_calls) == 7
