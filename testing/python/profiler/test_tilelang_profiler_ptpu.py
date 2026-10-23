import types

import pytest
import torch

from tilelang import profiler
from tilelang.profiler import bench
from tilelang.utils.tensor import TensorSupplyType


@pytest.mark.parametrize("requested, expected", [(None, "ptpu:2"), (3, "ptpu:3"), ("ptpu:1", "ptpu:1")])
def test_ptpu_profiler_device_selection(monkeypatch, requested, expected):
    from contextlib import nullcontext

    monkeypatch.setattr(bench, "IS_PTPU", True)
    monkeypatch.setattr(torch, "ptpu", types.SimpleNamespace(current_device=lambda: 2, device=lambda d: nullcontext()), raising=False)
    monkeypatch.setattr(bench, "_do_bench_impl", lambda fn, **kwargs: kwargs["device_idx"])

    assert bench.do_bench(lambda: None, device=requested) == torch.device(expected)


def test_cuda_profiler_device_selection_is_preserved(monkeypatch):
    from contextlib import nullcontext

    monkeypatch.setattr(bench, "IS_PTPU", False)
    monkeypatch.setattr(torch.cuda, "device", lambda d: nullcontext())
    monkeypatch.setattr(bench, "_do_bench_impl", lambda fn, **kwargs: kwargs["device_idx"])

    assert bench.do_bench(lambda: None, device=torch.device("cuda:3")) == 3


@pytest.mark.parametrize("requested, expected", [(None, "ptpu:2"), (3, "ptpu:3"), ("ptpu:1", "ptpu:1")])
@pytest.mark.parametrize("fails", [False, True])
def test_ptpu_wall_waits_for_selected_device_and_restores_context(monkeypatch, requested, expected, fails):
    from contextlib import contextmanager
    from tilelang.profiler import wall

    selected = torch.device(expected)
    state = types.SimpleNamespace(now=0.0, pending=0.0, active=None)
    synchronized = []

    @contextmanager
    def context(device):
        assert device == selected
        state.active = device
        try:
            yield
        finally:
            state.active = None

    def synchronize(device):
        assert device == state.active == selected
        synchronized.append(device)
        state.now += state.pending
        state.pending = 0.0

    def kernel():
        assert state.active == selected
        if fails:
            raise RuntimeError("kernel failure")
        state.now += 0.002
        state.pending += 0.003

    monkeypatch.setattr(bench, "IS_PTPU", True)
    monkeypatch.setattr(torch, "ptpu", types.SimpleNamespace(current_device=lambda: 2, device=context), raising=False)
    monkeypatch.setattr(bench, "device_synchronize", synchronize)
    monkeypatch.setattr(wall, "perf_counter", lambda: state.now)
    if fails:
        with pytest.raises(RuntimeError, match="kernel failure"):
            bench.do_bench(kernel, backend="wall", device=requested)
    else:
        latency = bench.do_bench(kernel, backend="wall", device=requested, warmup=0, _n_repeat=2)
        assert latency == pytest.approx(5.0)
        assert synchronized
        assert state.pending == 0
    assert state.active is None


@pytest.mark.parametrize("device", ["cpu", "cuda:3"])
def test_ptpu_wall_preserves_explicit_other_devices(monkeypatch, device):
    from contextlib import nullcontext

    def unexpected(*args):
        raise AssertionError("Explicit non-PTPU devices must not select PTPU")

    monkeypatch.setattr(bench, "IS_PTPU", True)
    monkeypatch.setattr(torch, "ptpu", types.SimpleNamespace(current_device=unexpected, device=unexpected), raising=False)
    monkeypatch.setattr(torch.cuda, "device", lambda d: nullcontext())
    synchronized = []
    monkeypatch.setattr(bench, "device_synchronize", lambda d: synchronized.append(d))
    bench.do_bench(lambda: None, backend="wall", device=torch.device(device), warmup=0, _n_repeat=1)
    assert bool(synchronized) == (device != "cpu")
    assert all(d == torch.device(device) for d in synchronized)


def test_ptpu_rejects_cuda_graph_backend(monkeypatch):
    from contextlib import nullcontext

    monkeypatch.setattr(bench, "IS_PTPU", True)
    monkeypatch.setattr(torch, "ptpu", types.SimpleNamespace(device=lambda d: nullcontext()), raising=False)

    def unexpected():
        raise AssertionError("Unsupported timing must fail before running the callable")

    with pytest.raises(ValueError, match="requires CUDA/HIP"):
        bench.do_bench(unexpected, backend="cudagraph", device=torch.device("ptpu:0"))


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


@pytest.mark.parametrize("callable_cache", [False, True])
def test_perf_counter_accepts_tensor_and_callable_cache(monkeypatch, callable_cache):
    from tilelang.profiler import torch_bench

    monkeypatch.setattr(torch_bench, "_cuda_synchronize", lambda device: None)
    cache = torch.ones(4)
    flush = (lambda: cache.zero_()) if callable_cache else cache
    calls = []

    def kernel():
        assert torch.equal(cache, torch.zeros(4))
        calls.append(None)
        cache.fill_(1)

    latency = torch_bench.bench_with_perf_counter(kernel, flush, 3, None, "mean", None)
    assert len(calls) == 3
    assert latency >= 0


def test_ptpu_profiler_excludes_cpu_annotated_flush_without_dropping_user_copies(monkeypatch):
    from contextlib import nullcontext
    from tilelang.profiler import torch_bench

    def event(key, device, duration, annotation=False, inclusive=None):
        return types.SimpleNamespace(
            key=key,
            device_type=types.SimpleNamespace(name=device),
            self_device_time_total=duration,
            device_time_total=duration if inclusive is None else inclusive,
            is_user_annotation=annotation,
        )

    events = [
        event(torch_bench._CACHE_FLUSH_ID, "CPU", 0, annotation=True, inclusive=1000),
        event("Memcpy D2D", "PrivateUse1", 1000),
        event("gemm_kernel", "PrivateUse1", 500),
        event("Memcpy D2D", "PrivateUse1", 200),  # A user copy outside the flush range.
        event("unrelated", "CPU", 9000),
    ]

    class Profile:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def step(self):
            pass

        def events(self):
            return events

    monkeypatch.setattr(torch_bench, "suppress_stdout_stderr", nullcontext)
    monkeypatch.setattr(torch.profiler, "profile", lambda **kwargs: Profile())
    monkeypatch.setattr(torch.profiler, "schedule", lambda **kwargs: None)
    monkeypatch.setattr(torch.profiler, "record_function", lambda name: nullcontext())
    latency = torch_bench.bench_with_cupti(lambda: None, lambda: None, 2, torch.device("ptpu:0"))
    assert latency == pytest.approx(0.35)
