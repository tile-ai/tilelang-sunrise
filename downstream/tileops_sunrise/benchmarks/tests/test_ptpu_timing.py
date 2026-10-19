"""S2 timing keeps profiler readings and fallback metadata distinguishable."""

import pytest
import torch

from benchmarks.timing import _capture_bench_meta, bench_kernel

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "ptpu") or not torch.ptpu.is_available(),
    reason="PTPU required",
)


def test_projection_failure_falls_back_to_ptpu_events():
    samples = bench_kernel(lambda: sum(range(64)), min_iters=2, max_iters=2)
    assert len(samples) == 2
    assert all(s.n_kernels is None for s in samples)
    assert _capture_bench_meta()["timing"] == "ptpu-events"
    assert _capture_bench_meta()["fallback_reason"]


def test_ptpu_kernel_runtime_error_propagates():
    def boom():
        raise RuntimeError("kernel failure")

    with pytest.raises(RuntimeError, match="kernel failure"):
        bench_kernel(boom, min_iters=2, max_iters=2)


def test_ptpu_samples_and_clone_metadata():
    x = torch.ones(256, device="ptpu")
    samples = bench_kernel(lambda a: a + 1, args=(x,), min_iters=3, max_iters=3)
    assert len(samples) == 3
    assert all(s.latency_ms >= s.device_busy_ms >= 0 for s in samples)
    meta = _capture_bench_meta()
    assert meta["timing"] in {"privateuse1-profiler", "ptpu-events"}
    assert meta["inputs_cloned"] is True
    assert meta["copy_policy"] == "all-projected-device-work"
