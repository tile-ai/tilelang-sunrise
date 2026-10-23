"""PyTorch GPU timing implementations."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable

import torch

from tilelang.utils.device import IS_PTPU, Event, device_synchronize as _cuda_synchronize

_CACHE_FLUSH_ID = "tilelang::cache_flush"


class suppress_stdout_stderr:
    """Context manager to suppress stdout and stderr output.

    Source: https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/testing/bench.py
    """

    def __enter__(self):
        # Open null device files
        self.outnull_file = open(os.devnull, "w")
        self.errnull_file = open(os.devnull, "w")

        # Save original file descriptors
        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()
        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        # Save original stdout/stderr objects
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        # Redirect file descriptors and streams to null device
        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)
        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file

        return self

    def __exit__(self, *_):
        # Restore original stdout/stderr objects
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        # Restore original file descriptors
        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        # Close duplicated file descriptors
        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        # Close null device files
        self.outnull_file.close()
        self.errnull_file.close()


def bench_with_cuda_events(
    fn: Callable,
    cache: torch.Tensor,
    n_repeat: int,
    quantiles: list[float] | None,
    return_mode: str,
    device_idx: int | torch.device | None,
) -> float | list[float]:
    """Benchmark using CUDA/HIP or MPS events for timing."""
    # Create timing events
    event_type = Event if device_idx is None else getattr(torch, "cuda" if isinstance(device_idx, int) else device_idx.type).Event
    start_events = [event_type(enable_timing=True) for _ in range(n_repeat)]
    end_events = [event_type(enable_timing=True) for _ in range(n_repeat)]

    # Run benchmark iterations
    for i in range(n_repeat):
        _flush_cache(cache)  # Clear L2 cache
        start_events[i].record()
        fn()
        end_events[i].record()

    # Synchronize and collect timings
    _cuda_synchronize(device_idx)
    times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_events, end_events)],
        dtype=torch.float,
    )

    # Return quantiles if requested
    if quantiles is not None:
        quantile_values = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        return quantile_values[0] if len(quantile_values) == 1 else quantile_values

    # Return aggregated result
    return getattr(torch, return_mode)(times).item()


def bench_with_cupti(
    fn: Callable,
    cache: torch.Tensor,
    n_repeat: int,
    device_idx: int | torch.device | None = None,
) -> float:
    """Benchmark using CUPTI profiler for detailed kernel timing."""
    use_ptpu = IS_PTPU if device_idx is None else isinstance(device_idx, torch.device) and device_idx.type == "ptpu"
    with suppress_stdout_stderr():
        schedule = torch.profiler.schedule(wait=1, warmup=0, active=1, repeat=1)
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.PrivateUse1 if use_ptpu else torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=schedule,
        )

        with profiler:
            for _ in range(2):
                for _ in range(n_repeat):
                    with torch.profiler.record_function(_CACHE_FLUSH_ID):
                        _flush_cache(cache)
                    fn()
                profiler.step()

    # `_flush_cache(cache)` and user code such as `torch.zeros` can share the same
    # generated kernel name, so exclude only the annotated cache flush range.
    def is_cuda_event(event):
        return getattr(getattr(event, "device_type", None), "name", "") == ("PrivateUse1" if use_ptpu else "CUDA")

    total_cuda_time = 0.0
    excluded_time = 0.0

    events = profiler.events()
    for event in events:
        if not is_cuda_event(event):
            continue

        if not event.is_user_annotation:
            total_cuda_time += event.self_device_time_total
        elif event.key == _CACHE_FLUSH_ID:
            excluded_time += event.self_device_time_total

    if use_ptpu:
        # PTPU associates the copy with a CPU range, whose inclusive device time
        # contains the flush. Prefer that range to avoid counting it twice.
        flush_ranges = [
            event for event in events if event.is_user_annotation and event.key == _CACHE_FLUSH_ID and event.device_type.name == "CPU"
        ]
        if flush_ranges:
            excluded_time = sum(event.device_time_total for event in flush_ranges)

    kernel_time_us = (total_cuda_time - excluded_time) / n_repeat
    return kernel_time_us * 1e-3  # Convert microseconds to milliseconds


def bench_with_cudagraph(
    fn: Callable,
    cache: torch.Tensor,
    n_repeat: int,
    quantiles: list[float] | None,
    return_mode: str,
    device_idx: int | None,
) -> float | list[float]:
    """Benchmark using CUDA graph for minimal launch overhead.

    This implementation follows triton.testing.do_bench_cudagraph.
    It captures the kernel execution in a CUDA graph and replays it multiple
    times to minimize host overhead and provide accurate timing measurements.

    Note: Cache flushing is done before graph replay, not within the graph,
    since CUDA graphs require fixed execution patterns.
    """
    n_retries = 10
    stream = torch.cuda.Stream(device=device_idx) if device_idx is not None else torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # Construct a CUDA graph with `n_repeat` unrolled function calls to minimize host overhead.
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n_repeat):
                fn()

        _cuda_synchronize(device_idx)

        # Measure time by replaying the graph multiple times.
        # Clear cache before each replay for consistent measurements.
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_retries)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_retries)]
        for i in range(n_retries):
            _flush_cache(cache)  # Clear L2 cache before replay
            start_events[i].record()
            g.replay()
            end_events[i].record()

        _cuda_synchronize(device_idx)
        times = torch.tensor(
            [s.elapsed_time(e) / n_repeat for s, e in zip(start_events, end_events)],
            dtype=torch.float,
        )

        # Return quantiles if requested
        if quantiles is not None:
            quantile_values = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
            return quantile_values[0] if len(quantile_values) == 1 else quantile_values

        # Return aggregated result
        return getattr(torch, return_mode)(times).item()


def _flush_cache(cache):
    if callable(cache):
        cache()
    else:
        cache.zero_()


def bench_with_perf_counter(
    fn: Callable,
    flush_cache: Callable,
    n_repeat: int,
    quantiles: list[float] | None,
    return_mode: str,
    device_idx: int | None,
) -> float | list[float]:
    """Benchmark using host wall-clock time (``time.perf_counter``).

    Unlike the event backend, this measures host wall-clock time (launch overhead
    included) rather than device-side event time. Synchronizing before and after
    each timed call brackets the full kernel duration.
    """
    import time

    times = []
    for _ in range(n_repeat):
        _flush_cache(flush_cache)  # Clear L2 cache
        _cuda_synchronize(device_idx)
        start = time.perf_counter()
        fn()
        _cuda_synchronize(device_idx)
        end = time.perf_counter()
        times.append((end - start) * 1e3)  # seconds -> milliseconds

    times = torch.tensor(times, dtype=torch.float)

    # Return quantiles if requested
    if quantiles is not None:
        quantile_values = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        return quantile_values[0] if len(quantile_values) == 1 else quantile_values

    # Return aggregated result
    return getattr(torch, return_mode)(times).item()
