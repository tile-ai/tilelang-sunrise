"""S2 profiler timing with isolated iterations and bounded input cloning."""

import contextlib
import logging
import sys
import time
from typing import Any, Callable, Optional

import torch
from torch.autograd.profiler import DeviceType

from benchmarks.timing import (
    Sample,
    _bench_meta,
    _clamp_iters,
    _kernel_busy_us,
    _kernel_span_us,
)

_logger = logging.getLogger("tileops.bench")


class _ProfilerProjectionError(Exception):
    """Profiler trace lacked a projected annotation window for every repeat."""


# Name of the ``record_function`` annotation wrapping the timed call. Kineto
# projects this scope onto the device timeline. The L2-flush ``cache.zero_()``
# is synchronized to completion before the window opens (see ``bench_kernel``),
# so its device event cannot fall inside a window regardless of how the
# projection behaves; kernels the timed call launches do.
_KERNEL_REGION = "tileops_bench_kernel"


def _project_samples(kineto_results, n_repeat: int, count_copies: bool) -> list[Sample]:
    windows = []
    events = []
    for evt in kineto_results.events():
        if evt.device_type() != DeviceType.PrivateUse1:
            continue
        if evt.is_user_annotation():
            if evt.name() == _KERNEL_REGION:
                windows.append((evt.start_ns(), evt.end_ns()))
        else:
            events.append({"start_ns": evt.start_ns(), "end_ns": evt.end_ns()})
    windows.sort()
    if len(windows) != n_repeat:
        raise _ProfilerProjectionError(f"{len(windows)}/{n_repeat} annotation windows projected")
    samples = []
    for start, end in windows:
        kernels = [e for e in events if start <= e["start_ns"] < end]
        if not kernels:
            raise _ProfilerProjectionError("a timed iteration has no projected device events")
        # PrivateUse1 does not expose a reliable kernel/copy activity discriminator.
        # Include all projected device work and disclose this in the measurement metadata.
        samples.append(
            Sample(_kernel_busy_us(kernels) * 1e-3, _kernel_span_us(kernels) * 1e-3, len(kernels))
        )
    return samples


# L2 cache flush buffer (sized to actual L2, allocated lazily)

_l2_flush_cache: Optional[torch.Tensor] = None


def _get_l2_flush_cache() -> torch.Tensor:
    global _l2_flush_cache
    if _l2_flush_cache is None:
        device_properties = torch.ptpu.get_device_properties(0)
        l2_bytes = getattr(device_properties, "L2_cache_size", 0)
        if l2_bytes <= 0:
            _logger.warning(
                "L2 cache size is unavailable or non-positive (%d); "
                "flushing a 256 MB buffer instead",
                l2_bytes,
            )
            l2_bytes = int(256e6)
        _l2_flush_cache = torch.empty(l2_bytes // 4, dtype=torch.int, device="ptpu")
    return _l2_flush_cache


def _native_output_suppressor():
    """Return an fd-level output suppressor that is safe under pytest capture.

    tilelang's ``suppress_stdout_stderr`` dup2's ``/dev/null`` over
    ``sys.stdout.fileno()``; under pytest fd capture that fileno is the
    capture tmpfile and the redirect corrupts it (``EBADF`` on later reads).
    Suppress only when stdout/stderr are the process fds 1/2.
    """
    try:
        native = sys.stdout.fileno() == 1 and sys.stderr.fileno() == 2
    except (AttributeError, OSError, ValueError):
        # Streams without a real descriptor (io.StringIO, capsys) or with
        # fileno() unsupported: fd-level suppression is impossible.
        native = False
    if not native:
        return contextlib.nullcontext()
    from tilelang.profiler.bench import suppress_stdout_stderr

    return suppress_stdout_stderr()


def bench_ptpu(
    fn: Callable,
    args: tuple[Any, ...],
    dry_run_ms: float,
    repeat_ms: float,
    max_iters: int,
    min_iters: int,
    count_copies: bool,
) -> list[Sample]:
    """Measure each iteration using projected PrivateUse1 events or PTPU events."""
    _bench_meta.copy_policy = "all-projected-device-work"
    if not isinstance(args, tuple):
        raise TypeError(
            f"bench_kernel expects a tuple of args, got {type(args).__name__}. "
            "Check that gen_inputs() returns a tuple."
        )

    cache = _get_l2_flush_cache()
    has_args = len(args) > 0

    # Pre-clone a small pool of input tensors so the kernel sees different
    # addresses across iterations.  Skip cloning if total tensor memory
    # exceeds 1 GB to avoid OOM on large workloads.
    _N_CLONES = 3
    _MAX_CLONE_BYTES = 1 << 30  # 1 GB
    if has_args:
        tensor_mask = tuple(isinstance(a, torch.Tensor) for a in args)
        total_bytes = sum(
            a.nelement() * a.element_size() for a, m in zip(args, tensor_mask, strict=True) if m
        )
        if total_bytes * _N_CLONES <= _MAX_CLONE_BYTES:
            arg_pool = [
                tuple(a.clone() if m else a for a, m in zip(args, tensor_mask, strict=True))
                for _ in range(_N_CLONES)
            ]

            def _run(i):
                return fn(*arg_pool[i % _N_CLONES])
        else:
            _logger.warning(
                "bench_kernel: inputs total %.2f GiB; skipping per-iteration "
                "cloning (kernel sees identical addresses)",
                total_bytes / (1 << 30),
            )
            arg_pool = None

            def _run(i):
                return fn(*args)
    else:
        arg_pool = None

        def _run(i):
            return fn()

    _bench_meta.inputs_cloned = arg_pool is not None or not has_args

    cache.zero_()
    _run(0)
    torch.ptpu.synchronize()
    started = time.perf_counter()
    for i in range(3):
        cache.zero_()
        _run(i)
        torch.ptpu.synchronize()
    per_iter_ms = max((time.perf_counter() - started) * 1000 / 3, 1e-6)
    n_warmup = _clamp_iters(dry_run_ms / per_iter_ms, max_iters, min_iters)
    n_repeat = _clamp_iters(repeat_ms / per_iter_ms, max_iters, min_iters)
    n_trials = 1
    # Warmup (no profiling)
    for i in range(n_warmup):
        cache.zero_()
        _run(i)
    torch.ptpu.synchronize()

    # One plain profiler context per trial; torch.profiler.schedule is avoided
    # because queued launches leak across its warmup/active boundary.
    # Kineto's window projection may include a flush merely enqueued before
    # the window, so the flush is drained (sync) before the timed call and
    # the call is drained before the next flush; the syncs add host-side
    # latency only.
    samples: list[Sample] = []
    try:
        with _native_output_suppressor():
            for _ in range(n_trials):
                with torch.profiler.profile(
                    # CPU activity is required for Kineto to project the
                    # annotation window; it never adds device time.
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.PrivateUse1,
                    ],
                ) as profiler:
                    for i in range(n_repeat):
                        cache.zero_()
                        torch.ptpu.synchronize()
                        with torch.profiler.record_function(_KERNEL_REGION):
                            _run(i)
                        torch.ptpu.synchronize()
                samples.extend(
                    _project_samples(
                        profiler.profiler.kineto_results,
                        n_repeat,
                        count_copies,
                    )
                )
        _bench_meta.timing = "privateuse1-profiler"
    except _ProfilerProjectionError as exc:
        _logger.warning(
            "PrivateUse1 profiler projection failed (%s); falling back to PTPU-events "
            "timing, which includes launch overhead",
            exc,
        )
        samples = []
        _bench_meta.fallback_reason = str(exc)

    # Fall back to PTPU events if PrivateUse1 profiling failed.
    if not samples:
        _bench_meta.timing = "ptpu-events"
        for _ in range(n_trials):
            start_events = [torch.ptpu.Event(enable_timing=True) for _ in range(n_repeat)]
            end_events = [torch.ptpu.Event(enable_timing=True) for _ in range(n_repeat)]
            for i in range(n_repeat):
                cache.zero_()
                start_events[i].record()
                _run(i)
                end_events[i].record()
            torch.ptpu.synchronize()
            times = [s.elapsed_time(e) for s, e in zip(start_events, end_events, strict=True)]
            samples.extend(Sample(t, t, None) for t in times)

    # Free the arg pool and release cached GPU memory to prevent
    # accumulation across hundreds of benchmark calls.
    if arg_pool is not None:
        del arg_pool
    torch.ptpu.empty_cache()

    return samples
