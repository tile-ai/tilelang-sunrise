"""PTPU / CUDA event timing verification for the tvm_ffi adapter.

These tests verify that the stream binding in TVMFFIKernelAdapter.__call__ is
effective: kernel launches should observe the caller's PyTorch stream so that
accelerator event timing (torch.ptpu.Event / torch.cuda.Event) reports correct,
non-zero elapsed times.

Background
----------
The adapter calls ``tvm.device(...).set_raw_stream(raw_stream)`` before each
launch, which internally calls ``TVMFFIEnvSetStream()`` — the same thread-local
environment that the TANG/CUDA kernel launcher reads via ``TVMFFIEnvGetStream()``.
If stream binding is broken (e.g., old tvm_ffi without ``_env_set_current_stream``),
the kernel launches on a different stream and event timings between PyTorch
operations and kernel calls become unreliable.

Test methodology
----------------
1. Baseline: ``do_bench(backend='event', return_mode='median')`` produces a
   finite, non-zero latency.
2. Consistency: repeated ``do_bench`` calls produce similar latencies.
3. Stream isolation: a kernel launched on a non-default stream is not visible
   to the default stream, and events recorded on the kernel's stream measure
   the actual kernel execution time.
4. Multi-stream: independent streams can run kernel invocations concurrently
   without interfering.

The ``perf`` marker gates tests that use ``do_bench`` — pass ``--run-perf``
to include them.
"""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.profiler import do_bench
from tilelang.utils.device import get_current_device, is_ptpu_available


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _accelerator():
    """Return the active accelerator module (ptpu or cuda)."""
    return torch.ptpu if is_ptpu_available() else torch.cuda


def _accelerator_kind():
    """Return the active accelerator type string."""
    return "ptpu" if is_ptpu_available() else "cuda"


def _make_simple_kernel(M=1024, N=1024):
    """Build a lightweight elementwise-add kernel via tvm_ffi."""

    @T.prim_func
    def kernel(
        A: T.Tensor((M, N), T.float32),
        B: T.Tensor((M, N), T.float32),
        C: T.Tensor((M, N), T.float32),
    ):
        block_size = 256
        with T.Kernel(M * N // block_size, threads=block_size) as bx:
            for i in T.Parallel(block_size):
                idx = bx * block_size + i
                if idx < M * N:
                    row = idx // N
                    col = idx % N
                    C[row, col] = A[row, col] + B[row, col]

    return kernel


def _make_ge_kernel(M=512, N=1024, K=768):
    """Build a GEMM kernel via tvm_ffi for realistic workload timing."""

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float16),
    ):
        block_M, block_N, block_K = 128, 256, 32
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), T.float16)
            B_shared = T.alloc_shared((block_K, block_N), T.float16)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return kernel


# ---------------------------------------------------------------------------
# Event timing baseline
# ---------------------------------------------------------------------------
@pytest.mark.perf
def test_event_timing_produces_nonzero_latency():
    """do_bench(backend='event') must return non-zero latency.

    A zero-latency measurement indicates the kernel launched on a different
    stream than the one the events were recorded on (e.g., stream binding
    failure). This is a regression test for the set_raw_stream fix.
    """
    M, N = 1024, 1024
    program = _make_simple_kernel(M, N)
    kernel = tilelang.compile(program, out_idx=[-1], execution_backend="tvm_ffi")
    device = get_current_device()

    A = torch.randn(M, N, dtype=torch.float32, device=device)
    B = torch.randn(M, N, dtype=torch.float32, device=device)

    latency_ms = do_bench(
        lambda: kernel(A, B),
        backend="event",
        warmup=10,
        rep=50,
        return_mode="median",
    )

    assert latency_ms is not None, "do_bench returned None"
    assert latency_ms > 0.0, (
        f"Event timing returned {latency_ms} ms — this indicates the kernel launch "
        f"may not be visible to the event stream. Check stream binding."
    )
    # Basic sanity: a 1024x1024 elementwise add should complete in well under 10ms
    assert latency_ms < 50.0, f"Latency {latency_ms} ms is implausibly high for a simple elementwise kernel"


# ---------------------------------------------------------------------------
# Event timing consistency
# ---------------------------------------------------------------------------
@pytest.mark.perf
def test_event_timing_consistent():
    """Repeated do_bench calls should produce consistent latencies.

    Large variance across repeated runs may indicate timing jitter from
    incorrect stream binding.
    """
    M, N = 512, 512
    program = _make_simple_kernel(M, N)
    kernel = tilelang.compile(program, out_idx=[-1], execution_backend="tvm_ffi")
    device = get_current_device()

    A = torch.randn(M, N, dtype=torch.float32, device=device)
    B = torch.randn(M, N, dtype=torch.float32, device=device)

    latencies = []
    for _ in range(5):
        lat = do_bench(
            lambda: kernel(A, B),
            backend="event",
            warmup=5,
            rep=50,
            return_mode="median",
        )
        latencies.append(lat)

    # All latencies should be within 3x of the minimum
    min_lat = min(latencies)
    for lat in latencies:
        assert lat > 0, f"Got zero/negative latency: {lat}"
        assert lat < 10.0, f"Implausibly high latency: {lat}"
        # Allow 3x variance for warm-up / system jitter
        assert lat < min_lat * 3.0, f"Inconsistent latencies: min={min_lat:.4f}ms, got {lat:.4f}ms"


# ---------------------------------------------------------------------------
# Per-stream event timing: kernel on non-default stream
# ---------------------------------------------------------------------------
def test_event_timing_on_nontrivial_stream():
    """Kernel launched on a non-default stream should still produce correct
    event timing, confirming that set_raw_stream binds the kernel to the
    caller's active stream (not the default stream)."""
    M, N = 512, 512
    program = _make_simple_kernel(M, N)
    kernel = tilelang.compile(program, out_idx=[-1], execution_backend="tvm_ffi")
    device = get_current_device()
    acc = _accelerator()

    A = torch.randn(M, N, dtype=torch.float32, device=device)
    B = torch.randn(M, N, dtype=torch.float32, device=device)

    s = acc.Stream()
    with acc.stream(s):
        start = acc.Event(enable_timing=True)
        end = acc.Event(enable_timing=True)
        start.record()
        kernel(A, B)
        end.record()

    acc.synchronize()
    elapsed = start.elapsed_time(end)
    assert elapsed > 0.0, (
        f"Event elapsed_time={elapsed}ms on non-default stream. "
        f"This may indicate the kernel launched on the default stream "
        f"instead of the active non-default stream."
    )


# ---------------------------------------------------------------------------
# Multi-stream event timing correctness
# ---------------------------------------------------------------------------
def test_event_timing_multi_stream():
    """Multiple independent streams should all produce non-zero event timings.

    Each stream gets its own kernel invocation; events recorded on each stream
    should measure that stream's kernel, confirming per-stream binding.
    """
    M, N = 512, 512
    program = _make_simple_kernel(M, N)
    kernel = tilelang.compile(program, out_idx=[-1], execution_backend="tvm_ffi")
    device = get_current_device()
    acc = _accelerator()

    A = torch.randn(M, N, dtype=torch.float32, device=device)
    B = torch.randn(M, N, dtype=torch.float32, device=device)

    num_streams = 4
    streams = [acc.Stream() for _ in range(num_streams)]
    events_start = [acc.Event(enable_timing=True) for _ in range(num_streams)]
    events_end = [acc.Event(enable_timing=True) for _ in range(num_streams)]

    for i in range(num_streams):
        with acc.stream(streams[i]):
            events_start[i].record()
            kernel(A, B)
            events_end[i].record()

    acc.synchronize()
    for i in range(num_streams):
        elapsed = events_start[i].elapsed_time(events_end[i])
        assert elapsed > 0.0, f"Stream {i}: elapsed_time={elapsed}ms — kernel may not be bound to this stream"


# ---------------------------------------------------------------------------
# GEMM event timing (heavier workload)
# ---------------------------------------------------------------------------
@pytest.mark.perf
def test_event_timing_gemm():
    """GEMM kernel event timing should produce reasonable TFlops.

    If the kernel launches on the wrong stream, do_bench reports very low or
    zero time, leading to implausibly high TFlops. This test uses a realistic
    GEMM workload to verify that timing is in the expected range.
    """
    M, N, K = 512, 1024, 768
    program = _make_ge_kernel(M, N, K)
    kernel_fn = tilelang.compile(program, out_idx=[-1], execution_backend="tvm_ffi")
    device = get_current_device()

    dtype = torch.float16
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)

    C = kernel_fn(A, B)
    assert C.shape == (M, N)

    # Warmup
    for _ in range(5):
        kernel_fn(A, B)

    lat_ms = do_bench(
        lambda: kernel_fn(A, B),
        backend="event",
        warmup=10,
        rep=100,
        return_mode="median",
    )

    tflops = (2.0 * M * N * K) / (lat_ms / 1000.0) / 1e12
    assert lat_ms > 0.001, f"GEMM latency unrealistically low: {lat_ms:.6f} ms"
    assert lat_ms < 1000.0, f"GEMM latency implausibly high: {lat_ms:.2f} ms"
    assert 0.1 < tflops < 1000.0, f"GEMM TFlops implausible: {tflops:.1f}"


if __name__ == "__main__":
    tilelang.testing.main()
