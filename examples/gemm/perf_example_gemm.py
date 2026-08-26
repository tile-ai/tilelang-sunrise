import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench
from tilelang.transform.pass_config import PassConfigKey

import torch
from typing import Callable

import subprocess
import threading
import time
import sys
import statistics

# Enable cop4 async DMA widening (16 bytes/load vs cop2's 8 bytes/load)
tilelang.transform.PassContext(
    config={
        PassConfigKey.TL_ENABLE_ASYNC_COPY: True,
    }
).__enter__()


# ===========================================================================
# GPU Monitor: background pt_smi dmon reader with latest-value cache
# ===========================================================================


class GPUMonitor:
    """Background GPU monitor that runs ``pt_smi dmon`` and caches the latest
    reading per GPU.  The dmon process outputs one sample per GPU every ~1 s;
    callers polling at higher frequency simply see the most-recent value
    repeated until the next hardware update arrives.

    Usage as context manager::

        with GPUMonitor(gpu_ids=[0]) as mon:
            for each_iteration:
                stats = mon.latest(gpu_id=0)
                print(stats.temp_c, stats.power_w, stats.sm_clk_mhz)
    """

    def __init__(self, gpu_ids: list[int] | None = None):
        self.gpu_ids = gpu_ids
        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._running = False
        # {gpu_id: dict} — latest reading per GPU
        self._latest: dict[int, dict] = {}
        # Full history: list of dicts (same format)
        self._samples: list[dict] = []

    # -- context manager ----------------------------------------------------
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    # -- lifecycle ----------------------------------------------------------
    def start(self):
        if self._running:
            return
        try:
            self._process = subprocess.Popen(
                ["pt_smi", "dmon"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError:
            print("[GPUMonitor] pt_smi not found — GPU monitoring disabled.", file=sys.stderr)
            return
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=3)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                self._process.kill()
                self._process.wait(timeout=2)
            self._process = None
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None

    # -- internals ----------------------------------------------------------
    def _reader(self):
        """Parse dmon output lines continuously."""
        assert self._process is not None
        for line in self._process.stdout:
            if not self._running:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            try:
                gpu_id = int(parts[0])
                if self.gpu_ids is not None and gpu_id not in self.gpu_ids:
                    continue
                sample = {
                    "ts": time.time(),
                    "gpu": gpu_id,
                    "power_w": float(parts[1]),
                    "temp_c": float(parts[2]),
                    "sm_util_pct": float(parts[4]),
                    "mem_util_pct": float(parts[5]),
                    "mem_clk_mhz": float(parts[8]),
                    "sm_clk_mhz": float(parts[9]),
                }
                with self._lock:
                    self._latest[gpu_id] = sample
                    self._samples.append(sample)
            except (ValueError, IndexError):
                pass

    # -- query API ----------------------------------------------------------
    def latest(self, gpu_id: int | None = None) -> dict | None:
        """Return the most recent sample for *gpu_id* (or first available).

        Returns None if no sample has been collected yet.
        """
        with self._lock:
            if gpu_id is not None:
                return self._latest.get(gpu_id)
            if self._latest:
                return next(iter(self._latest.values()))
            return None

    def clear(self):
        """Discard all collected samples (but keep latest cache)."""
        with self._lock:
            self._samples.clear()

    def get_samples_since(self, since_ts: float) -> list[dict]:
        """Return full-sample history since *since_ts*."""
        with self._lock:
            return [s for s in self._samples if s["ts"] >= since_ts]

    def summarize(self, samples: list[dict] | None = None, gpu_id: int | None = None) -> dict | None:
        """Compute min / max / avg for each metric."""
        if samples is None:
            with self._lock:
                samples = list(self._samples)
        if gpu_id is not None:
            samples = [s for s in samples if s["gpu"] == gpu_id]
        if not samples:
            return None

        def _stats(key):
            vals = [s[key] for s in samples]
            return {"min": min(vals), "max": max(vals), "avg": sum(vals) / len(vals)}

        return {
            "temp": _stats("temp_c"),
            "power": _stats("power_w"),
            "sm_clk": _stats("sm_clk_mhz"),
            "mem_clk": _stats("mem_clk_mhz"),
            "sm_util": _stats("sm_util_pct"),
            "mem_util": _stats("mem_util_pct"),
            "n_samples": len(samples),
        }

    @property
    def sample_count(self) -> int:
        with self._lock:
            return len(self._samples)


# ===========================================================================
# GEMM kernel definition
# ===========================================================================


@tilelang.jit(out_idx=[-1])
def matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    num_threads,
    a_local_load_type,
    b_local_load_type,
    enable_threadblock_swizzle,
    panel_size,
    k_step,
    num_stages=2,
    policy=T.GemmWarpPolicy.FullCol,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    """2-stage async DMA GEMM kernel with copy/compute overlap.

    Pipeline:
      1. Prefetch block 0 via async DMA (async_scope=1)
      2. Loop: async copy block k while gemm runs on block k-1

    Key optimizations:
      - async_scope=1: non-blocking DMA for true copy/gemm overlap
      - FullRow 128³ for >=4096: higher grid parallelism, lower register pressure
      - Scalar LDS (gemm_tmma.h): avoids __shfl_sync serialization
      - k_step=4: balanced B_reg(8 uint32) ←→ reg pressure
    """
    num_iters = T.ceildiv(K, block_K)

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=num_threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared")
            B_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared")
            T.use_swizzle(panel_size=panel_size, enable=enable_threadblock_swizzle)

            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)

            a_base = by * block_M
            b_base = bx * block_N

            with T.attr("default", "async_scope", 1):
                T.copy(A[a_base, 0 * block_K], A_shared)
                T.copy(B[0 * block_K, b_base], B_shared)
            T.gemm(
                A_shared,
                B_shared,
                C_local,
                k_step=k_step,
                policy=policy,
                a_local_load_type=a_local_load_type,
                b_local_load_type=b_local_load_type,
            )

            for k in T.serial(1, num_iters):
                with T.attr("default", "async_scope", 1):
                    T.copy(A[a_base, k * block_K], A_shared)
                    T.copy(B[k * block_K, b_base], B_shared)
                T.gemm(
                    A_shared,
                    B_shared,
                    C_local,
                    k_step=k_step,
                    policy=policy,
                    a_local_load_type=a_local_load_type,
                    b_local_load_type=b_local_load_type,
                )

            C_shared = T.alloc_shared((block_M, block_N), dtype, scope="shared")
            T.copy(C_local, C_shared)
            with T.attr("default", "async_scope", 1):
                T.copy(C_shared, C[by * block_M, bx * block_N])

    return gemm


# ===========================================================================
# Shape-aware auto-config selection
# ===========================================================================


def _ensure_k_step(config: dict, dtype) -> dict:
    """Adjust k_step so inner_k %% k_step == 0 for the given dtype."""
    # tile_size_k = 16 / sizeof(A_type)
    if dtype == torch.int8:
        tile_size_k = 16  # sizeof(int8) == 1 byte
    else:
        tile_size_k = 8  # fp16/bf16: sizeof(__fp16) == 2 bytes

    block_K = config["block_K"]
    inner_k = block_K // tile_size_k
    k_step = config["k_step"]

    if inner_k % k_step != 0:
        # Find the largest compatible k_step <= current k_step
        for candidate in (16, 8, 4, 2, 1):
            if k_step >= candidate and inner_k % candidate == 0:
                config = {**config, "k_step": candidate}
                break
    return config


def get_config(M: int, N: int, K: int, dtype=torch.float16) -> dict:
    """Return an optimized config for the given GEMM shape."""
    base = {
        "num_threads": 256,
        "k_step": 4,
        "num_stages": 2,
        "a_local_load_type": "load_overlap_mma",
        "b_local_load_type": "load_overlap_mma",
        "enable_threadblock_swizzle": True,
        "panel_size": 4,
    }

    if M >= 4096 and N >= 4096:
        cfg = {**base, "block_M": 128, "block_N": 128, "block_K": 128, "policy": T.GemmWarpPolicy.FullRow}
    elif M >= 2048 and N >= 2048:
        cfg = {**base, "block_M": 64, "block_N": 64, "block_K": 128, "policy": T.GemmWarpPolicy.FullRow}
    elif M >= 1024 and N >= 1024:
        cfg = {
            **base,
            "block_M": 64,
            "block_N": 64,
            "block_K": 128,
            "a_local_load_type": "load_before_mma",
            "b_local_load_type": "load_before_mma",
            "enable_threadblock_swizzle": False,
            "policy": T.GemmWarpPolicy.FullRow,
        }
    elif M > N * 4:
        cfg = {**base, "block_M": 128, "block_N": 64, "block_K": 256, "policy": T.GemmWarpPolicy.FullRow}
    elif N > M * 4:
        cfg = {**base, "block_M": 64, "block_N": 128, "block_K": 256, "policy": T.GemmWarpPolicy.FullRow}
    else:
        cfg = {**base, "block_M": 64, "block_N": 64, "block_K": 128, "policy": T.GemmWarpPolicy.FullRow}

    return _ensure_k_step(cfg, dtype)


def create_kernel(M: int, N: int, K: int, config: dict = None, dtype=torch.float16) -> Callable:
    """Create and return a compiled TileLang GEMM kernel."""
    if config is None:
        config = get_config(M, N, K, dtype=dtype)
    else:
        config = _ensure_k_step(config, dtype)
    return matmul(M, N, K, **config)


def _bench(fn) -> float:
    """Benchmark using tilelang's do_bench with event backend and median."""
    return do_bench(fn, backend="event", warmup=25, rep=100, return_mode="median")


def _compare_outputs(tl_out, ref_out, atol=1e-2, rtol=1e-2) -> bool:
    return torch.allclose(tl_out.cpu(), ref_out.cpu(), atol=atol, rtol=rtol)


# ===========================================================================
# Per-iteration benchmark with GPU monitoring
# ===========================================================================


def _fmt_gpu_latest(s: dict | None) -> tuple[str, str, str, str, str]:
    """Return (temp_str, power_str, sm_clk_str, mem_clk_str, sm_util_str)
    for a single GPU sample, each pre-formatted to a fixed width."""
    if s is None:
        return ("  N/A ", "  N/A ", "   N/A ", "   N/A ", "  N/A ")
    return (
        f"{s['temp_c']:5.0f}°C",
        f"{s['power_w']:5.0f}W",
        f"{s['sm_clk_mhz']:5.0f}MHz",
        f"{s['mem_clk_mhz']:5.0f}MHz",
        f"{s['sm_util_pct']:5.0f}%",
    )


def _fmt_gpu_summary(stats: dict | None) -> str:
    """Format GPU stats summary dict into a one-line string."""
    if stats is None or stats["n_samples"] == 0:
        return "GPU: (no samples)"
    n = stats["n_samples"]
    t = stats["temp"]
    p = stats["power"]
    sm = stats["sm_clk"]
    mem = stats["mem_clk"]
    su = stats["sm_util"]
    return (
        f"GPU ({n} samples): "
        f"temp {t['min']:.0f}→{t['max']:.0f}°C (avg {t['avg']:.1f})  "
        f"power {p['min']:.0f}→{p['max']:.0f}W (avg {p['avg']:.1f})  "
        f"SM {sm['avg']:.0f}MHz  Mem {mem['avg']:.0f}MHz  "
        f"util {su['min']:.0f}→{su['max']:.0f}%"
    )


def bench_with_gpu_monitor(
    fn: Callable,
    monitor: GPUMonitor,
    m: int,
    n: int,
    k: int,
    gpu_id: int = 0,
    report_interval: int = 50,
    description: str = "",
) -> tuple[float, dict | None]:
    """Run GEMM iterations in a tight loop, printing per-iteration GPU stats.

    Each line shows: iter#, kernel time, TFLOPS, GPU temp, power, SM/Mem clocks.

    Args:
        fn: The GEMM kernel callable.
        monitor: Running GPUMonitor instance.
        m, n, k: GEMM dimensions (for TFLOPS calculation).
        gpu_id: Which GPU to report stats for.
        report_interval: Print a line every N iterations.
        description: Label for the output header.

    Returns:
        (median_per_call_ms, gpu_summary_dict)
    """
    IS_PTPU = hasattr(torch, "ptpu") and torch.ptpu.is_available()
    IS_CUDA = hasattr(torch, "cuda") and torch.cuda.is_available()

    if IS_PTPU:
        sync = torch.ptpu.synchronize
    elif IS_CUDA:
        sync = torch.cuda.synchronize
    else:
        # CPU fallback: simple host-side timing, no GPU monitoring
        t0 = time.perf_counter()
        times = []
        for _ in range(100):
            t1 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t1) * 1000)
        return statistics.median(times), None

    # --- Warmup ---
    for _ in range(10):
        fn()
    sync()

    # --- Estimate per-call latency for progress reporting ---
    calib_iters = 20
    t0 = time.perf_counter()
    for _ in range(calib_iters):
        fn()
    sync()
    per_call_ms = max((time.perf_counter() - t0) / calib_iters * 1000, 1e-6)

    # --- Run at least long enough to capture several dmon samples ---
    # dmon updates ~1 Hz, so target 5+ seconds for 5+ GPU samples
    target_s = 5.0
    n_iters = max(200, int(target_s * 1000 / per_call_ms))
    # Cap at a reasonable maximum to avoid excessive runtime
    n_iters = min(n_iters, 50000)

    # --- Print per-iteration header ---
    label = description if description else f"({m},{n},{k})"
    # Column widths: iter=6, time=9, tflops=7, temp=7, power=7, smclk=8, memclk=8, smutil=7
    header_line = (
        f"  {'iter':>6s}  {'time(ms)':>9s}  {'TFLOPS':>7s}  {'temp':>7s}  {'power':>7s}  {'SMclk':>8s}  {'Memclk':>8s}  {'SM util':>7s}"
    )
    sep_line = f"  {'-' * 6}  {'-' * 9}  {'-' * 7}  {'-' * 7}  {'-' * 7}  {'-' * 8}  {'-' * 8}  {'-' * 7}"
    print(f"\n{'=' * 110}")
    print(
        f"  {label}: per-iteration GPU monitoring ({n_iters} iters, "
        f"~{n_iters * per_call_ms / 1000:.1f}s, report every {report_interval} iters)"
    )
    print(header_line)
    print(sep_line)

    monitor.clear()
    times_ms: list[float] = []
    bench_start = time.time()
    last_gpu_ts = None  # track when GPU sample last changed

    for i in range(n_iters):
        t1 = time.perf_counter()
        fn()
        sync()
        elapsed_ms = (time.perf_counter() - t1) * 1000
        times_ms.append(elapsed_ms)

        if (i + 1) % report_interval == 0 or i == 0:
            gpu = monitor.latest(gpu_id)
            tflops = (2.0 * m * n * k) / (elapsed_ms / 1000.0) / 1e12
            temp_s, pwr_s, sm_s, mem_s, util_s = _fmt_gpu_latest(gpu)
            new_marker = ""
            if gpu and gpu["ts"] != last_gpu_ts:
                new_marker = " *"  # mark when GPU sample updated
                last_gpu_ts = gpu["ts"]
            print(f"  {i + 1:6d}  {elapsed_ms:9.4f}  {tflops:7.1f}  {temp_s}  {pwr_s}  {sm_s}  {mem_s}  {util_s}{new_marker}")

    # --- Summary ---
    median_ms = statistics.median(times_ms)
    gpu_samples = monitor.get_samples_since(bench_start)
    gpu_summary = monitor.summarize(gpu_samples, gpu_id=gpu_id)

    # Add n_iters info
    avg_ms = sum(times_ms) / len(times_ms)
    min_ms = min(times_ms)
    max_ms = max(times_ms)
    tf = (2.0 * m * n * k) / (median_ms / 1000.0) / 1e12
    print(sep_line)
    print(f"  SUMMARY  {median_ms:9.4f}  {tf:7.1f}  (min={min_ms:.4f} avg={avg_ms:.4f} max={max_ms:.4f} ms, {len(times_ms)} iters)")
    if gpu_summary:
        print(f"  {_fmt_gpu_summary(gpu_summary)}")
    print()

    return median_ms, gpu_summary


# ===========================================================================
# Main
# ===========================================================================


def main(report_interval: int = 50):
    shapes = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096), (8192, 8192, 8192)]
    device = "ptpu" if hasattr(torch, "ptpu") and torch.ptpu.is_available() else "cpu"
    dtype = torch.float16

    print(f"Device: {device}, dtype: {dtype}")
    print("GPU monitoring via pt_smi dmon (~1 Hz sample rate)")

    with GPUMonitor() as monitor:
        # Wait for dmon to produce first samples
        print("Waiting for GPU monitor to initialise (2 s)...")
        time.sleep(2.0)

        # Print baseline GPU state
        baseline = monitor.latest()
        if baseline:
            print(
                f"Baseline GPU state: temp={baseline['temp_c']:.0f}°C, "
                f"power={baseline['power_w']:.0f}W, "
                f"SM={baseline['sm_clk_mhz']:.0f}MHz, "
                f"Mem={baseline['mem_clk_mhz']:.0f}MHz"
            )
        else:
            print("No GPU samples received — is pt_smi available?")

        # --- Phase 1: Per-iteration monitoring for TileLang GEMM (each shape) ---
        results: list[dict] = []

        for m, n, k in shapes:
            config = get_config(m, n, k, dtype=dtype)
            kernel = create_kernel(m, n, k, config, dtype=dtype)
            A = torch.randn(m, k, dtype=dtype, device=device)
            B = torch.randn(k, n, dtype=dtype, device=device)

            # --- TileLang per-iteration bench ---
            tl_ms, tl_gpu = bench_with_gpu_monitor(
                lambda kernel=kernel, A=A, B=B: kernel(A, B),
                monitor,
                m,
                n,
                k,
                gpu_id=0,
                report_interval=report_interval,
                description=f"TileLang ({m},{n},{k}) bk={config['block_M']}x{config['block_N']}x{config['block_K']}",
            )
            tl_tflops = (2.0 * m * n * k) / (tl_ms / 1000.0) / 1e12

            # --- Reference: torch.mm (standard do_bench, fast) ---
            ref_a = A.clone()
            ref_b = B.clone()
            matmul_ms = _bench(lambda ref_a=ref_a, ref_b=ref_b: torch.matmul(ref_a, ref_b))
            matmul_tflops = (2.0 * m * n * k) / (matmul_ms / 1000.0) / 1e12

            # --- Reference: torch.Linear (standard do_bench, fast) ---
            linear = torch.nn.Linear(k, n, bias=False, dtype=dtype, device=device)
            linear.weight.data = B.T.contiguous().clone()
            A_linear = A.clone()
            ref_linear_out = linear(A_linear)
            linear_ms = _bench(lambda linear=linear, A_linear=A_linear: linear(A_linear))
            linear_tflops = (2.0 * m * n * k) / (linear_ms / 1000.0) / 1e12

            # --- Correctness ---
            tl_out = kernel(A, B)
            ref_matmul = torch.matmul(A, B)
            correct = _compare_outputs(tl_out, ref_matmul) and _compare_outputs(tl_out, ref_linear_out)
            mark = "PASS" if correct else "FAIL"

            results.append(
                {
                    "shape": (m, n, k),
                    "tl_ms": tl_ms,
                    "tl_tflops": tl_tflops,
                    "matmul_ms": matmul_ms,
                    "matmul_tflops": matmul_tflops,
                    "linear_ms": linear_ms,
                    "linear_tflops": linear_tflops,
                    "correct": mark,
                    "gpu_summary": tl_gpu,
                }
            )

        # --- Phase 2: Summary table ---
        print(f"\n{'=' * 110}")
        print("PERFORMANCE SUMMARY")
        header = "{:20s} {:>12s} {:>12s} {:>12s} {:>7s} {:>7s} {:>8s}".format(
            "Shape", "TileLang", "torch.mm", "torch.Linear", "vs mm", "vs Lin", "correct"
        )
        print(header)
        print("-" * len(header))
        for r in results:
            m, n, k = r["shape"]
            vs_mm = r["tl_tflops"] / r["matmul_tflops"] * 100
            vs_lin = r["tl_tflops"] / r["linear_tflops"] * 100
            print(
                f"({m},{n},{k})  {r['tl_ms']:.4f}ms/{r['tl_tflops']:.1f}T  "
                f"{r['matmul_ms']:.4f}ms/{r['matmul_tflops']:.1f}T  "
                f"{r['linear_ms']:.4f}ms/{r['linear_tflops']:.1f}T  "
                f"{vs_mm:.1f}%  {vs_lin:.1f}%  {r['correct']}"
            )

        # --- Phase 3: GPU stats across all TileLang benchmarks ---
        print("\nGPU TELEMETRY SUMMARY")
        for r in results:
            m, n, k = r["shape"]
            gs = r["gpu_summary"]
            if gs and gs["n_samples"] > 0:
                t = gs["temp"]
                p = gs["power"]
                print(
                    f"  ({m},{n},{k}): temp {t['min']:.0f}→{t['max']:.0f}°C  "
                    f"power {p['min']:.0f}→{p['max']:.0f}W  "
                    f"({gs['n_samples']} dmon samples)"
                )
            else:
                print(f"  ({m},{n},{k}): (no GPU samples)")

        # Overall aggregate
        final = monitor.summarize()
        if final and final["n_samples"] > 0:
            print(f"\nAggregate (entire run): {_fmt_gpu_summary(final)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TileLang GEMM benchmark with per-iteration GPU monitoring")
    parser.add_argument("--report-interval", type=int, default=50, help="Print GPU stats every N iterations (default: 50)")
    parser.add_argument("--quick", action="store_true", help="Original fast do_bench timing (no GPU monitoring)")
    args = parser.parse_args()

    if args.quick:
        shapes = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096), (8192, 8192, 8192)]
        device = "ptpu" if hasattr(torch, "ptpu") and torch.ptpu.is_available() else "cpu"
        dtype = torch.float16
        header = "{:20s} {:>12s} {:>12s} {:>12s} {:>7s} {:>7s} {:>8s}".format(
            "Shape", "TileLang", "torch.mm", "torch.Linear", "vs mm", "vs Lin", "correct"
        )
        print(header)
        print("-" * len(header))
        for m, n, k in shapes:
            config = get_config(m, n, k, dtype=dtype)
            kernel = create_kernel(m, n, k, config, dtype=dtype)
            A = torch.randn(m, k, dtype=dtype, device=device)
            B = torch.randn(k, n, dtype=dtype, device=device)
            tl_ms = _bench(lambda kernel=kernel, A=A, B=B: kernel(A, B))
            tl_tflops = (2.0 * m * n * k) / (tl_ms / 1000.0) / 1e12
            ref_matmul = torch.matmul(A, B)
            matmul_ms = _bench(lambda A=A, B=B: torch.matmul(A, B))
            matmul_tflops = (2.0 * m * n * k) / (matmul_ms / 1000.0) / 1e12
            linear = torch.nn.Linear(k, n, bias=False, dtype=dtype, device=device)
            linear.weight.data = B.T.contiguous().clone()
            A_linear = A.clone()
            ref_linear = linear(A_linear)
            linear_ms = _bench(lambda linear=linear, A_linear=A_linear: linear(A_linear))
            linear_tflops = (2.0 * m * n * k) / (linear_ms / 1000.0) / 1e12
            tl_out = kernel(A, B)
            correct = _compare_outputs(tl_out, ref_matmul) and _compare_outputs(tl_out, ref_linear)
            mark = "PASS" if correct else "FAIL"
            print(
                f"({m},{n},{k})  {tl_ms:.4f}ms/{tl_tflops:.1f}T  "
                f"{matmul_ms:.4f}ms/{matmul_tflops:.1f}T  "
                f"{linear_ms:.4f}ms/{linear_tflops:.1f}T  "
                f"{tl_tflops / matmul_tflops * 100:.1f}%  "
                f"{tl_tflops / linear_tflops * 100:.1f}%  {mark}"
            )
    else:
        main(args.report_interval)
