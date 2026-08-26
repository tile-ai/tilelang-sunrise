"""Block-sparse GEMM benchmark with per-iteration GPU monitoring.

Mirrors the structure of ``examples/gemm/perf_example_gemm.py``: a background
``pt_smi dmon`` reader feeds per-iteration telemetry (temperature, power, SM and
memory clocks) alongside kernel latency, so thermal or clock throttling is
visible in the same table as the timings.

The kernel skips every block whose mask bit is false, so FLOPs are counted over
the blocks actually selected. Two derived numbers make that concrete:

  * ``TFLOPS``   -- effective throughput over the selected blocks
  * ``vs dense`` -- speedup against ``torch.matmul`` on the same shape, which is
    the number that says whether exploiting the sparsity paid off

Usage::

    python perf_example_blocksparse_gemm.py                  # full monitoring
    python perf_example_blocksparse_gemm.py --quick          # do_bench only
    python perf_example_blocksparse_gemm.py --tune           # sweep tile configs
    python perf_example_blocksparse_gemm.py --sparsity 0.9
    python perf_example_blocksparse_gemm.py --shape 4096
"""

import statistics
import subprocess
import sys
import threading
import time
from typing import Callable

import torch

from tilelang.profiler import do_bench
from tilelang.utils.device import get_current_device, is_ptpu_available

from example_blocksparse_gemm import (
    DEFAULT_BLOCK_K,
    DEFAULT_BLOCK_M,
    DEFAULT_BLOCK_N,
    DEFAULT_ENABLE_RASTERIZATION,
    DEFAULT_NUM_STAGES,
    DEFAULT_THREAD_NUM,
    blocksparse_matmul,
)

# The example's module-level defaults (128x128x32, 128 threads) are a poor fit
# for TANG, so use a 64x64x64 tile with 128 threads here.
# Keep the example's values reachable via --default-tile for comparison.
DEFAULT_CONFIG = {
    "block_M": 64,
    "block_N": 64,
    "block_K": 64,
    "num_stages": DEFAULT_NUM_STAGES,
    "thread_num": 128,
    "enable_rasteration": DEFAULT_ENABLE_RASTERIZATION,
}

EXAMPLE_CONFIG = {
    "block_M": DEFAULT_BLOCK_M,
    "block_N": DEFAULT_BLOCK_N,
    "block_K": DEFAULT_BLOCK_K,
    "num_stages": DEFAULT_NUM_STAGES,
    "thread_num": DEFAULT_THREAD_NUM,
    "enable_rasteration": DEFAULT_ENABLE_RASTERIZATION,
}

# Sweep space for --tune: tile geometry dominates, so vary it before threads.
TUNE_CONFIGS = [
    {"block_M": m, "block_N": n, "block_K": k, "num_stages": 2, "thread_num": t, "enable_rasteration": True}
    for m, n, k, t in [
        (64, 64, 32, 128),
        (64, 64, 64, 128),
        (64, 128, 64, 128),
        (64, 128, 64, 256),
        (128, 128, 32, 128),
        (128, 128, 64, 128),
        (128, 128, 64, 256),
        (128, 256, 64, 256),
    ]
]

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
                print(stats['temp_c'], stats['power_w'], stats['sm_clk_mhz'])
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
        """Return the most recent sample for *gpu_id* (or first available)."""
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
# Workload construction
# ===========================================================================


def build_workload(M: int, N: int, K: int, sparsity: float, seed: int = 42, config: dict | None = None):
    """Compile the kernel and return (run, dense_run, flops, membytes, density).

    ``flops`` counts only the mask-selected blocks, matching what the kernel
    actually computes. ``dense_run`` is the torch.matmul baseline on the same
    shape, so the caller can report the sparsity speedup.

    ``config`` overrides the tile shape; the module defaults are far from
    optimal on TANG (see ``--tune``).
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg = dict(DEFAULT_CONFIG if config is None else config)
    block_M, block_N, block_K = cfg["block_M"], cfg["block_N"], cfg["block_K"]
    device = get_current_device()

    a = torch.randn(M, K).to(device).half()
    b = torch.randn(K, N).to(device).half()

    mask_shape = (M // block_M, N // block_N, K // block_K)
    block_mask = torch.rand(mask_shape).to(device) > sparsity

    kernel = blocksparse_matmul.compile(M=M, N=N, K=K, **cfg)

    # Count the blocks the kernel does not skip: 2 flops per MAC.
    selected = int(block_mask.sum().item())
    total_blocks = block_mask.numel()
    density = selected / total_blocks if total_blocks else 0.0
    flops = 2.0 * selected * block_M * block_N * block_K
    # A and B tiles read per selected block, plus the C write, fp16.
    membytes = 2.0 * (selected * block_K * (block_M + block_N) + M * N)

    def run():
        return kernel(a, b, block_mask)

    def dense_run():
        return torch.matmul(a, b)

    return run, dense_run, flops, membytes, density


def _bench(fn) -> float:
    """Benchmark using tilelang's do_bench with event backend and median."""
    return do_bench(fn, backend="event", warmup=5, rep=20, return_mode="median")


# ===========================================================================
# Per-iteration benchmark with GPU monitoring
# ===========================================================================


def _fmt_gpu_latest(s: dict | None) -> tuple[str, str, str, str, str]:
    """Return (temp, power, sm_clk, mem_clk, sm_util) pre-formatted to fixed width."""
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
    t, p = stats["temp"], stats["power"]
    sm, mem, su = stats["sm_clk"], stats["mem_clk"], stats["sm_util"]
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
    flops: float,
    membytes: float,
    gpu_id: int = 0,
    report_interval: int = 50,
    description: str = "",
) -> tuple[float, dict | None]:
    """Run kernel iterations in a tight loop, printing per-iteration GPU stats.

    Each line shows: iter#, kernel time, TFLOPS, GB/s, GPU temp, power, clocks.

    Note the per-iteration times include one host-side ``synchronize()`` each,
    so they run consistently slower than ``--quick``'s ``do_bench`` numbers.
    Use this mode to watch telemetry drift across a long run; use ``--quick``
    when the absolute latency is what matters.

    Returns:
        (median_per_call_ms, gpu_summary_dict)
    """
    if is_ptpu_available():
        sync = torch.ptpu.synchronize
    elif torch.cuda.is_available():
        sync = torch.cuda.synchronize
    else:
        # CPU fallback: simple host-side timing, no GPU monitoring
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
    n_iters = min(n_iters, 50000)

    label = description if description else "kernel"
    header_line = (
        f"  {'iter':>6s}  {'time(ms)':>9s}  {'TFLOPS':>7s}  {'GB/s':>7s}  "
        f"{'temp':>7s}  {'power':>7s}  {'SMclk':>8s}  {'Memclk':>8s}  {'SM util':>7s}"
    )
    sep_line = f"  {'-' * 6}  {'-' * 9}  {'-' * 7}  {'-' * 7}  {'-' * 7}  {'-' * 7}  {'-' * 8}  {'-' * 8}  {'-' * 7}"
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
            secs = elapsed_ms / 1000.0
            tflops = flops / secs / 1e12
            gbps = membytes / secs / 1e9
            temp_s, pwr_s, sm_s, mem_s, util_s = _fmt_gpu_latest(gpu)
            new_marker = ""
            if gpu and gpu["ts"] != last_gpu_ts:
                new_marker = " *"  # mark when GPU sample updated
                last_gpu_ts = gpu["ts"]
            print(f"  {i + 1:6d}  {elapsed_ms:9.4f}  {tflops:7.2f}  {gbps:7.1f}  {temp_s}  {pwr_s}  {sm_s}  {mem_s}  {util_s}{new_marker}")

    # --- Summary ---
    median_ms = statistics.median(times_ms)
    gpu_samples = monitor.get_samples_since(bench_start)
    gpu_summary = monitor.summarize(gpu_samples, gpu_id=gpu_id)

    avg_ms = sum(times_ms) / len(times_ms)
    min_ms, max_ms = min(times_ms), max(times_ms)
    tf = flops / (median_ms / 1000.0) / 1e12
    gbps = membytes / (median_ms / 1000.0) / 1e9
    print(sep_line)
    print(
        f"  SUMMARY  {median_ms:9.4f}  {tf:7.2f}  {gbps:7.1f}  "
        f"(min={min_ms:.4f} avg={avg_ms:.4f} max={max_ms:.4f} ms, {len(times_ms)} iters)"
    )
    if gpu_summary:
        print(f"  {_fmt_gpu_summary(gpu_summary)}")
    print()

    return median_ms, gpu_summary


# ===========================================================================
# Main
# ===========================================================================

DEFAULT_SHAPES = [1024, 2048, 4096]
DEFAULT_SPARSITIES = [0.5, 0.9]


def _fmt_tile(cfg: dict) -> str:
    return f"{cfg['block_M']}x{cfg['block_N']}x{cfg['block_K']}/{cfg['thread_num']}t"


def tune(shapes: list[int] | None = None, sparsities: list[float] | None = None):
    """Sweep TUNE_CONFIGS and report the best tile per shape/sparsity."""
    shapes = shapes if shapes else DEFAULT_SHAPES
    sparsities = sparsities if sparsities else DEFAULT_SPARSITIES

    print(f"Device: {get_current_device()}, dtype: torch.float16")
    print(f"Sweeping {len(TUNE_CONFIGS)} configs\n")

    for size in shapes:
        for sparsity in sparsities:
            header = "{:>20s} {:>11s} {:>10s} {:>9s}".format("tile", "latency", "TFLOPS", "vs dense")
            print(f"=== {size}³ sparsity={sparsity} ===")
            print(header)
            print("-" * len(header))
            rows = []
            for cfg in TUNE_CONFIGS:
                if size % cfg["block_M"] or size % cfg["block_N"] or size % cfg["block_K"]:
                    continue
                try:
                    run, dense_run, flops, _, _ = build_workload(size, size, size, sparsity, config=cfg)
                    ms = _bench(run)
                    dense_ms = _bench(dense_run)
                except Exception as e:  # a config may fail to compile
                    print(f"{_fmt_tile(cfg):>20s}  FAILED: {str(e)[:50]}")
                    continue
                rows.append((ms, cfg, flops, dense_ms))
                print(f"{_fmt_tile(cfg):>20s} {ms:9.4f}ms {flops / (ms / 1000.0) / 1e12:10.2f} {dense_ms / ms:8.2f}x")
            if rows:
                rows.sort(key=lambda r: r[0])
                best_ms, best_cfg, best_flops, _ = rows[0]
                print(f"  best: {_fmt_tile(best_cfg)} at {best_ms:.4f}ms ({best_flops / (best_ms / 1000.0) / 1e12:.2f} TFLOPS)\n")


def main(
    report_interval: int = 50,
    shapes: list[int] | None = None,
    sparsities: list[float] | None = None,
    config: dict | None = None,
):
    shapes = shapes if shapes else DEFAULT_SHAPES
    sparsities = sparsities if sparsities else DEFAULT_SPARSITIES
    cfg = config if config else DEFAULT_CONFIG

    print(f"Device: {get_current_device()}, dtype: torch.float16")
    print(f"Tile: {_fmt_tile(cfg)}")
    print("GPU monitoring via pt_smi dmon (~1 Hz sample rate)")

    with GPUMonitor() as monitor:
        # Wait for dmon to produce first samples
        print("Waiting for GPU monitor to initialise (2 s)...")
        time.sleep(2.0)

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

        # --- Phase 1: Per-iteration monitoring for each shape/sparsity ---
        results: list[dict] = []

        for size in shapes:
            for sparsity in sparsities:
                run, dense_run, flops, membytes, density = build_workload(size, size, size, sparsity, config=cfg)
                label = f"blocksparse_gemm {size}³ sparsity={sparsity} (density={density:.1%}) {_fmt_tile(cfg)}"
                ms, gpu = bench_with_gpu_monitor(
                    run,
                    monitor,
                    flops,
                    membytes,
                    gpu_id=0,
                    report_interval=report_interval,
                    description=label,
                )
                dense_ms = _bench(dense_run)
                results.append(
                    {
                        "size": size,
                        "sparsity": sparsity,
                        "density": density,
                        "ms": ms,
                        "tflops": flops / (ms / 1000.0) / 1e12,
                        "gbps": membytes / (ms / 1000.0) / 1e9,
                        "dense_ms": dense_ms,
                        "gpu_summary": gpu,
                    }
                )

        # --- Phase 2: Summary table ---
        print(f"\n{'=' * 110}")
        print("PERFORMANCE SUMMARY")
        print("(per-iteration timing includes a host synchronize(); see --quick for do_bench numbers)")
        header = "{:>7s} {:>9s} {:>9s} {:>11s} {:>10s} {:>11s} {:>9s}".format(
            "shape", "sparsity", "density", "latency", "TFLOPS", "torch dense", "vs dense"
        )
        print(header)
        print("-" * len(header))
        for r in results:
            speedup = r["dense_ms"] / r["ms"] if r["ms"] else 0.0
            print(
                f"{r['size']:6d}³ {r['sparsity']:9.2f} {r['density']:8.1%} "
                f"{r['ms']:9.4f}ms {r['tflops']:10.2f} {r['dense_ms']:9.4f}ms {speedup:8.2f}x"
            )

        # --- Phase 3: GPU stats across all benchmarks ---
        print("\nGPU TELEMETRY SUMMARY")
        for r in results:
            gs = r["gpu_summary"]
            tag = f"{r['size']}³ sparsity={r['sparsity']}"
            if gs and gs["n_samples"] > 0:
                t, p = gs["temp"], gs["power"]
                print(
                    f"  {tag:26s}: temp {t['min']:.0f}→{t['max']:.0f}°C  "
                    f"power {p['min']:.0f}→{p['max']:.0f}W  "
                    f"({gs['n_samples']} dmon samples)"
                )
            else:
                print(f"  {tag:26s}: (no GPU samples)")

        final = monitor.summarize()
        if final and final["n_samples"] > 0:
            print(f"\nAggregate (entire run): {_fmt_gpu_summary(final)}")


def quick(shapes: list[int] | None = None, sparsities: list[float] | None = None, config: dict | None = None):
    """Fast do_bench timing, no GPU monitoring."""
    shapes = shapes if shapes else DEFAULT_SHAPES
    sparsities = sparsities if sparsities else DEFAULT_SPARSITIES
    cfg = config if config else DEFAULT_CONFIG

    print(f"Device: {get_current_device()}, dtype: torch.float16")
    print(f"Tile: {_fmt_tile(cfg)}")
    header = "{:>7s} {:>9s} {:>9s} {:>11s} {:>10s} {:>9s} {:>11s} {:>9s}".format(
        "shape", "sparsity", "density", "latency", "TFLOPS", "GB/s", "torch dense", "vs dense"
    )
    print(header)
    print("-" * len(header))
    for size in shapes:
        for sparsity in sparsities:
            run, dense_run, flops, membytes, density = build_workload(size, size, size, sparsity, config=cfg)
            ms = _bench(run)
            dense_ms = _bench(dense_run)
            secs = ms / 1000.0
            print(
                f"{size:6d}³ {sparsity:9.2f} {density:8.1%} {ms:9.4f}ms "
                f"{flops / secs / 1e12:10.2f} {membytes / secs / 1e9:9.1f} "
                f"{dense_ms:9.4f}ms {dense_ms / ms:8.2f}x"
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Block-sparse GEMM benchmark with per-iteration GPU monitoring")
    parser.add_argument("--report-interval", type=int, default=50, help="Print GPU stats every N iterations (default: 50)")
    parser.add_argument("--quick", action="store_true", help="Fast do_bench timing (no GPU monitoring)")
    parser.add_argument("--tune", action="store_true", help="Sweep tile configs and report the best per shape")
    parser.add_argument(
        "--default-tile",
        action="store_true",
        help="Use the example module's tile (128x128x32/128t) instead of the tuned default",
    )
    parser.add_argument(
        "--shape",
        type=int,
        action="append",
        help=f"Square shape to benchmark (repeatable; default: {DEFAULT_SHAPES})",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        action="append",
        help=f"Sparsity ratio to benchmark (repeatable; default: {DEFAULT_SPARSITIES})",
    )
    args = parser.parse_args()

    cfg = EXAMPLE_CONFIG if args.default_tile else DEFAULT_CONFIG

    if args.tune:
        tune(args.shape, args.sparsity)
    elif args.quick:
        quick(args.shape, args.sparsity, cfg)
    else:
        main(args.report_interval, args.shape, args.sparsity, cfg)
