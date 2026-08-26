"""Block-sparse attention benchmark with per-iteration GPU monitoring.

Mirrors the structure of ``examples/gemm/perf_example_gemm.py``: a background
``pt_smi dmon`` reader feeds per-iteration telemetry (temperature, power, SM and
memory clocks) alongside kernel latency, so thermal or clock throttling is
visible in the same table as the timings.

Three kernels are covered:

  * ``blocksparse_attn``   -- top-k block-sparse prefill attention
  * ``gqa_decode_indice``  -- sparse GQA decode selected by block indices
  * ``gqa_decode_mask``    -- sparse GQA decode selected by a block mask

Reference tensors and the sparsity patterns are built on the host: PTPU has no
``scatter_``/``randperm``/``tril``, so keeping them on the CPU is what makes the
same script run on both CUDA and PTPU.

Usage::

    python perf_example_blocksparse_attention.py            # full monitoring
    python perf_example_blocksparse_attention.py --quick    # do_bench only
    python perf_example_blocksparse_attention.py --kernel gqa_decode_mask
"""

import math
import statistics
import subprocess
import sys
import threading
import time
from typing import Callable

import torch

from tilelang.profiler import do_bench
from tilelang.utils.device import get_current_device, is_ptpu_available

import example_tilelang_block_sparse_attn as bsa
import example_tilelang_sparse_gqa_decode_varlen_indice as gqa_indice
import example_tilelang_sparse_gqa_decode_varlen_mask as gqa_mask

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
# Workload builders — each returns (callable, flops, label, correctness)
# ===========================================================================


def _sync_device():
    if is_ptpu_available():
        torch.ptpu.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def build_blocksparse_attn(batch=1, heads=32, seq_len=256, dim=64, topk=2, block=64):
    """Top-k block-sparse prefill attention.

    FLOPs count only the blocks the mask actually selects, so the number is
    comparable against a dense attention baseline of the same shape.
    """
    torch.manual_seed(0)
    # Host-side construction: PTPU lacks scatter_/masked_select used by the
    # top-k mask helper.
    q = torch.randn(batch, heads, seq_len, dim, dtype=torch.float16)
    k = torch.randn(batch, heads, seq_len, dim, dtype=torch.float16)
    v = torch.randn(batch, heads, seq_len, dim, dtype=torch.float16)

    downsample_len = math.ceil(seq_len / block)
    x_ds = torch.randn([batch, heads, downsample_len, downsample_len], dtype=torch.bfloat16)
    x_ds[:, :, :, 0] = 100
    block_mask = bsa.get_sparse_attn_mask_from_topk(x_ds, topk=topk)

    kernel = bsa.blocksparse_flashattn(batch, heads, seq_len, dim, downsample_len, is_causal=True)

    device = get_current_device()
    q_d, k_d, v_d = q.to(device), k.to(device), v.to(device)
    mask_d = block_mask.to(device)

    # QK^T and PV over the selected blocks only: 2 gemms * 2 flops/MAC.
    selected_blocks = int(block_mask.sum().item())
    flops = 2 * 2 * selected_blocks * (block * block) * dim
    # Q/K/V reads + O write, fp16.
    membytes = 2 * (q.numel() + k.numel() + v.numel() + q.numel())

    def run():
        return kernel(q_d, k_d, v_d, mask_d)

    label = f"blocksparse_attn b{batch} h{heads} s{seq_len} d{dim} topk{topk}"
    return run, flops, membytes, label


def build_gqa_decode_indice(batch=1, heads=32, heads_kv=8, max_cache_seqlen=2048, dim=128, dim_v=128, sparse_ratio=0.8, block_size=32):
    """Sparse GQA decode driven by explicit block indices."""
    torch.manual_seed(0)
    max_selected_blocks = int(math.ceil(max_cache_seqlen * (1 - sparse_ratio) / block_size))

    # Host-side construction: randperm is not implemented on PTPU.
    Q = torch.randn((batch, heads, dim), dtype=torch.float16)
    K = torch.randn((batch, max_cache_seqlen, heads_kv, dim), dtype=torch.float16)
    V = torch.randn((batch, max_cache_seqlen, heads_kv, dim_v), dtype=torch.float16)
    cache_seqlens = torch.randint(1, max_cache_seqlen, (batch,), dtype=torch.int32)
    max_valid_num_blocks = torch.ceil(cache_seqlens / block_size).int()
    block_indices = torch.full((batch, heads_kv, max_selected_blocks), -1, dtype=torch.int32)

    for b in range(batch):
        max_valid_block = max_valid_num_blocks[b].item()
        if max_valid_block > 0:
            for h in range(heads_kv):
                valid = torch.randperm(max_valid_block, dtype=torch.int32)[:max_selected_blocks]
                block_indices[b, h, : len(valid)] = valid
    block_indices, _ = block_indices.sort(dim=-1, descending=True)

    device = get_current_device()
    Q_d, K_d, V_d = Q.to(device), K.to(device), V.to(device)
    idx_d, seqlens_d = block_indices.to(device), cache_seqlens.to(device)

    model = gqa_indice.SparseFlashAttn(batch, heads, heads_kv, dim, dim_v, block_size)

    # Only non-negative indices contribute; each selected block does QK^T + PV.
    selected = int((block_indices >= 0).sum().item())
    kv_per_head_group = heads // heads_kv
    flops = 2 * 2 * selected * kv_per_head_group * block_size * dim
    # Decode is memory bound: only the selected KV blocks are actually read.
    membytes = 2 * (Q.numel() + selected * block_size * (dim + dim_v) + Q.numel())

    def run():
        return model(Q_d, K_d, V_d, idx_d, seqlens_d)

    label = f"gqa_decode_indice b{batch} h{heads}/{heads_kv} cache{max_cache_seqlen} bs{block_size}"
    return run, flops, membytes, label


def build_gqa_decode_mask(batch=1, heads=32, heads_kv=8, max_cache_seqlen=2048, dim=128, dim_v=128, sparse_ratio=0.8, block_size=32):
    """Sparse GQA decode driven by a boolean block mask."""
    torch.manual_seed(0)

    # Host-side construction: randperm is not implemented on PTPU.
    Q = torch.randn((batch, heads, dim), dtype=torch.float16)
    K = torch.randn((batch, max_cache_seqlen, heads_kv, dim), dtype=torch.float16)
    V = torch.randn((batch, max_cache_seqlen, heads_kv, dim_v), dtype=torch.float16)
    cache_seqlens = torch.randint(1, max_cache_seqlen, (batch,), dtype=torch.int32)
    cache_seqlens[torch.randint(0, batch, (1,)).item()] = max_cache_seqlen

    num_blocks = (max_cache_seqlen + block_size - 1) // block_size
    valid_num_blocks = torch.ceil(cache_seqlens * (1 - sparse_ratio) / block_size).int()
    max_valid_num_blocks = torch.ceil(cache_seqlens / block_size).int()
    block_mask = torch.zeros((batch, heads_kv, num_blocks), dtype=torch.bool)

    for b in range(batch):
        max_valid_block = max_valid_num_blocks[b].item()
        valid_num_block = valid_num_blocks[b].item()
        if valid_num_block > 0:
            for h in range(heads_kv):
                perm = torch.randperm(max_valid_block)[:valid_num_block]
                block_mask[b, h, perm] = True

    device = get_current_device()
    Q_d, K_d, V_d = Q.to(device), K.to(device), V.to(device)
    mask_d, seqlens_d = block_mask.to(device), cache_seqlens.to(device)

    model = gqa_mask.SparseFlashAttn(batch, heads, heads_kv, dim, dim_v, block_size)

    selected = int(block_mask.sum().item())
    kv_per_head_group = heads // heads_kv
    flops = 2 * 2 * selected * kv_per_head_group * block_size * dim
    # Decode is memory bound: only the selected KV blocks are actually read.
    membytes = 2 * (Q.numel() + selected * block_size * (dim + dim_v) + Q.numel())

    def run():
        return model(Q_d, K_d, V_d, mask_d, seqlens_d)

    label = f"gqa_decode_mask b{batch} h{heads}/{heads_kv} cache{max_cache_seqlen} bs{block_size}"
    return run, flops, membytes, label


WORKLOADS: dict[str, Callable] = {
    "blocksparse_attn": build_blocksparse_attn,
    "gqa_decode_indice": build_gqa_decode_indice,
    "gqa_decode_mask": build_gqa_decode_mask,
}


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
    Sparse decode is memory bound, so GB/s is usually the meaningful column
    while TFLOPS stays low by construction.

    Note the per-iteration times include one host-side ``synchronize()`` each,
    so they run consistently slower than ``--quick``'s ``do_bench`` numbers.
    Use this mode to watch telemetry drift across a long run; use ``--quick``
    when the absolute latency is what matters.

    Returns:
        (median_per_call_ms, gpu_summary_dict)
    """
    IS_PTPU = is_ptpu_available()
    IS_CUDA = torch.cuda.is_available()

    if IS_PTPU:
        sync = torch.ptpu.synchronize
    elif IS_CUDA:
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


def main(report_interval: int = 50, kernels: list[str] | None = None):
    names = kernels if kernels else list(WORKLOADS)
    device = get_current_device()

    print(f"Device: {device}, dtype: torch.float16")
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

        # --- Phase 1: Per-iteration monitoring for each kernel ---
        results: list[dict] = []

        for name in names:
            run, flops, membytes, label = WORKLOADS[name]()
            ms, gpu = bench_with_gpu_monitor(
                run,
                monitor,
                flops,
                membytes,
                gpu_id=0,
                report_interval=report_interval,
                description=label,
            )
            results.append(
                {
                    "name": name,
                    "label": label,
                    "ms": ms,
                    "tflops": flops / (ms / 1000.0) / 1e12,
                    "gbps": membytes / (ms / 1000.0) / 1e9,
                    "gpu_summary": gpu,
                }
            )

        # --- Phase 2: Summary table ---
        print(f"\n{'=' * 110}")
        print("PERFORMANCE SUMMARY")
        print("(per-iteration timing includes a host synchronize(); see --quick for do_bench numbers)")
        header = "{:22s} {:>12s} {:>12s} {:>10s}".format("Kernel", "latency", "TFLOPS", "GB/s")
        print(header)
        print("-" * len(header))
        for r in results:
            print(f"{r['name']:22s} {r['ms']:9.4f}ms {r['tflops']:11.2f} {r['gbps']:10.1f}")

        # --- Phase 3: GPU stats across all benchmarks ---
        print("\nGPU TELEMETRY SUMMARY")
        for r in results:
            gs = r["gpu_summary"]
            if gs and gs["n_samples"] > 0:
                t, p = gs["temp"], gs["power"]
                print(
                    f"  {r['name']:22s}: temp {t['min']:.0f}→{t['max']:.0f}°C  "
                    f"power {p['min']:.0f}→{p['max']:.0f}W  "
                    f"({gs['n_samples']} dmon samples)"
                )
            else:
                print(f"  {r['name']:22s}: (no GPU samples)")

        final = monitor.summarize()
        if final and final["n_samples"] > 0:
            print(f"\nAggregate (entire run): {_fmt_gpu_summary(final)}")


def quick(kernels: list[str] | None = None):
    """Fast do_bench timing, no GPU monitoring."""
    names = kernels if kernels else list(WORKLOADS)
    print(f"Device: {get_current_device()}, dtype: torch.float16")
    header = "{:22s} {:>12s} {:>12s} {:>10s}".format("Kernel", "latency", "TFLOPS", "GB/s")
    print(header)
    print("-" * len(header))
    for name in names:
        run, flops, membytes, _ = WORKLOADS[name]()
        ms = _bench(run)
        secs = ms / 1000.0
        print(f"{name:22s} {ms:9.4f}ms {flops / secs / 1e12:11.2f} {membytes / secs / 1e9:10.1f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Block-sparse attention benchmark with per-iteration GPU monitoring")
    parser.add_argument("--report-interval", type=int, default=50, help="Print GPU stats every N iterations (default: 50)")
    parser.add_argument("--quick", action="store_true", help="Fast do_bench timing (no GPU monitoring)")
    parser.add_argument(
        "--kernel",
        action="append",
        choices=sorted(WORKLOADS),
        help="Benchmark only the given kernel (repeatable; default: all)",
    )
    args = parser.parse_args()

    if args.quick:
        quick(args.kernel)
    else:
        main(args.report_interval, args.kernel)
