"""Batch Normalization kernels (training forward, inference forward, backward).

Reference: Ioffe & Szegedy (2015) https://arxiv.org/abs/1502.03167

Input layout expected by all kernels: (C, L) where C is the channel count and
L = N * H * W * ... is the product of batch and spatial dimensions.  The op
layer is responsible for reshaping the user-facing tensor to this layout.

Performance notes:
  - Persistent path (block_l >= L_padded): loads all L_padded elements into a
    register fragment once, reduces, and normalizes from the fragment — single
    global read, eliminates the second pass. Used for non-power-of-2 L, whose
    power-of-2 padding keeps the per-thread element count vectorizable.
  - Non-persistent path (block_l < L_padded): two-pass direct global access,
    but block_l ~512 gives many blocks per channel, hiding the HBM read latency
    better than one persistent block. Used for power-of-2 L.
"""

import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

__all__ = [
    "BatchNormBwdKernel",
    "BatchNormBwdKernelNCHW",
    "BatchNormBwdSplitKernel",
    "BatchNormBwdSplitKernelNCHW",
    "BatchNormFwdInferKernel",
    "BatchNormFwdInferKernelNCHW",
    "BatchNormFwdTrainKernel",
    "BatchNormFwdTrainKernelNCHW",
]

# Config helpers

# L threshold for the persistent (single global read) training path.
# x_shared uses L * sizeof(dtype) bytes per block:
#   L=8192, fp16 → 16 KB — well within H100 shared memory limits.
_PERSISTENT_THRESHOLD = 8192


def _align_up_pow2(n: int, min_align: int = 256) -> int:
    """Round ``n`` up to the next power of 2 (at least ``min_align``).

    A power-of-2 padded row keeps the per-thread element count a power of two,
    so the fragment stays 128-bit vectorizable and layout inference does not
    spill to local memory (the non-power-of-two collapse).
    """
    p = min_align
    while p < n:
        p <<= 1
    return p


def _find_best_threads(L: int) -> int:
    """Largest power-of-2 t in [256, 128, 64, 32] that evenly divides L.

    TileLang's AllReduce template requires a power-of-2 thread count.
    """
    for t in [256, 128, 64, 32]:
        if L % t == 0:
            return t
    return 32  # fallback


def _find_best_block_l(L: int) -> dict:
    """Find best non-persistent block_l config for given L.

    Uses power-of-2 thread counts only (required by TileLang's AllReduce).
    Block_l can be any multiple of `threads` that divides L — including
    non-power-of-2 values such as 448 for L=3136 — giving more tiles per
    channel and better GPU utilization than the strict power-of-2 search.
    block_l is capped at 512 to limit register pressure.
    """
    for threads in [256, 128, 64, 32]:
        for k in range(512 // threads, 0, -1):
            bl = threads * k
            if bl >= L:
                continue
            if L % bl == 0:
                return {"block_l": bl, "num_stages": 0, "threads": threads}
    # Fallback (should rarely be reached).
    for bl in [512, 256, 128, 64, 32, 16]:
        if L % bl == 0:
            return {"block_l": bl, "num_stages": 0, "threads": min(256, bl)}
    raise ValueError(
        f"L={L} is not divisible by any supported block_l. "
        "L must be divisible by at least 16 for the current kernel implementation."
    )


# Training forward

@functools.lru_cache(maxsize=32)
def _batch_norm_fwd_train_kernel(
    C: int,
    L: int,
    dtype: str = "float16",
    eps: float = 1e-5,
    momentum: float = 0.1,
) -> Callable:
    """Return the JIT-compiled training-forward kernel factory.

    Kernel computes, per channel:
      1. mean   = sum(x) / L
      2. var    = sum((x - mean)^2) / L
      3. rstd   = 1 / sqrt(var + eps)
      4. y      = weight * (x - mean) * rstd + bias
      5. running_mean/var updated with *momentum*.

    Saved mean and rstd are needed by the backward pass.

    Persistent path (block_l >= L): loads all L elements into a register
    fragment once, reduces, and normalizes from the fragment — single global
    read, and the trailing 128-bit y stores hide the read latency (TANG
    read-latency exposure).  mean/rstd are written via a (C, 2) loop instead of
    scalar stores to avoid the 32-bit store read-modify-write penalty.

    Non-persistent path (block_l < L): two global reads (classic two-pass BN).

    Requirements: L must be divisible by block_l; threads must divide block_l.
    """
    accum_dtype = "float32"
    L_padded = _align_up_pow2(L)
    pad_count = L_padded - L

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _bn_fwd_train_func(block_l: int, threads: int) -> Callable:

        @T.prim_func
        def _bn_fwd_train(
            x: T.Tensor([C, L], dtype),
            weight: T.Tensor([C], accum_dtype),
            bias: T.Tensor([C], accum_dtype),
            running_mean: T.Tensor([C], accum_dtype),
            running_var: T.Tensor([C], accum_dtype),
            mean_rstd: T.Tensor([C, 2], accum_dtype),
            y: T.Tensor([C, L], dtype),
        ):
            with T.Kernel(C, threads=threads) as (bc):
                if block_l >= L_padded:
                    # Persistent path: masked-load the power-of-2 padded row into
                    # a register fragment, reduce, and normalize from the
                    # fragment.  The trailing 128-bit y stores overlap the read
                    # latency; masked load avoids the F.pad memcpy.
                    x_local = T.alloc_fragment([1, L_padded], dtype)
                    x_f32 = T.alloc_fragment([1, L_padded], accum_dtype)
                    for _i, j in T.Parallel(1, L_padded):
                        x_local[_i, j] = T.if_then_else(
                            j < L, x[bc, j], T.cast(0.0, dtype))
                    for _i, j in T.Parallel(1, L_padded):
                        x_f32[_i, j] = T.cast(x_local[_i, j], accum_dtype)

                    sum_result = T.alloc_fragment([1], accum_dtype)
                    T.reduce_sum(x_f32, sum_result, dim=1)
                    mean_val = sum_result[0] / T.cast(L, accum_dtype)

                    for _i, j in T.Parallel(1, L_padded):
                        x_f32[_i, j] = (x_f32[_i, j] - mean_val) * (x_f32[_i, j] - mean_val)
                    sq_result = T.alloc_fragment([1], accum_dtype)
                    T.reduce_sum(x_f32, sq_result, dim=1)
                    var_val = (sq_result[0] - T.cast(pad_count, accum_dtype) * mean_val * mean_val) / T.cast(L, accum_dtype)
                    rstd_val = T.cast(1.0, accum_dtype) / T.sqrt(
                        var_val + T.cast(eps, accum_dtype))

                    # Loop-write mean/rstd to avoid 32-bit scalar-store
                    # read-modify-write (all threads writing one address).
                    for k in T.Parallel(2):
                        mean_rstd[bc, k] = T.if_then_else(k == 0, mean_val, rstd_val)

                    # Update running statistics (single writer).
                    mom = T.cast(momentum, accum_dtype)
                    unbiased_var = var_val * T.cast(L, accum_dtype) / (
                        T.cast(L, accum_dtype) - T.cast(1.0, accum_dtype))
                    if T.get_thread_binding() == 0:
                        running_mean[bc] = (T.cast(1.0, accum_dtype) - mom) * running_mean[bc] + mom * mean_val
                        running_var[bc] = (T.cast(1.0, accum_dtype) - mom) * running_var[bc] + mom * unbiased_var

                    # Normalize from the fragment (masked store).
                    w = weight[bc]
                    b = bias[bc]
                    for _i, j in T.Parallel(1, L_padded):
                        if j < L:
                            y[bc, j] = T.cast(
                                (T.cast(x_local[_i, j], accum_dtype) - mean_val) * rstd_val * w + b, dtype)
                else:
                    # Non-persistent path: classic two-pass with direct global
                    # access (T.copy inside T.Pipelined races on async copy).
                    xsum_frag = T.alloc_fragment([1, block_l], accum_dtype)
                    xsq_frag = T.alloc_fragment([1, block_l], accum_dtype)
                    T.clear(xsum_frag)
                    T.clear(xsq_frag)

                    for l_tile in T.Pipelined(L // block_l, num_stages=0):
                        for _i, j in T.Parallel(1, block_l):
                            xval = T.cast(x[bc, l_tile * block_l + j], accum_dtype)
                            xsum_frag[_i, j] += xval
                            xsq_frag[_i, j] += xval * xval

                    sum_result = T.alloc_fragment([1], accum_dtype)
                    sq_result = T.alloc_fragment([1], accum_dtype)
                    T.reduce_sum(xsum_frag, sum_result, dim=1)
                    T.reduce_sum(xsq_frag, sq_result, dim=1)

                    mean_val = sum_result[0] / T.cast(L, accum_dtype)
                    var_val = sq_result[0] / T.cast(L, accum_dtype) - mean_val * mean_val
                    rstd_val = T.cast(1.0, accum_dtype) / T.sqrt(
                        var_val + T.cast(eps, accum_dtype))

                    for k in T.Parallel(2):
                        mean_rstd[bc, k] = T.if_then_else(k == 0, mean_val, rstd_val)

                    mom = T.cast(momentum, accum_dtype)
                    unbiased_var = var_val * T.cast(L, accum_dtype) / (
                        T.cast(L, accum_dtype) - T.cast(1.0, accum_dtype))
                    if T.get_thread_binding() == 0:
                        running_mean[bc] = (T.cast(1.0, accum_dtype) - mom) * running_mean[bc] + mom * mean_val
                        running_var[bc] = (T.cast(1.0, accum_dtype) - mom) * running_var[bc] + mom * unbiased_var

                    for l_tile in T.Pipelined(L // block_l, num_stages=0):
                        for _i, j in T.Parallel(1, block_l):
                            xval = T.cast(x[bc, l_tile * block_l + j], accum_dtype)
                            y[bc, l_tile * block_l + j] = T.cast(
                                weight[bc] * (xval - mean_val) * rstd_val + bias[bc], dtype)

        return _bn_fwd_train

    return _bn_fwd_train_func


class BatchNormFwdTrainKernel(Kernel):
    """Training-mode batch normalization forward kernel.

    Args:
        C: Number of channels.
        L: Total reduction length = N * H * W * ... (must be divisible by block_l).
        dtype: Input/output data type.
        eps: Numerical stability constant.
        momentum: Running-stat update momentum.
        config: Optional tile config dict.
        tune: If True, autotune tile config.
    """
    supported_archs: list[int] = [80, 89, 90]

    def __init__(
        self,
        C: int,
        L: int,
        dtype: torch.dtype = torch.float16,
        eps: float = 1e-5,
        momentum: float = 0.1,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.C = C
        self.L = L
        self.L_padded = _align_up_pow2(L)
        self.dtype = dtype
        self.eps = eps
        self.momentum = momentum
        self.kernel = _batch_norm_fwd_train_kernel(C, L, self.dtype_str, eps, momentum)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        if self.L_padded == self.L:
            # Power-of-2 L: a non-persistent ~512 block_l yields many blocks per
            # channel, whose extra tiles hide the HBM read latency (the single
            # persistent block idles at low occupancy and drops IPC).
            cfg = _find_best_block_l(self.L)
            return {"block_l": cfg["block_l"], "threads": cfg["threads"]}
        # Non-power-of-2 L: the persistent path's power-of-2 padding keeps the
        # per-thread element count clean (a non-persistent divisor would spill).
        t = _find_best_threads(self.L_padded)
        return {"block_l": self.L_padded, "threads": t}

    @property
    def autotune_configs(self) -> list[dict]:
        seen: set = set()
        configs = []

        def _add(cfg: dict) -> None:
            key = (cfg["block_l"], cfg["threads"])
            if key not in seen:
                seen.add(key)
                configs.append(cfg)

        # Persistent config (block_l = L_padded); power-of-2 threads only.
        for t in [256, 128, 64, 32]:
            if self.L_padded % t == 0:
                _add({"block_l": self.L_padded, "threads": t})

        # Non-persistent configs: block_l divides the real L (not L_padded) —
        # the kernel's tile loop is `L // block_l`, and num_stages=0 disables
        # T.Pipelined's async prefetch in the multi-tile loop.
        for threads in [256, 128, 64, 32]:
            for k in range(512 // threads, 0, -1):
                bl = threads * k
                if bl >= self.L or self.L % bl != 0:
                    continue
                _add({"block_l": bl, "threads": threads})

        return configs if configs else [self.default_config]

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
    ):
        """Run training forward pass.

        Returns:
            y: Normalized output tensor.
            mean_out: Per-channel batch mean (saved for backward).
            rstd_out: Per-channel reciprocal std (saved for backward).
        """
        mean_rstd = torch.empty((self.C, 2), device=x.device, dtype=torch.float32)
        y = self.kernel(
            self.config["block_l"],
            self.config["threads"],
        )(x, weight, bias, running_mean, running_var, mean_rstd)
        return y, mean_rstd[:, 0], mean_rstd[:, 1]


# Training forward (natural NCHW layout)

@functools.lru_cache(maxsize=32)
def _batch_norm_fwd_train_kernel_nchw(
    N: int,
    C: int,
    S: int,
    dtype: str = "float16",
    eps: float = 1e-5,
    momentum: float = 0.1,
) -> Callable:
    """Strided variant of :func:`_batch_norm_fwd_train_kernel`.

    Identical math and tile structure, but reads/writes the natural contiguous
    ``(N, C, S)`` layout directly (``S = prod(spatial)``), so the op layer can
    feed the user tensor through a zero-copy ``view`` instead of a
    ``permute + contiguous`` transpose.  A channel ``bc``'s ``L = N * S``
    elements are strided: ``N`` contiguous blocks of ``S`` elements, each at
    ``x[n, bc, s]``.  The fragment layout and reduction are unchanged from the
    ``(C, L)`` kernel; only the global index mapping ``j -> (j // S, j % S)``
    differs (``//`` and ``%`` by a compile-time constant S lower to
    shift/reciprocal-multiply, not a full division).
    """
    accum_dtype = "float32"
    L = N * S
    L_padded = _align_up_pow2(L)
    pad_count = L_padded - L

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _bn_fwd_train_nchw_func(block_l: int, threads: int) -> Callable:

        @T.prim_func
        def _bn_fwd_train_nchw(
            x: T.Tensor([N, C, S], dtype),
            weight: T.Tensor([C], accum_dtype),
            bias: T.Tensor([C], accum_dtype),
            running_mean: T.Tensor([C], accum_dtype),
            running_var: T.Tensor([C], accum_dtype),
            mean_rstd: T.Tensor([C, 2], accum_dtype),
            y: T.Tensor([N, C, S], dtype),
        ):
            with T.Kernel(C, threads=threads) as (bc):
                if block_l >= L_padded:
                    # Persistent path: masked strided load into a register
                    # fragment, reduce, and normalize from the fragment.  The
                    # ``T.min`` clamps the batch index so the (discarded)
                    # padding read stays in bounds.
                    x_local = T.alloc_fragment([1, L_padded], dtype)
                    x_f32 = T.alloc_fragment([1, L_padded], accum_dtype)
                    for _i, j in T.Parallel(1, L_padded):
                        x_local[_i, j] = T.if_then_else(
                            j < L,
                            x[T.min(j // S, N - 1), bc, j % S],
                            T.cast(0.0, dtype))
                    for _i, j in T.Parallel(1, L_padded):
                        x_f32[_i, j] = T.cast(x_local[_i, j], accum_dtype)

                    sum_result = T.alloc_fragment([1], accum_dtype)
                    T.reduce_sum(x_f32, sum_result, dim=1)
                    mean_val = sum_result[0] / T.cast(L, accum_dtype)

                    for _i, j in T.Parallel(1, L_padded):
                        x_f32[_i, j] = (x_f32[_i, j] - mean_val) * (x_f32[_i, j] - mean_val)
                    sq_result = T.alloc_fragment([1], accum_dtype)
                    T.reduce_sum(x_f32, sq_result, dim=1)
                    var_val = (sq_result[0] - T.cast(pad_count, accum_dtype) * mean_val * mean_val) / T.cast(L, accum_dtype)
                    rstd_val = T.cast(1.0, accum_dtype) / T.sqrt(
                        var_val + T.cast(eps, accum_dtype))

                    # Loop-write mean/rstd to avoid 32-bit scalar-store
                    # read-modify-write (all threads writing one address).
                    for k in T.Parallel(2):
                        mean_rstd[bc, k] = T.if_then_else(k == 0, mean_val, rstd_val)

                    # Update running statistics (single writer).
                    mom = T.cast(momentum, accum_dtype)
                    unbiased_var = var_val * T.cast(L, accum_dtype) / (
                        T.cast(L, accum_dtype) - T.cast(1.0, accum_dtype))
                    if T.get_thread_binding() == 0:
                        running_mean[bc] = (T.cast(1.0, accum_dtype) - mom) * running_mean[bc] + mom * mean_val
                        running_var[bc] = (T.cast(1.0, accum_dtype) - mom) * running_var[bc] + mom * unbiased_var

                    # Normalize from the fragment (masked strided store).
                    w = weight[bc]
                    b = bias[bc]
                    for _i, j in T.Parallel(1, L_padded):
                        if j < L:
                            y[j // S, bc, j % S] = T.cast(
                                (T.cast(x_local[_i, j], accum_dtype) - mean_val) * rstd_val * w + b, dtype)
                else:
                    # Non-persistent path: classic two-pass with direct strided
                    # global access (T.copy inside T.Pipelined races on async copy).
                    xsum_frag = T.alloc_fragment([1, block_l], accum_dtype)
                    xsq_frag = T.alloc_fragment([1, block_l], accum_dtype)
                    T.clear(xsum_frag)
                    T.clear(xsq_frag)

                    for l_tile in T.Pipelined(L // block_l, num_stages=0):
                        for _i, j in T.Parallel(1, block_l):
                            g = l_tile * block_l + j
                            xval = T.cast(x[g // S, bc, g % S], accum_dtype)
                            xsum_frag[_i, j] += xval
                            xsq_frag[_i, j] += xval * xval

                    sum_result = T.alloc_fragment([1], accum_dtype)
                    sq_result = T.alloc_fragment([1], accum_dtype)
                    T.reduce_sum(xsum_frag, sum_result, dim=1)
                    T.reduce_sum(xsq_frag, sq_result, dim=1)

                    mean_val = sum_result[0] / T.cast(L, accum_dtype)
                    var_val = sq_result[0] / T.cast(L, accum_dtype) - mean_val * mean_val
                    rstd_val = T.cast(1.0, accum_dtype) / T.sqrt(
                        var_val + T.cast(eps, accum_dtype))

                    for k in T.Parallel(2):
                        mean_rstd[bc, k] = T.if_then_else(k == 0, mean_val, rstd_val)

                    mom = T.cast(momentum, accum_dtype)
                    unbiased_var = var_val * T.cast(L, accum_dtype) / (
                        T.cast(L, accum_dtype) - T.cast(1.0, accum_dtype))
                    if T.get_thread_binding() == 0:
                        running_mean[bc] = (T.cast(1.0, accum_dtype) - mom) * running_mean[bc] + mom * mean_val
                        running_var[bc] = (T.cast(1.0, accum_dtype) - mom) * running_var[bc] + mom * unbiased_var

                    for l_tile in T.Pipelined(L // block_l, num_stages=0):
                        for _i, j in T.Parallel(1, block_l):
                            g = l_tile * block_l + j
                            xval = T.cast(x[g // S, bc, g % S], accum_dtype)
                            y[g // S, bc, g % S] = T.cast(
                                weight[bc] * (xval - mean_val) * rstd_val + bias[bc], dtype)

        return _bn_fwd_train_nchw

    return _bn_fwd_train_nchw_func


class BatchNormFwdTrainKernelNCHW(BatchNormFwdTrainKernel):
    """Training forward kernel over the natural ``(N, C, S)`` contiguous layout.

    Reuses the ``(C, L)`` config search (it depends only on ``L = N * S`` and
    ``L_padded``) but compiles a strided kernel that reads/writes ``x[n, bc, s]``
    directly, so the op layer avoids the ``permute + contiguous`` transpose.

    Args:
        N: Batch size.
        C: Number of channels.
        S: Spatial extent per channel = ``prod(spatial dims)``; ``L = N * S``.
        dtype: Input/output data type.
        eps: Numerical stability constant.
        momentum: Running-stat update momentum.
        config: Optional tile config dict.
        tune: If True, autotune tile config.
    """

    def __init__(
        self,
        N: int,
        C: int,
        S: int,
        dtype: torch.dtype = torch.float16,
        eps: float = 1e-5,
        momentum: float = 0.1,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        Kernel.__init__(self)
        self.N = N
        self.C = C
        self.S = S
        self.L = N * S
        self.L_padded = _align_up_pow2(self.L)
        self.dtype = dtype
        self.eps = eps
        self.momentum = momentum
        self.kernel = _batch_norm_fwd_train_kernel_nchw(
            N, C, S, self.dtype_str, eps, momentum)
        self.init_config(config, tune)

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
    ):
        """Run training forward pass on ``x`` of shape ``(N, C, S)``.

        Returns:
            y: Normalized output tensor of shape ``(N, C, S)``.
            mean_out: Per-channel batch mean (saved for backward).
            rstd_out: Per-channel reciprocal std (saved for backward).
        """
        mean_rstd = torch.empty((self.C, 2), device=x.device, dtype=torch.float32)
        y = self.kernel(
            self.config["block_l"],
            self.config["threads"],
        )(x, weight, bias, running_mean, running_var, mean_rstd)
        return y, mean_rstd[:, 0], mean_rstd[:, 1]


# Inference forward

@functools.lru_cache(maxsize=32)
def _batch_norm_fwd_infer_kernel(
    C: int,
    L: int,
    dtype: str = "float16",
    eps: float = 1e-5,
) -> Callable:
    """Return the JIT-compiled inference-forward kernel factory.

    Single pass: y = weight * (x - running_mean) / sqrt(running_var + eps) + bias.
    Fused into a pre-computed scale/shift per channel to minimize arithmetic.
    """
    accum_dtype = "float32"

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _bn_fwd_infer_func(block_l: int, num_stages: int, threads: int) -> Callable:

        @T.prim_func
        def _bn_fwd_infer(
            x: T.Tensor([C, L], dtype),
            weight: T.Tensor([C], accum_dtype),
            bias: T.Tensor([C], accum_dtype),
            running_mean: T.Tensor([C], accum_dtype),
            running_var: T.Tensor([C], accum_dtype),
            y: T.Tensor([C, L], dtype),
        ):
            with T.Kernel(C, threads=threads) as (bc):
                # Fused scale/shift: avoids recomputing per element.
                scale = weight[bc] / T.sqrt(
                    running_var[bc] + T.cast(eps, accum_dtype))
                shift = bias[bc] - running_mean[bc] * scale

                # Non-persistent: direct global memory access avoids async-copy
                # data race that occurs when T.copy is used inside T.Pipelined.
                for l_tile in T.Pipelined(L // block_l, num_stages=0):
                    for _i, j in T.Parallel(1, block_l):
                        y[bc, l_tile * block_l + j] = T.cast(
                            T.cast(x[bc, l_tile * block_l + j], accum_dtype) * scale + shift, dtype)

        return _bn_fwd_infer

    return _bn_fwd_infer_func


class BatchNormFwdInferKernel(Kernel):
    """Inference-mode batch normalization forward kernel.

    Args:
        C: Number of channels.
        L: Total reduction length = N * H * W * ... (must be divisible by block_l).
        dtype: Input/output data type.
        eps: Numerical stability constant.
        config: Optional tile config dict.
        tune: If True, autotune tile config.
    """
    supported_archs: list[int] = [80, 89, 90]

    def __init__(
        self,
        C: int,
        L: int,
        dtype: torch.dtype = torch.float16,
        eps: float = 1e-5,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.C = C
        self.L = L
        self.dtype = dtype
        self.eps = eps
        self.kernel = _batch_norm_fwd_infer_kernel(C, L, self.dtype_str, eps)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return _find_best_block_l(self.L)

    @property
    def autotune_configs(self) -> list[dict]:
        seen: set = set()
        configs = []

        def _add(cfg: dict) -> None:
            key = (cfg["block_l"], cfg["num_stages"], cfg["threads"])
            if key not in seen:
                seen.add(key)
                configs.append(cfg)

        for threads in [256, 128, 64, 32]:
            for k in range(512 // threads, 0, -1):
                bl = threads * k
                if self.L % bl != 0:
                    continue
                _add({"block_l": bl, "num_stages": 0, "threads": threads})

        return configs if configs else [self.default_config]

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
    ) -> torch.Tensor:
        return self.kernel(
            self.config["block_l"],
            self.config["num_stages"],
            self.config["threads"],
        )(x, weight, bias, running_mean, running_var)


# Inference forward (natural NCHW layout)

@functools.lru_cache(maxsize=32)
def _batch_norm_fwd_infer_kernel_nchw(
    N: int,
    C: int,
    S: int,
    dtype: str = "float16",
    eps: float = 1e-5,
) -> Callable:
    """Strided variant of :func:`_batch_norm_fwd_infer_kernel` over ``(N, C, S)``.

    Same fused scale/shift, reading/writing ``x[n, bc, s]`` directly so the op
    layer can skip the ``permute + contiguous`` transpose.
    """
    accum_dtype = "float32"
    L = N * S

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _bn_fwd_infer_nchw_func(block_l: int, num_stages: int, threads: int) -> Callable:

        @T.prim_func
        def _bn_fwd_infer_nchw(
            x: T.Tensor([N, C, S], dtype),
            weight: T.Tensor([C], accum_dtype),
            bias: T.Tensor([C], accum_dtype),
            running_mean: T.Tensor([C], accum_dtype),
            running_var: T.Tensor([C], accum_dtype),
            y: T.Tensor([N, C, S], dtype),
        ):
            with T.Kernel(C, threads=threads) as (bc):
                scale = weight[bc] / T.sqrt(
                    running_var[bc] + T.cast(eps, accum_dtype))
                shift = bias[bc] - running_mean[bc] * scale

                for l_tile in T.Pipelined(L // block_l, num_stages=0):
                    for _i, j in T.Parallel(1, block_l):
                        g = l_tile * block_l + j
                        y[g // S, bc, g % S] = T.cast(
                            T.cast(x[g // S, bc, g % S], accum_dtype) * scale + shift, dtype)

        return _bn_fwd_infer_nchw

    return _bn_fwd_infer_nchw_func


class BatchNormFwdInferKernelNCHW(BatchNormFwdInferKernel):
    """Inference forward kernel over the natural ``(N, C, S)`` contiguous layout.

    Reuses the ``(C, L)`` config search (``L = N * S``) but compiles a strided
    kernel that reads/writes ``x[n, bc, s]`` directly.
    """

    def __init__(
        self,
        N: int,
        C: int,
        S: int,
        dtype: torch.dtype = torch.float16,
        eps: float = 1e-5,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        Kernel.__init__(self)
        self.N = N
        self.C = C
        self.S = S
        self.L = N * S
        self.dtype = dtype
        self.eps = eps
        self.kernel = _batch_norm_fwd_infer_kernel_nchw(
            N, C, S, self.dtype_str, eps)
        self.init_config(config, tune)

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
    ) -> torch.Tensor:
        return self.kernel(
            self.config["block_l"],
            self.config["num_stages"],
            self.config["threads"],
        )(x, weight, bias, running_mean, running_var)


# Backward

@functools.lru_cache(maxsize=32)
def _batch_norm_bwd_kernel(
    C: int,
    L: int,
    dtype: str = "float16",
) -> Callable:
    """Return the JIT-compiled backward kernel factory.

    Given saved mean and rstd from the training forward pass, computes:
      grad_bias[c]   = sum_i( grad_out[c, i] )
      grad_weight[c] = sum_i( grad_out[c, i] * x_hat[c, i] )
      grad_x[c, i]   = weight[c] * rstd[c] / L
                       * ( L * grad_out[c, i]
                           - grad_bias[c]
                           - x_hat[c, i] * grad_weight[c] )

    where x_hat[c, i] = (x[c, i] - mean[c]) * rstd[c].

    Persistent path (block_l >= L): after pass 1 accumulates grad_bias /
    grad_weight while loading grad_out and x into shared memory, pass 2 computes
    grad_x directly from shared memory — eliminates the second global read.

    Non-persistent path (block_l < L): two global reads (classic two-pass BN bwd).

    Requirements: L must be divisible by block_l.
    """
    accum_dtype = "float32"

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _bn_bwd_func(block_l: int, threads: int) -> Callable:

        @T.prim_func
        def _bn_bwd(
            grad_out: T.Tensor([C, L], dtype),
            x: T.Tensor([C, L], dtype),
            weight: T.Tensor([C], accum_dtype),
            mean: T.Tensor([C], accum_dtype),
            rstd: T.Tensor([C], accum_dtype),
            grad_weight: T.Tensor([C], accum_dtype),
            grad_bias: T.Tensor([C], accum_dtype),
            grad_x: T.Tensor([C, L], dtype),
        ):
            with T.Kernel(C, threads=threads) as (bc):
                go_shared = T.alloc_shared([block_l], dtype)
                x_shared = T.alloc_shared([block_l], dtype)

                mean_val = mean[bc]
                rstd_val = rstd[bc]
                w_val = weight[bc]

                # Accumulators for sum(grad_out) and sum(grad_out * x_hat).
                do_frag = T.alloc_fragment([1, block_l], accum_dtype)
                do_xhat_frag = T.alloc_fragment([1, block_l], accum_dtype)
                T.clear(do_frag)
                T.clear(do_xhat_frag)

                # Pass 1 – accumulate grad_bias and grad_weight contributions.
                if block_l >= L:
                    # Persistent path has exactly one tile, so a pipelined loop
                    # cannot overlap producer/consumer work.
                    T.copy(grad_out[bc, 0:block_l], go_shared)
                    T.copy(x[bc, 0:block_l], x_shared)
                    for _i, j in T.Parallel(1, block_l):
                        go_val = T.cast(go_shared[j], accum_dtype)
                        x_hat = (T.cast(x_shared[j], accum_dtype) - mean_val) * rstd_val
                        do_frag[_i, j] += go_val
                        do_xhat_frag[_i, j] += go_val * x_hat
                else:
                    # Non-persistent path: direct global memory access avoids async-copy
                    # data race that occurs when T.copy is used inside T.Pipelined.
                    for l_tile in T.Pipelined(L // block_l, num_stages=0):
                        for _i, j in T.Parallel(1, block_l):
                            go_val = T.cast(grad_out[bc, l_tile * block_l + j], accum_dtype)
                            x_hat = (T.cast(x[bc, l_tile * block_l + j], accum_dtype) - mean_val) * rstd_val
                            do_frag[_i, j] += go_val
                            do_xhat_frag[_i, j] += go_val * x_hat

                # Cross-thread reduction.
                sum_do = T.alloc_fragment([1], accum_dtype)
                sum_do_xhat = T.alloc_fragment([1], accum_dtype)
                T.reduce_sum(do_frag, sum_do, dim=1)
                T.reduce_sum(do_xhat_frag, sum_do_xhat, dim=1)

                # Write grad_bias and grad_weight.
                grad_bias[bc] = sum_do[0]
                grad_weight[bc] = sum_do_xhat[0]

                # Precompute per-channel constant.
                w_rstd_over_L = w_val * rstd_val / T.cast(L, accum_dtype)

                # Pass 2 – compute grad_x.
                if block_l >= L:
                    # Persistent path: go_shared and x_shared hold all L elements.
                    # No second global read needed.
                    for _i, j in T.Parallel(1, block_l):
                        go_val = T.cast(go_shared[j], accum_dtype)
                        x_hat = (T.cast(x_shared[j], accum_dtype) - mean_val) * rstd_val
                        gx = w_rstd_over_L * (
                            T.cast(L, accum_dtype) * go_val
                            - sum_do[0]
                            - x_hat * sum_do_xhat[0]
                        )
                        grad_x[bc, j] = T.cast(gx, dtype)
                else:
                    # Non-persistent path: direct global memory access avoids async-copy
                    # data race that occurs when T.copy is used inside T.Pipelined.
                    for l_tile in T.Pipelined(L // block_l, num_stages=0):
                        for _i, j in T.Parallel(1, block_l):
                            go_val = T.cast(grad_out[bc, l_tile * block_l + j], accum_dtype)
                            x_hat = (T.cast(x[bc, l_tile * block_l + j], accum_dtype) - mean_val) * rstd_val
                            gx = w_rstd_over_L * (
                                T.cast(L, accum_dtype) * go_val
                                - sum_do[0]
                                - x_hat * sum_do_xhat[0]
                            )
                            grad_x[bc, l_tile * block_l + j] = T.cast(gx, dtype)

        return _bn_bwd

    return _bn_bwd_func


class BatchNormBwdKernel(Kernel):
    """Batch normalization backward kernel.

    Args:
        C: Number of channels.
        L: Total reduction length = N * H * W * ... (must be divisible by block_l).
        dtype: grad_out/x/grad_x data type.
        config: Optional tile config dict.
        tune: If True, autotune tile config.
    """
    supported_archs: list[int] = [80, 89, 90]

    def __init__(
        self,
        C: int,
        L: int,
        dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.C = C
        self.L = L
        self.dtype = dtype
        self.kernel = _batch_norm_bwd_kernel(C, L, self.dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        if self.L <= _PERSISTENT_THRESHOLD:
            # Persistent path: block_l = L, single global read.
            # go_shared and x_shared together use 2 * L * sizeof(dtype) SMEM.
            t = _find_best_threads(self.L)
            return {"block_l": self.L, "threads": t}
        cfg = _find_best_block_l(self.L)
        return {"block_l": cfg["block_l"], "threads": cfg["threads"]}

    @property
    def autotune_configs(self) -> list[dict]:
        seen: set = set()
        configs = []

        def _add(cfg: dict) -> None:
            key = (cfg["block_l"], cfg["threads"])
            if key not in seen:
                seen.add(key)
                configs.append(cfg)

        # Persistent configs (block_l = L); power-of-2 threads only.
        if self.L <= _PERSISTENT_THRESHOLD:
            for t in [256, 128, 64, 32]:
                if self.L % t == 0:
                    _add({"block_l": self.L, "threads": t})

        # Non-persistent configs: power-of-2 threads, block_l can be non-power-of-2.
        # num_stages=0 disables pipelining for correctness in multi-tile loops.
        for threads in [256, 128, 64, 32]:
            for k in range(512 // threads, 0, -1):
                bl = threads * k
                if bl >= self.L or self.L % bl != 0:
                    continue
                _add({"block_l": bl, "threads": threads})

        return configs if configs else [self.default_config]

    def forward(
        self,
        grad_out: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        mean: torch.Tensor,
        rstd: torch.Tensor,
    ):
        """Run backward pass.

        Returns:
            grad_x: Gradient w.r.t. input.
            grad_weight: Gradient w.r.t. affine scale (gamma).
            grad_bias: Gradient w.r.t. affine shift (beta).
        """
        grad_weight = torch.empty(self.C, device=grad_out.device, dtype=torch.float32)
        grad_bias = torch.empty(self.C, device=grad_out.device, dtype=torch.float32)
        grad_x = self.kernel(
            self.config["block_l"],
            self.config["threads"],
        )(grad_out, x, weight, mean, rstd, grad_weight, grad_bias)
        return grad_x, grad_weight, grad_bias


# Multi-block split backward: two kernels with atomic cross-block reduction.
#
# The persistent single-block-per-channel path launches only C blocks (33%
# occupancy for C=128) and stages go/x through shared memory behind a barrier,
# so the HBM read latency is fully exposed.  Splitting each channel into
# `split` blocks raises occupancy and reads go/x directly (no barrier): the
# first kernel reduces each chunk and atomically adds the partial sums into
# grad_bias/grad_weight; the second reads the finalized sums and computes
# grad_x.  Two sequential launches make the atomics visible without a grid sync.

@functools.lru_cache(maxsize=32)
def _batch_norm_bwd_split_reduce_kernel(
    C: int,
    L: int,
    L_padded: int,
    split: int,
    dtype: str = "float16",
) -> Callable:
    """Reduce kernel: partial sums of grad_out and grad_out*x_hat per chunk.

    ``L_padded`` (>= L) is the power-of-two padded length; the padded tail is
    masked so it contributes nothing to either reduction (grad_out padded with
    0, x padded with mean so its x_hat is 0).
    """
    accum_dtype = "float32"
    chunk_len = L_padded // split

    @tilelang.jit(compile_flags=["-O3", "-DENABLE_BF16"])
    def _reduce_func() -> Callable:

        @T.prim_func
        def _bn_bwd_split_reduce(
            grad_out: T.Tensor([C, L], dtype),
            x: T.Tensor([C, L], dtype),
            mean: T.Tensor([C], accum_dtype),
            rstd: T.Tensor([C], accum_dtype),
            grad_weight: T.Tensor([C], accum_dtype),
            grad_bias: T.Tensor([C], accum_dtype),
        ):
            with T.Kernel(C * split, threads=256) as bid:
                bc = bid // split
                chunk_id = bid % split
                do_frag = T.alloc_fragment([1, chunk_len], accum_dtype)
                do_xhat_frag = T.alloc_fragment([1, chunk_len], accum_dtype)
                T.clear(do_frag)
                T.clear(do_xhat_frag)
                for _i, j in T.Parallel(1, chunk_len):
                    idx = chunk_id * chunk_len + j
                    go_val = T.cast(T.if_then_else(
                        idx < L, grad_out[bc, idx], T.cast(0.0, dtype)), accum_dtype)
                    x_hat = (T.cast(T.if_then_else(
                        idx < L, x[bc, idx], mean[bc]), accum_dtype) - mean[bc]) * rstd[bc]
                    do_frag[_i, j] += go_val
                    do_xhat_frag[_i, j] += go_val * x_hat
                sum_do = T.alloc_fragment([1], accum_dtype)
                sum_do_xhat = T.alloc_fragment([1], accum_dtype)
                T.reduce_sum(do_frag, sum_do, dim=1)
                T.reduce_sum(do_xhat_frag, sum_do_xhat, dim=1)
                if T.get_thread_binding() == 0:
                    T.atomic_add(grad_bias[bc], sum_do[0])
                    T.atomic_add(grad_weight[bc], sum_do_xhat[0])

        return _bn_bwd_split_reduce

    return _reduce_func()


@functools.lru_cache(maxsize=32)
def _batch_norm_bwd_split_gradx_kernel(
    C: int,
    L: int,
    L_padded: int,
    split: int,
    dtype: str = "float16",
) -> Callable:
    """grad_x kernel: read finalized grad_bias/grad_weight, compute grad_x."""
    accum_dtype = "float32"
    chunk_len = L_padded // split

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _gradx_func() -> Callable:

        @T.prim_func
        def _bn_bwd_split_gradx(
            grad_out: T.Tensor([C, L], dtype),
            x: T.Tensor([C, L], dtype),
            weight: T.Tensor([C], accum_dtype),
            mean: T.Tensor([C], accum_dtype),
            rstd: T.Tensor([C], accum_dtype),
            grad_weight: T.Tensor([C], accum_dtype),
            grad_bias: T.Tensor([C], accum_dtype),
            grad_x: T.Tensor([C, L], dtype),
        ):
            with T.Kernel(C * split, threads=256) as bid:
                bc = bid // split
                chunk_id = bid % split
                w_rstd_over_L = weight[bc] * rstd[bc] / T.cast(L, accum_dtype)
                for _i, j in T.Parallel(1, chunk_len):
                    idx = chunk_id * chunk_len + j
                    go_val = T.cast(T.if_then_else(
                        idx < L, grad_out[bc, idx], T.cast(0.0, dtype)), accum_dtype)
                    x_hat = (T.cast(T.if_then_else(
                        idx < L, x[bc, idx], mean[bc]), accum_dtype) - mean[bc]) * rstd[bc]
                    gx = w_rstd_over_L * (
                        T.cast(L, accum_dtype) * go_val
                        - grad_bias[bc]
                        - x_hat * grad_weight[bc]
                    )
                    if idx < L:
                        grad_x[bc, idx] = T.cast(gx, dtype)

        return _bn_bwd_split_gradx

    return _gradx_func()


class BatchNormBwdSplitKernel(Kernel):
    """Backward kernel split across ``C * split`` blocks.

    Used for ``L <= 8192`` where the persistent single-block path is
    latency-bound.  ``L`` is padded to ``L_padded = align_up_pow2(L)`` and the
    padded tail is masked so ``chunk_len = L_padded // split`` is a power of two
    (each block's ``[1, chunk_len]`` accumulator fragment stays ~8 fp32/thread,
    no local spill), and ``split`` blocks per channel raise the grid occupancy
    enough to hide the HBM read latency.

    Args:
        C: Number of channels.
        L: Total reduction length (padded to a power of two internally).
        dtype: grad_out/x/grad_x data type.
    """

    _CHUNK = 2048

    def __init__(
        self,
        C: int,
        L: int,
        dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.C = C
        self.L = L
        self.dtype = dtype
        self.L_padded = _align_up_pow2(L)
        self.split = max(1, self.L_padded // self._CHUNK)
        self.reduce_kernel = _batch_norm_bwd_split_reduce_kernel(
            C, L, self.L_padded, self.split, self.dtype_str)
        self.gradx_kernel = _batch_norm_bwd_split_gradx_kernel(
            C, L, self.L_padded, self.split, self.dtype_str)

    def forward(
        self,
        grad_out: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        mean: torch.Tensor,
        rstd: torch.Tensor,
    ):
        """Run backward pass.

        Returns:
            grad_x: Gradient w.r.t. input.
            grad_weight: Gradient w.r.t. affine scale (gamma).
            grad_bias: Gradient w.r.t. affine shift (beta).
        """
        grad_weight = torch.zeros(self.C, device=grad_out.device, dtype=torch.float32)
        grad_bias = torch.zeros(self.C, device=grad_out.device, dtype=torch.float32)
        self.reduce_kernel(grad_out, x, mean, rstd, grad_weight, grad_bias)
        grad_x = self.gradx_kernel(
            grad_out, x, weight, mean, rstd, grad_weight, grad_bias)
        return grad_x, grad_weight, grad_bias


# Strided (N, C, S) variants of the split backward kernels — read/write the
# natural contiguous layout directly so the op layer skips the transpose.


@functools.lru_cache(maxsize=32)
def _batch_norm_bwd_split_reduce_kernel_nchw(
    N: int,
    C: int,
    S: int,
    L: int,
    L_padded: int,
    split: int,
    dtype: str = "float16",
) -> Callable:
    """Strided (N, C, S) variant of the split reduce kernel.

    ``L = N * S``; a flat index ``idx`` maps to ``(idx // S, idx % S)``.  The
    ``T.min`` clamps the batch index so the masked padding tail stays in bounds.
    """
    accum_dtype = "float32"
    chunk_len = L_padded // split

    @tilelang.jit(compile_flags=["-O3", "-DENABLE_BF16"])
    def _reduce_func() -> Callable:

        @T.prim_func
        def _bn_bwd_split_reduce_nchw(
            grad_out: T.Tensor([N, C, S], dtype),
            x: T.Tensor([N, C, S], dtype),
            mean: T.Tensor([C], accum_dtype),
            rstd: T.Tensor([C], accum_dtype),
            grad_weight: T.Tensor([C], accum_dtype),
            grad_bias: T.Tensor([C], accum_dtype),
        ):
            with T.Kernel(C * split, threads=256) as bid:
                bc = bid // split
                chunk_id = bid % split
                do_frag = T.alloc_fragment([1, chunk_len], accum_dtype)
                do_xhat_frag = T.alloc_fragment([1, chunk_len], accum_dtype)
                T.clear(do_frag)
                T.clear(do_xhat_frag)
                for _i, j in T.Parallel(1, chunk_len):
                    idx = chunk_id * chunk_len + j
                    n = T.min(idx // S, N - 1)
                    s = idx % S
                    go_val = T.cast(T.if_then_else(
                        idx < L, grad_out[n, bc, s], T.cast(0.0, dtype)), accum_dtype)
                    x_hat = (T.cast(T.if_then_else(
                        idx < L, x[n, bc, s], mean[bc]), accum_dtype) - mean[bc]) * rstd[bc]
                    do_frag[_i, j] += go_val
                    do_xhat_frag[_i, j] += go_val * x_hat
                sum_do = T.alloc_fragment([1], accum_dtype)
                sum_do_xhat = T.alloc_fragment([1], accum_dtype)
                T.reduce_sum(do_frag, sum_do, dim=1)
                T.reduce_sum(do_xhat_frag, sum_do_xhat, dim=1)
                if T.get_thread_binding() == 0:
                    T.atomic_add(grad_bias[bc], sum_do[0])
                    T.atomic_add(grad_weight[bc], sum_do_xhat[0])

        return _bn_bwd_split_reduce_nchw

    return _reduce_func()


@functools.lru_cache(maxsize=32)
def _batch_norm_bwd_split_gradx_kernel_nchw(
    N: int,
    C: int,
    S: int,
    L: int,
    L_padded: int,
    split: int,
    dtype: str = "float16",
) -> Callable:
    """Strided (N, C, S) variant of the split grad_x kernel."""
    accum_dtype = "float32"
    chunk_len = L_padded // split

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _gradx_func() -> Callable:

        @T.prim_func
        def _bn_bwd_split_gradx_nchw(
            grad_out: T.Tensor([N, C, S], dtype),
            x: T.Tensor([N, C, S], dtype),
            weight: T.Tensor([C], accum_dtype),
            mean: T.Tensor([C], accum_dtype),
            rstd: T.Tensor([C], accum_dtype),
            grad_weight: T.Tensor([C], accum_dtype),
            grad_bias: T.Tensor([C], accum_dtype),
            grad_x: T.Tensor([N, C, S], dtype),
        ):
            with T.Kernel(C * split, threads=256) as bid:
                bc = bid // split
                chunk_id = bid % split
                w_rstd_over_L = weight[bc] * rstd[bc] / T.cast(L, accum_dtype)
                for _i, j in T.Parallel(1, chunk_len):
                    idx = chunk_id * chunk_len + j
                    n = T.min(idx // S, N - 1)
                    s = idx % S
                    go_val = T.cast(T.if_then_else(
                        idx < L, grad_out[n, bc, s], T.cast(0.0, dtype)), accum_dtype)
                    x_hat = (T.cast(T.if_then_else(
                        idx < L, x[n, bc, s], mean[bc]), accum_dtype) - mean[bc]) * rstd[bc]
                    gx = w_rstd_over_L * (
                        T.cast(L, accum_dtype) * go_val
                        - grad_bias[bc]
                        - x_hat * grad_weight[bc]
                    )
                    if idx < L:
                        grad_x[n, bc, s] = T.cast(gx, dtype)

        return _bn_bwd_split_gradx_nchw

    return _gradx_func()


class BatchNormBwdSplitKernelNCHW(BatchNormBwdSplitKernel):
    """Split backward kernel over the natural ``(N, C, S)`` contiguous layout.

    Reuses the ``BatchNormBwdSplitKernel`` config logic (``L_padded``/``split``
    depend only on ``L = N * S``) but compiles strided kernels that read/write
    ``grad_out[n, bc, s]`` directly, so the op layer avoids the transpose.
    """

    def __init__(
        self,
        N: int,
        C: int,
        S: int,
        dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        Kernel.__init__(self)
        self.N = N
        self.C = C
        self.S = S
        self.L = N * S
        self.dtype = dtype
        self.L_padded = _align_up_pow2(self.L)
        self.split = max(1, self.L_padded // self._CHUNK)
        self.reduce_kernel = _batch_norm_bwd_split_reduce_kernel_nchw(
            N, C, S, self.L, self.L_padded, self.split, self.dtype_str)
        self.gradx_kernel = _batch_norm_bwd_split_gradx_kernel_nchw(
            N, C, S, self.L, self.L_padded, self.split, self.dtype_str)


# Strided (N, C, S) variant of the single-block backward kernel (used for
# L > 8192, the memory-bound large shapes).

@functools.lru_cache(maxsize=32)
def _batch_norm_bwd_kernel_nchw(
    N: int,
    C: int,
    S: int,
    dtype: str = "float16",
) -> Callable:
    """Strided variant of :func:`_batch_norm_bwd_kernel`.

    Identical math and tile structure, but reads/writes the natural contiguous
    ``(N, C, S)`` layout directly (``S = prod(spatial)``), so the op layer feeds
    the user tensor through a zero-copy ``view`` instead of a transpose.  Only
    the index mapping ``j -> (j // S, j % S)`` differs from the ``(C, L)`` kernel.
    """
    accum_dtype = "float32"
    L = N * S

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _bn_bwd_nchw_func(block_l: int, threads: int) -> Callable:

        @T.prim_func
        def _bn_bwd_nchw(
            grad_out: T.Tensor([N, C, S], dtype),
            x: T.Tensor([N, C, S], dtype),
            weight: T.Tensor([C], accum_dtype),
            mean: T.Tensor([C], accum_dtype),
            rstd: T.Tensor([C], accum_dtype),
            grad_weight: T.Tensor([C], accum_dtype),
            grad_bias: T.Tensor([C], accum_dtype),
            grad_x: T.Tensor([N, C, S], dtype),
        ):
            with T.Kernel(C, threads=threads) as (bc):
                go_shared = T.alloc_shared([block_l], dtype)
                x_shared = T.alloc_shared([block_l], dtype)

                mean_val = mean[bc]
                rstd_val = rstd[bc]
                w_val = weight[bc]

                do_frag = T.alloc_fragment([1, block_l], accum_dtype)
                do_xhat_frag = T.alloc_fragment([1, block_l], accum_dtype)
                T.clear(do_frag)
                T.clear(do_xhat_frag)

                if block_l >= L:
                    # Persistent: strided load of the whole channel into shared.
                    for _i, j in T.Parallel(1, block_l):
                        go_shared[j] = grad_out[j // S, bc, j % S]
                        x_shared[j] = x[j // S, bc, j % S]
                    for _i, j in T.Parallel(1, block_l):
                        go_val = T.cast(go_shared[j], accum_dtype)
                        x_hat = (T.cast(x_shared[j], accum_dtype) - mean_val) * rstd_val
                        do_frag[_i, j] += go_val
                        do_xhat_frag[_i, j] += go_val * x_hat
                else:
                    for l_tile in T.Pipelined(L // block_l, num_stages=0):
                        for _i, j in T.Parallel(1, block_l):
                            g = l_tile * block_l + j
                            go_val = T.cast(grad_out[g // S, bc, g % S], accum_dtype)
                            x_hat = (T.cast(x[g // S, bc, g % S], accum_dtype) - mean_val) * rstd_val
                            do_frag[_i, j] += go_val
                            do_xhat_frag[_i, j] += go_val * x_hat

                sum_do = T.alloc_fragment([1], accum_dtype)
                sum_do_xhat = T.alloc_fragment([1], accum_dtype)
                T.reduce_sum(do_frag, sum_do, dim=1)
                T.reduce_sum(do_xhat_frag, sum_do_xhat, dim=1)

                grad_bias[bc] = sum_do[0]
                grad_weight[bc] = sum_do_xhat[0]

                w_rstd_over_L = w_val * rstd_val / T.cast(L, accum_dtype)

                if block_l >= L:
                    for _i, j in T.Parallel(1, block_l):
                        go_val = T.cast(go_shared[j], accum_dtype)
                        x_hat = (T.cast(x_shared[j], accum_dtype) - mean_val) * rstd_val
                        gx = w_rstd_over_L * (
                            T.cast(L, accum_dtype) * go_val
                            - sum_do[0]
                            - x_hat * sum_do_xhat[0]
                        )
                        grad_x[j // S, bc, j % S] = T.cast(gx, dtype)
                else:
                    for l_tile in T.Pipelined(L // block_l, num_stages=0):
                        for _i, j in T.Parallel(1, block_l):
                            g = l_tile * block_l + j
                            go_val = T.cast(grad_out[g // S, bc, g % S], accum_dtype)
                            x_hat = (T.cast(x[g // S, bc, g % S], accum_dtype) - mean_val) * rstd_val
                            gx = w_rstd_over_L * (
                                T.cast(L, accum_dtype) * go_val
                                - sum_do[0]
                                - x_hat * sum_do_xhat[0]
                            )
                            grad_x[g // S, bc, g % S] = T.cast(gx, dtype)

        return _bn_bwd_nchw

    return _bn_bwd_nchw_func


class BatchNormBwdKernelNCHW(BatchNormBwdKernel):
    """Single-block backward kernel over the natural ``(N, C, S)`` layout.

    Reuses the ``(C, L)`` config search (it depends only on ``L = N * S``) but
    compiles a strided kernel that reads/writes ``x[n, bc, s]`` directly.
    """

    def __init__(
        self,
        N: int,
        C: int,
        S: int,
        dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        Kernel.__init__(self)
        self.N = N
        self.C = C
        self.S = S
        self.L = N * S
        self.dtype = dtype
        self.kernel = _batch_norm_bwd_kernel_nchw(
            N, C, S, self.dtype_str)
        self.init_config(config, tune)
