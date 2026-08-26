"""GroupNorm forward kernel using TileLang.

y = (x - mean) / sqrt(var + eps) * weight + bias

where mean and var are computed over (C/G, *spatial) dimensions for each of
the G groups independently. The input (N, C, *spatial) is reshaped to
(N*G, D) where D = (C/G) * spatial_size, enabling row-wise normalization
identical to LayerNorm.

256-element alignment (512 bytes for fp16/bf16) required by T.copy() shared
memory instructions. Padding zeros contribute 0 to sum; the centered two-pass
variance computation subtracts the exact padding bias.

Weight and bias are per-channel (C elements). After reshaping, each row of
length D = (C/G) * spatial_size has its own weight/bias slice of length D,
which is tiled from the weight/bias vectors accordingly.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

from ._config import select_row_config, select_row_configs

__all__ = ["GroupNormKernel", "GroupNormNoAffineKernel", "align_up_pow2"]

ALIGNMENT = 256


def align_up_pow2(n: int, min_alignment: int = ALIGNMENT) -> int:
    """Return ``n`` rounded up to the next power of 2 (at least ``min_alignment``).

    GroupNorm/InstanceNorm reduce over a row of ``D`` elements; a power-of-2
    padded row keeps the per-thread element count a power of two (and thus
    128-bit vectorizable), avoiding the local-memory spill + scalar-access
    collapse that non-power-of-two ``D_padded`` triggers in layout inference.
    """
    p = min_alignment
    while p < n:
        p <<= 1
    return p


@functools.lru_cache(maxsize=32)
def _group_norm_kernel(M, G, cpg, spatial_size, eps, dtype):
    """Build a row-wise normalization kernel for shape (M, D).

    This is the core computation shared by GroupNorm and InstanceNorm.
    The caller is responsible for reshaping input into (M, D) and
    weight/bias into (G, cpg). The input is NOT padded by the caller: when
    D is not a power of two the kernel pads internally via masked loads,
    eliminating the F.pad kernel launch in the Op layer.

    Affine is per-channel (not per-column): for element at (row m, col j),
    the group is ``g = m % G`` and the channel-within-group is
    ``c = j // spatial_size``. This fuses the affine into the kernel so the
    caller does not need a separate per-channel broadcast elementwise pass.

    Args:
        M: Number of rows = N * G.
        G: Number of groups.
        cpg: Channels per group = C / G.
        spatial_size: Spatial extent per channel; D = cpg * spatial_size.
        eps: Epsilon for numerical stability.
        dtype: TileLang dtype string.
    """
    D = cpg * spatial_size
    D_padded = align_up_pow2(D)
    pad_count = D_padded - D

    @tilelang.jit(out_idx=[3])
    def _func(block_m, threads):

        if pad_count > 0:
            # Masked-load kernel: accepts unpadded (M, D) input, zero-fills
            # the power-of-2 tail via if_then_else, and stores only valid
            # (j < D) columns. Eliminates the F.pad memcpy in the Op layer.
            @T.prim_func
            def main(
                x: T.Tensor[(M, D), dtype],
                weight: T.Tensor[(G, cpg), dtype],
                bias: T.Tensor[(G, cpg), dtype],
                y: T.Tensor[(M, D), dtype],
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    x_local = T.alloc_fragment((block_m, D_padded), dtype)
                    x_f32 = T.alloc_fragment((block_m, D_padded), "float32")
                    acc = T.alloc_fragment((block_m,), "float32")
                    mean_val = T.alloc_fragment((block_m,), "float32")
                    rstd = T.alloc_fragment((block_m,), "float32")

                    for i, j in T.Parallel(block_m, D_padded):
                        x_local[i, j] = T.if_then_else(
                            j < D, x[pid_m * block_m + i, j], T.cast(0.0, dtype)
                        )

                    # Cast to fp32 once -- reused across all passes
                    for i, j in T.Parallel(block_m, D_padded):
                        x_f32[i, j] = T.cast(x_local[i, j], "float32")

                    # --- Mean reduction ---
                    T.reduce_sum(x_f32, acc, dim=1)
                    for i in T.Parallel(block_m):
                        mean_val[i] = acc[i] / float(D)

                    # --- Centered variance reduction ---
                    for i, j in T.Parallel(block_m, D_padded):
                        x_f32[i, j] = (x_f32[i, j] - mean_val[i]) * (x_f32[i, j] - mean_val[i])

                    T.reduce_sum(x_f32, acc, dim=1)
                    for i in T.Parallel(block_m):
                        rstd[i] = T.rsqrt(
                            (acc[i] - float(pad_count) * mean_val[i] * mean_val[i])
                            / float(D)
                            + eps
                        )

                    # --- Output: y = (x - mean) * rstd * weight[g, c] + bias[g, c] ---
                    for i, j in T.Parallel(block_m, D_padded):
                        g = (pid_m * block_m + i) % G
                        c = T.min(j // spatial_size, cpg - 1)
                        x_local[i, j] = (
                            (T.cast(x_local[i, j], "float32") - mean_val[i])
                            * rstd[i]
                            * T.cast(weight[g, c], "float32")
                            + T.cast(bias[g, c], "float32")
                        )

                    # Masked store: write only valid columns.
                    for i, j in T.Parallel(block_m, D_padded):
                        if j < D:
                            y[pid_m * block_m + i, j] = x_local[i, j]

        else:
            # D == D_padded (already power-of-2): fast T.copy + shared path.
            @T.prim_func
            def main(
                x: T.Tensor[(M, D), dtype],
                weight: T.Tensor[(G, cpg), dtype],
                bias: T.Tensor[(G, cpg), dtype],
                y: T.Tensor[(M, D), dtype],
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    shared_buf = T.alloc_shared((block_m, D), dtype)
                    x_local = T.alloc_fragment((block_m, D), dtype)
                    x_f32 = T.alloc_fragment((block_m, D), "float32")
                    acc = T.alloc_fragment((block_m,), "float32")
                    mean_val = T.alloc_fragment((block_m,), "float32")
                    rstd = T.alloc_fragment((block_m,), "float32")

                    # Load input row block via shared memory
                    T.copy(x[pid_m * block_m, 0], shared_buf)
                    T.copy(shared_buf, x_local)

                    # Cast to fp32 once -- reused across all passes
                    for i, j in T.Parallel(block_m, D):
                        x_f32[i, j] = T.cast(x_local[i, j], "float32")

                    # --- Mean reduction ---
                    T.reduce_sum(x_f32, acc, dim=1)
                    for i in T.Parallel(block_m):
                        mean_val[i] = acc[i] / float(D)

                    # --- Centered variance reduction ---
                    for i, j in T.Parallel(block_m, D):
                        x_f32[i, j] = (x_f32[i, j] - mean_val[i]) * (x_f32[i, j] - mean_val[i])

                    T.reduce_sum(x_f32, acc, dim=1)
                    for i in T.Parallel(block_m):
                        rstd[i] = T.rsqrt(acc[i] / float(D) + eps)

                    # --- Output: y = (x - mean) * rstd * weight[g, c] + bias[g, c] ---
                    for i, j in T.Parallel(block_m, D):
                        g = (pid_m * block_m + i) % G
                        c = T.min(j // spatial_size, cpg - 1)
                        x_local[i, j] = (
                            (T.cast(x_local[i, j], "float32") - mean_val[i])
                            * rstd[i]
                            * T.cast(weight[g, c], "float32")
                            + T.cast(bias[g, c], "float32")
                        )

                    # Write output via shared memory
                    T.copy(x_local, shared_buf)
                    T.copy(shared_buf, y[pid_m * block_m, 0])

        return main

    return _func


@torch.library.custom_op("top::group_norm_fwd", mutates_args=())
def _group_norm_wrapped(
    M: int,
    G: int,
    cpg: int,
    spatial_size: int,
    eps: float,
    dtype_str: str,
    block_m: int,
    threads: int,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return _group_norm_kernel(M, G, cpg, spatial_size, eps, dtype_str)(block_m, threads)(x, weight, bias)


@_group_norm_wrapped.register_fake
def _(M, G, cpg, spatial_size, eps, dtype_str, block_m, threads, x, weight, bias):
    D = cpg * spatial_size
    return torch.empty((M, D), dtype=x.dtype, device=x.device)


class GroupNormKernel(Kernel):
    """GroupNorm forward kernel.

    Normalizes each group's (C/G, *spatial) slice independently.
    Input is pre-reshaped to (M, D) where M = N*G, D = (C/G)*spatial_size;
    weight/bias are pre-reshaped to (G, cpg). The per-channel affine is
    fused into the kernel (``weight[g, j // spatial_size]``).

    Supports SM80+ architectures. Uses 256-element alignment for shared
    memory copies. Single shared buffer reused for input load and output store.

    Args:
        M: Number of rows = N * G.
        G: Number of groups.
        cpg: Channels per group = C / G.
        spatial_size: Spatial extent per channel; D = cpg * spatial_size.
        eps: Epsilon for numerical stability.
        dtype: Data type (float32, float16, or bfloat16).
        config: Optional tile config dict.
        tune: If True, autotune tile config.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        M: int,
        G: int,
        cpg: int,
        spatial_size: int,
        eps: float,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        self.M = M
        self.G = G
        self.cpg = cpg
        self.spatial_size = spatial_size
        self.D = cpg * spatial_size
        self.eps = eps
        self.dtype = dtype
        self.D_padded = align_up_pow2(self.D)
        self.kernel = _group_norm_kernel(
            self.M, self.G, self.cpg, self.spatial_size, self.eps, self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return select_row_config(self.D_padded, self.dtype)

    @property
    def autotune_configs(self) -> list[dict]:
        return select_row_configs(self.D_padded, self.dtype)

    def forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return _group_norm_wrapped(
            self.M,
            self.G,
            self.cpg,
            self.spatial_size,
            self.eps,
            self.dtype_str,
            self.config["block_m"],
            self.config["threads"],
            x,
            weight,
            bias,
        )


@functools.lru_cache(maxsize=32)
def _group_norm_no_affine_kernel(M, D, eps, dtype):
    """Build a row-wise normalization kernel for shape (M, D) without affine.

    Same numerics as :func:`_group_norm_kernel` but omits the trailing
    weight/bias multiply/add — output is ``(x - mean) * rstd``. Used for the
    no-affine variants of GroupNorm and InstanceNorm.

    The input is NOT padded by the caller: when D is not a power of two the
    kernel pads internally via masked loads, eliminating the F.pad kernel
    launch in the Op layer.

    Args:
        M: Number of rows = N * G (already padded to a block_m multiple by
            the caller for the T.copy fast path).
        D: Row length = (C / G) * spatial_size (before padding).
        eps: Epsilon for numerical stability.
        dtype: TileLang dtype string.
    """
    D_padded = align_up_pow2(D)
    pad_count = D_padded - D

    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):

        if pad_count > 0:
            # Masked-load kernel: accepts unpadded (M, D) input, zero-fills
            # the power-of-2 tail via if_then_else, and stores only valid
            # (j < D) columns. Eliminates the F.pad memcpy in the Op layer.
            @T.prim_func
            def main(
                x: T.Tensor[(M, D), dtype],
                y: T.Tensor[(M, D), dtype],
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    x_local = T.alloc_fragment((block_m, D_padded), dtype)
                    x_f32 = T.alloc_fragment((block_m, D_padded), "float32")
                    acc = T.alloc_fragment((block_m,), "float32")
                    mean_val = T.alloc_fragment((block_m,), "float32")
                    rstd = T.alloc_fragment((block_m,), "float32")

                    for i, j in T.Parallel(block_m, D_padded):
                        x_local[i, j] = T.if_then_else(
                            j < D, x[pid_m * block_m + i, j], T.cast(0.0, dtype)
                        )

                    # Cast to fp32 once -- reused across all passes
                    for i, j in T.Parallel(block_m, D_padded):
                        x_f32[i, j] = T.cast(x_local[i, j], "float32")

                    # --- Mean reduction ---
                    T.reduce_sum(x_f32, acc, dim=1)
                    for i in T.Parallel(block_m):
                        mean_val[i] = acc[i] / float(D)

                    # --- Centered variance reduction ---
                    for i, j in T.Parallel(block_m, D_padded):
                        x_f32[i, j] = (x_f32[i, j] - mean_val[i]) * (x_f32[i, j] - mean_val[i])

                    T.reduce_sum(x_f32, acc, dim=1)
                    for i in T.Parallel(block_m):
                        rstd[i] = T.rsqrt(
                            (acc[i] - float(pad_count) * mean_val[i] * mean_val[i])
                            / float(D)
                            + eps
                        )

                    # --- Output: y = (x - mean) * rstd ---
                    for i, j in T.Parallel(block_m, D_padded):
                        x_local[i, j] = T.cast(
                            (T.cast(x_local[i, j], "float32") - mean_val[i]) * rstd[i],
                            dtype,
                        )

                    # Masked store: write only valid columns.
                    for i, j in T.Parallel(block_m, D_padded):
                        if j < D:
                            y[pid_m * block_m + i, j] = x_local[i, j]

        else:
            # D == D_padded (already power-of-2): fast T.copy + shared path.
            @T.prim_func
            def main(
                x: T.Tensor[(M, D), dtype],
                y: T.Tensor[(M, D), dtype],
            ):
                with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                    shared_buf = T.alloc_shared((block_m, D), dtype)
                    x_local = T.alloc_fragment((block_m, D), dtype)
                    x_f32 = T.alloc_fragment((block_m, D), "float32")
                    acc = T.alloc_fragment((block_m,), "float32")
                    mean_val = T.alloc_fragment((block_m,), "float32")
                    rstd = T.alloc_fragment((block_m,), "float32")

                    # Load input row block via shared memory
                    T.copy(x[pid_m * block_m, 0], shared_buf)
                    T.copy(shared_buf, x_local)

                    # Cast to fp32 once -- reused across all passes
                    for i, j in T.Parallel(block_m, D):
                        x_f32[i, j] = T.cast(x_local[i, j], "float32")

                    # --- Mean reduction ---
                    T.reduce_sum(x_f32, acc, dim=1)
                    for i in T.Parallel(block_m):
                        mean_val[i] = acc[i] / float(D)

                    # --- Centered variance reduction ---
                    for i, j in T.Parallel(block_m, D):
                        x_f32[i, j] = (x_f32[i, j] - mean_val[i]) * (x_f32[i, j] - mean_val[i])

                    T.reduce_sum(x_f32, acc, dim=1)
                    for i in T.Parallel(block_m):
                        rstd[i] = T.rsqrt(acc[i] / float(D) + eps)

                    # --- Output: y = (x - mean) * rstd ---
                    for i, j in T.Parallel(block_m, D):
                        x_local[i, j] = T.cast(
                            (T.cast(x_local[i, j], "float32") - mean_val[i]) * rstd[i],
                            dtype,
                        )

                    # Write output via shared memory
                    T.copy(x_local, shared_buf)
                    T.copy(shared_buf, y[pid_m * block_m, 0])

        return main

    return _func


@torch.library.custom_op("top::group_norm_no_affine_fwd", mutates_args=())
def _group_norm_no_affine_wrapped(
    M: int,
    D: int,
    eps: float,
    dtype_str: str,
    block_m: int,
    threads: int,
    x: torch.Tensor,
) -> torch.Tensor:
    return _group_norm_no_affine_kernel(M, D, eps, dtype_str)(block_m, threads)(x)


@_group_norm_no_affine_wrapped.register_fake
def _(M, D, eps, dtype_str, block_m, threads, x):
    return torch.empty((M, D), dtype=x.dtype, device=x.device)


class GroupNormNoAffineKernel(Kernel):
    """GroupNorm forward kernel without affine scale/shift.

    Computes ``y = (x - mean) * rstd`` row-wise for shape ``(M, D)`` reshaped
    inputs. Shares the build/launch parameters and shared-memory layout of
    :class:`GroupNormKernel`; only the output stage differs (no weight/bias
    multiply-add). Used by the no-affine variants of GroupNorm and
    InstanceNorm.

    Args:
        M: Number of rows = N * G.
        D: Row length = (C / G) * spatial_size.
        eps: Epsilon for numerical stability.
        dtype: Data type (float32, float16, or bfloat16).
        config: Optional tile config dict.
        tune: If True, autotune tile config.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        M: int,
        D: int,
        eps: float,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        self.M = M
        self.D = D
        self.eps = eps
        self.dtype = dtype
        self.D_padded = align_up_pow2(D)
        self.kernel = _group_norm_no_affine_kernel(
            self.M, self.D, self.eps, self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return select_row_config(self.D_padded, self.dtype)

    @property
    def autotune_configs(self) -> list[dict]:
        return select_row_configs(self.D_padded, self.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _group_norm_no_affine_wrapped(
            self.M,
            self.D,
            self.eps,
            self.dtype_str,
            self.config["block_m"],
            self.config["threads"],
            x,
        )
