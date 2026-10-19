"""RMS Norm kernel using TileLang.

y = x * rsqrt(mean(x^2) + eps) * weight

The normalized row goes from the register fragment straight to global memory. Only the
partial reduction reads shared memory, because a thread walking a strided run of the row
can reach it there and cannot reach another thread's registers.

256-element alignment (512 bytes for fp16/bf16) required by the T.copy() that fills that
shared buffer. Padding zeros don't affect sum of squares; division uses original N for
correct mean computation.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.tiling import ALIGNMENT, align_up
from tileops.utils import get_sm_count

from ._config import select_row_config, select_row_configs

__all__ = ["RMSNormKernel"]


@functools.lru_cache(maxsize=32)
def _rms_norm_kernel(M, N, eps, dtype, partial_min_elements, sm_count):
    N_padded = align_up(N, ALIGNMENT)

    @tilelang.jit(out_idx=[2])
    def _func(block_m, threads):
        # A partial per thread trades the fp32 fragment's N/threads registers,
        # which cap the resident warps, for a serial walk of shared memory. Only
        # a grid that oversubscribes the device is paid back for the walk.
        # A tail row block runs past the end unless every index is guarded.
        row_guard = M % block_m != 0
        per_thread_partial = (
            -(-M // block_m) > sm_count
            # A thread count that does not divide the row truncates the walk.
            and N_padded % threads == 0
            and N_padded // threads >= partial_min_elements
        )

        @T.prim_func
        def main(
            x: T.Tensor[(M, N_padded), dtype],
            weight: T.Tensor[(N_padded,), dtype],
            y: T.Tensor[(M, N_padded), dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                x_local = T.alloc_fragment((block_m, N_padded), dtype)
                reduce_width = threads if per_thread_partial else N_padded
                xsq_f32 = T.alloc_fragment((block_m, reduce_width), "float32")
                sumsq = T.alloc_fragment((block_m,), "float32")
                rrms = T.alloc_fragment((block_m,), "float32")

                if per_thread_partial:
                    shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                    T.copy(x[pid_m * block_m, 0], shared_buf)
                    T.clear(xsq_f32)
                    for i, j in T.Parallel(block_m, threads):
                        for k in T.serial(N_padded // threads):
                            v = T.cast(shared_buf[i, k * threads + j], "float32")
                            xsq_f32[i, j] += v * v

                    # Order spilled square stores before the reduction reads.
                    T.call_extern("void", "__threadfence_all")
                    T.reduce_sum(xsq_f32, sumsq, dim=1)

                    # N, not N_padded: the pad contributes zero to the sum.
                    for i in T.Parallel(block_m):
                        rrms[i] = T.rsqrt(sumsq[i] / float(N) + eps)

                    T.copy(shared_buf, x_local)
                    T.call_extern("void", "__threadfence_all")
                else:
                    if row_guard:
                        for i, j in T.Parallel(block_m, N_padded):
                            x_local[i, j] = T.if_then_else(
                                pid_m * block_m + i < M,
                                x[pid_m * block_m + i, j],
                                T.cast(0.0, dtype),
                            )
                    else:
                        for i, j in T.Parallel(block_m, N_padded):
                            x_local[i, j] = x[pid_m * block_m + i, j]

                    # Order spilled x_local stores before the square reads on TANG.
                    T.call_extern("void", "__threadfence_all")
                    for i, j in T.Parallel(block_m, N_padded):
                        xsq_f32[i, j] = T.cast(x_local[i, j], "float32") * T.cast(
                            x_local[i, j], "float32"
                        )

                    # Order spilled square stores before the reduction reads.
                    T.call_extern("void", "__threadfence_all")
                    T.reduce_sum(xsq_f32, sumsq, dim=1)

                    # N, not N_padded: the pad contributes zero to the sum.
                    for i in T.Parallel(block_m):
                        rrms[i] = T.rsqrt(sumsq[i] / float(N) + eps)

                # y = x * rrms * weight, written from the fragment holding the row.
                for i, j in T.Parallel(block_m, N_padded):
                    if (not row_guard) or pid_m * block_m + i < M:
                        y[pid_m * block_m + i, j] = T.cast(
                            T.cast(x_local[i, j], "float32")
                            * rrms[i]
                            * T.cast(weight[j], "float32"),
                            dtype,
                        )

        return main

    return _func


class RMSNormKernel(Kernel):
    """RMS Norm kernel.

    Supports SM80+ architectures. Uses 256-element alignment (512 bytes for
    fp16/bf16) for the shared buffer the partial reduction walks; the row itself
    is held in a register fragment from the load through the store.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    # Row elements a thread must own before the walk pays. One reduction here,
    # so one walk.
    PARTIAL_MIN_ELEMENTS_PER_THREAD = 32

    def __init__(
        self,
        N: int,
        eps: float,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        """Build for a hidden size and dtype.

        The program for a given row count is resolved in ``forward``, memoized by
        ``_rms_norm_kernel``.
        """
        super().__init__()
        self.N = N
        self.eps = eps
        self.dtype = dtype
        self.N_padded = align_up(N, ALIGNMENT)
        self._sm_count_device = None
        self._sm_count = None
        self._tune_pending = tune  # tuning needs a program, so it waits for the first call
        self.init_config(config, tune=False)

    @property
    def default_config(self) -> dict:
        return select_row_config(self.N_padded, self.dtype)

    @property
    def autotune_configs(self) -> list[dict]:
        return select_row_configs(self.N_padded, self.dtype)

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Normalize ``x`` over its trailing ``N`` elements.

        Flattening to 2-D rows and a flat weight happens here, as does the alignment
        padding the prim_func requires.

        Args:
            x: Input whose trailing axes multiply to ``N``, contiguous, on a CUDA device.
            weight: Affine scale holding ``N`` elements, contiguous, on the same device.

        Returns:
            Tensor shaped like *x*.

        Raises:
            ValueError: Either input is not on a CUDA device.
        """
        device = x.device
        if device.type not in {"cuda", "ptpu"} or weight.device.type not in {"cuda", "ptpu"}:
            raise ValueError(
                f"{type(self).__name__} requires GPU tensors; got x on {x.device} and weight on "
                f"{weight.device}. Another target's backend serves other devices."
            )

        original_shape = x.shape
        rows = x.reshape(-1, self.N)
        weight = weight.reshape(self.N)
        m = rows.shape[0]

        if device != self._sm_count_device:
            self._sm_count = get_sm_count(device.index)
            self._sm_count_device = device

        # Exposed as ``self.kernel`` because that is what autotune and profiling read.
        self.kernel = _rms_norm_kernel(
            m,
            self.N,
            self.eps,
            self.dtype_str,
            self.PARTIAL_MIN_ELEMENTS_PER_THREAD,
            # The device the input is on, not whichever is current.
            self._sm_count,
        )
        if self._tune_pending:
            self._tune_pending = False
            self.autotune()

        pad = self.N_padded - self.N
        if pad:
            rows = F.pad(rows, (0, pad))
            weight = F.pad(weight, (0, pad))
        y = self.kernel(self.config["block_m"], self.config["threads"])(rows, weight)
        if pad:
            y = y[:, : self.N]
        return y.reshape(original_shape)
