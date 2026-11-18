"""W4A16 GEMM and M=1 GEMV over row-major packed INT4 weights with FP32 group scales, for PTPU.

The programs read the row-major packing (even K in the low nibble) directly, so they take
no repack, and are not implementations of `GEMMW4A16FwdInterface`, whose weight is in the
`W4A16RepackKernel` order and whose scale is in the activation dtype.
"""

import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.kernel_base import Kernel
from tileops._utils import get_device

GROUP_SIZE = 128

__all__ = ["GEMMW4A16RowMajorDecodeKernel", "GEMMW4A16RowMajorKernel"]


@functools.lru_cache(maxsize=32)
def _gemm_w4a16_row_major_kernel(
    m: int,
    n: int,
    k: int,
    dtype: str,
    group_size: int = GROUP_SIZE,
) -> Callable:
    if k % group_size != 0:
        raise ValueError(f"K must be divisible by group_size={group_size}, got {k}")

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def build(
        block_m: int = 64,
        block_n: int = 64,
        block_k: int = 64,
        num_stages: int = 2,
        threads: int = 128,
    ) -> Callable:
        if group_size % block_k != 0:
            raise ValueError(f"group_size={group_size} must be divisible by block_k={block_k}")
        if (block_n * block_k // 2) % (threads * 4) != 0:
            raise ValueError("packed W4 tile must divide into four-byte decode chunks per thread")

        @T.prim_func
        def main(
            activation: T.Tensor((m, k), dtype),  # type: ignore
            packed_weight: T.Tensor((n, k // 2), "uint8"),  # type: ignore
            weight_scale: T.Tensor((n, k // group_size), "float32"),  # type: ignore
            weight_zero: T.Tensor((n, k // group_size), "uint8"),  # type: ignore
            output: T.Tensor((m, n), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(
                T.ceildiv(n, block_n),
                T.ceildiv(m, block_m),
                threads=threads,
            ) as (bx, by):
                activation_shared = T.alloc_shared((block_m, block_k), dtype)
                weight_shared = T.alloc_shared((block_n, block_k), dtype)
                packed_weight_shared = T.alloc_shared((block_n, block_k // 2), "uint8")
                weight_scale_shared = T.alloc_shared((block_n,), "float32")
                weight_zero_shared = T.alloc_shared((block_n,), "uint8")
                output_local = T.alloc_fragment((block_m, block_n), "float")
                packed_local = T.alloc_local((4,), "uint8")
                dequantized_local = T.alloc_local((8,), dtype)
                scale_local = T.alloc_local((1,), dtype)
                zero_local = T.alloc_local((1,), dtype)

                if get_device() != "ptpu":
                    T.annotate_layout(
                        {
                            activation_shared: tilelang.layout.make_swizzled_layout(
                                activation_shared
                            ),
                            weight_shared: tilelang.layout.make_swizzled_layout(weight_shared),
                        }
                    )

                m_start = by * block_m
                n_start = bx * block_n
                tx = T.get_thread_binding(0)
                T.clear(output_local)

                for kk in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    k_start = kk * block_k

                    for i, j in T.Parallel(block_m, block_k):
                        activation_shared[i, j] = T.if_then_else(
                            (m_start + i < m) & (k_start + j < k),
                            activation[m_start + i, k_start + j],
                            T.cast(0, dtype),
                        )

                    if n % block_n == 0:
                        T.copy(
                            packed_weight[n_start, k_start // 2],
                            packed_weight_shared,
                        )
                    else:
                        for i, j in T.Parallel(block_n, block_k // 2):
                            packed_weight_shared[i, j] = T.if_then_else(
                                n_start + i < n,
                                packed_weight[n_start + i, k_start // 2 + j],
                                T.cast(0, "uint8"),
                            )

                    for i in T.Parallel(block_n):
                        weight_scale_shared[i] = T.if_then_else(
                            n_start + i < n,
                            weight_scale[n_start + i, k_start // group_size],
                            T.cast(0, "float32"),
                        )
                        weight_zero_shared[i] = T.if_then_else(
                            n_start + i < n,
                            weight_zero[n_start + i, k_start // group_size],
                            T.cast(0, "uint8"),
                        )

                    for chunk in T.serial(block_n * block_k // 2 // (threads * 4)):
                        packed_base = chunk * threads * 4 + tx * 4
                        scale_row = packed_base // (block_k // 2)
                        scale_local[0] = T.cast(weight_scale_shared[scale_row], dtype)
                        zero_local[0] = T.cast(weight_zero_shared[scale_row], dtype)

                        for v in T.vectorized(4):
                            packed_index = packed_base + v
                            packed_local[v] = packed_weight_shared[
                                packed_index // (block_k // 2),
                                packed_index % (block_k // 2),
                            ]

                        for v in T.serial(8):
                            # Nibble v % 2 of byte v // 2.
                            dequantized_local[v] = (
                                T.cast(
                                    (packed_local[v // 2] >> T.cast((v % 2) * 4, "uint8"))
                                    & T.uint8(15),
                                    dtype,
                                )
                                - zero_local[0]
                            ) * scale_local[0]

                        for v in T.vectorized(8):
                            output_index = chunk * threads * 8 + tx * 8 + v
                            weight_shared[
                                output_index // block_k,
                                output_index % block_k,
                            ] = dequantized_local[v]

                    T.gemm(
                        activation_shared,
                        weight_shared,
                        output_local,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                for i, j in T.Parallel(block_m, block_n):
                    if m_start + i < m and n_start + j < n:
                        output[m_start + i, n_start + j] = output_local[i, j]

        return main

    return build


class GEMMW4A16RowMajorKernel(Kernel):
    """W4A16 NT GEMM: dequantize a row-major packed weight tile in shared memory, then MMA.

    Weight ``(n, k)`` is ``(q - zero[n, g]) * scale[n, g]`` with ``g = k // group_size``,
    computed in the activation dtype (the FP32 scale is rounded to it first); the product
    accumulates in FP32.

    Args:
        m: Token rows.
        n: Output columns.
        k: Contraction dim, a multiple of *group_size*.
        dtype: Activation/output torch dtype.
        config: Optional ``{block_m, block_n, block_k, num_stages, threads}`` override.
        group_size: Weights per dequantization group; a multiple of ``block_k``.
        device_index: The device the kernel runs on.
    """

    devices = frozenset({"ptpu"})

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        group_size: int = GROUP_SIZE,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.group_size = group_size
        self.kernel = _gemm_w4a16_row_major_kernel(m, n, k, self.dtype_str, group_size)
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 64,
            "block_n": 64,
            "block_k": 64,
            "num_stages": 2,
            "threads": 128,
        }

    def forward(
        self,
        activation: torch.Tensor,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zero: torch.Tensor,
    ) -> torch.Tensor:
        """Multiply by a row-major packed INT4 weight.

        Args:
            activation: Activations, ``[M, K]``.
            packed_weight: Row-major packed weights, ``[N, K/2]``, ``torch.uint8``, even K
                in the low nibble.
            weight_scale: Group scales, ``[N, K/group_size]``, ``torch.float32``.
            weight_zero: Group zero points, ``[N, K/group_size]``, ``torch.uint8``.

        Returns:
            The product, ``[M, N]``, in the activation dtype.
        """
        compiled = _gemm_w4a16_row_major_kernel(
            self.m, self.n, self.k, self.dtype_str, self.group_size
        )(**self.config)
        return compiled(activation, packed_weight, weight_scale, weight_zero)


# Packed bytes each carried partial sum covers. Four keeps the accumulator fragment small
# enough to stay in registers at the tile sizes that saturate the weight stream; the
# builder and the autotune filter must agree on it.
BYTES_PER_SLOT = 4


@functools.lru_cache(maxsize=32)
def _gemm_w4a16_row_major_decode_kernel(n: int, k: int, dtype: str) -> Callable:
    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def build(
        block_n: int = 32,
        block_k: int = 512,
        threads: int = 128,
        num_stages: int = 4,
    ) -> Callable:
        if block_k % GROUP_SIZE != 0:
            raise ValueError(f"block_k={block_k} must be a multiple of {GROUP_SIZE}")
        packed_k = block_k // 2
        tile_groups = block_k // GROUP_SIZE

        @T.prim_func
        def main(
            activation: T.Tensor((1, k), dtype),  # type: ignore
            packed_weight: T.Tensor((n, k // 2), "uint8"),  # type: ignore
            weight_scale: T.Tensor((n, k // GROUP_SIZE), "float32"),  # type: ignore
            weight_zero: T.Tensor((n, k // GROUP_SIZE), "uint8"),  # type: ignore
            output: T.Tensor((1, n), dtype),  # type: ignore
        ) -> None:
            with T.Kernel(T.ceildiv(n, block_n), threads=threads) as bx:
                activation_shared = T.alloc_shared((1, block_k), dtype)
                packed_shared = T.alloc_shared((block_n, packed_k), "uint8")
                scale_shared = T.alloc_shared((block_n, tile_groups), "float32")
                zero_shared = T.alloc_shared((block_n, tile_groups), "uint8")
                # Partial sums carried across the whole K loop, so the
                # cross-thread reduction runs once instead of once per tile.
                products = T.alloc_fragment((block_n, packed_k // BYTES_PER_SLOT), "float")
                partial = T.alloc_fragment((block_n,), "float")

                n_start = bx * block_n
                T.clear(products)

                for kk in T.Pipelined(k // block_k, num_stages=num_stages):
                    for j in T.Parallel(block_k):
                        activation_shared[0, j] = activation[0, kk * block_k + j]
                    if n % block_n == 0:
                        T.copy(
                            packed_weight[
                                n_start : n_start + block_n,
                                kk * packed_k : (kk + 1) * packed_k,
                            ],
                            packed_shared,
                        )
                        T.copy(
                            weight_scale[
                                n_start : n_start + block_n,
                                kk * tile_groups : (kk + 1) * tile_groups,
                            ],
                            scale_shared,
                        )
                        T.copy(
                            weight_zero[
                                n_start : n_start + block_n,
                                kk * tile_groups : (kk + 1) * tile_groups,
                            ],
                            zero_shared,
                        )
                    else:
                        for i, j in T.Parallel(block_n, packed_k):
                            packed_shared[i, j] = T.if_then_else(
                                n_start + i < n,
                                packed_weight[n_start + i, kk * packed_k + j],
                                T.cast(0, "uint8"),
                            )
                        for i, g in T.Parallel(block_n, tile_groups):
                            scale_shared[i, g] = T.if_then_else(
                                n_start + i < n,
                                weight_scale[n_start + i, kk * tile_groups + g],
                                T.cast(0, "float32"),
                            )
                            zero_shared[i, g] = T.if_then_else(
                                n_start + i < n,
                                weight_zero[n_start + i, kk * tile_groups + g],
                                T.cast(0, "uint8"),
                            )

                    for i, j in T.Parallel(block_n, packed_k // BYTES_PER_SLOT):
                        for v in T.serial(BYTES_PER_SLOT):
                            byte_k = j * BYTES_PER_SLOT + v
                            byte = T.cast(packed_shared[i, byte_k], "int32")
                            scale = scale_shared[i, byte_k // (GROUP_SIZE // 2)]
                            zero = T.cast(zero_shared[i, byte_k // (GROUP_SIZE // 2)], "float")
                            # Round to the storage dtype before the product: the
                            # contract dequantizes to A16 and only then multiplies.
                            low = T.cast((T.cast(byte % 16, "float") - zero) * scale, dtype)
                            high = T.cast((T.cast(byte // 16, "float") - zero) * scale, dtype)
                            products[i, j] += T.cast(
                                activation_shared[0, byte_k * 2], "float"
                            ) * T.cast(low, "float") + T.cast(
                                activation_shared[0, byte_k * 2 + 1], "float"
                            ) * T.cast(high, "float")

                T.reduce_sum(products, partial, dim=1)
                for i in T.Parallel(block_n):
                    if n_start + i < n:
                        output[0, n_start + i] = T.cast(partial[i], dtype)

        return main

    return build


class GEMMW4A16RowMajorDecodeKernel(Kernel):
    """M=1 W4A16 GEMV over a row-major packed weight: unpack, dequantize, and FMA in FP32.

    Weight ``(n, k)`` is ``(q - zero[n, g]) * scale[n, g]`` computed in FP32 and rounded to
    the activation dtype, ``g = k // 128``; the products accumulate in FP32.

    Args:
        m: Token rows; must be 1.
        n: Output columns.
        k: Contraction dim, a multiple of 128.
        dtype: Activation/output torch dtype.
        config: Optional ``{block_n, block_k, threads, num_stages}`` override.
        group_size: Must be 128.
        device_index: The device the kernel runs on.
    """

    devices = frozenset({"ptpu"})
    # ``packed_weight`` and ``weight_zero`` are uint8 payloads; the K loop is ``k // block_k``.
    autotune_accepts_random_int_inputs: bool = True

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        group_size: int = GROUP_SIZE,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        if m != 1:
            raise ValueError(f"GEMMW4A16RowMajorDecodeKernel requires M=1, got {m}")
        if group_size != GROUP_SIZE:
            raise ValueError(f"only group_size={GROUP_SIZE} is supported")
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.group_size = group_size
        self.kernel = _gemm_w4a16_row_major_decode_kernel(n, k, self.dtype_str)
        self.init_config(config)
        # block_k must tile K exactly; K is a multiple of GROUP_SIZE, so halving
        # a power-of-two block_k always lands on a divisor.
        while k % self.config["block_k"]:
            self.config["block_k"] //= 2

    @property
    def default_config(self) -> dict:
        return {"block_n": 32, "block_k": 512, "threads": 128, "num_stages": 4}

    @property
    def autotune_configs(self) -> list[dict]:
        return [
            {"block_n": block_n, "block_k": block_k, "threads": threads, "num_stages": num_stages}
            for block_n in (8, 16, 32)
            for block_k in (256, 512)
            for threads in (128, 256)
            for num_stages in (2, 4)
            # Tile K exactly, and keep the carried accumulator off the stack.
            if self.k % block_k == 0
            and threads <= block_n * block_k // (2 * BYTES_PER_SLOT) <= threads * 32
        ]

    def forward(
        self,
        activation: torch.Tensor,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zero: torch.Tensor,
    ) -> torch.Tensor:
        """Multiply one activation row by a row-major packed INT4 weight.

        Args:
            activation: Activations, ``[1, K]``.
            packed_weight: Row-major packed weights, ``[N, K/2]``, ``torch.uint8``, even K
                in the low nibble.
            weight_scale: Group scales, ``[N, K/128]``, ``torch.float32``.
            weight_zero: Group zero points, ``[N, K/128]``, ``torch.uint8``.

        Returns:
            The product, ``[1, N]``, in the activation dtype.
        """
        compiled = _gemm_w4a16_row_major_decode_kernel(self.n, self.k, self.dtype_str)(
            **self.config
        )
        return compiled(activation, packed_weight, weight_scale, weight_zero)
