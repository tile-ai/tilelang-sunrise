"""PTPU DeltaNet inference, keeping a sequence's recurrent state in FP32."""

import functools

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.kernel_base import Entry, Kernel
from tileops._kernels.linear_attention.call_spec import (
    DeltaNetCall,
    DeltaNetFwdInterface,
    head_count_refusal,
)
from tileops._utils import WARP_LANES, WARP_SHUFFLE_STAGES

__all__ = ["DeltaNetPTPUFwdKernel"]


@functools.lru_cache(maxsize=32)
def _deltanet_inference_ptpu(
    batch: int,
    seq_len: int,
    heads: int,
    dim: int,
    sequences: int,
    dtype: str,
    scale: float,
    l2norm: bool,
    packed: bool,
    has_initial_state: bool,
):
    # A group of four lanes owns one value column, as in the shared decode program.
    threads, lane_group = 128, 4
    value_tile = threads // lane_group
    value_tiles = dim // value_tile
    key_chunk = dim // lane_group

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False},
    )
    def build():
        state_shape = [sequences, heads, dim, dim] if has_initial_state else [1]
        offsets_shape = [sequences + 1] if packed else [1]

        @T.prim_func
        def main(
            q: T.Tensor([batch, seq_len, heads, dim], dtype),
            k: T.Tensor([batch, seq_len, heads, dim], dtype),
            v: T.Tensor([batch, seq_len, heads, dim], dtype),
            beta: T.Tensor([batch, seq_len, heads], dtype),
            initial: T.Tensor(state_shape, "float32"),
            offsets: T.Tensor(offsets_shape, "int64"),
            o: T.Tensor([batch, seq_len, heads, dim], dtype),
            final: T.Tensor([sequences, heads, dim, dim], "float32"),
        ):
            with T.Kernel(sequences * heads * value_tiles, threads=threads) as block:
                tx = T.get_thread_binding()
                column = (block % value_tiles) * value_tile + tx // lane_group
                head = block // value_tiles % heads
                sequence = block // (value_tiles * heads)
                key_begin = (tx % lane_group) * key_chunk
                row = 0 if packed else sequence
                begin = offsets[sequence] if packed else 0
                end = offsets[sequence + 1] if packed else seq_len
                q_shared = T.alloc_shared([dim], "float32")
                k_shared = T.alloc_shared([dim], "float32")
                state = T.alloc_local([key_chunk], "float32")

                for i in T.serial(key_chunk):
                    if has_initial_state:
                        state[i] = initial[sequence, head, key_begin + i, column]
                    else:
                        state[i] = 0.0

                for token in T.serial(begin, end):
                    if tx < dim:
                        q_shared[tx] = T.cast(q[row, token, head, tx], "float32")
                        k_shared[tx] = T.cast(k[row, token, head, tx], "float32")
                    T.sync_threads()
                    if l2norm:
                        q_square = T.alloc_var("float32", init=0.0)
                        k_square = T.alloc_var("float32", init=0.0)
                        lane = tx % WARP_LANES
                        for i in T.serial(dim // WARP_LANES):
                            key = lane + i * WARP_LANES
                            q_square += q_shared[key] * q_shared[key]
                            k_square += k_shared[key] * k_shared[key]
                        for stage in T.unroll(WARP_SHUFFLE_STAGES):
                            q_square += T.shfl_xor(q_square, 1 << stage, width=WARP_LANES)
                            k_square += T.shfl_xor(k_square, 1 << stage, width=WARP_LANES)
                        T.sync_threads()
                        if tx < dim:
                            # FLA materializes normalized Q/K in the input dtype.
                            q_shared[tx] = T.cast(
                                T.cast(q_shared[tx] * T.rsqrt(q_square + 1e-6), dtype),
                                "float32",
                            )
                            k_shared[tx] = T.cast(
                                T.cast(k_shared[tx] * T.rsqrt(k_square + 1e-6), dtype),
                                "float32",
                            )
                        T.sync_threads()

                    old_value = T.alloc_var("float32", init=0.0)
                    for i in T.serial(key_chunk):
                        old_value += state[i] * k_shared[key_begin + i]
                    for stage in T.unroll(2):
                        old_value += T.shfl_xor(old_value, 1 << stage, width=lane_group)
                    update = T.cast(beta[row, token, head], "float32") * (
                        T.cast(v[row, token, head, column], "float32") - old_value
                    )
                    result = T.alloc_var("float32", init=0.0)
                    for i in T.serial(key_chunk):
                        key = key_begin + i
                        state[i] += k_shared[key] * update
                        result += state[i] * (q_shared[key] * scale)
                    for stage in T.unroll(2):
                        result += T.shfl_down(result, 1 << stage, width=lane_group)
                    if tx % lane_group == 0:
                        o[row, token, head, column] = T.cast(result, dtype)
                    # No lane may overwrite a token's Q/K while another still reads them.
                    T.sync_threads()

                for i in T.serial(key_chunk):
                    final[sequence, head, key_begin + i, column] = state[i]

        return main

    return build()


class DeltaNetPTPUFwdKernel(Kernel, DeltaNetFwdInterface):
    """Dense/packed prefill and decode with a caller-owned, read-only FP32 initial state."""

    devices = frozenset({"ptpu"})

    @classmethod
    def refusal(cls, call: DeltaNetCall) -> str | None:
        reason = head_count_refusal(call.heads)
        if reason is not None:
            return reason
        if call.dtype not in (torch.float16, torch.bfloat16):
            return "requires float16 or bfloat16 inputs"
        if call.dim_k != call.dim_v or call.dim_k not in (64, 128):
            return "requires matching K/V dimensions of 64 or 128"
        return None

    @classmethod
    def entry_for(cls, call: DeltaNetCall) -> Entry:
        return call, lambda: cls(call)

    def __init__(self, call: DeltaNetCall):
        super().__init__(device_index=None if call.device is None else call.device.index)
        self._compiled_fn = _deltanet_inference_ptpu(
            call.batch,
            call.seq_len,
            call.heads,
            call.dim_k,
            call.num_sequences,
            self.dtype_to_str(call.dtype),
            call.scale,
            call.l2norm,
            call.varlen,
            call.has_initial_state,
        )

    @property
    def default_config(self):
        return {}

    def forward(self, q, k, v, beta, initial_state=None, cu_seqlens=None, cu_seqlens_cpu=None):
        del cu_seqlens_cpu
        self._require_devices(q=q, k=k, v=v, beta=beta, initial_state=initial_state)
        initial = (
            torch.empty(1, dtype=torch.float32, device=q.device)
            if initial_state is None
            else initial_state
        )
        offsets = (
            torch.empty(1, dtype=torch.int64, device=q.device) if cu_seqlens is None else cu_seqlens
        )
        return self._compiled_fn(q, k, v, beta, initial, offsets)
