"""Packed variable-length GQA prefill forward kernel.

Inputs use THD layout:
  q: [T_q, H, D]
  k/v: [T_kv, H_kv, D]

``cu_seqlens_q`` and ``cu_seqlens_kv`` describe per-request packed ranges.
Causal masking and the sliding window use bottom-right alignment per request,
matching the dense prefill contract when q_len may be smaller than kv_len.
"""

import functools
import itertools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.attention.online_softmax import (
    make_apply_softcap,
    make_online_softmax_with_mask_guard,
    make_rescale,
    make_sink_scale,
)
from tileops._kernels.attention.varlen import VarlenKernel
from tileops._kernels.attention.varlen_rope import make_varlen_query_rope
from tileops._kernels.constants import (
    LOG2E,
    SHARED_BUFFER_ALIGN_BYTES,
    WARPGROUP_THREADS,
    WGMMA_ROWS,
)
from tileops._kernels.grouped_tiling import GroupTiling
from tileops._utils import get_device, get_shared_memory_optin

__all__ = ["GQAPrefillVarlenFwdKernel"]


def _stages_score_tile(block_m: int, threads: int) -> bool:
    """Whether the score tile goes through shared memory: TileLang finds no register
    layout for it when a warpgroup holds fewer than ``WGMMA_ROWS`` rows."""
    warpgroups = threads // WARPGROUP_THREADS
    return block_m // warpgroups < WGMMA_ROWS


@functools.lru_cache(maxsize=32)
def _gqa_prefill_varlen_fwd_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    dim: int,
    is_causal: bool,
    sm_scale: Optional[float] = None,
    softcap: float = 0.0,
    dtype: str = "float16",
    window_size_left: int = -1,
    window_size_right: int = -1,
    fuse_rope: bool = False,
    max_position: int = 1,
    rotary_dim: int = 0,
    rope_layout: str = "neox",
    rope_dtype: str = "",
    has_sinks: bool = False,
) -> Callable:
    score_scale = dim**-0.5 if sm_scale is None else sm_scale
    use_softcap = softcap > 0.0
    has_left = window_size_left >= 0
    has_right = window_size_right >= 0
    scale = LOG2E if use_softcap or score_scale <= 0 else score_scale * LOG2E
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    groups = heads // heads_kv
    accum_dtype = "float"
    ptpu = get_device() == "ptpu"

    @tilelang.jit(
        out_idx=[7] if fuse_rope else [5],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: not (
                ptpu and (has_sinks or use_softcap or score_scale <= 0)
            ),
            **(
                {tilelang.PassConfigKey.TL_TANG_DISABLE_WARP_ALU: True}
                if ptpu and heads == heads_kv
                else {}
            ),
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gqa_prefill_varlen_fwd_func(
        block_m: int, block_n: int, num_stages: int, threads: int, k_step: int = 2
    ) -> Callable:
        total_q = T.dynamic("total_q")
        total_kv = T.dynamic("total_kv")
        q_shape = (total_q, heads, dim)
        kv_shape = (total_kv, heads_kv, dim)

        # Pair row reductions on TANG; candidate layouts must keep an even per-thread count.
        reduce_batch = 2 if ptpu else 1

        if ptpu:

            @T.macro
            def online_softmax(
                acc_s, scores_max, scores_max_prev, scores_scale, scores_sum, logsum
            ):
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False, batch=reduce_batch)
                for i in T.Parallel(block_m):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_m):
                    scores_max[i] = T.max(scores_max[i], T.cast(-1e38, accum_dtype))
                for i in T.Parallel(block_m):
                    scores_scale[i] = T.exp2((scores_max_prev[i] - scores_max[i]) * scale)
                for i, j in T.Parallel(block_m, block_n):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1, batch=reduce_batch)
                for i in T.Parallel(block_m):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

        else:
            online_softmax = make_online_softmax_with_mask_guard(
                scale, accum_dtype, block_m, block_n
            )

        apply_softcap = (
            make_apply_softcap(score_scale, softcap, accum_dtype, block_m, block_n)
            if use_softcap
            else None
        )
        rescale = make_rescale(block_m, dim)
        p_via_shared = ptpu or _stages_score_tile(block_m, threads)
        q_tiling = GroupTiling(batch, block_m)
        num_q_tiles = q_tiling.tile_upper_bound(total_q)
        rotate_query_tile = (
            make_varlen_query_rope(
                block_m, rotary_dim, rope_layout, max_position, dtype, rope_dtype, threads=threads
            )
            if fuse_rope
            else None
        )

        sink_scale = make_sink_scale(scale, accum_dtype, block_m)

        @T.macro
        def attend(
            q, k, v, cu_seqlens_q, cu_seqlens_kv, output, sinks, q_tile, by,
            rope_cos=None, rope_sin=None,
        ):  # fmt: skip
            """The scan, with the query tile rotated in place when the tables are passed."""
            q_shared = T.alloc_shared([block_m, dim], dtype)
            k_shared = T.alloc_shared([block_n, dim], dtype)
            v_shared = T.alloc_shared([block_n, dim], dtype)
            if ptpu:
                k_reg = T.alloc_fragment([block_n, dim], dtype)
                v_reg = T.alloc_fragment([block_n, dim], dtype)
                # A padded row stride retains the TANG epilogue's bank-conflict protection.
                o_shared = T.alloc_shared([block_m, dim + 8], dtype)
            tile_cum = T.alloc_shared([batch + 1], "int32")
            acc_s = T.alloc_fragment([block_m, block_n], accum_dtype)
            if p_via_shared:
                acc_s_cast = T.alloc_shared([block_m, block_n], dtype)
            else:
                acc_s_cast = T.alloc_fragment([block_m, block_n], dtype)
            acc_o = T.alloc_fragment([block_m, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_m], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_m], accum_dtype)
            scores_scale = T.alloc_fragment([block_m], accum_dtype)
            scores_sum = T.alloc_fragment([block_m], accum_dtype)
            logsum = T.alloc_fragment([block_m], accum_dtype)
            inv_logsum = T.alloc_fragment([block_m], accum_dtype)
            lo = T.alloc_local([1], "int32")
            hi = T.alloc_local([1], "int32")
            q_row = T.alloc_local([1], "int32")
            request = T.alloc_local([1], "int32")

            if ptpu:
                seq_meta = T.alloc_shared([6], "int32")
                cu_q = T.alloc_shared([batch + 1], "int32")
                cu_kv = T.alloc_shared([batch + 1], "int32")
                tx = T.get_thread_binding()
                if tx == 0:
                    for request_i in T.serial(batch + 1):
                        cu_q[request_i] = cu_seqlens_q[request_i]
                        cu_kv[request_i] = cu_seqlens_kv[request_i]
                    tile_cum[0] = 0
                    for request_i in T.serial(batch):
                        tile_cum[request_i + 1] = tile_cum[request_i] + T.ceildiv(
                            cu_q[request_i + 1] - cu_q[request_i], block_m
                        )
                # Only thread zero reads this prefix while decoding below; its
                # metadata barrier publishes both copies and the decoded bounds.
            else:
                q_tiling.cumsum_offsets(cu_seqlens_q, tile_cum)
            if ptpu or q_tile < tile_cum[batch]:
                if ptpu:
                    # Decode and load request metadata once per CTA, then broadcast it.
                    if tx == 0:
                        seq_meta[5] = 0
                        for request_i in T.serial(batch - 1):
                            if q_tile >= tile_cum[request_i + 1]:
                                seq_meta[5] = request_i + 1
                        seq_meta[0] = cu_q[seq_meta[5]]
                        seq_meta[1] = cu_kv[seq_meta[5]]
                        seq_meta[2] = cu_q[seq_meta[5] + 1] - seq_meta[0]
                        seq_meta[3] = cu_kv[seq_meta[5] + 1] - seq_meta[1]
                        seq_meta[4] = (q_tile - tile_cum[seq_meta[5]]) * block_m
                    T.sync_threads()
                    q_start = seq_meta[0]
                    kv_start = seq_meta[1]
                    q_len = seq_meta[2]
                    kv_len = seq_meta[3]
                    # A scalar row base, not a local-buffer load, lets the per-element mask
                    # loops fully unroll on TANG, so acc_s stays in registers.
                    q_row_base = seq_meta[4]
                else:
                    q_tiling.decode(q_tile, tile_cum, lo, hi, request, q_row)
                    q_start = cu_seqlens_q[request[0]]
                    kv_start = cu_seqlens_kv[request[0]]
                    q_len = cu_seqlens_q[request[0] + 1] - q_start
                    kv_len = cu_seqlens_kv[request[0] + 1] - kv_start
                causal_offset = kv_len - q_len
                cur_kv_head = by // groups

                if (q_row_base if ptpu else q_row[0]) + block_m <= q_len:
                    if ptpu:
                        # Folding block-uniform bounds avoids a redundant guard per loaded lane.
                        q_base = T.min(T.max(q_start + q_row_base, 0), total_q - block_m)
                        T.copy(
                            q[q_base : q_base + block_m, by, :],
                            q_shared,
                            disable_tma=True,
                        )
                    else:
                        T.copy(
                            q[q_start + q_row[0] : q_start + q_row[0] + block_m, by, :],
                            q_shared,
                            disable_tma=True,
                        )
                else:
                    for i, d in T.Parallel(block_m, dim):
                        q_pos = (q_row_base if ptpu else q_row[0]) + i
                        if q_pos < q_len:
                            q_shared[i, d] = q[q_start + q_pos, by, d]
                        else:
                            q_shared[i, d] = T.cast(0, dtype)
                if rope_cos is not None and (not ptpu or q_tile < tile_cum[batch]):
                    # Query token i of a request sits at position kv_len - q_len + i.
                    rotate_query_tile(
                        q_shared,
                        rope_cos,
                        rope_sin,
                        causal_offset + (q_row_base if ptpu else q_row[0]),
                    )

                T.clear(acc_o)
                T.clear(logsum)
                T.fill(scores_max, -T.infinity(accum_dtype))

                if is_causal:
                    loop_range = T.max(
                        0,
                        T.ceildiv(
                            T.min(
                                kv_len, causal_offset + (q_row_base if ptpu else q_row[0]) + block_m
                            ),
                            block_n,
                        ),
                    )
                elif has_right:
                    loop_range = T.max(
                        0,
                        T.ceildiv(
                            T.min(
                                kv_len,
                                causal_offset
                                + (q_row_base if ptpu else q_row[0])
                                + block_m
                                + window_size_right,
                            ),
                            block_n,
                        ),
                    )
                else:
                    loop_range = T.ceildiv(kv_len, block_n)
                # Key tiles wholly left of the window are skipped, not masked.
                if has_left:
                    k_first = (
                        T.max(
                            0, causal_offset + (q_row_base if ptpu else q_row[0]) - window_size_left
                        )
                        // block_n
                    )
                    loop_range = T.max(0, loop_range - k_first)

                if ptpu:
                    # Inactive upper-bound CTAs keep stores predicated off and skip the scan.
                    loop_range = T.if_then_else(q_tile < tile_cum[batch], loop_range, 0)

                for k_idx in T.Pipelined(loop_range, num_stages=num_stages):
                    if has_left:
                        tile_start = (k_first + k_idx) * block_n
                        tile_end = (k_first + k_idx + 1) * block_n
                    else:
                        tile_start = k_idx * block_n
                        tile_end = (k_idx + 1) * block_n
                    if ptpu:
                        # On TANG, block_n <= total_kv (implied by tile_end <= kv_len) proves the
                        # clamped base below non-negative, dropping a branch per loaded element.
                        if tile_end <= kv_len and block_n <= total_kv:
                            # The full tile is in range; fold its uniform bounds once.
                            kv_row_base = T.min(T.max(kv_start + tile_start, 0), total_kv - block_n)
                            T.copy(
                                k[kv_row_base : kv_row_base + block_n, cur_kv_head, :],
                                k_reg,
                                disable_tma=True,
                            )
                            T.copy(
                                v[kv_row_base : kv_row_base + block_n, cur_kv_head, :],
                                v_reg,
                                disable_tma=True,
                            )
                            T.copy(k_reg, k_shared)
                            T.copy(v_reg, v_shared)
                        else:
                            for j, d in T.Parallel(block_n, dim):
                                kv_pos = tile_start + j
                                if kv_pos < kv_len:
                                    k_reg[j, d] = k[kv_start + kv_pos, cur_kv_head, d]
                                    v_reg[j, d] = v[kv_start + kv_pos, cur_kv_head, d]
                                else:
                                    k_reg[j, d] = T.cast(0, dtype)
                                    v_reg[j, d] = T.cast(0, dtype)
                            T.copy(k_reg, k_shared)
                            T.copy(v_reg, v_shared)
                    elif tile_end <= kv_len:
                        T.copy(
                            k[kv_start + tile_start : kv_start + tile_end, cur_kv_head, :],
                            k_shared,
                            disable_tma=True,
                        )
                        T.copy(
                            v[kv_start + tile_start : kv_start + tile_end, cur_kv_head, :],
                            v_shared,
                            disable_tma=True,
                        )
                    else:
                        for j, d in T.Parallel(block_n, dim):
                            kv_pos = tile_start + j
                            if kv_pos < kv_len:
                                k_shared[j, d] = k[kv_start + kv_pos, cur_kv_head, d]
                                v_shared[j, d] = v[kv_start + kv_pos, cur_kv_head, d]
                            else:
                                k_shared[j, d] = T.cast(0, dtype)
                                v_shared[j, d] = T.cast(0, dtype)

                    if (
                        ptpu
                        and not use_softcap
                        and not has_left
                        and not has_right
                        and q_row_base + block_m <= q_len
                        and tile_end <= kv_len
                        and (not is_causal or tile_end <= q_row_base + causal_offset + 1)
                    ):
                        # Every score is visible on an interior tile.
                        T.clear(acc_s)
                    elif ptpu:
                        for i, j in T.Parallel(block_m, block_n):
                            q_pos = q_row_base + i
                            kv_pos = tile_start + j
                            valid = (q_pos < q_len) & (kv_pos < kv_len)
                            if is_causal:
                                valid = valid & (kv_pos <= q_pos + causal_offset)
                            elif has_right:
                                valid = valid & (
                                    kv_pos <= q_pos + causal_offset + window_size_right
                                )
                            if has_left:
                                valid = valid & (kv_pos >= q_pos + causal_offset - window_size_left)
                            acc_s[i, j] = T.if_then_else(valid, 0, -T.infinity(acc_s.dtype))
                    else:
                        for i, j in T.Parallel(block_m, block_n):
                            q_pos = q_row[0] + i
                            kv_pos = tile_start + j
                            if is_causal:
                                valid = (
                                    (q_pos < q_len)
                                    & (kv_pos < kv_len)
                                    & (kv_pos <= q_pos + causal_offset)
                                )
                            elif has_right:
                                valid = (
                                    (q_pos < q_len)
                                    & (kv_pos < kv_len)
                                    & (kv_pos <= q_pos + causal_offset + window_size_right)
                                )
                            else:
                                valid = (q_pos < q_len) & (kv_pos < kv_len)
                            if has_left:
                                valid = valid & (kv_pos >= q_pos + causal_offset - window_size_left)
                            acc_s[i, j] = T.if_then_else(valid, 0, -T.infinity(acc_s.dtype))
                    if ptpu:
                        T.gemm(
                            q_shared,
                            k_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            k_step=4 if dim % 32 == 0 else 2,
                        )
                    else:
                        T.gemm(
                            q_shared,
                            k_shared,
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                    if use_softcap:
                        apply_softcap(acc_s)
                    elif score_scale <= 0:
                        for i, j in T.Parallel(block_m, block_n):
                            acc_s[i, j] = T.if_then_else(
                                acc_s[i, j] == -T.infinity(accum_dtype),
                                -T.infinity(accum_dtype),
                                acc_s[i, j] * score_scale,
                            )
                    online_softmax(
                        acc_s,
                        scores_max,
                        scores_max_prev,
                        scores_scale,
                        scores_sum,
                        logsum,
                    )
                    T.copy(acc_s, acc_s_cast)
                    rescale(acc_o, scores_scale)
                    if ptpu:
                        T.gemm(
                            acc_s_cast,
                            v_shared,
                            acc_o,
                            policy=T.GemmWarpPolicy.FullRow,
                            k_step=k_step,
                        )
                    else:
                        T.gemm(acc_s_cast, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                for i in T.Parallel(block_m):
                    inv_logsum[i] = T.if_then_else(logsum[i] > 0, 1.0 / logsum[i], 0)
                if has_sinks:
                    sink_scale(scores_max, logsum, sinks, by, inv_logsum)
                if (q_row_base if ptpu else q_row[0]) + block_m <= q_len:
                    # The query tile is dead after the last key tile, so the output
                    # goes through its shared buffer and leaves as whole rows.
                    if ptpu:
                        for i, j in T.Parallel(block_m, dim):
                            o_shared[i, j] = (
                                T.if_then_else(inv_logsum[i] > 0, acc_o[i, j] * inv_logsum[i], 0)
                                if has_sinks
                                else acc_o[i, j] * inv_logsum[i]
                            )
                        T.copy(
                            o_shared[:, :dim],
                            output[q_start + q_row_base : q_start + q_row_base + block_m, by, :],
                            disable_tma=True,
                        )
                    else:
                        for i, j in T.Parallel(block_m, dim):
                            q_shared[i, j] = (
                                T.if_then_else(inv_logsum[i] > 0, acc_o[i, j] * inv_logsum[i], 0)
                                if has_sinks
                                else acc_o[i, j] * inv_logsum[i]
                            )
                        T.copy(
                            q_shared,
                            output[q_start + q_row[0] : q_start + q_row[0] + block_m, by, :],
                            disable_tma=True,
                        )
                else:
                    for i, j in T.Parallel(block_m, dim):
                        q_pos = (q_row_base if ptpu else q_row[0]) + i
                        if q_pos < q_len:
                            output[q_start + q_pos, by, j] = (
                                T.if_then_else(inv_logsum[i] > 0, acc_o[i, j] * inv_logsum[i], 0)
                                if has_sinks
                                else acc_o[i, j] * inv_logsum[i]
                            )

        if fuse_rope:
            half = rotary_dim // 2
            table_dtype = rope_dtype or dtype

            @T.prim_func
            def _gqa_prefill_varlen_rope_fwd_main(
                q: T.Tensor(q_shape, dtype),  # type: ignore
                k: T.Tensor(kv_shape, dtype),  # type: ignore
                v: T.Tensor(kv_shape, dtype),  # type: ignore
                cu_seqlens_q: T.Tensor([batch + 1], T.int32),  # type: ignore
                cu_seqlens_kv: T.Tensor([batch + 1], T.int32),  # type: ignore
                rope_cos: T.Tensor([max_position, half], table_dtype),  # type: ignore
                rope_sin: T.Tensor([max_position, half], table_dtype),  # type: ignore
                output: T.Tensor(q_shape, dtype),  # type: ignore
                sinks: T.Tensor([heads], "float32"),  # type: ignore
            ) -> None:
                with T.Kernel(num_q_tiles, heads, threads=threads) as (q_tile, by):
                    attend(
                        q, k, v, cu_seqlens_q, cu_seqlens_kv, output, sinks, q_tile, by,
                        rope_cos, rope_sin,
                    )  # fmt: skip

            return _gqa_prefill_varlen_rope_fwd_main

        @T.prim_func
        def _gqa_prefill_varlen_fwd_main(
            q: T.Tensor(q_shape, dtype),  # type: ignore
            k: T.Tensor(kv_shape, dtype),  # type: ignore
            v: T.Tensor(kv_shape, dtype),  # type: ignore
            cu_seqlens_q: T.Tensor([batch + 1], T.int32),  # type: ignore
            cu_seqlens_kv: T.Tensor([batch + 1], T.int32),  # type: ignore
            output: T.Tensor(q_shape, dtype),  # type: ignore
            sinks: T.Tensor([heads], "float32"),  # type: ignore
        ) -> None:
            with T.Kernel(num_q_tiles, heads, threads=threads) as (q_tile, by):
                attend(q, k, v, cu_seqlens_q, cu_seqlens_kv, output, sinks, q_tile, by)

        return _gqa_prefill_varlen_fwd_main

    return _gqa_prefill_varlen_fwd_func


def prefill_shared_bytes(
    config: dict, *, batch: int, dim: int, element: int, ptpu: bool = False
) -> int:
    """Count the one-stage program's shared buffers with allocation alignment."""
    block_m, block_n = config["block_m"], config["block_n"]
    tile = block_n * dim * element
    buffers = [block_m * dim * element, tile, tile, 4 * (batch + 1)]
    if ptpu:
        buffers += [24, 4 * (batch + 1), 4 * (batch + 1), block_m * (dim + 8) * element]
    if ptpu or _stages_score_tile(block_m, config["threads"]):
        buffers += [block_m * block_n * element, 4 * config["threads"], 4 * config["threads"]]
    align = SHARED_BUFFER_ALIGN_BYTES
    return sum(-(-size // align) * align for size in buffers)


def prefill_default_config(*, batch: int, dim: int, element: int, budget: int, ptpu: bool = False):
    """Select the existing one-stage plan, or report that no candidate fits."""
    narrow = {"block_m": 64, "block_n": 64 if dim <= 128 else 32, "num_stages": 1, "threads": 128}
    if ptpu:
        narrow["k_step"] = 2
    candidates = [narrow, {**narrow, "block_m": 32}, {**narrow, "block_m": 32, "block_n": 16}]
    if not ptpu and 256 < dim <= 512:
        candidates.insert(0, {"block_m": 64, "block_n": 64, "num_stages": 1, "threads": 256})
    return next(
        (
            config
            for config in candidates
            if prefill_shared_bytes(config, batch=batch, dim=dim, element=element, ptpu=ptpu)
            <= budget
        ),
        None,
    )


class GQAPrefillVarlenFwdKernel(VarlenKernel):
    """Packed prefill over per-request ranges of any length, with or without a sliding window."""

    supported_archs: list[int] = [80, 89, 90]
    devices = frozenset({"cuda", "ptpu"})
    general: bool = True

    @classmethod
    def refusal(cls, call) -> "str | None":
        if call.is_fp8:
            return "does not serve FP8"
        reason = super().refusal(call)
        if reason is not None:
            return reason
        if call.device is not None and call.device.type == "ptpu":
            plan = prefill_default_config(
                batch=call.batch,
                dim=call.dim,
                element=call.dtype.itemsize,
                budget=call.smem_budget,
                ptpu=True,
            )
            if plan is None:
                return "no TANG prefill tile fits the device shared-memory budget"
        return None

    def _make_kernel(self) -> Callable:
        return _gqa_prefill_varlen_fwd_kernel(
            self.batch,
            self.heads,
            self.heads_kv,
            self.dim,
            self.is_causal,
            self.sm_scale,
            self.softcap,
            self.dtype_str,
            self.window_size_left,
            self.window_size_right,
            self.fuse_rope,
            self.max_position,
            self.rotary_dim,
            self.rope_layout,
            self.rope_table_dtype_str,
            self.has_sinks,
        )

    @property
    def default_config(self) -> dict:
        """The first candidate the device's shared memory holds."""
        plan = prefill_default_config(
            batch=self.batch,
            dim=self.dim,
            element=self.dtype.itemsize,
            budget=get_shared_memory_optin(self.device_index),
            ptpu=get_device() == "ptpu",
        )
        return plan or {
            "block_m": 64,
            "block_n": 64 if self.dim <= 128 else 32,
            "num_stages": 1,
            "threads": 128,
            **({"k_step": 2} if get_device() == "ptpu" else {}),
        }

    def _shared_bytes(self, config: dict) -> int:
        """Shared memory a one-stage *config* allocates, buffer by buffer as TileLang does."""
        return prefill_shared_bytes(
            config,
            batch=self.batch,
            dim=self.dim,
            element=self.dtype.itemsize,
            ptpu=get_device() == "ptpu",
        )

    def tuning_candidates(self, configs, untuned=None) -> list[dict]:
        candidates = super().tuning_candidates(configs, untuned)
        budget = get_shared_memory_optin(self.device_index)
        # Q and K are simultaneously live at the score product. This lower bound
        # rejects impossible tiles without assuming pipeline allocation or aliasing.
        return [
            config
            for config in candidates
            if (config["block_m"] + config["block_n"]) * self.dim * self.dtype.itemsize <= budget
        ]

    @property
    def autotune_configs(self) -> list[dict]:
        if get_device() == "ptpu":
            # Keep the retained row-reduction layout and one-stage load schedule.
            return [dict(self.default_config)]
        configs = list(itertools.product([32, 64, 128], [32, 64, 128], [1, 2, 3], [128, 256]))
        return [
            {"block_m": c[0], "block_n": c[1], "num_stages": c[2], "threads": c[3]} for c in configs
        ]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        q_scale: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        program = self.program(self._compile)
        if self.key_rope is None:
            return program(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_kv,
                sinks if self.has_sinks else self._unused_sinks,
            )
        k_rot = self.key_rope(k, cu_seqlens_kv, rope_cos, rope_sin)
        return program(
            q,
            k_rot,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            rope_cos,
            rope_sin,
            sinks if self.has_sinks else self._unused_sinks,
        )

    def _compile(self, cfg: dict):
        """The program for *cfg*."""
        return self.kernel(**cfg)
