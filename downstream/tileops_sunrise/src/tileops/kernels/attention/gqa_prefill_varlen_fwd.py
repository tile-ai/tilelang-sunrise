"""Packed variable-length GQA prefill forward kernel.

Inputs use THD layout:
  q: [T_q, H, D]
  k/v: [T_kv, H_kv, D]

``cu_seqlens_q`` and ``cu_seqlens_kv`` describe per-request packed ranges.
Causal masking uses bottom-right alignment per request, matching the dense
prefill contract when q_len may be smaller than kv_len.
"""

import functools
import itertools
from typing import Callable, Optional

import tilelang.language as T
import torch

import tilelang

from .call_spec import uses_sliding_window
from .online_softmax import (
    LOG2E,
    make_apply_softcap,
    make_rescale,
)
from .packed_prefill import PackedPrefillKernel

__all__ = ["GQAPrefillVarlenFwdKernel"]


@functools.lru_cache(maxsize=32)
def _gqa_prefill_varlen_fwd_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    total_q: int,
    total_kv: int,
    dim: int,
    is_causal: bool,
    sm_scale: Optional[float] = None,
    softcap: float = 0.0,
    dtype: str = "float16",
) -> Callable:
    score_scale = dim**-0.5 if sm_scale is None else sm_scale
    use_softcap = softcap > 0.0
    scale = LOG2E if use_softcap else score_scale * LOG2E
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    groups = heads // heads_kv
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[7],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _gqa_prefill_varlen_fwd_func(
        block_m: int, block_n: int, num_stages: int, threads: int, k_step: int = 2
    ) -> Callable:
        q_shape = (total_q, heads, dim)
        kv_shape = (total_kv, heads_kv, dim)

        # Reduction topology: both row (dim=1) reductions of the online softmax are
        # block-wide shared-memory butterflies.  With the default batch=1 the lowering
        # emits one scalar AllReduce per per-thread output channel, each carrying its own
        # pair of shared-memory barriers and workspace round trips.  batch=2 packs the
        # channels into a single AllReduce call sharing one barrier pair, halving the
        # reduction barrier count while leaving the per-channel thread set and fold order
        # untouched.  Body below is make_online_softmax_with_mask_guard verbatim
        # (monotonic max + NaN guard) so the numerics are unchanged.
        #
        # Contract of the hard-coded batch=2: batch must evenly divide the per-thread
        # output element count N (tilelang/language/reduce_op.py, reduce_max/reduce_sum
        # "batch" doc).  The backend enforces it loudly rather than mis-computing --
        # ICHECK_EQ(N_total % batch, 0) in src/backend/common/op/reduce.h and
        # finalize_reducer.h abort the build -- so a future block_m / block_n / threads
        # candidate that breaks the divisibility dies in lowering for a reason that
        # belongs to this constant, NOT to its own mechanism.  batch is deliberately not
        # in tuning_keys; if a candidate fails to compile, check this contract before
        # attributing the failure to the candidate's direction.
        @T.macro
        def online_softmax(acc_s, scores_max, scores_max_prev, scores_scale, scores_sum, logsum):
            T.copy(scores_max, scores_max_prev)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.reduce_max(acc_s, scores_max, dim=1, clear=False, batch=2)
            for i in T.Parallel(block_m):
                scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
            for i in T.Parallel(block_m):
                scores_max[i] = T.max(scores_max[i], T.cast(-1e38, accum_dtype))
            for i in T.Parallel(block_m):
                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
            for i, j in T.Parallel(block_m, block_n):
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
            T.reduce_sum(acc_s, scores_sum, dim=1, batch=2)
            for i in T.Parallel(block_m):
                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

        apply_softcap = (
            make_apply_softcap(score_scale, softcap, accum_dtype, block_m, block_n)
            if use_softcap
            else None
        )
        rescale = make_rescale(block_m, dim)

        @T.prim_func
        def _gqa_prefill_varlen_fwd_main(
            q: T.Tensor(q_shape, dtype),  # type: ignore
            k: T.Tensor(kv_shape, dtype),  # type: ignore
            v: T.Tensor(kv_shape, dtype),  # type: ignore
            cu_seqlens_q: T.Tensor([batch + 1], T.int32),  # type: ignore
            cu_seqlens_kv: T.Tensor([batch + 1], T.int32),  # type: ignore
            max_seqlen_q: T.int32,  # type: ignore
            max_seqlen_kv: T.int32,  # type: ignore
            output: T.Tensor(q_shape, dtype),  # type: ignore
        ) -> None:
            # The q axis enumerates one tile per *real* request q-block, not
            # batch * ceildiv(max_seqlen_q, block_m): with ragged q_lens the latter
            # allocates q-tiles that no request owns, and each of those CTAs still
            # spends a residency slot plus its whole prologue/epilogue to run a
            # loop_range of 0.  The real count is sum_b ceildiv(q_len_b, block_m);
            # ceildiv(total_q, block_m) + batch - 1 is a statically computable upper
            # bound on it (every ceildiv(q_b, m) <= q_b/m + 1), so the grid reaches
            # the real task count with no host round trip.  The prefix-sum decode in
            # the prologue recovers (request, tile-in-request) from this linear tile id.
            with T.Kernel(T.ceildiv(total_q, block_m) + batch - 1, heads, threads=threads) as (
                bx,
                by,
            ):
                q_shared = T.alloc_shared([block_m, dim], dtype)
                k_shared = T.alloc_shared([block_n, dim], dtype)
                v_shared = T.alloc_shared([block_n, dim], dtype)
                k_reg = T.alloc_fragment([block_n, dim], dtype)
                v_reg = T.alloc_fragment([block_n, dim], dtype)
                acc_s = T.alloc_fragment([block_m, block_n], accum_dtype)
                acc_s_cast = T.alloc_shared([block_m, block_n], dtype)
                acc_o = T.alloc_fragment([block_m, dim], accum_dtype)
                # Row stride padded from dim to dim + 8 halfwords (+16 B = +4 word, 36 word per row
                # = 4 mod 32).  At stride = dim = 64 halfwords = 32 word the stride is a whole
                # multiple of the bank count, so the four row blocks one warp writes in the epilogue
                # collapse onto the same four banks (8 lanes per bank, 7 extra cycles per store).
                # +8 halves that to 2 lanes per bank.  +8 fp16 elements is the permutation floor for
                # a 16-bit store: any smaller gap still leaves two lanes sharing one 4 B word.  The
                # write side and the read-back side below must both address the same [:, 0:dim]
                # sub-region, or the read silently picks up the wrong columns.
                o_shared = T.alloc_shared([block_m, dim + 8], dtype)
                scores_max = T.alloc_fragment([block_m], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_m], accum_dtype)
                scores_scale = T.alloc_fragment([block_m], accum_dtype)
                scores_sum = T.alloc_fragment([block_m], accum_dtype)
                logsum = T.alloc_fragment([block_m], accum_dtype)
                inv_logsum = T.alloc_fragment([block_m], accum_dtype)

                # Block-uniform sequence metadata (q_start/kv_start/q_len/kv_len plus the
                # decoded (request, tile-in-request) pair) is evaluated by thread 0 only
                # and broadcast through shared memory, so the cu_seqlens scalar global
                # loads and the decode execute once per block instead of once per thread
                # (1024x).  causal_offset / cur_kv_head / loop_range are arithmetic-only
                # (no global load), so they stay per-thread.
                #
                # seq_meta: [0]=q_start [1]=kv_start [2]=q_len [3]=kv_len [4]=bz [5]=tile base.
                # tile_off is the per-request tile prefix sum: request b owns linear q-tiles
                # [tile_off[b], tile_off[b+1]).  Its last entry is <= gridDim.x because the
                # grid is the ceildiv(total_q, block_m) + batch - 1 upper bound above, so
                # any tile id past the last request decodes to a tile-in-request at or past
                # that request's q_len and every downstream q_pos < q_len guard retires it.
                seq_meta = T.alloc_shared([6], T.int32)
                tile_off = T.alloc_shared([batch + 1], T.int32)
                cu_q = T.alloc_shared([batch + 1], T.int32)
                cu_kv = T.alloc_shared([batch + 1], T.int32)
                tx = T.get_thread_binding(0)
                if tx == 0:
                    # batch is a build-time constant, so reading all of cu_seqlens once
                    # into shared keeps the global-load count at batch+1 per tensor
                    # (the prefix sum and the metadata both read the copies).
                    for b in T.serial(batch + 1):
                        cu_q[b] = cu_seqlens_q[b]
                        cu_kv[b] = cu_seqlens_kv[b]
                    tile_off[0] = 0
                    for b in T.serial(batch):
                        tile_off[b + 1] = tile_off[b] + T.ceildiv(cu_q[b + 1] - cu_q[b], block_m)
                    seq_meta[4] = 0
                    # The scan stops one request short of the end on purpose.  The grid's
                    # upper bound can exceed the real tile count, and a tile id past the
                    # last request would otherwise decode to request `batch`, whose
                    # cu_q[batch + 1] read runs one past the [batch + 1] shared copy and
                    # yields a q_len out of uninitialized shared.  Landing such a tile on
                    # the last request instead is correct: bxr there is >= that request's
                    # tile count, so loop_range and both store predicates retire it.
                    for b in T.serial(batch - 1):
                        if bx >= tile_off[b + 1]:
                            seq_meta[4] = b + 1
                    seq_meta[5] = tile_off[seq_meta[4]]
                    seq_meta[0] = cu_q[seq_meta[4]]
                    seq_meta[1] = cu_kv[seq_meta[4]]
                    seq_meta[2] = cu_q[seq_meta[4] + 1] - cu_q[seq_meta[4]]
                    seq_meta[3] = cu_kv[seq_meta[4] + 1] - cu_kv[seq_meta[4]]
                T.sync_threads()
                q_start = seq_meta[0]
                kv_start = seq_meta[1]
                q_len = seq_meta[2]
                kv_len = seq_meta[3]
                # Tile index inside the owning request; replaces the old blockIdx.x.
                bxr = bx - seq_meta[5]
                causal_offset = kv_len - q_len
                cur_kv_head = by // groups

                # Row bases of the full-tile (fast) paths below, clamped into the
                # tensor's leading dimension.  q_start/kv_start arrive as shared-memory
                # loads with an unknown range, so the interval analysis behind
                # LegalizeSafeMemoryAccess cannot bound `q_start + bx*block_m` and wraps
                # every lane's load of every unrolled element in a two-sided bounds guard
                # whose predicate only involves block-uniform terms (bx, k_idx, warpid,
                # q_start/kv_start) -- i.e. it is recomputed identically by all 32 lanes.
                # Folding the same block-uniform bound into the tile base gives
                # const_int_bound a concrete `[0, total - tile]` interval for the whole
                # row range, so the guard becomes provable and disappears, leaving one
                # scalar clamp per block (q) or per tile (kv) instead of one predicate,
                # two zero materialisations and a branch per lane per unrolled element.
                # The clamp is inert wherever it is used: both paths are entered only
                # when a full tile fits inside the request, and the op contract
                # (cu_seqlens non-decreasing, last entry == total_*) gives
                # q_start + q_len <= total_q and kv_start + kv_len <= total_kv.  The
                # partial-tile paths below keep the raw indices -- there the read window
                # may legitimately extend past `total - tile` and a base clamp would
                # shift it onto the wrong rows.
                q_row_base = T.min(T.max(q_start + bxr * block_m, 0), total_q - block_m)

                if (bxr + 1) * block_m <= q_len:
                    T.copy(
                        q[q_row_base : q_row_base + block_m, by, :],
                        q_shared,
                        disable_tma=True,
                    )
                else:
                    for i, d in T.Parallel(block_m, dim):
                        q_pos = bxr * block_m + i
                        if q_pos < q_len:
                            q_shared[i, d] = q[q_start + q_pos, by, d]
                        else:
                            q_shared[i, d] = T.cast(0, dtype)

                T.clear(acc_o)
                T.clear(logsum)
                T.fill(scores_max, -T.infinity(accum_dtype))

                kv_tiles = (
                    T.ceildiv(T.min(kv_len, causal_offset + (bxr + 1) * block_m), block_n)
                    if is_causal
                    else T.ceildiv(kv_len, block_n)
                )
                # A q-block whose first row is already past q_len carries no valid q row,
                # so its KV trip count is mathematically 0.  Clamping the bound with a
                # scalar select (no branch, no extra barrier) retires the whole pipelined
                # body - both gemms and the online-softmax work - while the epilogue's
                # q_pos < q_len store predicates keep the (empty) boundary writes correct.
                loop_range = T.if_then_else(bxr * block_m < q_len, kv_tiles, 0)

                for k_idx in T.Pipelined(loop_range, num_stages=num_stages):
                    tile_start = k_idx * block_n
                    tile_end = (k_idx + 1) * block_n
                    if tile_end <= kv_len:
                        # Same bound-folding as q_row_base above; the tile base is
                        # block-uniform, so this costs one clamp per tile rather than a
                        # bounds guard per lane per unrolled element.  tile_end <= kv_len
                        # makes it inert (see the note at q_row_base).
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

                    if use_softcap:
                        for i, j in T.Parallel(block_m, block_n):
                            acc_s[i, j] = 0
                    elif is_causal:
                        # Pure interior tile: every element satisfies
                        # q_pos<q_len & kv_pos<kv_len & kv_pos<=q_pos+causal_offset,
                        # so the per-element mask predicate is a no-op.  Zero the
                        # accumulator directly and skip the elementwise select.
                        if (
                            ((bxr + 1) * block_m <= q_len)
                            & (tile_end <= kv_len)
                            & (tile_end <= bxr * block_m + causal_offset + 1)
                        ):
                            T.clear(acc_s)
                        else:
                            for i, j in T.Parallel(block_m, block_n):
                                q_pos = bxr * block_m + i
                                kv_pos = tile_start + j
                                valid = (
                                    (q_pos < q_len)
                                    & (kv_pos < kv_len)
                                    & (kv_pos <= q_pos + causal_offset)
                                )
                                acc_s[i, j] = T.if_then_else(valid, 0, -T.infinity(acc_s.dtype))
                    else:
                        for i, j in T.Parallel(block_m, block_n):
                            q_pos = bxr * block_m + i
                            kv_pos = tile_start + j
                            valid = (q_pos < q_len) & (kv_pos < kv_len)
                            acc_s[i, j] = T.if_then_else(valid, 0, -T.infinity(acc_s.dtype))
                    # k_step here is a literal, not the `k_step` config key, so this
                    # QK-side dose stays attributable on its own and the PV gemm below
                    # keeps its independently measured setting.  Contract: K_Tile is
                    # dim = 64, A = fp16 -> tile_size_k = 8, inner_k = 8, and 8 % 4 == 0
                    # satisfies gemm_tmma.h's static_assert(inner_k % kStep == 0).
                    T.gemm(
                        q_shared,
                        k_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                        k_step=4,
                    )
                    if use_softcap:
                        apply_softcap(acc_s)
                        for i, j in T.Parallel(block_m, block_n):
                            q_pos = bxr * block_m + i
                            kv_pos = tile_start + j
                            if is_causal:
                                valid = (
                                    (q_pos < q_len)
                                    & (kv_pos < kv_len)
                                    & (kv_pos <= q_pos + causal_offset)
                                )
                            else:
                                valid = (q_pos < q_len) & (kv_pos < kv_len)
                            acc_s[i, j] = T.if_then_else(
                                valid,
                                acc_s[i, j],
                                -T.infinity(acc_s.dtype),
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
                    # Contract of k_step (default 2): the TANG TMMA k-loop below walks
                    # inner_k = K_Tile / tile_size_k in steps of kStep and requires both
                    # K_Tile % tile_size_k == 0 and inner_k % kStep == 0
                    # (src/tl_templates/tang/gemm_tmma.h static_asserts, "choose a k_step
                    # that divides K/tile_size_k"); a partial-K tail would otherwise truncate
                    # or over-read shared tiles instead of failing loudly.  The accepted config
                    # satisfies it (K_Tile = block_n = 64, A=fp16 -> tile_size_k = 16/2 = 8,
                    # inner_k = 8, 8 % 2 == 0), as does every block_n in autotune_configs
                    # ({32, 64, 128}); a future block_n = 8 candidate (inner_k = 8/8 = 1) fails
                    # in lowering for a reason that belongs to this constant, NOT to the
                    # candidate's own mechanism.  The QK gemm above carries its own
                    # explicit literal k_step (see the comment there); the config key
                    # below drives this PV gemm only.
                    #
                    # k_step is a real tuning key, not a source literal: it is a parameter of
                    # this jit factory AND a key of GQAPrefillVarlenFwdKernel.default_config
                    # (default 2), so a perf_config `config.k_step` reaches the build
                    # (Kernel.init_config copies only default_config keys into self.config).
                    # The reduce batch=2 above stays a contract constant -- batch is not in
                    # the adapter's tuning vocabulary, k_step is.
                    T.gemm(
                        acc_s_cast,
                        v_shared,
                        acc_o,
                        policy=T.GemmWarpPolicy.FullRow,
                        k_step=k_step,
                    )

                if (bxr + 1) * block_m <= q_len:
                    for i in T.Parallel(block_m):
                        inv_logsum[i] = T.cast(1, accum_dtype) / logsum[i]
                    for i, j in T.Parallel(block_m, dim):
                        acc_o[i, j] *= inv_logsum[i]
                    T.copy(acc_o, o_shared[:, 0:dim])
                    T.copy(
                        o_shared[:, 0:dim],
                        output[q_row_base : q_row_base + block_m, by, :],
                        disable_tma=True,
                    )
                else:
                    for i in T.Parallel(block_m):
                        q_pos = bxr * block_m + i
                        if q_pos < q_len:
                            inv_logsum[i] = T.cast(1, accum_dtype) / logsum[i]
                    for i, j in T.Parallel(block_m, dim):
                        q_pos = bxr * block_m + i
                        if q_pos < q_len:
                            output[q_start + q_pos, by, j] = acc_o[i, j] * inv_logsum[i]

        return _gqa_prefill_varlen_fwd_main

    return _gqa_prefill_varlen_fwd_func


@torch.library.custom_op("tileops::gqa_prefill_varlen_fwd_wrapped_kernel", mutates_args=())
def _gqa_prefill_varlen_fwd_wrapped_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    total_q: int,
    total_kv: int,
    dim: int,
    is_causal: bool,
    sm_scale: float,
    softcap: float,
    dtype: str,
    block_m: int,
    block_n: int,
    num_stages: int,
    threads: int,
    k_step: int,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
) -> torch.Tensor:
    return _gqa_prefill_varlen_fwd_kernel(
        batch, heads, heads_kv, total_q, total_kv, dim, is_causal, sm_scale, softcap, dtype
    )(block_m, block_n, num_stages, threads, k_step)(
        q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv
    )


@_gqa_prefill_varlen_fwd_wrapped_kernel.register_fake
def _(
    batch: int,
    heads: int,
    heads_kv: int,
    total_q: int,
    total_kv: int,
    dim: int,
    is_causal: bool,
    sm_scale: float,
    softcap: float,
    dtype: str,
    block_m: int,
    block_n: int,
    num_stages: int,
    threads: int,
    k_step: int,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
) -> torch.Tensor:
    return torch.empty([total_q, heads, dim], dtype=q.dtype, device=q.device)


class GQAPrefillVarlenFwdKernel(PackedPrefillKernel):
    """Ragged packed prefill: per-request ranges of unequal length.

    Serves the requests the dense implementations cannot: a caller that asked
    for the varlen algorithm explicitly, or an automatic choice whose packed
    ranges are not uniform.
    """

    supported_archs: list[int] = [80, 89, 90]

    @classmethod
    def applies(cls, call) -> bool:
        if call.is_fp8 or uses_sliding_window(call):
            return False
        if call.backend == "varlen":
            return True
        return call.backend == "auto" and not call.is_uniform

    def _build_program(self) -> None:
        # The program is specialized on the packed totals, which are known per
        # call, so there is nothing to build until forward runs.
        self.kernel = None

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 64,
            "block_n": 64 if self.dim <= 128 else 32,
            "num_stages": 1,
            "threads": 128,
            "k_step": 2,
        }

    @property
    def autotune_configs(self) -> list[dict]:
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
    ) -> torch.Tensor:
        total_q, total_kv = q.shape[0], k.shape[0]
        output = _gqa_prefill_varlen_fwd_wrapped_kernel(
            self.batch,
            self.heads,
            self.heads_kv,
            total_q,
            total_kv,
            self.dim,
            self.is_causal,
            self.sm_scale,
            self.softcap,
            self.dtype_str,
            self.config["block_m"],
            self.config["block_n"],
            self.config["num_stages"],
            self.config["threads"],
            self.config["k_step"],
            self.max_seqlen_q,
            self.max_seqlen_kv,
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
        )
        return output
