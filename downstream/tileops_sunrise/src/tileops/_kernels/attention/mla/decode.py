import functools
import itertools
import re
from typing import Optional

import tilelang
import tilelang.language as T
import torch
import tvm_ffi

from tileops._csrc import csrc_include
from tileops._kernels.attention.call_spec import (
    ATTENTION_DTYPES,
    MLADecodeCall,
    MLADecodeFwdInterface,
)
from tileops._kernels.attention.mla.seesaw import (
    _mla_seesaw_combine_kernel,
    _mla_seesaw_dense_kernel,
)
from tileops._kernels.attention.online_softmax import make_rescale
from tileops._kernels.constants import LOG2E
from tileops._kernels.kernel_base import CompiledProgram, Entry, Kernel
from tileops._utils import get_device, get_shared_memory_optin, get_sm_count, get_sm_version

__all__ = ["MLADecodeMMAKernel", "MLADecodeSeesawKernel", "MLADecodeWSKernel"]


def _split_combine(batch, heads, num_split, dim, dtype, lse_dtype):
    """Merge the splits' partial outputs, each weighted by its share of the log-sum-exp."""
    accum_dtype = "float"

    @T.macro
    def combine(
        glse: T.Tensor([batch, heads, num_split], lse_dtype),
        Output_partial: T.Tensor([batch, heads, num_split, dim], dtype),
        Output: T.Tensor([batch, heads, dim], dtype),
    ):
        with T.Kernel(heads, batch, threads=128) as (hid, bz):
            po_local = T.alloc_fragment([dim], dtype)
            o_accum_local = T.alloc_fragment([dim], accum_dtype)
            lse_local_split = T.alloc_local([1], accum_dtype)
            lse_logsum_local = T.alloc_local([1], accum_dtype)
            lse_max_local = T.alloc_local([1], accum_dtype)
            scale_local = T.alloc_local([1], accum_dtype)

            T.annotate_layout(
                {
                    lse_logsum_local: T.Fragment(
                        lse_logsum_local.shape, forward_thread_fn=lambda i: i
                    ),
                }
            )

            T.clear(lse_logsum_local)
            T.clear(o_accum_local)
            lse_max_local[0] = -T.infinity(accum_dtype)
            for k in T.serial(num_split):
                lse_max_local[0] = T.max(lse_max_local[0], glse[bz, hid, k])
            for k in T.Pipelined(num_split, num_stages=1):
                lse_local_split[0] = glse[bz, hid, k]
                lse_logsum_local[0] += T.exp2(lse_local_split[0] - lse_max_local[0])
            lse_logsum_local[0] = T.log2(lse_logsum_local[0]) + lse_max_local[0]
            for k in T.serial(num_split):
                for i in T.Parallel(dim):
                    po_local[i] = Output_partial[bz, hid, k, i]
                lse_local_split[0] = glse[bz, hid, k]
                scale_local[0] = T.exp2(lse_local_split[0] - lse_logsum_local[0])
                for i in T.Parallel(dim):
                    o_accum_local[i] += po_local[i] * scale_local[0]
            for i in T.Parallel(dim):
                Output[bz, hid, i] = o_accum_local[i]

    return combine


# The warp-specialized kernel issues this call once as a bare statement;
# _uniform_wgmma_bases rewrites only code that holds it, so the two must agree.
_UNIFORM_BASES_CALL = "tileops::uniform_wgmma_bases"
_DESCRIPTOR_BASE = re.compile(
    r"(tl::initialize_wgmma_descriptor<[^>]*>\(\w+, )(\(&\(\(\(\w+\*\)\w+\)\[0\]\)\))\);"
)


def _uniform_wgmma_bases(code: str) -> str:
    """Build the warp-specialized kernel's WGMMA descriptors from lane-0 shared-memory bases.

    The consumers hold up to 240 registers each, and ptxas keeps the shared-memory bases
    in ordinary registers, so every WGMMA issue pays an R2UR. A base broadcast from lane 0
    is provably uniform, which keeps the descriptors in uniform registers. TileLang has no
    way to express that broadcast, so the generated CUDA is rewritten instead. Code
    without the marker statement is returned unchanged.
    """
    marker = re.compile(rf"^[ \t]*{re.escape(_UNIFORM_BASES_CALL)}\(\);$", re.MULTILINE)
    if len(marker.findall(code)) != 1:
        return code
    return _DESCRIPTOR_BASE.sub(r"\1tileops::warp_uniform_ptr\2);", code)


def _register_uniform_wgmma_bases() -> None:
    """Chain the rewrite after whatever holds TileLang's one CUDA post-processing slot."""
    previous = tvm_ffi.get_global_func("tilelang_callback_cuda_postproc", allow_missing=True)

    def postproc(code, target):
        if previous is not None:
            code = previous(code, target)
        return _uniform_wgmma_bases(code)

    tilelang.register_cuda_postproc(postproc)


_register_uniform_wgmma_bases()


@functools.lru_cache(maxsize=32)
def _mla_decode_ws_kernel(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim, dtype="float16"):
    sm_scale = (1.0 / (dim + pe_dim)) ** 0.5 * LOG2E
    accum_dtype = "float"
    kv_group_num = heads // kv_head_num

    @tilelang.jit(
        out_idx=[6],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=[
            "-O3",
            "-Wno-deprecated-declarations",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--ptxas-options=-v,--register-usage-level=10",
            "-DNDEBUG",
            *csrc_include("warp_uniform.h"),
        ],
    )
    def _mla_decode_ws_func(block_H, block_N, num_split, num_stages, threads=384):
        # A head count block_H does not divide leaves the last block partly filled; its
        # loads past the heads read zero and its stores there are dropped.
        VALID_BLOCK_H = min(block_H, kv_group_num)
        # Two 128-thread consumer warpgroups and one producer warpgroup, which gathers
        # KV 8 threads to a row, 16 rows a pass.
        if threads != 384:
            raise ValueError(f"the warp-specialized schedule runs 384 threads, threads={threads}")
        if block_N % 16 != 0:
            raise ValueError(f"the KV gather copies 16 rows a pass, block_N={block_N}")
        # Each split owns kv_per_split keys and a loop step consumes two tiles. A
        # length that does not fill every step masks the keys past its split's end.
        kv_per_split = tilelang.cdiv(seqlen_kv, num_split)
        ragged = seqlen_kv % (num_split * 2 * block_N) != 0
        may_leave_a_split_empty = (num_split - 1) * kv_per_split >= seqlen_kv

        @T.macro
        def load_row(KV, K_pe, kv_l, kv_r, k_tail, bid, cur_kv_head, r, kv_index, tx):
            with T.attr("default", "async_scope", 1):
                for u in T.serial(dim // 128):
                    for v in T.vectorized(8):
                        kv_l[r * 16 + (tx - 256) // 8, 64 * u + (tx - 256) % 8 * 8 + v] = KV[
                            bid, kv_index, cur_kv_head, 64 * u + (tx - 256) % 8 * 8 + v
                        ]
                        kv_r[r * 16 + (tx - 256) // 8, 64 * u + (tx - 256) % 8 * 8 + v] = KV[
                            bid, kv_index, cur_kv_head, dim // 2 + 64 * u + (tx - 256) % 8 * 8 + v
                        ]
            with T.attr("default", "async_scope", 1):
                for v in T.vectorized(8):
                    k_tail[r * 16 + (tx - 256) // 8, (tx - 256) % 8 * 8 + v] = K_pe[
                        bid, kv_index, cur_kv_head, (tx - 256) % 8 * 8 + v
                    ]

        @T.macro
        def gather_tile(KV, K_pe, kv_l, kv_r, k_tail, bid, cur_kv_head, tile_start, kv_end, tx):
            for r in T.serial(block_N // 16):
                kv_index = tile_start + r * 16 + (tx - 256) // 8
                if ragged:
                    # Zero, not stale: the row's softmax weight is zero, and zero
                    # times a NaN an earlier kernel left in shared memory is NaN.
                    if kv_index < kv_end:
                        load_row(KV, K_pe, kv_l, kv_r, k_tail, bid, cur_kv_head, r, kv_index, tx)
                    else:
                        for u in T.serial(dim // 128):
                            for v in T.vectorized(8):
                                kv_l[r * 16 + (tx - 256) // 8, 64 * u + (tx - 256) % 8 * 8 + v] = 0
                                kv_r[r * 16 + (tx - 256) // 8, 64 * u + (tx - 256) % 8 * 8 + v] = 0
                        for v in T.vectorized(8):
                            k_tail[r * 16 + (tx - 256) // 8, (tx - 256) % 8 * 8 + v] = 0
                else:
                    load_row(KV, K_pe, kv_l, kv_r, k_tail, bid, cur_kv_head, r, kv_index, tx)

        @T.macro
        def init_scores(acc_s, tile_start, kv_end):
            if ragged:
                for h_i, bi_i in T.Parallel(block_H, block_N):
                    acc_s[h_i, bi_i] = T.if_then_else(
                        tile_start + bi_i < kv_end, 0, -T.infinity(accum_dtype)
                    )
            else:
                T.clear(acc_s)

        @T.macro
        def flash_attn(
            Q: T.Tensor([batch, heads, dim], dtype),
            Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
            KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
            K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
            Output: T.Tensor([batch, heads, dim], dtype),
        ):
            with T.Kernel(T.ceildiv(heads, VALID_BLOCK_H), batch, threads=threads) as (
                hid,
                bid,
            ):
                Q_shared_l = T.alloc_shared([block_H, dim // 2], dtype)
                Q_shared_r = T.alloc_shared([block_H, dim // 2], dtype)
                Q_tail_shared = T.alloc_shared([block_H, pe_dim], dtype)
                KV_shared_0_l = T.alloc_shared([block_N, dim // 2], dtype)
                KV_shared_0_r = T.alloc_shared([block_N, dim // 2], dtype)
                KV_shared_1_l = T.alloc_shared([block_N, dim // 2], dtype)
                KV_shared_1_r = T.alloc_shared([block_N, dim // 2], dtype)
                K_tail_shared_0 = T.alloc_shared([block_N, pe_dim], dtype)
                K_tail_shared_1 = T.alloc_shared([block_N, pe_dim], dtype)
                O_shared_l = Q_shared_l
                O_shared_r = Q_shared_r

                acc_o_l = T.alloc_fragment([block_H, dim // 2], accum_dtype)
                acc_o_r = T.alloc_fragment([block_H, dim // 2], accum_dtype)
                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                S_shared = T.alloc_shared([block_H, block_N], dtype)
                sumexp = T.alloc_fragment([block_H], accum_dtype)
                sum_exp_shared = T.alloc_shared([block_H], accum_dtype)
                sumexp_i = T.alloc_fragment([block_H], accum_dtype)
                alpha_shared = T.alloc_shared([block_H], accum_dtype, scope="shared")
                alpha_local = T.alloc_fragment([block_H], accum_dtype)
                m_i = T.alloc_fragment([block_H], accum_dtype)
                m_i_prev = T.alloc_fragment([block_H], accum_dtype)

                bar_k_0_ready = T.alloc_barrier(arrive_count=128)
                bar_k_1_ready = T.alloc_barrier(arrive_count=128)
                bar_k_0_free = T.alloc_barrier(arrive_count=256)
                bar_k_1_free = T.alloc_barrier(arrive_count=256)
                bar_sScale_and_sS_ready = T.alloc_barrier(arrive_count=256)
                bar_sScale_and_sS_free = T.alloc_barrier(arrive_count=256)

                cur_kv_head = hid * VALID_BLOCK_H // kv_group_num
                kv_start = 0
                kv_end = seqlen_kv
                NI = T.ceildiv(seqlen_kv, block_N)

                tx = T.get_thread_binding()
                T.evaluate(T.call_extern("handle", _UNIFORM_BASES_CALL))

                # Q/Q_pe -> shared copies must stay inside tx < 128 so that the copy
                # and the subsequent T.wgmma_gemm share the same 128-thread bounds.
                # Mixing a 384-thread copy with a 128-thread WGMMA on the same shared
                # buffer causes TileLang 0.1.9 layout inference to fail with
                # "no available layout found".
                if tx < 128:
                    T.set_max_nreg(240, 1)
                    T.copy(
                        Q[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, 0 : dim // 2],
                        Q_shared_l,
                    )
                    T.copy(
                        Q[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, dim // 2 : dim],
                        Q_shared_r,
                    )
                    T.copy(
                        Q_pe[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, :], Q_tail_shared
                    )
                    T.fill(sumexp, 0)
                    T.fill(m_i, -(2**30))  # avoid -inf - inf to cause nan
                    T.fill(acc_o_l, 0)

                    for i_i in T.serial(T.ceildiv(NI, 2)):
                        T.barrier_wait(bar_k_0_ready[0], (i_i & 1))

                        init_scores(acc_s, kv_start + (i_i * 2) * block_N, kv_end)
                        T.wgmma_gemm(Q_shared_l, KV_shared_0_l, acc_s, transpose_B=True)
                        T.wgmma_gemm(Q_shared_r, KV_shared_0_r, acc_s, transpose_B=True)
                        T.wgmma_gemm(Q_tail_shared, K_tail_shared_0, acc_s, transpose_B=True)

                        T.wait_wgmma(0)

                        if i_i != 0:
                            T.barrier_arrive(bar_sScale_and_sS_free)
                            T.barrier_wait(bar_sScale_and_sS_free, ((i_i * 2) & 1) ^ 1)

                        T.copy(m_i, m_i_prev)
                        T.reduce_max(acc_s, m_i, dim=1, clear=False)
                        for h_i in T.Parallel(block_H):
                            alpha_local[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                        for h_i, bi_i in T.Parallel(block_H, block_N):
                            acc_s[h_i, bi_i] = T.exp2(
                                acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                            )
                        T.reduce_sum(acc_s, sumexp_i, dim=1)
                        for h_i in T.Parallel(block_H):
                            sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                        for h_i, d_i in T.Parallel(block_H, dim // 2):
                            acc_o_l[h_i, d_i] *= alpha_local[h_i]
                        T.copy(alpha_local, alpha_shared)

                        T.copy(acc_s, S_shared)
                        T.gemm(S_shared, KV_shared_0_l, acc_o_l)

                        T.barrier_arrive(bar_sScale_and_sS_ready)
                        T.barrier_arrive(bar_k_0_free[0])

                        T.barrier_wait(bar_k_1_ready[0], (i_i & 1))

                        init_scores(acc_s, kv_start + (i_i * 2 + 1) * block_N, kv_end)
                        T.wgmma_gemm(Q_shared_l, KV_shared_1_l, acc_s, transpose_B=True)
                        T.wgmma_gemm(Q_shared_r, KV_shared_1_r, acc_s, transpose_B=True)
                        T.wgmma_gemm(Q_tail_shared, K_tail_shared_1, acc_s, transpose_B=True)

                        T.wait_wgmma(0)

                        T.barrier_arrive(bar_sScale_and_sS_free)
                        T.barrier_wait(bar_sScale_and_sS_free, ((i_i * 2 + 1) & 1) ^ 1)

                        T.copy(m_i, m_i_prev)
                        T.reduce_max(acc_s, m_i, dim=1, clear=False)
                        for h_i in T.Parallel(block_H):
                            alpha_local[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                        for h_i, bi_i in T.Parallel(block_H, block_N):
                            acc_s[h_i, bi_i] = T.exp2(
                                acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                            )
                        T.reduce_sum(acc_s, sumexp_i, dim=1)
                        for h_i in T.Parallel(block_H):
                            sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                        for h_i, d_i in T.Parallel(block_H, dim // 2):
                            acc_o_l[h_i, d_i] *= alpha_local[h_i]
                        T.copy(alpha_local, alpha_shared)

                        T.copy(acc_s, S_shared)
                        T.gemm(S_shared, KV_shared_1_l, acc_o_l)

                        T.barrier_arrive(bar_sScale_and_sS_ready)
                        T.barrier_arrive(bar_k_1_free[0])

                    for h_i in T.Parallel(block_H):
                        sum_exp_shared[h_i] = sumexp[h_i]
                    for h_i, d_i in T.Parallel(block_H, dim // 2):
                        acc_o_l[h_i, d_i] /= sumexp[h_i]
                    for h_i in T.Parallel(block_H):
                        sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale
                    T.copy(acc_o_l, O_shared_l)
                    T.copy(
                        O_shared_l,
                        Output[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, 0 : dim // 2],
                    )

                elif tx >= 128 and tx < 256:
                    T.set_max_nreg(168, 1)
                    T.fill(acc_o_r, 0)
                    for i_i in T.serial(T.ceildiv(NI, 2)):
                        T.barrier_arrive(bar_sScale_and_sS_ready)
                        T.barrier_wait(bar_sScale_and_sS_ready, ((i_i * 2) & 1))
                        for h_i, d_i in T.Parallel(block_H, dim // 2):
                            acc_o_r[h_i, d_i] *= alpha_shared[h_i]
                        T.gemm(S_shared, KV_shared_0_r, acc_o_r)
                        T.barrier_arrive(bar_k_0_free[0])
                        T.barrier_arrive(bar_sScale_and_sS_free)

                        T.barrier_arrive(bar_sScale_and_sS_ready)
                        T.barrier_wait(bar_sScale_and_sS_ready, ((i_i * 2 + 1) & 1))
                        for h_i, d_i in T.Parallel(block_H, dim // 2):
                            acc_o_r[h_i, d_i] *= alpha_shared[h_i]
                        T.gemm(S_shared, KV_shared_1_r, acc_o_r)
                        T.barrier_arrive(bar_k_1_free[0])
                        if i_i != T.ceildiv(NI, 2) - 1:
                            T.barrier_arrive(bar_sScale_and_sS_free)

                    for h_i, d_i in T.Parallel(block_H, dim // 2):
                        acc_o_r[h_i, d_i] /= sum_exp_shared[h_i]

                    T.copy(acc_o_r, O_shared_r)
                    T.copy(
                        O_shared_r,
                        Output[
                            bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, dim // 2 : dim
                        ],
                    )

                elif tx >= 256:
                    T.set_max_nreg(80, 0)
                    for i_i in T.serial(T.ceildiv(NI, 2)):
                        T.barrier_wait(bar_k_0_free[0], ((i_i & 1) ^ 1))
                        gather_tile(
                            KV,
                            K_pe,
                            KV_shared_0_l,
                            KV_shared_0_r,
                            K_tail_shared_0,
                            bid,
                            cur_kv_head,
                            kv_start + (i_i * 2) * block_N,
                            kv_end,
                            tx,
                        )
                        T.cp_async_barrier_noinc(bar_k_0_ready[0])

                        T.barrier_wait(bar_k_1_free[0], ((i_i & 1) ^ 1))
                        gather_tile(
                            KV,
                            K_pe,
                            KV_shared_1_l,
                            KV_shared_1_r,
                            K_tail_shared_1,
                            bid,
                            cur_kv_head,
                            kv_start + (i_i * 2 + 1) * block_N,
                            kv_end,
                            tx,
                        )
                        T.cp_async_barrier_noinc(bar_k_1_ready[0])

        @T.macro
        def flash_attn_split(
            Q: T.Tensor([batch, heads, dim], dtype),
            Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
            KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
            K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
            glse: T.Tensor([batch, heads, num_split], dtype),
            Output_partial: T.Tensor([batch, heads, num_split, dim], dtype),
        ):
            with T.Kernel(batch, T.ceildiv(heads, VALID_BLOCK_H), num_split, threads=threads) as (
                bid,
                hid,
                bz,
            ):
                Q_shared_l = T.alloc_shared([block_H, dim // 2], dtype)
                Q_shared_r = T.alloc_shared([block_H, dim // 2], dtype)
                Q_tail_shared = T.alloc_shared([block_H, pe_dim], dtype)
                KV_shared_0_l = T.alloc_shared([block_N, dim // 2], dtype)
                KV_shared_0_r = T.alloc_shared([block_N, dim // 2], dtype)
                KV_shared_1_l = T.alloc_shared([block_N, dim // 2], dtype)
                KV_shared_1_r = T.alloc_shared([block_N, dim // 2], dtype)
                K_tail_shared_0 = T.alloc_shared([block_N, pe_dim], dtype)
                K_tail_shared_1 = T.alloc_shared([block_N, pe_dim], dtype)
                O_shared_l = Q_shared_l
                O_shared_r = Q_shared_r

                acc_o_l = T.alloc_fragment([block_H, dim // 2], accum_dtype)
                acc_o_r = T.alloc_fragment([block_H, dim // 2], accum_dtype)
                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                S_shared = T.alloc_shared([block_H, block_N], dtype)
                sumexp = T.alloc_fragment([block_H], accum_dtype)
                sum_exp_shared = T.alloc_shared([block_H], accum_dtype)
                sumexp_i = T.alloc_fragment([block_H], accum_dtype)
                alpha_shared = T.alloc_shared([block_H], accum_dtype, scope="shared")
                alpha_local = T.alloc_fragment([block_H], accum_dtype)
                m_i = T.alloc_fragment([block_H], accum_dtype)
                m_i_prev = T.alloc_fragment([block_H], accum_dtype)

                bar_k_0_ready = T.alloc_barrier(arrive_count=128)
                bar_k_1_ready = T.alloc_barrier(arrive_count=128)
                bar_k_0_free = T.alloc_barrier(arrive_count=256)
                bar_k_1_free = T.alloc_barrier(arrive_count=256)
                bar_sScale_and_sS_ready = T.alloc_barrier(arrive_count=256)
                bar_sScale_and_sS_free = T.alloc_barrier(arrive_count=256)

                cur_kv_head = hid * VALID_BLOCK_H // kv_group_num
                kv_start = kv_per_split * bz
                kv_end = T.min(kv_start + kv_per_split, seqlen_kv)
                NI = T.ceildiv(kv_per_split, block_N)

                tx = T.get_thread_binding()
                T.evaluate(T.call_extern("handle", _UNIFORM_BASES_CALL))

                # Q/Q_pe -> shared copies must stay inside tx < 128 so that the copy
                # and the subsequent T.wgmma_gemm share the same 128-thread bounds.
                # Mixing a 384-thread copy with a 128-thread WGMMA on the same shared
                # buffer causes TileLang 0.1.9 layout inference to fail with
                # "no available layout found".
                if tx < 128:
                    T.set_max_nreg(240, 1)
                    T.copy(
                        Q[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, 0 : dim // 2],
                        Q_shared_l,
                    )
                    T.copy(
                        Q[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, dim // 2 : dim],
                        Q_shared_r,
                    )
                    T.copy(
                        Q_pe[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, :], Q_tail_shared
                    )
                    T.fill(sumexp, 0)
                    T.fill(m_i, -(2**30))  # avoid -inf - inf to cause nan
                    T.fill(acc_o_l, 0)

                    for i_i in T.serial(T.ceildiv(NI, 2)):
                        T.barrier_wait(bar_k_0_ready[0], (i_i & 1))

                        init_scores(acc_s, kv_start + (i_i * 2) * block_N, kv_end)
                        T.wgmma_gemm(Q_shared_l, KV_shared_0_l, acc_s, transpose_B=True)
                        T.wgmma_gemm(Q_shared_r, KV_shared_0_r, acc_s, transpose_B=True)
                        T.wgmma_gemm(Q_tail_shared, K_tail_shared_0, acc_s, transpose_B=True)

                        T.wait_wgmma(0)

                        if i_i != 0:
                            T.barrier_arrive(bar_sScale_and_sS_free)
                            T.barrier_wait(bar_sScale_and_sS_free, ((i_i * 2) & 1) ^ 1)

                        T.copy(m_i, m_i_prev)
                        T.reduce_max(acc_s, m_i, dim=1, clear=False)
                        for h_i in T.Parallel(block_H):
                            alpha_local[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                        for h_i, bi_i in T.Parallel(block_H, block_N):
                            acc_s[h_i, bi_i] = T.exp2(
                                acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                            )
                        T.reduce_sum(acc_s, sumexp_i, dim=1)
                        for h_i in T.Parallel(block_H):
                            sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                        for h_i, d_i in T.Parallel(block_H, dim // 2):
                            acc_o_l[h_i, d_i] *= alpha_local[h_i]
                        T.copy(alpha_local, alpha_shared)

                        T.copy(acc_s, S_shared)
                        T.gemm(S_shared, KV_shared_0_l, acc_o_l)

                        T.barrier_arrive(bar_sScale_and_sS_ready)
                        T.barrier_arrive(bar_k_0_free[0])

                        T.barrier_wait(bar_k_1_ready[0], (i_i & 1))

                        init_scores(acc_s, kv_start + (i_i * 2 + 1) * block_N, kv_end)
                        T.wgmma_gemm(Q_shared_l, KV_shared_1_l, acc_s, transpose_B=True)
                        T.wgmma_gemm(Q_shared_r, KV_shared_1_r, acc_s, transpose_B=True)
                        T.wgmma_gemm(Q_tail_shared, K_tail_shared_1, acc_s, transpose_B=True)

                        T.wait_wgmma(0)

                        T.barrier_arrive(bar_sScale_and_sS_free)
                        T.barrier_wait(bar_sScale_and_sS_free, ((i_i * 2 + 1) & 1) ^ 1)

                        T.copy(m_i, m_i_prev)
                        T.reduce_max(acc_s, m_i, dim=1, clear=False)
                        for h_i in T.Parallel(block_H):
                            alpha_local[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                        for h_i, bi_i in T.Parallel(block_H, block_N):
                            acc_s[h_i, bi_i] = T.exp2(
                                acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                            )
                        T.reduce_sum(acc_s, sumexp_i, dim=1)
                        for h_i in T.Parallel(block_H):
                            sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                        for h_i, d_i in T.Parallel(block_H, dim // 2):
                            acc_o_l[h_i, d_i] *= alpha_local[h_i]
                        T.copy(alpha_local, alpha_shared)

                        T.copy(acc_s, S_shared)
                        T.gemm(S_shared, KV_shared_1_l, acc_o_l)

                        T.barrier_arrive(bar_sScale_and_sS_ready)
                        T.barrier_arrive(bar_k_1_free[0])

                    for h_i in T.Parallel(block_H):
                        sum_exp_shared[h_i] = sumexp[h_i]
                    if may_leave_a_split_empty:
                        # An empty split has no key to divide by; its zero weight in the combine
                        # needs a finite partial.
                        for h_i, d_i in T.Parallel(block_H, dim // 2):
                            acc_o_l[h_i, d_i] = T.if_then_else(
                                sumexp[h_i] > 0, acc_o_l[h_i, d_i] / sumexp[h_i], 0
                            )
                    else:
                        for h_i, d_i in T.Parallel(block_H, dim // 2):
                            acc_o_l[h_i, d_i] /= sumexp[h_i]
                    for h_i in T.Parallel(block_H):
                        sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale
                    T.copy(acc_o_l, O_shared_l)
                    T.copy(
                        O_shared_l,
                        Output_partial[
                            bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, bz, 0 : dim // 2
                        ],
                    )
                    T.copy(sumexp, glse[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, bz])

                elif tx >= 128 and tx < 256:
                    T.set_max_nreg(168, 1)
                    T.fill(acc_o_r, 0)
                    for i_i in T.serial(T.ceildiv(NI, 2)):
                        T.barrier_arrive(bar_sScale_and_sS_ready)
                        T.barrier_wait(bar_sScale_and_sS_ready, ((i_i * 2) & 1))
                        for h_i, d_i in T.Parallel(block_H, dim // 2):
                            acc_o_r[h_i, d_i] *= alpha_shared[h_i]
                        T.gemm(S_shared, KV_shared_0_r, acc_o_r)
                        T.barrier_arrive(bar_k_0_free[0])
                        T.barrier_arrive(bar_sScale_and_sS_free)

                        T.barrier_arrive(bar_sScale_and_sS_ready)
                        T.barrier_wait(bar_sScale_and_sS_ready, ((i_i * 2 + 1) & 1))
                        for h_i, d_i in T.Parallel(block_H, dim // 2):
                            acc_o_r[h_i, d_i] *= alpha_shared[h_i]
                        T.gemm(S_shared, KV_shared_1_r, acc_o_r)
                        T.barrier_arrive(bar_k_1_free[0])
                        if i_i != T.ceildiv(NI, 2) - 1:
                            T.barrier_arrive(bar_sScale_and_sS_free)

                    if may_leave_a_split_empty:
                        # An empty split has no key to divide by; its zero weight in the combine
                        # needs a finite partial.
                        for h_i, d_i in T.Parallel(block_H, dim // 2):
                            acc_o_r[h_i, d_i] = T.if_then_else(
                                sum_exp_shared[h_i] > 0, acc_o_r[h_i, d_i] / sum_exp_shared[h_i], 0
                            )
                    else:
                        for h_i, d_i in T.Parallel(block_H, dim // 2):
                            acc_o_r[h_i, d_i] /= sum_exp_shared[h_i]

                    T.copy(acc_o_r, O_shared_r)
                    T.copy(
                        O_shared_r,
                        Output_partial[
                            bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, bz, dim // 2 : dim
                        ],
                    )

                elif tx >= 256:
                    T.set_max_nreg(80, 0)
                    for i_i in T.serial(T.ceildiv(NI, 2)):
                        T.barrier_wait(bar_k_0_free[0], ((i_i & 1) ^ 1))
                        gather_tile(
                            KV,
                            K_pe,
                            KV_shared_0_l,
                            KV_shared_0_r,
                            K_tail_shared_0,
                            bid,
                            cur_kv_head,
                            kv_start + (i_i * 2) * block_N,
                            kv_end,
                            tx,
                        )
                        T.cp_async_barrier_noinc(bar_k_0_ready[0])

                        T.barrier_wait(bar_k_1_free[0], ((i_i & 1) ^ 1))
                        gather_tile(
                            KV,
                            K_pe,
                            KV_shared_1_l,
                            KV_shared_1_r,
                            K_tail_shared_1,
                            bid,
                            cur_kv_head,
                            kv_start + (i_i * 2 + 1) * block_N,
                            kv_end,
                            tx,
                        )
                        T.cp_async_barrier_noinc(bar_k_1_ready[0])

        combine = _split_combine(batch, heads, num_split, dim, dtype, dtype)

        @T.prim_func
        def main_split(
            Q: T.Tensor([batch, heads, dim], dtype),
            Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
            KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
            K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
            glse: T.Tensor([batch, heads, num_split], dtype),
            Output_partial: T.Tensor([batch, heads, num_split, dim], dtype),
            Output: T.Tensor([batch, heads, dim], dtype),
        ):
            flash_attn_split(Q, Q_pe, KV, K_pe, glse, Output_partial)
            combine(glse, Output_partial, Output)

        @T.prim_func
        def main_no_split(
            Q: T.Tensor([batch, heads, dim], dtype),
            Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
            KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
            K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
            glse: T.Tensor([batch, heads, num_split], dtype),
            Output_partial: T.Tensor([batch, heads, num_split, dim], dtype),
            Output: T.Tensor([batch, heads, dim], dtype),
        ):
            flash_attn(Q, Q_pe, KV, K_pe, Output)

        if num_split > 1:
            return main_split
        else:
            return main_no_split

    return _mla_decode_ws_func


def _mla_no_split_ptpu(
    batch,
    heads,
    kv_head_num,
    seqlen_kv,
    dim,
    pe_dim,
    dtype,
    scale,
    block_H,
    block_N,
    num_stages,
    threads,
    swizzle,
    swizzle_order,
    qk_k_step,
    pe_k_step,
    pv_k_step,
):
    """The PTPU single-split program: the macro to run and the pass configs it needs.

    One CTA per head block runs the whole KV span, so the barriers of a short loop can be
    placed by hand and the row sum deferred to the end.
    """
    accum_dtype = "float"
    VALID_BLOCK_H = block_H
    # With trip_count <= num_stages every KV copy is hoisted into the prologue.
    # The async form removes register-staged LDG->STS and turns shared-memory
    # reuse off, keeping Q, KV and O tiles resident.
    loop_tiles = (seqlen_kv + block_N - 1) // block_N
    async_kv_copy = loop_tiles <= num_stages and block_H * dim <= 8192
    # `T.Pipelined` rotates each copy to the earliest point after its last reader, which
    # lands it *before* an in-body barrier, so the loop-carried WAR cannot be hand-placed
    # through the pipelined form. The long span therefore uses an explicit two-buffer
    # steady loop (_mla_no_split_steady) whose statement order is emitted verbatim.
    steady_two_buffer = (
        not async_kv_copy
        and block_H == threads // 32
        and loop_tiles % 2 == 0
        and not (dim == 128 and seqlen_kv <= 4096)
    )
    # The no-split path may place its block-wide barriers by hand wherever the loop's
    # happens-before edges are enumerable: the fully unrolled short span, or the explicit
    # steady loop above. Elsewhere the compiler's ThreadSync pass stays in charge.
    long_loop_hand_placed = steady_two_buffer
    hand_placed_sync = async_kv_copy or long_loop_hand_placed
    # Group k_step k-steps of MMA operand fragment loads before issuing the paired MMAs.
    # The pe gemm has inner_k == 2 and PV inner_k == 4, so each caps at that; QK reaches 8.
    narrow_rows = block_H <= 8
    # A zero step derives the grouping from the tile shape; explicit values override it.
    qk_k_step = qk_k_step or (8 if narrow_rows else 1)
    pe_k_step = pe_k_step or (2 if narrow_rows else 1)
    pv_k_step = pv_k_step or (4 if narrow_rows else 1)

    # Each warp owns block_H // (threads // 32) output rows. When that is exactly 2,
    # one batched collective replaces the two per-row scalar AllReduce ladders (5
    # block barriers each).
    rows_per_warp = threads // 32
    reduce_batch = block_H // rows_per_warp if block_H % rows_per_warp == 0 else 1
    # The row sum only feeds logsum, which is consumed after the KV loop, so it does not
    # need a collective per block: accumulate the rescaled probabilities element-wise in a
    # fragment and run one row reduction at the end. That drops one workspace ladder (and
    # its two block barriers) per KV block beyond the first, for any loop length > 1.
    if reduce_batch == 2:

        @T.macro
        def online_softmax_deferred(acc_s, scores_max, scores_max_prev, scores_scale, p_acc):
            T.copy(scores_max, scores_max_prev)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.reduce_max(acc_s, scores_max, dim=1, clear=False, batch=2)
            for i in T.Parallel(block_H):
                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
            for i, j in T.Parallel(block_H, block_N):
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                p_acc[i, j] = p_acc[i, j] * scores_scale[i] + acc_s[i, j]

    else:

        @T.macro
        def online_softmax_deferred(acc_s, scores_max, scores_max_prev, scores_scale, p_acc):
            T.copy(scores_max, scores_max_prev)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
            for i in T.Parallel(block_H):
                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
            for i, j in T.Parallel(block_H, block_N):
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                p_acc[i, j] = p_acc[i, j] * scores_scale[i] + acc_s[i, j]

    rescale = make_rescale(block_H, dim)
    rescale_half = make_rescale(block_H, dim // 2)

    @T.macro
    def _mla_no_split(
        Q: T.Tensor([batch, heads, dim], dtype),
        Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
        KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
        K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
        Output: T.Tensor([batch, heads, dim], dtype),
    ):
        with T.Kernel(batch, T.ceildiv(heads, VALID_BLOCK_H), threads=threads) as (bx, by):
            # A 128-wide shared row puts the two 64-half halves exactly one bank rotation
            # apart (64 halves = 32 words), so a 4-byte store's 32 lanes cover 16 banks
            # twice. Two 64-wide buffers keep the 64-column tile layout (72-half pitch),
            # where one warp spans 8 rows x 4 words and lands on 32 distinct banks.
            # Split copies read each KV row as two 128-byte halves instead of one 256-byte
            # row, which is free while the per-batch KV span still fits cache but costs
            # DRAM efficiency once it does not, so gate on the long spans.
            split_kv = dim == 128 and seqlen_kv <= 4096 and not async_kv_copy
            if split_kv:
                Q_shared_l = T.alloc_shared([block_H, dim // 2], dtype)
                Q_shared_r = T.alloc_shared([block_H, dim // 2], dtype)
                KV_shared_l = T.alloc_shared([block_N, dim // 2], dtype)
                KV_shared_r = T.alloc_shared([block_N, dim // 2], dtype)
                O_shared_l = T.alloc_shared([block_H, dim // 2], dtype)
                O_shared_r = T.alloc_shared([block_H, dim // 2], dtype)
                acc_o_l = T.alloc_fragment([block_H, dim // 2], accum_dtype)
                acc_o_r = T.alloc_fragment([block_H, dim // 2], accum_dtype)
            else:
                Q_shared = T.alloc_shared([block_H, dim], dtype)
                KV_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([block_H, dim], dtype)
                acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
            S_shared = T.alloc_shared([block_H, block_N], dtype)
            Q_pe_shared = T.alloc_shared([block_H, pe_dim], dtype)
            K_pe_shared = T.alloc_shared([block_N, pe_dim], dtype)
            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)
            p_acc = T.alloc_fragment([block_H, block_N], accum_dtype)

            cur_kv_head = 0
            if swizzle > 0:
                T.use_swizzle(swizzle, order=swizzle_order)
            if not split_kv:
                T.annotate_layout(
                    {
                        O_shared: tilelang.layout.make_swizzled_layout(O_shared),
                    }
                )

            if async_kv_copy:
                # Stage Q through registers so the shared store can be issued after the
                # KV/K_pe DMA, overlapping the two independent memory chains.
                Q_local = T.alloc_fragment([block_H, dim], dtype)
                Q_pe_local = T.alloc_fragment([block_H, pe_dim], dtype)
                T.copy(Q[bx, by * VALID_BLOCK_H : (by + 1) * VALID_BLOCK_H, :], Q_local)
                T.copy(Q_pe[bx, by * VALID_BLOCK_H : (by + 1) * VALID_BLOCK_H, :], Q_pe_local)
            else:
                if split_kv:
                    T.copy(
                        Q[bx, by * VALID_BLOCK_H : (by + 1) * VALID_BLOCK_H, 0 : dim // 2],
                        Q_shared_l,
                    )
                    T.copy(
                        Q[bx, by * VALID_BLOCK_H : (by + 1) * VALID_BLOCK_H, dim // 2 : dim],
                        Q_shared_r,
                    )
                else:
                    T.copy(Q[bx, by * VALID_BLOCK_H : (by + 1) * VALID_BLOCK_H, :], Q_shared)
                T.copy(Q_pe[bx, by * VALID_BLOCK_H : (by + 1) * VALID_BLOCK_H, :], Q_pe_shared)
            if split_kv:
                T.fill(acc_o_l, 0)
                T.fill(acc_o_r, 0)
            else:
                T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.fill(p_acc, 0)

            loop_range = T.ceildiv(seqlen_kv, block_N)
            for k in T.Pipelined(loop_range, num_stages=num_stages):
                if async_kv_copy:
                    with T.attr("default", "async_scope", 1):
                        T.copy(
                            KV[bx, k * block_N : (k + 1) * block_N, cur_kv_head, :],
                            KV_shared,
                        )
                    with T.attr("default", "async_scope", 1):
                        T.copy(
                            K_pe[bx, k * block_N : (k + 1) * block_N, cur_kv_head, :],
                            K_pe_shared,
                        )
                    T.copy(Q_local, Q_shared)
                    T.copy(Q_pe_local, Q_pe_shared)
                else:
                    if split_kv:
                        T.copy(
                            KV[bx, k * block_N : (k + 1) * block_N, cur_kv_head, 0 : dim // 2],
                            KV_shared_l,
                        )
                        T.copy(
                            KV[
                                bx,
                                k * block_N : (k + 1) * block_N,
                                cur_kv_head,
                                dim // 2 : dim,
                            ],
                            KV_shared_r,
                        )
                    else:
                        T.copy(
                            KV[bx, k * block_N : (k + 1) * block_N, cur_kv_head, :],
                            KV_shared,
                        )
                    T.copy(
                        K_pe[bx, k * block_N : (k + 1) * block_N, cur_kv_head, :],
                        K_pe_shared,
                    )
                if hand_placed_sync:
                    T.sync_threads()
                T.clear(acc_s)
                if split_kv:
                    T.gemm(
                        Q_shared_l,
                        KV_shared_l,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol,
                        k_step=qk_k_step,
                    )
                    T.gemm(
                        Q_shared_r,
                        KV_shared_r,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol,
                        k_step=qk_k_step,
                    )
                else:
                    T.gemm(
                        Q_shared,
                        KV_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol,
                        k_step=qk_k_step,
                    )
                T.gemm(
                    Q_pe_shared,
                    K_pe_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    k_step=pe_k_step,
                )
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.if_then_else(
                        k * block_N + j < seqlen_kv, acc_s[i, j], -T.infinity(accum_dtype)
                    )
                online_softmax_deferred(acc_s, scores_max, scores_max_prev, scores_scale, p_acc)
                # The reduce workspace aliases S_shared, so a warp still gathering the max
                # must finish before any warp overwrites that shared range.
                if hand_placed_sync:
                    T.sync_threads()
                T.copy(acc_s, S_shared)
                if hand_placed_sync:
                    T.sync_threads()
                if split_kv:
                    rescale_half(acc_o_l, scores_scale)
                    rescale_half(acc_o_r, scores_scale)
                    T.gemm(
                        S_shared,
                        KV_shared_l,
                        acc_o_l,
                        policy=T.GemmWarpPolicy.FullCol,
                        k_step=pv_k_step,
                    )
                    T.gemm(
                        S_shared,
                        KV_shared_r,
                        acc_o_r,
                        policy=T.GemmWarpPolicy.FullCol,
                        k_step=pv_k_step,
                    )
                else:
                    rescale(acc_o, scores_scale)
                    T.gemm(
                        S_shared,
                        KV_shared,
                        acc_o,
                        policy=T.GemmWarpPolicy.FullCol,
                        k_step=pv_k_step,
                    )
            if reduce_batch == 2:
                T.reduce_sum(p_acc, logsum, dim=1, batch=2)
            else:
                T.reduce_sum(p_acc, logsum, dim=1)
            # The sum's workspace aliases O_shared, so a warp still gathering must finish
            # before any warp stores the output.
            if hand_placed_sync:
                T.sync_threads()
            if split_kv:
                for i, j in T.Parallel(block_H, dim // 2):
                    acc_o_l[i, j] /= logsum[i]
                    acc_o_r[i, j] /= logsum[i]
                T.copy(acc_o_l, O_shared_l)
                T.copy(acc_o_r, O_shared_r)
                if hand_placed_sync:
                    T.sync_threads()
                T.copy(
                    O_shared_l,
                    Output[bx, by * VALID_BLOCK_H : (by + 1) * VALID_BLOCK_H, 0 : dim // 2],
                )
                T.copy(
                    O_shared_r,
                    Output[bx, by * VALID_BLOCK_H : (by + 1) * VALID_BLOCK_H, dim // 2 : dim],
                )
            else:
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, O_shared)
                if hand_placed_sync:
                    T.sync_threads()
                T.copy(O_shared, Output[bx, by * VALID_BLOCK_H : (by + 1) * VALID_BLOCK_H, :])

    @T.macro
    def _mla_no_split_steady(
        Q: T.Tensor([batch, heads, dim], dtype),
        Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
        KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
        K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
        Output: T.Tensor([batch, heads, dim], dtype),
    ):
        # Long KV span, two explicit buffers, one tile of lookahead. ThreadSync is off for
        # this prim_func, so the barriers below are the only block-wide ones besides the
        # collectives' own: a WAR/RAW pair around the refill of the buffer the next tile
        # will read, and a RAW between the S_shared write and the PV gemm. The row-max
        # collective needs no barrier before that write: its workspace sits in its own
        # range beside S_shared (verified in the built kernel's shared layout).
        with T.Kernel(batch, T.ceildiv(heads, VALID_BLOCK_H), threads=threads) as (bx, by):
            Q_shared = T.alloc_shared([block_H, dim], dtype)
            Q_pe_shared = T.alloc_shared([block_H, pe_dim], dtype)
            KV_shared_0 = T.alloc_shared([block_N, dim], dtype)
            KV_shared_1 = T.alloc_shared([block_N, dim], dtype)
            K_pe_shared_0 = T.alloc_shared([block_N, pe_dim], dtype)
            K_pe_shared_1 = T.alloc_shared([block_N, pe_dim], dtype)
            S_shared = T.alloc_shared([block_H, block_N], dtype)
            O_shared = T.alloc_shared([block_H, dim], dtype)
            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)
            p_acc = T.alloc_fragment([block_H, block_N], accum_dtype)

            cur_kv_head = 0
            if swizzle > 0:
                T.use_swizzle(swizzle, order=swizzle_order)
            T.annotate_layout(
                {
                    O_shared: tilelang.layout.make_swizzled_layout(O_shared),
                }
            )

            T.copy(Q[bx, by * VALID_BLOCK_H : (by + 1) * VALID_BLOCK_H, :], Q_shared)
            T.copy(Q_pe[bx, by * VALID_BLOCK_H : (by + 1) * VALID_BLOCK_H, :], Q_pe_shared)
            T.copy(KV[bx, 0:block_N, cur_kv_head, :], KV_shared_0)
            T.copy(K_pe[bx, 0:block_N, cur_kv_head, :], K_pe_shared_0)
            T.copy(KV[bx, block_N : 2 * block_N, cur_kv_head, :], KV_shared_1)
            T.copy(K_pe[bx, block_N : 2 * block_N, cur_kv_head, :], K_pe_shared_1)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.fill(p_acc, 0)
            T.sync_threads()

            for kk in T.serial(loop_tiles // 2):
                T.clear(acc_s)
                T.gemm(
                    Q_shared,
                    KV_shared_0,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    k_step=qk_k_step,
                )
                T.gemm(
                    Q_pe_shared,
                    K_pe_shared_0,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    k_step=pe_k_step,
                )
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.if_then_else(
                        (2 * kk) * block_N + j < seqlen_kv,
                        acc_s[i, j],
                        -T.infinity(accum_dtype),
                    )
                online_softmax_deferred(acc_s, scores_max, scores_max_prev, scores_scale, p_acc)
                # The allocator lays the collective scratch next to S_shared, not over it:
                # in the built kernel the last workspace ends exactly where S_shared
                # begins, so the store below cannot outrun a warp still gathering the max.
                T.copy(acc_s, S_shared)
                T.sync_threads()
                rescale(acc_o, scores_scale)
                T.gemm(
                    S_shared,
                    KV_shared_0,
                    acc_o,
                    policy=T.GemmWarpPolicy.FullCol,
                    k_step=pv_k_step,
                )
                # WAR: tile 2*kk is done reading buffer 0; RAW: buffer 0 is ready for the
                # next step's tile 2*kk+2, and buffer 1 for this step's tile 2*kk+1.
                T.sync_threads()
                if 2 * kk + 2 < loop_tiles:
                    T.copy(
                        KV[
                            bx,
                            (2 * kk + 2) * block_N : (2 * kk + 3) * block_N,
                            cur_kv_head,
                            :,
                        ],
                        KV_shared_0,
                    )
                    T.copy(
                        K_pe[
                            bx,
                            (2 * kk + 2) * block_N : (2 * kk + 3) * block_N,
                            cur_kv_head,
                            :,
                        ],
                        K_pe_shared_0,
                    )
                T.sync_threads()
                T.clear(acc_s)
                T.gemm(
                    Q_shared,
                    KV_shared_1,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    k_step=qk_k_step,
                )
                T.gemm(
                    Q_pe_shared,
                    K_pe_shared_1,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    k_step=pe_k_step,
                )
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.if_then_else(
                        (2 * kk + 1) * block_N + j < seqlen_kv,
                        acc_s[i, j],
                        -T.infinity(accum_dtype),
                    )
                online_softmax_deferred(acc_s, scores_max, scores_max_prev, scores_scale, p_acc)
                T.copy(acc_s, S_shared)
                T.sync_threads()
                rescale(acc_o, scores_scale)
                T.gemm(
                    S_shared,
                    KV_shared_1,
                    acc_o,
                    policy=T.GemmWarpPolicy.FullCol,
                    k_step=pv_k_step,
                )
                T.sync_threads()
                if 2 * kk + 3 < loop_tiles:
                    T.copy(
                        KV[
                            bx,
                            (2 * kk + 3) * block_N : (2 * kk + 4) * block_N,
                            cur_kv_head,
                            :,
                        ],
                        KV_shared_1,
                    )
                    T.copy(
                        K_pe[
                            bx,
                            (2 * kk + 3) * block_N : (2 * kk + 4) * block_N,
                            cur_kv_head,
                            :,
                        ],
                        K_pe_shared_1,
                    )

            if reduce_batch == 2:
                T.reduce_sum(p_acc, logsum, dim=1, batch=2)
            else:
                T.reduce_sum(p_acc, logsum, dim=1)
            T.sync_threads()
            for i, j in T.Parallel(block_H, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.sync_threads()
            T.copy(O_shared, Output[bx, by * VALID_BLOCK_H : (by + 1) * VALID_BLOCK_H, :])

    # When the whole KV span is two block_N tiles (loop_range == 2) the two row-max
    # collectives reduce the same lane domain: fold both tiles' scores element-wise and
    # run one collective instead of two, on the same reduction tree.
    merge_two_tiles = hand_placed_sync and reduce_batch == 2 and loop_tiles == 2

    @T.macro
    def _mla_no_split_merged(
        Q: T.Tensor([batch, heads, dim], dtype),
        Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
        KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
        K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
        Output: T.Tensor([batch, heads, dim], dtype),
    ):
        with T.Kernel(batch, T.ceildiv(heads, VALID_BLOCK_H), threads=threads) as (bx, by):
            Q_shared = T.alloc_shared([block_H, dim], dtype)
            S_shared_0 = T.alloc_shared([block_H, block_N], dtype)
            S_shared_1 = T.alloc_shared([block_H, block_N], dtype)
            Q_pe_shared = T.alloc_shared([block_H, pe_dim], dtype)
            KV_shared_0 = T.alloc_shared([block_N, dim], dtype)
            KV_shared_1 = T.alloc_shared([block_N, dim], dtype)
            K_pe_shared_0 = T.alloc_shared([block_N, pe_dim], dtype)
            K_pe_shared_1 = T.alloc_shared([block_N, pe_dim], dtype)
            O_shared = T.alloc_shared([block_H, dim], dtype)
            acc_s0 = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_s1 = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_s_row_max = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)
            p_acc = T.alloc_fragment([block_H, block_N], accum_dtype)
            Q_local = T.alloc_fragment([block_H, dim], dtype)
            Q_pe_local = T.alloc_fragment([block_H, pe_dim], dtype)

            cur_kv_head = 0
            if swizzle > 0:
                T.use_swizzle(swizzle, order=swizzle_order)
            T.annotate_layout(
                {
                    O_shared: tilelang.layout.make_swizzled_layout(O_shared),
                }
            )

            T.copy(Q[bx, by * VALID_BLOCK_H : (by + 1) * VALID_BLOCK_H, :], Q_local)
            T.copy(Q_pe[bx, by * VALID_BLOCK_H : (by + 1) * VALID_BLOCK_H, :], Q_pe_local)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.fill(p_acc, 0)

            with T.attr("default", "async_scope", 1):
                T.copy(KV[bx, 0:block_N, cur_kv_head, :], KV_shared_0)
            with T.attr("default", "async_scope", 1):
                T.copy(K_pe[bx, 0:block_N, cur_kv_head, :], K_pe_shared_0)
            with T.attr("default", "async_scope", 1):
                T.copy(KV[bx, block_N : 2 * block_N, cur_kv_head, :], KV_shared_1)
            with T.attr("default", "async_scope", 1):
                T.copy(K_pe[bx, block_N : 2 * block_N, cur_kv_head, :], K_pe_shared_1)
            T.copy(Q_local, Q_shared)
            T.copy(Q_pe_local, Q_pe_shared)
            T.sync_threads()

            T.clear(acc_s0)
            T.clear(acc_s1)
            T.gemm(
                Q_shared,
                KV_shared_0,
                acc_s0,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullCol,
                k_step=qk_k_step,
            )
            T.gemm(
                Q_pe_shared,
                K_pe_shared_0,
                acc_s0,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullCol,
                k_step=pe_k_step,
            )
            T.gemm(
                Q_shared,
                KV_shared_1,
                acc_s1,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullCol,
                k_step=qk_k_step,
            )
            T.gemm(
                Q_pe_shared,
                K_pe_shared_1,
                acc_s1,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullCol,
                k_step=pe_k_step,
            )
            for i, j in T.Parallel(block_H, block_N):
                acc_s0[i, j] = T.if_then_else(j < seqlen_kv, acc_s0[i, j], -T.infinity(accum_dtype))
                acc_s1[i, j] = T.if_then_else(
                    block_N + j < seqlen_kv, acc_s1[i, j], -T.infinity(accum_dtype)
                )

            T.copy(scores_max, scores_max_prev)
            T.fill(scores_max, -T.infinity(accum_dtype))
            for i, j in T.Parallel(block_H, block_N):
                acc_s_row_max[i, j] = T.max(acc_s0[i, j], acc_s1[i, j])
            T.reduce_max(acc_s_row_max, scores_max, dim=1, clear=False, batch=2)
            for i in T.Parallel(block_H):
                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
            for i, j in T.Parallel(block_H, block_N):
                acc_s0[i, j] = T.exp2(acc_s0[i, j] * scale - scores_max[i] * scale)
                acc_s1[i, j] = T.exp2(acc_s1[i, j] * scale - scores_max[i] * scale)
                p_acc[i, j] = p_acc[i, j] * scores_scale[i] + acc_s0[i, j] + acc_s1[i, j]
            T.copy(acc_s0, S_shared_0)
            T.copy(acc_s1, S_shared_1)
            T.sync_threads()
            rescale(acc_o, scores_scale)
            T.gemm(
                S_shared_0,
                KV_shared_0,
                acc_o,
                policy=T.GemmWarpPolicy.FullCol,
                k_step=pv_k_step,
            )
            T.gemm(
                S_shared_1,
                KV_shared_1,
                acc_o,
                policy=T.GemmWarpPolicy.FullCol,
                k_step=pv_k_step,
            )

            T.reduce_sum(p_acc, logsum, dim=1, batch=2)
            for i, j in T.Parallel(block_H, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.sync_threads()
            T.copy(O_shared, Output[bx, by * VALID_BLOCK_H : (by + 1) * VALID_BLOCK_H, :])

    # Where the no-split path places its block-wide barriers by hand, ThreadSync's
    # automatic pass stays off; only the collective templates keep their own barriers.
    no_split_pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    }
    if async_kv_copy:
        # Without reuse the reduce scratch no longer aliases KV/K_pe/S/O, so the two
        # pre-store barriers guard nothing. The long loop keeps the allocator default:
        # its collective scratch, KV/K_pe and S_shared are all live across the body
        # already, so disabling reuse only raises the shared footprint and risks
        # dropping the second resident block.
        no_split_pass_configs[tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE] = True

    no_split = (
        _mla_no_split_merged
        if merge_two_tiles
        else _mla_no_split_steady
        if long_loop_hand_placed
        else _mla_no_split
    )
    return no_split, (no_split_pass_configs if hand_placed_sync else None)


@functools.lru_cache(maxsize=32)
def _mla_decode_mma_kernel(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim, dtype="float16"):
    sm_scale = (1.0 / (dim + pe_dim)) ** 0.5 * LOG2E
    accum_dtype = "float"
    ptpu = get_device() == "ptpu"

    @tilelang.jit(
        out_idx=[6],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _mla_decode_mma_func(
        block_H,
        block_N,
        num_split,
        num_stages,
        threads=128,
        swizzle=10,
        swizzle_order="row",
        qk_k_step=0,
        pe_k_step=0,
        pv_k_step=0,
    ):
        no_split, no_split_pass_configs = (
            _mla_no_split_ptpu(
                batch,
                heads,
                kv_head_num,
                seqlen_kv,
                dim,
                pe_dim,
                dtype,
                sm_scale,
                block_H,
                block_N,
                num_stages,
                threads,
                swizzle,
                swizzle_order,
                qk_k_step,
                pe_k_step,
                pv_k_step,
            )
            if ptpu and num_split == 1
            else (None, None)
        )
        # A length that does not fill every tile masks the keys past its split's end; loads
        # past the cache read zero.
        kv_per_split = tilelang.cdiv(seqlen_kv, num_split)
        ragged = seqlen_kv % (num_split * block_N) != 0
        may_leave_a_split_empty = (num_split - 1) * kv_per_split >= seqlen_kv

        @T.macro
        def flash_attn(
            Q: T.Tensor([batch, heads, dim], dtype),
            Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
            KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
            K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
            glse: T.Tensor([batch, heads, num_split], accum_dtype),
            Output_partial: T.Tensor([batch, heads, num_split, dim], dtype),
            Output: T.Tensor([batch, heads, dim], dtype),
        ):
            # Head blocks vary fastest, so the blocks reading one request's cache run together.
            with T.Kernel(T.ceildiv(heads, block_H), batch, num_split, threads=threads) as (
                hid,
                bid,
                bz,
            ):
                Q_shared = T.alloc_shared([block_H, dim], dtype)
                Q_pe_shared = T.alloc_shared([block_H, pe_dim], dtype)
                KV_shared = T.alloc_shared([block_N, dim], dtype)
                K_pe_shared = T.alloc_shared([block_N, pe_dim], dtype)
                S_shared = T.alloc_shared([block_H, block_N], dtype)
                O_shared = Q_shared

                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
                sumexp = T.alloc_fragment([block_H], accum_dtype)
                sumexp_i = T.alloc_fragment([block_H], accum_dtype)
                alpha = T.alloc_fragment([block_H], accum_dtype)
                m_i = T.alloc_fragment([block_H], accum_dtype)
                m_i_prev = T.alloc_fragment([block_H], accum_dtype)

                kv_start = kv_per_split * bz
                kv_end = T.min(kv_start + kv_per_split, seqlen_kv)

                T.copy(Q[bid, hid * block_H : (hid + 1) * block_H, :], Q_shared)
                T.copy(Q_pe[bid, hid * block_H : (hid + 1) * block_H, :], Q_pe_shared)
                T.fill(sumexp, 0)
                T.fill(m_i, -(2**30))  # avoid -inf - inf to cause nan
                T.fill(acc_o, 0)

                for k in T.Pipelined(T.ceildiv(kv_per_split, block_N), num_stages=num_stages):
                    tile_start = kv_start + k * block_N
                    T.copy(KV[bid, tile_start : tile_start + block_N, 0, :], KV_shared)
                    T.copy(K_pe[bid, tile_start : tile_start + block_N, 0, :], K_pe_shared)
                    if ragged:
                        for h_i, n_i in T.Parallel(block_H, block_N):
                            acc_s[h_i, n_i] = T.if_then_else(
                                tile_start + n_i < kv_end, 0, -T.infinity(accum_dtype)
                            )
                    else:
                        T.clear(acc_s)
                    T.gemm(
                        Q_shared,
                        KV_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
                    T.gemm(
                        Q_pe_shared,
                        K_pe_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )

                    T.copy(m_i, m_i_prev)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)
                    for h_i in T.Parallel(block_H):
                        alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                    for h_i, n_i in T.Parallel(block_H, block_N):
                        acc_s[h_i, n_i] = T.exp2(acc_s[h_i, n_i] * sm_scale - m_i[h_i] * sm_scale)
                    T.reduce_sum(acc_s, sumexp_i, dim=1)
                    for h_i in T.Parallel(block_H):
                        sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                    for h_i, d_i in T.Parallel(block_H, dim):
                        acc_o[h_i, d_i] *= alpha[h_i]

                    T.copy(acc_s, S_shared)
                    T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)

                if num_split == 1:
                    for h_i, d_i in T.Parallel(block_H, dim):
                        acc_o[h_i, d_i] /= sumexp[h_i]
                    T.copy(acc_o, O_shared)
                    T.copy(O_shared, Output[bid, hid * block_H : (hid + 1) * block_H, :])
                else:
                    if may_leave_a_split_empty:
                        # An empty split has no key to divide by; its zero weight in the combine
                        # needs a finite partial.
                        for h_i, d_i in T.Parallel(block_H, dim):
                            acc_o[h_i, d_i] = T.if_then_else(
                                sumexp[h_i] > 0, acc_o[h_i, d_i] / sumexp[h_i], 0
                            )
                    else:
                        for h_i, d_i in T.Parallel(block_H, dim):
                            acc_o[h_i, d_i] /= sumexp[h_i]
                    for h_i in T.Parallel(block_H):
                        sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale
                    T.copy(sumexp, glse[bid, hid * block_H : (hid + 1) * block_H, bz])
                    T.copy(acc_o, O_shared)
                    T.copy(
                        O_shared, Output_partial[bid, hid * block_H : (hid + 1) * block_H, bz, :]
                    )

        # The split log-sum-exps stay in float32; rounded to the operand type they would
        # shift each split's weight in the combine.
        combine = _split_combine(batch, heads, num_split, dim, dtype, accum_dtype)

        @T.prim_func
        def main(
            Q: T.Tensor([batch, heads, dim], dtype),
            Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
            KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
            K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
            glse: T.Tensor([batch, heads, num_split], accum_dtype),
            Output_partial: T.Tensor([batch, heads, num_split, dim], dtype),
            Output: T.Tensor([batch, heads, dim], dtype),
        ):
            if no_split is not None:
                if no_split_pass_configs is not None:
                    T.annotate_pass_configs(no_split_pass_configs)
                no_split(Q, Q_pe, KV, K_pe, Output)
            else:
                flash_attn(Q, Q_pe, KV, K_pe, glse, Output_partial, Output)
                if num_split > 1:
                    combine(glse, Output_partial, Output)

        return main

    return _mla_decode_mma_func


class MLADecodeWSKernel(Kernel, MLADecodeFwdInterface):
    supported_archs: list[int] = [90]
    # Where both run, a caller's replacement of this key wins over the MMA kernel.
    preferred_over = frozenset({"mla_decode_mma_kernel"})
    _build = staticmethod(_mla_decode_ws_kernel)

    @classmethod
    def refusal(cls, call: MLADecodeCall) -> Optional[str]:
        """Why *call* is outside the shapes the warp-specialized schedule serves."""
        if call.heads_kv != 1:
            return f"serves one KV head, got {call.heads_kv}"
        if call.heads < 64:
            return f"query heads fill at least one 64-row WGMMA tile, got {call.heads}"
        if call.dim % 128 != 0:
            return f"the KV gather walks dim in 128-column steps, got {call.dim}"
        if call.pe_dim != 64:
            return f"the KV tail gather copies exactly 64 columns, got {call.pe_dim}"
        return None

    @classmethod
    def entry_for(cls, call: MLADecodeCall) -> Entry:
        return call, lambda: cls(
            call.batch,
            call.heads,
            call.heads_kv,
            call.seqlen_kv,
            call.dim,
            call.pe_dim,
            call.dtype,
            device_index=call.device.index if call.device is not None else None,
        )

    def __init__(
        self,
        batch,
        heads,
        kv_head_num,
        seqlen_kv,
        dim,
        pe_dim,
        dtype,
        config: Optional[dict] = None,
        *,
        device_index: Optional[int] = None,
    ):
        super().__init__(device_index=device_index)
        self.batch = batch
        self.heads = heads
        self.kv_head_num = kv_head_num
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.pe_dim = pe_dim
        self.dtype = dtype

        self.kernel = self._build(
            self.batch,
            self.heads,
            self.kv_head_num,
            self.seqlen_kv,
            self.dim,
            self.pe_dim,
            self.dtype_str,
        )

        self.init_config(config)

    @property
    def lse_dtype(self) -> torch.dtype:
        return self.dtype

    @property
    def default_config(self) -> dict:
        return {
            "block_H": min(64, self.heads // self.kv_head_num),
            "block_N": 64,
            "num_split": 2,
            "num_stages": 1,
            "threads": 384,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_H = [64, 128]
        block_N = [64, 128]
        num_split = [1, 2, 4]
        num_stages = [1, 2, 3]
        threads = [384]
        _configs = list(itertools.product(block_H, block_N, num_split, num_stages, threads))

        configs = [
            {
                "block_H": c[0],
                "block_N": c[1],
                "num_split": c[2],
                "num_stages": c[3],
                "threads": c[4],
            }
            for c in _configs
        ]
        return configs

    def forward(self, q: torch.Tensor, q_pe: torch.Tensor, k: torch.Tensor, k_pe: torch.Tensor):
        if self.seqlen_kv == 0:
            # No keys: the softmax normalizer is zero, so the program would divide 0 by 0.
            # Attention over no keys is the empty sum, as torch's reference returns.
            return torch.zeros(
                (self.batch, self.heads, self.dim), dtype=self.dtype, device=q.device
            )
        glse = torch.empty(
            (self.batch, self.heads, self.config["num_split"]),
            dtype=self.lse_dtype,
            device=q.device,
        )
        Output_partial = torch.empty(
            (self.batch, self.heads, self.config["num_split"], self.dim),
            dtype=self.dtype,
            device=q.device,
        )
        return self.program(self._compile)(q, q_pe, k, k_pe, glse, Output_partial)

    def _compile(self, cfg: dict):
        """The program ``forward`` launches, compiled from *cfg*."""
        return self.kernel(
            cfg["block_H"], cfg["block_N"], cfg["num_split"], cfg["num_stages"], cfg["threads"]
        )


class MLADecodeMMAKernel(MLADecodeWSKernel):
    """The same decode on MMA, for GPUs without WGMMA."""

    devices = frozenset({"cuda", "ptpu"})
    supported_archs: list[int] = [80, 86, 89]
    preferred_over = frozenset()
    _build = staticmethod(_mla_decode_mma_kernel)
    # Head rows of the default block, widest first, over a 32-key tile and four warps.
    _BLOCK_HS = (32, 16)
    _BLOCK_N = 32
    _THREADS = 128
    _NUM_SPLIT = 2

    @classmethod
    def refusal(cls, call: MLADecodeCall) -> Optional[str]:
        """Why *call* is outside the shapes this schedule serves. An empty cache builds no program;
        otherwise shared memory is read at the narrowest head block."""
        if call.heads_kv != 1:
            return f"serves one KV head, got {call.heads_kv}"
        if call.seqlen_kv == 0:
            return None
        if call.pe_dim % 16 != 0:
            return f"the score GEMM steps pe_dim by 16, got {call.pe_dim}"
        # The output GEMM splits dim over the four warps, 8 columns at a time; TileLang lays a
        # warp's share out only up to 32 columns, or in multiples of 32.
        if call.dim % 32 != 0 or (call.dim > 128 and call.dim % 128 != 0):
            return f"dim must be a multiple of 32 up to 128, or of 128 past it; got {call.dim}"
        need = cls._shared_bytes(
            cls._BLOCK_HS[-1], call.dim, call.pe_dim, call.dtype.itemsize, call.seqlen_kv
        )
        if call.smem_budget and need > call.smem_budget:
            return (
                f"needs {need} bytes of shared memory per block at dim {call.dim}, "
                f"pe_dim {call.pe_dim} in {call.dtype}; the device gives {call.smem_budget}"
            )
        return None

    @property
    def lse_dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def default_config(self) -> dict:
        config = self._default_config_for(
            get_shared_memory_optin(self.device_index),
            self.dim,
            self.pe_dim,
            self.dtype.itemsize,
            self.seqlen_kv,
        )
        if get_device() == "ptpu":
            # The head tile follows the query group instead of padding a small group up to
            # 32 rows; 8 rows is the narrowest run.
            rows = tilelang.math.next_power_of_2(self.heads // self.kv_head_num)
            config["block_H"] = min(config["block_H"], max(8, rows))
            return {
                **config,
                "num_split": 1,
                "num_stages": 2,
                # Block-rasterization panel size and order; swizzle=0 disables T.use_swizzle.
                "swizzle": 10,
                "swizzle_order": "row",
                # MMA operand-load grouping; 0 derives it from the tile shape.
                "qk_k_step": 0,
                "pe_k_step": 0,
                "pv_k_step": 0,
            }
        return {**config, "num_split": self._split_count(config["block_H"])}

    def _compile(self, cfg: dict):
        """The program ``forward`` launches, compiled from *cfg*."""
        ptpu_keys = ("swizzle", "swizzle_order", "qk_k_step", "pe_k_step", "pv_k_step")
        return self.kernel(
            cfg["block_H"],
            cfg["block_N"],
            cfg["num_split"],
            cfg["num_stages"],
            cfg["threads"],
            **{k: cfg[k] for k in ptpu_keys if k in cfg},
        )

    def _split_count(self, block_h: int) -> int:
        minimum_cache = 256
        latent_dim, rope_dim = 512, 64
        target_blocks, maximum_splits = 128, 16
        if (
            self.seqlen_kv < minimum_cache
            or self.dim != latent_dim
            or self.pe_dim != rope_dim
            or get_sm_version(self.device_index) != 89
        ):
            return self._NUM_SPLIT
        # Avoid extra partial outputs once the batch already supplies enough CTAs.
        head_blocks = self.batch * tilelang.cdiv(self.heads, block_h)
        available = max(target_blocks // head_blocks, self._NUM_SPLIT)
        splits = 1 << (available.bit_length() - 1)
        return min(splits, maximum_splits)

    @classmethod
    def _default_config_for(
        cls, budget: int, dim: int, pe_dim: int, itemsize: int, seqlen_kv: int
    ) -> dict:
        """The config this kernel builds at *budget* bytes of shared memory per block."""
        block_h = next(
            (
                h
                for h in cls._BLOCK_HS
                if cls._shared_bytes(h, dim, pe_dim, itemsize, seqlen_kv) <= budget
            ),
            cls._BLOCK_HS[-1],
        )
        return {
            "block_H": block_h,
            "block_N": cls._BLOCK_N,
            "num_split": cls._NUM_SPLIT,
            "num_stages": 1,
            "threads": cls._THREADS,
        }

    @classmethod
    def _shared_bytes(
        cls, block_h: int, dim: int, pe_dim: int, itemsize: int, seqlen_kv: int
    ) -> int:
        """Shared memory of the default program: the query and key rows with their rope parts and
        the scores, plus the two reductions' workspaces of 4 bytes a thread, which TileLang folds
        into freed space when a split runs one key tile (both) or two (one)."""
        n = cls._BLOCK_N
        tiles = tilelang.cdiv(tilelang.cdiv(seqlen_kv, cls._NUM_SPLIT), n)
        workspaces = max(min(tiles, 3) - 1, 0)
        return (
            (block_h + n) * (dim + pe_dim) * itemsize
            + block_h * n * itemsize
            + workspaces * 4 * cls._THREADS
        )

    @property
    def autotune_configs(self) -> list[dict]:
        # Four warps: with eight, or a 16-key tile, the score and output GEMMs split the
        # head rows differently and the row statistics have no common layout.
        budget = get_shared_memory_optin(self.device_index)
        # Query tiles and staged KV tiles coexist; reject only when these alone overflow.
        row_bytes = (self.dim + self.pe_dim) * self.dtype.itemsize
        # Only the PTPU single-split program tunes its block rasterization.
        swizzles = [0, 10, 16] if get_device() == "ptpu" else [None]
        return [
            {"block_H": h, "block_N": n, "num_split": s, "num_stages": st, "threads": 128}
            | ({} if z is None else {"swizzle": z, "swizzle_order": "row"})
            for h, n, s, st, z in itertools.product(
                [16, 32], [32, 64], [1, 2, 4, 8, 16], [1, 2], swizzles
            )
            if (h + n * st) * row_bytes <= budget and (s == 1 or z in (None, 0))
        ]


class MLADecodeSeesawKernel(Kernel, MLADecodeFwdInterface):
    """SM90 staggered seesaw over 64-key tile pairs of each request's cache; see ``mla.seesaw``.

    A request's keys are cut into runs, one block per run and 64-row head block, merged by
    their log-sum-exp; ``num_splits`` picks the run count.
    """

    supported_archs: list[int] = [90]
    preferred_over = frozenset({"mla_decode_kernel"})

    @classmethod
    def refusal(cls, call: MLADecodeCall) -> Optional[str]:
        """Why *call* is outside the shapes this schedule serves."""
        if call.dtype not in ATTENTION_DTYPES:
            return f"serves float16 and bfloat16, got {call.dtype}"
        if call.heads_kv != 1:
            return f"serves one KV head, got {call.heads_kv}"
        if (call.dim, call.pe_dim) != (512, 64):
            # The query, two key tiles and the exchange buffers fill the shared memory of an
            # SM90 block at a 512-wide latent and a 64-wide rope part.
            return (
                f"its buffers are sized for a 512 + 64 latent row, got {call.dim} + {call.pe_dim}"
            )
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: MLADecodeCall) -> Entry:
        index = call.device.index if call.device is not None else None
        return (call.heads, call.dim, call.pe_dim, call.dtype, index), lambda: cls(
            call.heads, call.dim, call.pe_dim, call.dtype, device_index=index
        )

    def __init__(
        self,
        heads: int,
        dim: int,
        pe_dim: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        *,
        device_index: Optional[int] = None,
    ):
        """Build the program for one model's decode geometry; the batch and length vary.

        Args:
            heads: Query heads.
            dim: Width of the latent, the value.
            pe_dim: Width of the rope part of the key.
            dtype: Query, cache and output dtype.
            config: Unused; the schedule's geometry is fixed.
            device_index: The device the program is built for, or None for the current one.
        """
        super().__init__(device_index=device_index)
        self.heads, self.dim, self.pe_dim, self.dtype = heads, dim, pe_dim, dtype
        self.sm_count = get_sm_count(device_index)
        self.init_config(config)
        self.program(self._compile, 1)

    def _compile(self, cfg: dict, splits: int):
        """The program over *splits* runs a request and, past one run, the merge of its runs;
        the schedule reads no configuration."""
        main = _mla_seesaw_dense_kernel(
            self.heads, self.dim + self.pe_dim, self.dim, self.dtype_str, splits
        )
        if splits == 1:
            return CompiledProgram(main), None
        combine = _mla_seesaw_combine_kernel(
            self.heads, 1, self.dim, splits, self.dtype_str, self.dtype_str
        )
        return CompiledProgram(main), CompiledProgram(combine)

    def num_splits(self, batch: int, seqlen: int) -> int:
        """Runs a request's tiles are cut into: the most that fit one wave of blocks, each
        run keeping a pair of tiles for the two warpgroups."""
        # The program's geometry: 64-row head blocks, 64-key tiles.
        blocks = batch * -(-self.heads // 64)
        tiles = -(-seqlen // 64)
        return max(1, min(self.sm_count // blocks, tiles // 2))

    def forward(self, q: torch.Tensor, q_pe: torch.Tensor, k: torch.Tensor, k_pe: torch.Tensor):
        batch, seqlen = k.shape[0], k.shape[1]
        if seqlen == 0:
            # No keys: attention over no keys is the empty sum, as torch's reference returns.
            return q.new_zeros((batch, self.heads, self.dim))
        k, k_pe = k.view(batch, seqlen, self.dim), k_pe.view(batch, seqlen, self.pe_dim)
        main, combine = self.program(self._compile, self.num_splits(batch, seqlen))
        if combine is None:
            return main(q, q_pe, k, k_pe)[0]
        o_part, lse_part = main(q, q_pe, k, k_pe)
        return combine(o_part, lse_part)[0]
