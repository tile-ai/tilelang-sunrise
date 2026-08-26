# ruff: noqa
import torch
import tilelang
import tilelang.language as T
import tilelang.testing
import argparse
from tilelang.profiler import do_bench
from tilelang.autotuner import set_autotune_inputs, autotune
from tilelang.utils.device import get_current_device

import torch
from varlen_utils import generate_random_padding_mask, generate_qkv
import itertools


def get_configs():
    iter_params = dict(block_M=[64], block_N=[64], num_stages=[1], threads=[128])
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]


@autotune(configs=get_configs())
@tilelang.jit(
    out_idx=[6],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashattn(batch_size, UQ, UKV, heads, dim, is_causal, block_M=64, block_N=64, num_stages=1, threads=128):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    q_shape = [UQ, heads, dim]
    k_shape = [UKV, heads, dim]
    v_shape = [UKV, heads, dim]
    o_shape = [UQ, heads, dim]

    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q_unpad: T.Tensor(q_shape, dtype),
        K_unpad: T.Tensor(k_shape, dtype),
        V_unpad: T.Tensor(v_shape, dtype),
        cu_seqlens_q: T.Tensor([batch_size + 1], T.int32),
        cu_seqlens_k: T.Tensor([batch_size + 1], T.int32),
        max_seqlen_q: T.int32,
        Output_unpad: T.Tensor(o_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(max_seqlen_q, block_M), heads, batch_size, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_shared([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            batch_idx = bz
            head_idx = by

            q_start_idx = cu_seqlens_q[batch_idx]
            kv_start_idx = cu_seqlens_k[batch_idx]
            q_end_idx = cu_seqlens_q[batch_idx + 1]
            kv_end_idx = cu_seqlens_k[batch_idx + 1]

            q_current_seqlen = q_end_idx - q_start_idx
            kv_current_seqlen = kv_end_idx - kv_start_idx

            T.copy(
                Q_unpad[q_start_idx + bx * block_M : q_start_idx + bx * block_M + block_M, head_idx, :], Q_shared
            )  # OOB positions will be handled below

            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            offset = kv_current_seqlen - q_current_seqlen  # always align on the right
            loop_range = (
                T.min(T.ceildiv(offset + (bx + 1) * block_M, block_N), T.ceildiv(kv_current_seqlen, block_N))
                if is_causal
                else T.ceildiv(kv_current_seqlen, block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                # Q * K
                T.copy(
                    K_unpad[kv_start_idx + k * block_N : kv_start_idx + k * block_N + block_N, head_idx, :], K_shared
                )  # OOB positions will be handled below
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            (bx * block_M + i + offset < k * block_N + j)
                            or (bx * block_M + i >= q_current_seqlen or k * block_N + j >= kv_current_seqlen),
                            -1e9,
                            0,
                        )
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            (bx * block_M + i >= q_current_seqlen or k * block_N + j >= kv_current_seqlen), -1e9, 0
                        )

                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                # Softmax
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                # To do causal softmax, we need to set the scores_max to 0 if it is -inf
                # This process is called Check_inf in FlashAttention3 code, and it only need to be done
                # in the first ceil_div(kBlockM, kBlockN) steps.
                # for i in T.Parallel(block_M):
                #     scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_M, block_N):
                    # Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                    # max * log_2(e)) This allows the compiler to use the ffma
                    # instruction instead of fadd and fmul separately.
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)

                # Rescale
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]

                # V * softmax(Q * K)
                T.copy(
                    V_unpad[kv_start_idx + k * block_N : kv_start_idx + k * block_N + block_N, head_idx, :], V_shared
                )  # OOB positions' weights are 0

                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_M, dim):
                # When sq > skv, some tokens can see nothing
                acc_o[i, j] = 0 if is_causal and bx * block_M + i + offset < 0 else acc_o[i, j] / logsum[i]

            T.copy(acc_o, O_shared)
            for i, d in T.Parallel(block_M, dim):
                if bx * block_M + i < q_current_seqlen:
                    Output_unpad[q_start_idx + bx * block_M + i, head_idx, d] = O_shared[i, d]

    return main


def ref_flash_attn_varlen(q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, causal=False):
    """Pure-torch reference for flash_attn_varlen_func.

    Computes softmax(Q K^T / sqrt(dim)) V per (batch, head) with right-aligned
    causal masking and per-batch key padding, matching the kernel's semantics.
    Runs on CPU because PTPU torch does not reliably support einsum/softmax.
    """
    q_cpu = q_unpad.cpu().float()
    k_cpu = k_unpad.cpu().float()
    v_cpu = v_unpad.cpu().float()
    cu_q = cu_seqlens_q.cpu().long()
    cu_k = cu_seqlens_k.cpu().long()

    heads = q_cpu.shape[1]
    dim = q_cpu.shape[2]
    scale = dim**-0.5
    batch = cu_q.numel() - 1

    outs = []
    for b in range(batch):
        q_start, q_end = int(cu_q[b]), int(cu_q[b + 1])
        k_start, k_end = int(cu_k[b]), int(cu_k[b + 1])
        q_len, k_len = q_end - q_start, k_end - k_start
        if q_len == 0:
            continue

        q_b = q_cpu[q_start:q_end]  # [q_len, heads, dim]
        k_b = k_cpu[k_start:k_end]  # [k_len, heads, dim]
        v_b = v_cpu[k_start:k_end]  # [k_len, heads, dim]

        scores = torch.einsum("qhd,khd->qhk", q_b, k_b) * scale

        if causal:
            offset = k_len - q_len  # right-aligned
            i_idx = torch.arange(q_len).view(q_len, 1, 1)
            j_idx = torch.arange(k_len).view(1, 1, k_len)
            mask = (i_idx + offset) >= j_idx  # [q_len, 1, k_len]
            scores = scores.masked_fill(~mask, float("-inf"))

        probs = torch.softmax(scores, dim=-1)
        # Rows fully masked (q_len > k_len and the token sees no key) produce
        # NaN softmax; the kernel writes 0 for those rows.
        probs = torch.nan_to_num(probs, nan=0.0)
        outs.append(torch.einsum("qhk,khd->qhd", probs, v_b))

    out = torch.cat(outs, dim=0) if outs else q_cpu.new_zeros(0, heads, dim)
    return out.to(q_unpad.dtype)


def main(batch: int = 8, heads: int = 64, seq_len: int = 2048, dim: int = 128, causal: bool = False, tune: bool = False):
    flops_per_matmul = 2.0 * batch * heads * seq_len * seq_len * dim
    total_flops = 2 * flops_per_matmul

    tilelang.testing.set_random_seed(0)

    if causal:
        total_flops *= 0.5

    dtype = torch.float16
    device = get_current_device()

    q = torch.randn(batch, seq_len, heads, dim, dtype=dtype, requires_grad=True).to(device)
    k = torch.randn(batch, seq_len, heads, dim, dtype=dtype, requires_grad=True).to(device)
    v = torch.randn(batch, seq_len, heads, dim, dtype=dtype, requires_grad=True).to(device)

    query_padding_mask = generate_random_padding_mask(seq_len, batch, device, mode="random")
    key_padding_mask = generate_random_padding_mask(seq_len, batch, device, mode="random")
    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        _,
        _,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)

    UQ = q_unpad.shape[0]  # unpadded query length
    UKV = k_unpad.shape[0]  # unpadded query key length

    if tune:
        with set_autotune_inputs(q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q):
            kernel = flashattn(batch, UQ, UKV, heads, dim, causal)
    else:
        kernel = flashattn(batch, UQ, UKV, heads, dim, causal, block_M=64, block_N=64, num_stages=1, threads=128)
        # NOTE: (128, 128, 2or3, 256) is recommended for Hopper

    out_unpad = kernel(q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q)

    try:
        import flash_attn

        _HAS_FLASH_ATTN = True
    except ImportError:
        _HAS_FLASH_ATTN = False

    # Pure-torch reference (no flash_attn dependency); computed on CPU because
    # PTPU torch does not reliably support einsum/softmax.
    ref_out_unpad = ref_flash_attn_varlen(q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, causal=causal)
    torch.testing.assert_close(out_unpad.cpu(), ref_out_unpad, rtol=1e-2, atol=1e-2)
    print("All checks passed.✅")

    # benchmark
    t = do_bench(
        lambda: kernel(q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q),
        warmup=5,
        rep=20,
        backend="event",
        return_mode="median",
    )
    print(f"Tilelang time: {t} ms")
    print(f"Tilelang: {total_flops / t * 1e-9} TFlops")
    if _HAS_FLASH_ATTN:
        t = do_bench(
            lambda: flash_attn.flash_attn_varlen_func(
                q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, 0.0, causal=causal
            ),
            warmup=5,
            rep=20,
            backend="event",
            return_mode="median",
        )
        print(f"FA2 time: {t} ms")
        print(f"FA2: {total_flops / t * 1e-9} TFlops")


def run_regression_perf(batch: int = 8, heads: int = 64, seq_len: int = 2048, dim: int = 128, causal: bool = False):
    flops_per_matmul = 2.0 * batch * heads * seq_len * seq_len * dim
    total_flops = 2 * flops_per_matmul
    tilelang.testing.set_random_seed(0)
    if causal:
        total_flops *= 0.5
    dtype = torch.float16
    device = get_current_device()
    q = torch.randn(batch, seq_len, heads, dim, dtype=dtype, requires_grad=True).to(device)
    k = torch.randn(batch, seq_len, heads, dim, dtype=dtype, requires_grad=True).to(device)
    v = torch.randn(batch, seq_len, heads, dim, dtype=dtype, requires_grad=True).to(device)
    query_padding_mask = generate_random_padding_mask(seq_len, batch, device, mode="random")
    key_padding_mask = generate_random_padding_mask(seq_len, batch, device, mode="random")
    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)
    UQ = q_unpad.shape[0]
    UKV = k_unpad.shape[0]
    kernel = flashattn(batch, UQ, UKV, heads, dim, causal, block_M=128, block_N=128, num_stages=2, threads=256)

    from tilelang.profiler import do_bench

    def run_kernel_only():
        kernel(q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q)

    return do_bench(run_kernel_only, warmup=5, rep=20, backend="event", return_mode="median")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument("--heads", type=int, default=64, help="heads")
    parser.add_argument("--seq_len", type=int, default=2048, help="sequence length")
    parser.add_argument("--dim", type=int, default=128, help="dim")
    parser.add_argument("--is_causal", action="store_true", default=False, help="causal attention")
    parser.add_argument("--tune", action="store_true", default=False, help="tune the kernel")

    args = parser.parse_args()
    main(args.batch, args.heads, args.seq_len, args.dim, args.is_causal, args.tune)
