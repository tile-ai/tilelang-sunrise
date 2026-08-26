import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune
from tilelang.utils.device import get_current_device
import torch
import torch.nn.functional as F
from test_utils_kda import assert_similar, build_kernel

DEVICE = get_current_device()

torch.random.manual_seed(42)


def prepare_input(
    B,
    S,
    H,
    DK,
    chunk_size,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
):
    q = torch.randn(B, S, H, DK, dtype=input_dtype).to(DEVICE)
    k = torch.randn(B, S, H, DK, dtype=input_dtype).to(DEVICE)
    beta = torch.randn(B, S, H, dtype=input_dtype).to(DEVICE)
    gk = F.logsigmoid(torch.randn(B, S, H, DK, dtype=gate_dtype))
    gk = gk.reshape(B, S // chunk_size, chunk_size, H, DK).cumsum(2).reshape(B, S, H, DK).to(DEVICE)
    return q, k, gk, beta


def prepare_output(
    B,
    S,
    H,
    chunk_size,
    sub_chunk_size,
    output_dtype,
):
    Aqk = torch.empty(B, S, H, chunk_size, dtype=output_dtype).cuda()
    Akk = torch.empty(B, S, H, sub_chunk_size, dtype=output_dtype).cuda()
    return Aqk, Akk


def get_configs():
    import itertools

    block_H = [1, 2, 4, 8]
    threads = [128, 256]
    num_stages = [0, 1, 2, 3]
    _configs = list(itertools.product(block_H, threads, num_stages))

    configs = [{"block_H": c[0], "threads": c[1], "num_stages": c[2]} for c in _configs]
    return configs


@autotune(configs=get_configs(), warmup=3, rep=5)
@tilelang.jit(out_idx=[-2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def tilelang_chunk_kda_fwd_intra_token_parallel(
    B,
    S,
    H,
    DK,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
    chunk_size,
    sub_chunk_size,
    block_H=1,
    threads=32,
    num_stages=1,
):
    CS = chunk_size
    SCS = sub_chunk_size
    Q_shape = (B, S, H, DK)
    K_shape = (B, S, H, DK)
    GK_shape = (B, S, H, DK)
    Beta_shape = (B, S, H)
    Aqk_shape = (B, S, H, CS)
    Akk_shape = (B, S, H, SCS)

    @T.prim_func
    def kernel(
        Q: T.Tensor(Q_shape, dtype=input_dtype),
        K: T.Tensor(K_shape, dtype=input_dtype),
        GK: T.Tensor(GK_shape, dtype=gate_dtype),
        Beta: T.Tensor(Beta_shape, dtype=input_dtype),
        Aqk: T.Tensor(Aqk_shape, dtype=output_dtype),
        Akk: T.Tensor(Akk_shape, dtype=output_dtype),
    ):
        with T.Kernel(B * S, T.ceildiv(H, block_H), threads=threads) as (bbs, bh):  # block_index_bs, block_index_dh
            bb, bs = bbs // S, bbs % S
            i_c = bs // CS  # indice chunk
            i_s = (bs % CS) // SCS  # indice subchunk
            i_tc = i_c * CS
            i_ts = i_tc + i_s * SCS
            loops = bs + 1 - i_ts

            Q_i_shared = T.alloc_shared((block_H, DK), dtype=input_dtype)
            K_i_shared = T.alloc_shared((block_H, DK), dtype=input_dtype)
            GK_i_shared = T.alloc_shared((block_H, DK), dtype=gate_dtype)
            Beta_shared = T.alloc_shared(
                (block_H,),
                dtype=input_dtype,
            )
            K_j_shared = T.alloc_shared((block_H, DK), dtype=input_dtype)
            GK_j_shared = T.alloc_shared((block_H, DK), dtype=gate_dtype)
            Aqk_shared = T.alloc_shared((block_H, DK), dtype=accum_dtype)
            Akk_shared = T.alloc_shared((block_H, DK), dtype=accum_dtype)
            Sum_Aqk_shared = T.alloc_shared((block_H, CS), dtype=output_dtype)
            Sum_Akk_shared = T.alloc_shared((block_H, SCS), dtype=output_dtype)

            Q_i_fragment = T.alloc_fragment(
                (block_H, DK),
                dtype=input_dtype,
            )
            K_i_fragment = T.alloc_fragment(
                (block_H, DK),
                dtype=input_dtype,
            )
            K_j_fragment = T.alloc_fragment(
                (block_H, DK),
                dtype=accum_dtype,
            )

            Sum_Aqk_fragment = T.alloc_fragment(
                (block_H,),
                dtype=accum_dtype,
            )
            Sum_Akk_fragment = T.alloc_fragment(
                (block_H,),
                dtype=accum_dtype,
            )

            T.copy(Q[bb, bs, bh * block_H : (bh + 1) * block_H, :], Q_i_shared)
            T.copy(K[bb, bs, bh * block_H : (bh + 1) * block_H, :], K_i_shared)
            T.copy(GK[bb, bs, bh * block_H : (bh + 1) * block_H, :], GK_i_shared)  # TMA

            T.disable_warp_group_reg_alloc()
            for i_h in T.Parallel(block_H):  # cannot use TMA
                Beta_shared[i_h] = Beta[bb, bs, bh * block_H + i_h]

            for i_h, i_k in T.Parallel(block_H, DK):
                K_i_fragment[i_h, i_k] = K_i_shared[i_h, i_k] * Beta_shared[i_h]
                Q_i_fragment[i_h, i_k] = Q_i_shared[i_h, i_k]

            T.clear(Sum_Akk_shared)
            T.clear(Sum_Aqk_shared)

            for d in T.Pipelined(loops, num_stages=num_stages):
                j = d + i_ts
                T.copy(K[bb, j, bh * block_H : (bh + 1) * block_H, :], K_j_shared)
                T.copy(GK[bb, j, bh * block_H : (bh + 1) * block_H, :], GK_j_shared)
                # T.copy(K_j_shared, K_j_fragment)
                for i_h, i_k in T.Parallel(block_H, DK):
                    K_j_fragment[i_h, i_k] = K_j_shared[i_h, i_k] * T.exp2(GK_i_shared[i_h, i_k] - GK_j_shared[i_h, i_k])
                    Aqk_shared[i_h, i_k] = Q_i_fragment[i_h, i_k] * K_j_fragment[i_h, i_k]
                    Akk_shared[i_h, i_k] = K_i_fragment[i_h, i_k] * K_j_fragment[i_h, i_k]

                T.reduce_sum(Aqk_shared, Sum_Aqk_fragment, dim=-1, clear=True)
                T.reduce_sum(Akk_shared, Sum_Akk_fragment, dim=-1, clear=True)

                T.copy(Sum_Aqk_fragment, Sum_Aqk_shared[:, j % CS])

                if j < bs:
                    T.copy(Sum_Akk_fragment, Sum_Akk_shared[:, d])

            T.copy(Sum_Aqk_shared, Aqk[bb, bs, bh * block_H : (bh + 1) * block_H, :])
            T.copy(Sum_Akk_shared, Akk[bb, bs, bh * block_H : (bh + 1) * block_H, :])

    return kernel


def run_test(
    B,
    S,
    H,
    DK,
    scale,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
    chunk_size,
    sub_chunk_size,
):
    q, k, gk, beta = prepare_input(
        B,
        S,
        H,
        DK,
        chunk_size,
        getattr(torch, input_dtype),
        getattr(torch, output_dtype),
        getattr(torch, accum_dtype),
        getattr(torch, gate_dtype),
    )
    kernel = build_kernel(
        tilelang_chunk_kda_fwd_intra_token_parallel,
        DEVICE,
        B,
        S,
        H,
        DK,
        input_dtype,
        output_dtype,
        accum_dtype,
        gate_dtype,
        chunk_size,
        sub_chunk_size,
        1,
        128,
        0,
    )
    Aqk_tilelang, Akk_tilelang = kernel(
        q,
        k,
        gk,
        beta,
    )
    # CPU reference (fp32): per-token intra-subchunk Aqk / Akk
    # kernel does not apply scale; keep example scale (=1.0) consistent
    cs, BC = chunk_size, sub_chunk_size
    qf, kf, gf, betaf = (t.cpu().float() for t in (q, k, gk, beta))
    Aqk_ref = torch.zeros(B, S, H, cs)
    Akk_ref = torch.zeros(B, S, H, BC)
    for b in range(B):
        for hh in range(H):
            for bs in range(S):
                i_tc = (bs // cs) * cs
                i_ts = i_tc + ((bs % cs) // BC) * BC
                gi = gf[b, bs, hh]
                for j in range(i_ts, bs + 1):
                    kg = kf[b, j, hh] * torch.exp2(gi - gf[b, j, hh])
                    Aqk_ref[b, bs, hh, j - i_tc] = scale * (qf[b, bs, hh] * kg).sum()
                    if j < bs:
                        Akk_ref[b, bs, hh, j - i_ts] = (betaf[b, bs, hh] * kf[b, bs, hh] * kg).sum()

    assert_similar(Aqk_tilelang.cpu().float(), Aqk_ref, eps=1e-2, name="Aqk", raise_assert=False)
    assert_similar(Akk_tilelang.cpu().float(), Akk_ref, eps=1e-2, name="Akk", raise_assert=False)


def main():
    run_test(
        B=1,
        S=64,
        H=1,
        DK=32,
        scale=1.0,
        input_dtype="bfloat16",
        output_dtype="bfloat16",
        accum_dtype="float32",
        gate_dtype="float32",
        chunk_size=64,
        sub_chunk_size=16,
    )


if __name__ == "__main__":
    main()
