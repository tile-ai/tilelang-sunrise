import sys  # noqa: F401

import tilelang
import tilelang.language as T
from test_utils import DEVICE, assert_similar

import torch
import torch.nn.functional as F

torch.random.manual_seed(0)
torch.set_printoptions(profile="full")


def prepare_input_fake(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
    state_dtype,
):
    BS = chunk_size
    K = torch.ones(B, S, H, DK, dtype=input_dtype).cuda()
    V = torch.ones(B, S, H, DV, dtype=input_dtype).cuda()
    Beta = torch.ones(B, S, H, dtype=input_dtype).cuda()
    G = torch.ones(B, S, H, dtype=gate_dtype).cuda()
    A = torch.ones(B, S, H, BS, dtype=input_dtype).cuda()
    dw = torch.ones(B, S, H, DK, dtype=input_dtype).cuda()
    du = torch.ones(B, S, H, DV, dtype=input_dtype).cuda()
    return K, V, Beta, G, A, dw, du


def prepare_input(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
    state_dtype,
):
    BS = chunk_size
    K = F.normalize(torch.randn(B, S, H, DK).float(), dim=-1, p=2).to(input_dtype).to(DEVICE)
    V = F.normalize(torch.randn(B, S, H, DV).float(), dim=-1, p=2).to(input_dtype).to(DEVICE)
    Beta = torch.randn(B, S, H, dtype=input_dtype).to(DEVICE)
    G = torch.randn(B, S, H, dtype=gate_dtype).to(DEVICE)
    A = torch.randn(B, S, H, BS, dtype=input_dtype).to(DEVICE)
    dw = torch.randn(B, S, H, DK, dtype=input_dtype).to(DEVICE)
    du = torch.randn(B, S, H, DV, dtype=input_dtype).to(DEVICE)
    return K, V, Beta, G, A, dw, du


def prepare_output(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size,
    output_dtype,
    gate_dtype,
    state_dtype,
):
    dk = torch.empty(B, S, H, DK, dtype=output_dtype).cuda()
    dv = torch.empty(B, S, H, DV, dtype=output_dtype).cuda()
    dbeta = torch.empty(B, S, H, dtype=output_dtype).cuda()
    dg = torch.empty(B, S, H, dtype=gate_dtype).cuda()
    return dk, dv, dbeta, dg


@tilelang.jit(
    out_idx=[-5, -4, -3, -2, -1],
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
)
def tilelang_wy_fast_bwd(
    # task config
    B,
    S,
    H,
    DK,
    DV,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
    state_dtype,
    chunk_size,
    # kernel config
    block_DK=64,
    block_DV=64,
    threads=128,
    num_stages=0,
):
    block_S = chunk_size
    BS = block_S

    K_shape = (B, S, H, DK)
    V_shape = (B, S, H, DV)
    Beta_shape = (B, S, H)
    G_shape = (B, S, H)
    A_shape = (B, S, H, BS)
    dw_shape = (B, S, H, DK)
    du_shape = (B, S, H, DV)

    dk_shape = (B, S, H, DK)
    dv_shape = (B, S, H, DV)
    dbeta_shape = (B, S, H)
    dg_shape = (B, S, H)
    dA_shape = (B, S, H, BS)

    @T.prim_func
    def kernel(
        # input
        K: T.Tensor(K_shape, dtype=input_dtype),
        V: T.Tensor(V_shape, dtype=input_dtype),
        Beta: T.Tensor(Beta_shape, dtype=input_dtype),
        G: T.Tensor(G_shape, dtype=gate_dtype),
        A: T.Tensor(A_shape, dtype=input_dtype),
        dw: T.Tensor(dw_shape, dtype=input_dtype),
        du: T.Tensor(du_shape, dtype=input_dtype),
        # output
        dA: T.Tensor(dA_shape, dtype=input_dtype),
        dk: T.Tensor(dk_shape, dtype=output_dtype),
        dv: T.Tensor(dv_shape, dtype=output_dtype),
        dbeta: T.Tensor(dbeta_shape, dtype=output_dtype),
        dg: T.Tensor(dg_shape, dtype=gate_dtype),
    ):
        with T.Kernel(T.ceildiv(S, block_S), B * H, threads=threads) as (bs, bbh):
            bb, bh = bbh // H, bbh % H

            A_shared = T.alloc_shared((block_S, block_S), dtype=input_dtype)
            K_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            K_shared_beta_g = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            V_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            V_shared_beta = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            Beta_shared = T.alloc_shared((block_S,), dtype=input_dtype)
            G_shared = T.alloc_shared((block_S,), dtype=gate_dtype)
            G_shared_exp = T.alloc_shared((block_S,), dtype=gate_dtype)
            dw_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            du_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)

            dA_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            dk_fragment = T.alloc_fragment((block_S, block_DK), dtype=accum_dtype)
            dk_fragment_beta_g = T.alloc_fragment((block_S, block_DK), dtype=accum_dtype)
            dv_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            dv_fragment_beta = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            dbeta_fragment_k = T.alloc_fragment((block_S,), dtype=accum_dtype)
            dbeta_fragment_v = T.alloc_fragment((block_S,), dtype=accum_dtype)
            dbeta_fragment_reduce_tmpk = T.alloc_fragment((block_S, block_DK), dtype=accum_dtype)
            dbeta_fragment_reduce_tmpv = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            dg_fragment = T.alloc_fragment((block_S,), dtype=gate_dtype)
            dg_fragment_reduce_tmp = T.alloc_fragment((block_S, block_DK), dtype=gate_dtype)

            T.use_swizzle(10)

            T.clear(dA_fragment)
            T.clear(dk_fragment)
            T.clear(dk_fragment_beta_g)
            T.clear(dv_fragment)
            T.clear(dv_fragment_beta)
            T.clear(dbeta_fragment_k)
            T.clear(dbeta_fragment_v)
            T.clear(dg_fragment)

            T.copy(A[bb, bs * block_S : (bs + 1) * block_S, bh, :], A_shared)
            for i_s in T.Parallel(block_S):
                Beta_shared[i_s] = Beta[bb, bs * block_S + i_s, bh]
                G_shared[i_s] = G[bb, bs * block_S + i_s, bh]
                G_shared_exp[i_s] = T.exp(G[bb, bs * block_S + i_s, bh])

            # Update dk
            for i_k in T.Pipelined(T.ceildiv(DK, block_DK), num_stages=num_stages):
                T.copy(K[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK], K_shared)
                for i_s, i_k2 in T.Parallel(block_S, block_DK):
                    K_shared_beta_g[i_s, i_k2] = K_shared[i_s, i_k2] * Beta_shared[i_s] * G_shared_exp[i_s]
                T.copy(dw[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK], dw_shared)
                T.gemm(dw_shared, K_shared_beta_g, dA_fragment, transpose_B=True)
                T.gemm(A_shared, dw_shared, dk_fragment_beta_g, clear_accum=True, transpose_A=True)
                for i_s, i_k2 in T.Parallel(block_S, block_DK):
                    dk_fragment[i_s, i_k2] = dk_fragment_beta_g[i_s, i_k2] * Beta_shared[i_s] * G_shared_exp[i_s]
                # for i_s, i_k2 in T.Parallel(block_S, block_DK):
                #     dbeta_fragment[i_s] = dbeta_fragment[i_s] + dk_fragment_beta_g[i_s, i_k2] * K_shared[i_s, i_k2] * G_shared_exp[i_s]
                for i_s, i_k2 in T.Parallel(block_S, block_DK):
                    dbeta_fragment_reduce_tmpk[i_s, i_k2] = dk_fragment_beta_g[i_s, i_k2] * K_shared[i_s, i_k2] * G_shared_exp[i_s]
                T.reduce_sum(dbeta_fragment_reduce_tmpk, dbeta_fragment_k, dim=1, clear=False)

                # for i_s, i_k2 in T.Parallel(block_S, block_DK):
                #     dg_fragment[i_s] = dg_fragment[i_s] + dk_fragment_beta_g[i_s, i_k2] * K_shared[i_s, i_k2] * G_shared_exp[i_s] * Beta_shared[i_s]
                for i_s, i_k2 in T.Parallel(block_S, block_DK):
                    dg_fragment_reduce_tmp[i_s, i_k2] = (
                        dk_fragment_beta_g[i_s, i_k2] * K_shared[i_s, i_k2] * G_shared_exp[i_s] * Beta_shared[i_s]
                    )
                T.reduce_sum(dg_fragment_reduce_tmp, dg_fragment, dim=1, clear=False)

                # correct dk
                T.copy(dk_fragment, dk[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK])

            # Update dv
            for i_v in T.Pipelined(T.ceildiv(DV, block_DV), num_stages=num_stages):
                T.copy(V[bb, bs * block_S : (bs + 1) * block_S, bh, i_v * block_DV : (i_v + 1) * block_DV], V_shared)
                for i_s, i_v2 in T.Parallel(block_S, block_DV):
                    V_shared_beta[i_s, i_v2] = V_shared[i_s, i_v2] * Beta_shared[i_s]
                T.copy(du[bb, bs * block_S : (bs + 1) * block_S, bh, i_v * block_DV : (i_v + 1) * block_DV], du_shared)
                T.gemm(du_shared, V_shared_beta, dA_fragment, transpose_B=True)
                T.gemm(A_shared, du_shared, dv_fragment_beta, clear_accum=True, transpose_A=True)
                for i_s, i_v2 in T.Parallel(block_S, block_DV):
                    dv_fragment[i_s, i_v2] = dv_fragment_beta[i_s, i_v2] * Beta_shared[i_s]
                # for i_s, i_v2 in T.Parallel(block_S, block_DV):
                #     dbeta_fragment[i_s] = dbeta_fragment[i_s] + dv_fragment_beta[i_s, i_v2] * V_shared[i_s, i_v2]
                for i_s, i_v2 in T.Parallel(block_S, block_DV):
                    dbeta_fragment_reduce_tmpv[i_s, i_v2] = dv_fragment_beta[i_s, i_v2] * V_shared[i_s, i_v2]
                T.reduce_sum(dbeta_fragment_reduce_tmpv, dbeta_fragment_v, dim=1, clear=False)

                T.copy(dv_fragment, dv[bb, bs * block_S : (bs + 1) * block_S, bh, i_v * block_DV : (i_v + 1) * block_DV])

            # Temporary store dbeta, dg and dA
            for i_s in T.Parallel(block_S):
                dbeta[bb, bs * block_S + i_s, bh] = dbeta_fragment_k[i_s] + dbeta_fragment_v[i_s]
                dg[bb, bs * block_S + i_s, bh] = dg_fragment[i_s]
            # correct dA
            T.copy(dA_fragment, dA[bb, bs * block_S : (bs + 1) * block_S, bh, :])

    return kernel


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def tilelang_wy_fast_bwd_split(
    # task config
    B,
    S,
    H,
    DK,
    DV,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
    state_dtype,
    chunk_size,
    # kernel config
    block_DK=64,
    block_DV=64,
    threads=128,
    num_stages=0,
):
    block_S = chunk_size
    BS = block_S

    K_shape = (B, S, H, DK)
    V_shape = (B, S, H, DV)
    Beta_shape = (B, S, H)
    G_shape = (B, S, H)
    A_shape = (B, S, H, BS)
    dw_shape = (B, S, H, DK)
    du_shape = (B, S, H, DV)

    dk_shape = (B, S, H, DK)
    dv_shape = (B, S, H, DV)
    dbeta_shape = (B, S, H)
    dA_shape = (B, S, H, BS)

    @T.prim_func
    def kernel(
        # input
        K: T.Tensor(K_shape, dtype=input_dtype),
        V: T.Tensor(V_shape, dtype=input_dtype),
        Beta: T.Tensor(Beta_shape, dtype=input_dtype),
        G: T.Tensor(G_shape, dtype=gate_dtype),
        A: T.Tensor(A_shape, dtype=input_dtype),
        dw: T.Tensor(dw_shape, dtype=input_dtype),
        du: T.Tensor(du_shape, dtype=input_dtype),
        dA: T.Tensor(dA_shape, dtype=input_dtype),
        dk: T.Tensor(dk_shape, dtype=output_dtype),
        dv: T.Tensor(dv_shape, dtype=output_dtype),
        dbeta_k: T.Tensor(dbeta_shape, dtype=output_dtype),
        dg_A_positive: T.Tensor(dA_shape, dtype=gate_dtype),
        dg_A_negative: T.Tensor(dA_shape, dtype=gate_dtype),
    ):
        with T.Kernel(T.ceildiv(S, block_S), B * H, threads=threads) as (bs, bbh):
            bb, bh = bbh // H, bbh % H

            A_shared = T.alloc_shared((block_S, block_S), dtype=input_dtype)
            A_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            K_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            K_shared_beta = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            dA_shared = T.alloc_shared((block_S, block_S), dtype=input_dtype)
            dA_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            dA_A_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            dk_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            dk_shared_beta = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            dk_fragment = T.alloc_fragment((block_S, block_DK), dtype=accum_dtype)
            dk_fragment_beta = T.alloc_fragment((block_S, block_DK), dtype=accum_dtype)
            dk_fragment_beta_shared = T.alloc_shared((block_S, block_DK), dtype=accum_dtype)
            Beta_shared = T.alloc_shared((block_S,), dtype=input_dtype)
            dbeta_fragment_k = T.alloc_fragment((block_S,), dtype=accum_dtype)
            G_shared = T.alloc_shared((block_S,), dtype=gate_dtype)
            G_shared_exp = T.alloc_shared((block_S,), dtype=gate_dtype)

            T.clear(dbeta_fragment_k)

            T.copy(A[bb, bs * block_S : (bs + 1) * block_S, bh, :], A_shared)
            for i_s in T.Parallel(block_S):
                Beta_shared[i_s] = Beta[bb, bs * block_S + i_s, bh]
                G_shared[i_s] = G[bb, bs * block_S + i_s, bh]
            for i_s in T.Parallel(block_S):
                G_shared_exp[i_s] = T.exp(G_shared[i_s])

            # Load intermediate results
            # for i_s in T.Parallel(block_S):
            # dbeta_fragment[i_s] = dbeta[bb, bs * block_S + i_s, bh]
            # dg_fragment[i_s] = dg[bb, bs * block_S + i_s, bh]
            T.copy(dA[bb, bs * block_S : (bs + 1) * block_S, bh, :], dA_shared)
            # T.copy(dA_shared, dA[bb, bs * block_S:(bs + 1) * block_S, bh, :])

            # Update dA
            T.copy(dA_shared, dA_fragment)

            for i_s1, i_s2 in T.Parallel(block_S, block_S):
                if i_s1 <= i_s2:
                    dA_fragment[i_s1, i_s2] = 0
            T.copy(dA_fragment, dA_shared)
            T.gemm(dA_shared, A_shared, dA_fragment, clear_accum=True, transpose_B=True)
            T.copy(dA_fragment, dA_shared)
            T.gemm(A_shared, dA_shared, dA_fragment, clear_accum=True, transpose_A=True)
            for i_s1, i_s2 in T.Parallel(block_S, block_S):
                dA_fragment[i_s1, i_s2] = T.if_then_else(
                    i_s1 <= i_s2,
                    0,
                    -dA_fragment[i_s1, i_s2],
                )

            for i_s1, i_s2 in T.Parallel(block_S, block_S):
                dA_fragment[i_s1, i_s2] = T.if_then_else(
                    G[bb, bs * block_S + i_s1, bh] - G[bb, bs * block_S + i_s2, bh] <= 0,
                    dA_fragment[i_s1, i_s2] * T.exp(G[bb, bs * block_S + i_s1, bh] - G[bb, bs * block_S + i_s2, bh]),
                    0,
                )
            T.copy(dA_fragment, dA_shared)

            # acceptable dA diff
            # T.copy(dA_fragment, dA[bb, bs * block_S:(bs + 1) * block_S, bh, :])

            # Update dk using previous dk
            T.clear(A_fragment)
            for i_k in T.Pipelined(T.ceildiv(DK, block_DK), num_stages=num_stages):
                T.copy(K[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK], K_shared)
                T.copy(dk[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK], dk_shared)
                T.copy(dk_shared, dk_fragment)
                for i_s, i_k2 in T.Parallel(block_S, block_DK):
                    K_shared_beta[i_s, i_k2] = K_shared[i_s, i_k2] * Beta_shared[i_s]
                T.gemm(K_shared_beta, K_shared, A_fragment, transpose_B=True)
                T.gemm(dA_shared, K_shared, dk_fragment_beta, clear_accum=True)
                T.copy(dk_fragment_beta, dk_fragment_beta_shared)
                T.sync_threads()
                for i_s in T.Parallel(block_S):
                    for i_k2 in T.serial(block_DK):
                        dbeta_fragment_k[i_s] += dk_fragment_beta_shared[i_s, i_k2] * K_shared[i_s, i_k2]
                T.gemm(dA_shared, K_shared_beta, dk_fragment, transpose_A=True)
                for i_s, i_k2 in T.Parallel(block_S, block_DK):
                    dk_shared_beta[i_s, i_k2] = dk_fragment_beta[i_s, i_k2] * Beta_shared[i_s]
                for i_s, i_k2 in T.Parallel(block_S, block_DK):
                    dk_fragment[i_s, i_k2] = dk_fragment[i_s, i_k2] + dk_shared_beta[i_s, i_k2]
                T.copy(dk_fragment, dk[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK])

            # Update dg and dbeta
            T.copy(A_fragment, A_shared)
            for i_s1, i_s2 in T.Parallel(block_S, block_S):
                dA_A_fragment[i_s1, i_s2] = dA_fragment[i_s1, i_s2] * A_fragment[i_s1, i_s2]

            for i_s1, i_s2 in T.Parallel(block_S, block_S):
                dg_A_positive[bb, bs * block_S + i_s1, bh, i_s2] = dA_A_fragment[i_s1, i_s2]
                dg_A_negative[bb, bs * block_S + i_s2, bh, i_s1] = dA_A_fragment[i_s1, i_s2]

            for i_s in T.Parallel(block_S):
                dbeta_k[bb, bs * block_S + i_s, bh] = dbeta_fragment_k[i_s]

    return kernel


def run_test(
    B,
    S,
    H,
    DK,
    DV,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
    state_dtype,
    chunk_size,
    block_DK=64,
    block_DV=64,
    threads=128,
    num_stages=0,
):
    K, V, Beta, G, A, dw, du = prepare_input(
        B,
        S,
        H,
        DK,
        DV,
        chunk_size,
        getattr(torch, input_dtype),
        getattr(torch, output_dtype),
        getattr(torch, accum_dtype),
        getattr(torch, gate_dtype),
        getattr(torch, state_dtype),
    )
    BS = chunk_size
    dA_tilelang = torch.empty(B, S, H, BS, dtype=getattr(torch, input_dtype)).to(DEVICE)
    dbeta_tilelang_k = torch.empty(B, S, H, dtype=getattr(torch, output_dtype)).to(DEVICE)
    dg_tilelang_A_positive = torch.empty(B, S, H, BS, dtype=getattr(torch, gate_dtype)).to(DEVICE)
    dg_tilelang_A_negative = torch.empty(B, S, H, BS, dtype=getattr(torch, gate_dtype)).to(DEVICE)

    kernel = tilelang_wy_fast_bwd(
        B,
        S,
        H,
        DK,
        DV,
        input_dtype,
        output_dtype,
        accum_dtype,
        gate_dtype,
        state_dtype,
        chunk_size,
        block_DK,
        block_DV,
        threads,
        num_stages,
    )
    dA_tilelang, dk_tilelang, dv_tilelang, dbeta_tilelang, dg_tilelang = kernel(K, V, Beta, G, A, dw, du)
    kernel_split = tilelang_wy_fast_bwd_split(
        B,
        S,
        H,
        DK,
        DV,
        input_dtype,
        output_dtype,
        accum_dtype,
        gate_dtype,
        state_dtype,
        chunk_size,
        block_DK,
        block_DV,
        threads,
        num_stages,
    )
    kernel_split(
        K, V, Beta, G, A, dw, du, dA_tilelang, dk_tilelang, dv_tilelang, dbeta_tilelang_k, dg_tilelang_A_positive, dg_tilelang_A_negative
    )

    dbeta_tilelang = dbeta_tilelang_k.cpu() + dbeta_tilelang.cpu()
    dg_tilelang = dg_tilelang.cpu() + dg_tilelang_A_positive.cpu().sum(dim=-1) - dg_tilelang_A_negative.cpu().sum(dim=-1)

    # CPU reference (fp32)
    NT = S // chunk_size
    CS = chunk_size
    Kc, Vc, Betac = K.cpu().float(), V.cpu().float(), Beta.cpu().float()
    Gc, Ac, dwc, duc = G.cpu().float(), A.cpu().float(), dw.cpu().float(), du.cpu().float()
    o_dk = torch.zeros(B, S, H, DK)
    o_dv = torch.zeros(B, S, H, DV)
    o_db = torch.zeros(B, S, H)
    o_dg = torch.zeros(B, S, H)
    strict = torch.arange(CS)[:, None] > torch.arange(CS)[None, :]
    for b in range(B):
        for hh in range(H):
            for c in range(NT):
                sl = slice(c * CS, (c + 1) * CS)
                k = Kc[b, sl, hh]
                v = Vc[b, sl, hh]
                dw_ = dwc[b, sl, hh]
                du_ = duc[b, sl, hh]
                beta = Betac[b, sl, hh]
                g = Gc[b, sl, hh]
                g_exp = torch.exp(g)
                bA = Ac[b, sl, hh].transpose(0, 1)

                dbeta = torch.zeros(CS)
                dA = torch.zeros(CS, CS)
                dg = torch.zeros(CS)

                k_beta_g = k * beta[:, None] * g_exp[:, None]
                dA = dA + dw_ @ k_beta_g.transpose(0, 1)
                dk_beta_g = bA @ dw_
                dk = dk_beta_g * beta[:, None] * g_exp[:, None]
                dbeta = dbeta + (dk_beta_g * k * g_exp[:, None]).sum(dim=1)
                dg = dg + (dk_beta_g * k * g_exp[:, None] * beta[:, None]).sum(dim=1)

                v_beta = v * beta[:, None]
                dA = dA + du_ @ v_beta.transpose(0, 1)
                dv_beta = bA @ du_
                dv = dv_beta * beta[:, None]
                dbeta = dbeta + (dv_beta * v).sum(dim=1)

                dA = torch.where(strict, dA, torch.zeros_like(dA))
                dA = dA @ bA
                dA = bA @ dA
                dA = torch.where(strict, -dA * torch.exp(g[:, None] - g[None, :]), torch.zeros_like(dA))

                A_acc = (k * beta[:, None]) @ k.transpose(0, 1)
                dk_beta = dA @ k
                dbeta = dbeta + (dk_beta * k).sum(dim=1)
                dk = dk + dA.transpose(0, 1) @ (k * beta[:, None])
                dk = dk + dk_beta * beta[:, None]

                dA_A = dA * A_acc
                dg = dg + dA_A.sum(dim=1) - dA_A.sum(dim=0)

                o_dk[b, sl, hh] = dk
                o_dv[b, sl, hh] = dv
                o_db[b, sl, hh] = dbeta
                o_dg[b, sl, hh] = dg

    # dA has no FLA reference; finiteness check only
    try:
        assert torch.isfinite(dA_tilelang.cpu()).all()
        print("dA isfinite passed")
    except Exception as e:
        print(f"dA isfinite failed: {e}")
    assert_similar(o_dk, dk_tilelang.cpu().float(), 1e-2, "torch-tilelang", data="dk", raise_assert=False)
    assert_similar(o_dv, dv_tilelang.cpu().float(), 1e-2, "torch-tilelang", data="dv", raise_assert=False)
    assert_similar(o_db, dbeta_tilelang.float(), 1e-2, "torch-tilelang", data="dbeta", raise_assert=False)
    assert_similar(o_dg, dg_tilelang.float(), 1e-2, "torch-tilelang", data="dg", raise_assert=False)


def main():
    DK = 32
    DV = 32
    run_test(
        B=1,
        S=64,
        H=1,
        DK=DK,
        DV=DV,
        input_dtype=T.bfloat16,
        output_dtype=T.bfloat16,
        accum_dtype=T.float32,
        gate_dtype=T.float32,
        state_dtype=T.float32,
        chunk_size=64,
        block_DK=32,
        block_DV=32,
        threads=128,
        num_stages=0,
    )


if __name__ == "__main__":
    main()
