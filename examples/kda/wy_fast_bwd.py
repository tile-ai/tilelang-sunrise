import sys  # noqa: F401

import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune
from tilelang.utils.device import get_current_device
from test_utils_kda import assert_similar, build_kernel

import torch

DEVICE = get_current_device()

torch.random.manual_seed(0)
torch.set_printoptions(profile="full")


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
    K = torch.randn(B, S, H, DK, dtype=input_dtype).to(DEVICE)
    V = torch.randn(B, S, H, DV, dtype=input_dtype).to(DEVICE)
    Beta = torch.randn(B, S, H, dtype=input_dtype).to(DEVICE)
    GK = torch.randn(B, S, H, DK, dtype=gate_dtype).to(DEVICE)
    A = torch.randn(B, S, H, BS, dtype=input_dtype).to(DEVICE)
    dw = torch.randn(B, S, H, DK, dtype=input_dtype).to(DEVICE)
    dv = torch.randn(B, S, H, DV, dtype=input_dtype).to(DEVICE)
    dk = torch.randn(B, S, H, DK, dtype=input_dtype).to(DEVICE)
    dg = torch.randn(B, S, H, DK, dtype=gate_dtype).to(DEVICE)

    return K, V, Beta, GK, A, dw, dv, dk, dg


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
    dk = torch.empty(B, S, H, DK, dtype=output_dtype).to(DEVICE)
    dv = torch.empty(B, S, H, DV, dtype=output_dtype).to(DEVICE)
    dbeta = torch.empty(B, S, H, dtype=output_dtype).to(DEVICE)
    dg = torch.empty(B, S, H, DK, dtype=gate_dtype).to(DEVICE)
    dA = torch.empty(B, S, H, DK, dtype=output_dtype).to(DEVICE)
    return dk, dv, dbeta, dg, dA


def get_configs():
    import itertools

    block_DK = [32, 64, 128]
    block_DV = [32, 64, 128]
    threads = [32, 64, 128, 256]
    num_stages = [0, 1, 2, 3]
    _configs = list(itertools.product(block_DK, block_DV, threads, num_stages))

    configs = [{"block_DK": c[0], "block_DV": c[1], "threads": c[2], "num_stages": c[3]} for c in _configs]
    return configs


@autotune(configs=get_configs(), warmup=3, rep=5)
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
    G_shape = (B, S, H, DK)
    A_shape = (B, S, H, BS)
    dw_shape = (B, S, H, DK)
    du_shape = (B, S, H, DV)

    dk_shape = (B, S, H, DK)
    dv_shape = (B, S, H, DV)
    dbeta_shape = (B, S, H)
    dg_shape = (B, S, H, DK)
    dA_shape = (B, S, H, BS)

    @T.prim_func
    def kernel(
        # input
        K: T.Tensor(K_shape, dtype=input_dtype),
        V: T.Tensor(V_shape, dtype=input_dtype),
        Beta: T.Tensor(Beta_shape, dtype=input_dtype),
        GK: T.Tensor(G_shape, dtype=gate_dtype),
        A: T.Tensor(A_shape, dtype=input_dtype),
        dw: T.Tensor(dw_shape, dtype=input_dtype),
        du: T.Tensor(du_shape, dtype=input_dtype),
        dk: T.Tensor(dk_shape, dtype=input_dtype),
        dg: T.Tensor(dg_shape, dtype=gate_dtype),
        # output
        dA: T.Tensor(dA_shape, dtype=input_dtype),
        dk2: T.Tensor(dk_shape, dtype=output_dtype),
        dv: T.Tensor(dv_shape, dtype=output_dtype),
        dbeta: T.Tensor(dbeta_shape, dtype=output_dtype),
        dg2: T.Tensor(dg_shape, dtype=gate_dtype),
    ):
        with T.Kernel(T.ceildiv(S, block_S), B * H, threads=threads) as (bs, bbh):
            bb, bh = bbh // H, bbh % H

            A_shared = T.alloc_shared((block_S, block_S), dtype=input_dtype)
            A_rhs_shared = T.alloc_shared((block_S, block_S), dtype=input_dtype)
            K_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            K_shared_beta_g = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            V_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            V_shared_beta = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            Beta_shared = T.alloc_shared((block_S,), dtype=input_dtype)
            GK_shared = T.alloc_shared((block_S, block_DK), dtype=gate_dtype)
            GK_shared_exp = T.alloc_shared((block_S, block_DK), dtype=gate_dtype)
            dw_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            dw_rhs_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            du_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            du_rhs_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)

            dk_old_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            dg_old_shared = T.alloc_shared((block_S, block_DK), dtype=gate_dtype)
            dA_shared = T.alloc_shared((block_S, block_S), dtype=input_dtype)
            dA_rhs_shared = T.alloc_shared((block_S, block_S), dtype=input_dtype)

            dA_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            dA_fragment_tmp1 = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            dA_fragment_tmp2 = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)

            dk_fragment = T.alloc_fragment((block_S, block_DK), dtype=accum_dtype)
            dk_fragment_beta_g = T.alloc_fragment((block_S, block_DK), dtype=accum_dtype)
            dv_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            dv_fragment_beta = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            dbeta_fragment = T.alloc_fragment((block_S,), dtype=accum_dtype)
            dbeta_fragment_reduce_tmpk = T.alloc_fragment((block_S, block_DK), dtype=accum_dtype)
            dbeta_fragment_reduce_tmpv = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            dg_fragment = T.alloc_fragment((block_S, block_DK), dtype=gate_dtype)

            T.clear(dA_fragment)
            T.clear(dk_fragment)
            T.clear(dk_fragment_beta_g)
            T.clear(dv_fragment)
            T.clear(dv_fragment_beta)
            T.clear(dbeta_fragment)
            T.clear(dg_fragment)

            T.copy(A[bb, bs * block_S : (bs + 1) * block_S, bh, :], A_shared)  # load A
            T.copy(A[bb, bs * block_S : (bs + 1) * block_S, bh, :], A_rhs_shared)
            T.copy(Beta[bb, bs * block_S : (bs + 1) * block_S, bh], Beta_shared)

            # Update dk
            for i_k in T.Pipelined(T.ceildiv(DK, block_DK), num_stages=num_stages):
                T.copy(K[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK], K_shared)
                T.copy(dk[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK], dk_old_shared)
                T.copy(dg[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK], dg_old_shared)
                T.copy(GK[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK], GK_shared)

                for i_s, i_k2 in T.Parallel(block_S, block_DK):
                    GK_shared_exp[i_s, i_k2] = T.exp2(GK_shared[i_s, i_k2])
                    K_shared_beta_g[i_s, i_k2] = K_shared[i_s, i_k2] * Beta_shared[i_s] * GK_shared_exp[i_s, i_k2]

                T.copy(dw[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK], dw_shared)
                T.copy(dw[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK], dw_rhs_shared)
                T.gemm(dw_shared, K_shared_beta_g, dA_fragment, transpose_B=True, clear_accum=False)
                T.gemm(A_shared, dw_rhs_shared, dk_fragment_beta_g, transpose_A=True, clear_accum=True)

                for i_s, i_k2 in T.Parallel(block_S, block_DK):
                    dk_fragment[i_s, i_k2] = (
                        dk_fragment_beta_g[i_s, i_k2] * GK_shared_exp[i_s, i_k2] * Beta_shared[i_s] + dk_old_shared[i_s, i_k2]
                    )

                for i_s, i_k2 in T.Parallel(block_S, block_DK):
                    dbeta_fragment_reduce_tmpk[i_s, i_k2] = dk_fragment_beta_g[i_s, i_k2] * K_shared[i_s, i_k2] * GK_shared_exp[i_s, i_k2]
                T.reduce_sum(dbeta_fragment_reduce_tmpk, dbeta_fragment, dim=1, clear=False)

                for i_s, i_k2 in T.Parallel(block_S, block_DK):
                    dg_fragment[i_s, i_k2] = dk_fragment_beta_g[i_s, i_k2] * K_shared_beta_g[i_s, i_k2] + dg_old_shared[i_s, i_k2]

                # correct dk, dg
                T.copy(dk_fragment, dk2[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK])
                T.copy(dg_fragment, dg2[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK])

            # Update dv
            for i_v in T.Pipelined(T.ceildiv(DV, block_DV), num_stages=num_stages):
                T.copy(V[bb, bs * block_S : (bs + 1) * block_S, bh, i_v * block_DV : (i_v + 1) * block_DV], V_shared)
                for i_s, i_v2 in T.Parallel(block_S, block_DV):
                    V_shared_beta[i_s, i_v2] = V_shared[i_s, i_v2] * Beta_shared[i_s]
                T.copy(du[bb, bs * block_S : (bs + 1) * block_S, bh, i_v * block_DV : (i_v + 1) * block_DV], du_shared)
                T.copy(du[bb, bs * block_S : (bs + 1) * block_S, bh, i_v * block_DV : (i_v + 1) * block_DV], du_rhs_shared)
                T.gemm(du_shared, V_shared_beta, dA_fragment, transpose_B=True)
                T.gemm(A_shared, du_rhs_shared, dv_fragment_beta, clear_accum=True, transpose_A=True)
                for i_s, i_v2 in T.Parallel(block_S, block_DV):
                    dv_fragment[i_s, i_v2] = dv_fragment_beta[i_s, i_v2] * Beta_shared[i_s]

                for i_s, i_v2 in T.Parallel(block_S, block_DV):
                    dbeta_fragment_reduce_tmpv[i_s, i_v2] = dv_fragment_beta[i_s, i_v2] * V_shared[i_s, i_v2]
                T.reduce_sum(dbeta_fragment_reduce_tmpv, dbeta_fragment, dim=1, clear=False)

                T.copy(dv_fragment, dv[bb, bs * block_S : (bs + 1) * block_S, bh, i_v * block_DV : (i_v + 1) * block_DV])

            T.copy(dbeta_fragment, dbeta[bb, bs * block_S : (bs + 1) * block_S, bh])

            # correct dA
            for i_s1, i_s2 in T.Parallel(block_S, block_S):
                dA_shared[i_s1, i_s2] = T.if_then_else(i_s1 > i_s2, dA_fragment[i_s1, i_s2], 0.0)
            T.gemm(dA_shared, A_rhs_shared, dA_fragment_tmp1, transpose_B=True, clear_accum=True)
            T.copy(dA_fragment_tmp1, dA_rhs_shared)
            T.gemm(A_shared, dA_rhs_shared, dA_fragment_tmp2, transpose_A=True, clear_accum=True)
            for i_s1, i_s2 in T.Parallel(block_S, block_S):
                dA_fragment_tmp2[i_s1, i_s2] = T.if_then_else(i_s1 > i_s2, -dA_fragment_tmp2[i_s1, i_s2], 0.0)
            T.copy(dA_fragment_tmp2, dA[bb, bs * block_S : (bs + 1) * block_S, bh, :])

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
    K, V, Beta, GK, A, dw, dv, dk, dg = prepare_input(
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

    dk_tilelang, dv_tilelang, dbeta_tilelang, dg_tilelang, dA_tilelang = prepare_output(
        B, S, H, DK, DV, chunk_size, getattr(torch, output_dtype), getattr(torch, gate_dtype), getattr(torch, state_dtype)
    )

    kernel = build_kernel(
        tilelang_wy_fast_bwd,
        DEVICE,
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
    dA_tilelang, dk_tilelang, dv_tilelang, dbeta_tilelang, dg_tilelang = kernel(K, V, Beta, GK, A, dw, dv, dk, dg)

    # CPU reference (fp32): per-chunk wy_repr backward; example dv is the kernel's du
    BT = chunk_size
    Kf, Vf, Bf, Gf, Af, dwf, duf, dkof, dgof = (t.cpu().float() for t in (K, V, Beta, GK, A, dw, dv, dk, dg))
    o_dk = torch.empty(B, S, H, DK)
    o_dv = torch.empty(B, S, H, DV)
    o_db = torch.empty(B, S, H)
    o_dg = torch.empty(B, S, H, DK)
    o_dA = torch.empty(B, S, H, BT)
    for b in range(B):
        for hh in range(H):
            for c0 in range(0, S, BT):
                sl = slice(c0, c0 + BT)
                Ac = Af[b, sl, hh]
                beta = Bf[b, sl, hh]
                gk_exp = torch.exp2(Gf[b, sl, hh])
                k_ = Kf[b, sl, hh]
                v_ = Vf[b, sl, hh]
                kbg = k_ * beta[:, None] * gk_exp
                dw_ = dwf[b, sl, hh]
                du_ = duf[b, sl, hh]
                dA_acc = dw_ @ kbg.T
                dkbg = Ac.T @ dw_
                dk_ = dkbg * gk_exp * beta[:, None] + dkof[b, sl, hh]
                db = (dkbg * k_ * gk_exp).sum(dim=1)
                dg_ = kbg * dkbg + dgof[b, sl, hh]
                vb = v_ * beta[:, None]
                dA_acc = dA_acc + du_ @ vb.T
                dvb = Ac.T @ du_
                dv_ = dvb * beta[:, None]
                db = db + (dvb * v_).sum(dim=1)
                i = torch.arange(BT)
                mA = i[:, None] > i[None, :]
                dA_l = torch.where(mA, dA_acc, torch.zeros_like(dA_acc))
                dA_f2 = Ac.T @ (dA_l @ Ac.T)
                dA_f2 = torch.where(mA, -dA_f2, torch.zeros_like(dA_f2))
                o_dk[b, sl, hh] = dk_
                o_dg[b, sl, hh] = dg_
                o_db[b, sl, hh] = db
                o_dv[b, sl, hh] = dv_
                o_dA[b, sl, hh] = dA_f2

    try:
        torch.testing.assert_close(dk_tilelang.cpu().float(), o_dk, rtol=1e-2, atol=1e-2)
        print("dk passed")
    except Exception as e:
        print(f"dk failed: {e}")
    try:
        torch.testing.assert_close(dv_tilelang.cpu().float(), o_dv, rtol=1e-2, atol=1e-2)
        print("dv passed")
    except Exception as e:
        print(f"dv failed: {e}")
    try:
        torch.testing.assert_close(dbeta_tilelang.cpu().float(), o_db, rtol=1e-2, atol=1e-2)
        print("dbeta passed")
    except Exception as e:
        print(f"dbeta failed: {e}")
    try:
        torch.testing.assert_close(dg_tilelang.cpu().float(), o_dg, rtol=1e-2, atol=1e-2)
        print("dg passed")
    except Exception as e:
        print(f"dg failed: {e}")
    assert_similar(dA_tilelang.cpu().float(), o_dA, eps=1e-2, name="dA", raise_assert=False)


def main():
    DK = 32
    DV = 32
    run_test(
        B=1,
        S=64,
        H=1,
        DK=DK,
        DV=DV,
        input_dtype=T.float32,
        output_dtype=T.float32,
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
