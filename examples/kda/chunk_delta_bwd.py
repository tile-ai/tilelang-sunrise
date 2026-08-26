import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune
from tilelang.utils.device import get_current_device

from test_utils_kda import assert_similar, build_kernel

import torch
import torch.nn.functional as F

DEVICE = get_current_device()

torch.random.manual_seed(42)


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
    Q = torch.randn(B, S, H, DK, dtype=input_dtype).to(DEVICE) * 0.01
    K = F.normalize(torch.randn(B, S, H, DK).float(), dim=-1, p=2).to(input_dtype).to(DEVICE)
    W = torch.randn(B, S, H, DK, dtype=input_dtype).to(DEVICE)
    G = F.logsigmoid(torch.randn(B, S, H, DK, dtype=gate_dtype))
    G = G.reshape(B, S // chunk_size, chunk_size, H, DK).cumsum(2).reshape(B, S, H, DK).to(DEVICE)

    h0 = torch.randn(B, H, DK, DV, dtype=input_dtype).to(DEVICE)
    dht = torch.randn(B, H, DK, DV, dtype=input_dtype).to(DEVICE)
    dO = torch.randn(B, S, H, DV, dtype=input_dtype).to(DEVICE) * 0.01

    dv = torch.randn(B, S, H, DV, dtype=input_dtype).to(DEVICE)
    return Q, K, W, G, h0, dht, dO, dv


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
    BS = S // chunk_size
    dh = torch.empty(B, BS, H, DK, DV, dtype=output_dtype).to(DEVICE)
    dh0 = torch.empty(B, H, DK, DV, dtype=state_dtype).to(DEVICE)
    dv2 = torch.empty(B, S, H, DV, dtype=output_dtype).to(DEVICE)
    return dh, dh0, dv2


def get_configs():
    import itertools

    block_DV = [32, 64, 128]
    threads = [32, 64, 128, 256]
    num_stages = [0, 1, 2, 3, 4]
    _configs = list(itertools.product(block_DV, threads, num_stages))

    configs = [{"block_DV": c[0], "threads": c[1], "num_stages": c[2]} for c in _configs]
    return configs


@autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(out_idx=[-3, -2, -1])
def tilelang_chunk_gated_delta_rule_bwd_dhu(
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
    scale,
    use_gk=True,
    use_initial_state=True,
    use_final_state_gradient=True,
    # kernel config
    block_DV=64,
    threads=256,
    num_stages=0,
):
    block_S = chunk_size
    # Should support cu_seqlen
    BS = S // block_S

    Q_shape = (B, S, H, DK)
    K_shape = (B, S, H, DK)
    W_shape = (B, S, H, DK)
    G_shape = (B, S, H, DK)
    h0_shape = (B, H, DK, DV)
    dht_shape = (B, H, DK, DV)
    dO_shape = (B, S, H, DV)
    dv_shape = (B, S, H, DV)

    dh_shape = (B, BS, H, DK, DV)
    dh0_shape = (B, H, DK, DV)
    dv2_shape = (B, S, H, DV)

    @T.prim_func
    def kernel(
        # Input
        Q: T.Tensor(Q_shape, dtype=input_dtype),
        K: T.Tensor(K_shape, dtype=input_dtype),
        W: T.Tensor(W_shape, dtype=input_dtype),
        GK: T.Tensor(G_shape, dtype=gate_dtype),
        h0: T.Tensor(h0_shape, dtype=input_dtype),
        dht: T.Tensor(dht_shape, dtype=input_dtype),
        dO: T.Tensor(dO_shape, dtype=input_dtype),
        dv: T.Tensor(dv_shape, dtype=input_dtype),
        # Output
        dh: T.Tensor(dh_shape, dtype=output_dtype),
        dh0: T.Tensor(dh0_shape, dtype=state_dtype),
        dv2: T.Tensor(dv2_shape, dtype=output_dtype),
    ):
        with T.Kernel(T.ceildiv(DV, block_DV), B * H, threads=threads) as (bv, bbh):
            bb, bh = bbh // H, bbh % H

            b_dh_shared = T.alloc_shared((DK, block_DV), dtype=output_dtype)
            b_dh_fragment = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
            b_dh_fragment_1 = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
            b_dh_fragment_2 = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
            dv_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            dv_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            dv_fragment_2 = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            dO_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            K_shared = T.alloc_shared((block_S, DK), dtype=input_dtype)

            Q_shared = T.alloc_shared((block_S, DK), dtype=input_dtype)
            W_shared = T.alloc_shared((block_S, DK), dtype=input_dtype)

            GK_last_shared = T.alloc_shared((DK,), dtype=gate_dtype)

            if use_final_state_gradient:
                T.copy(dht[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV], b_dh_shared)
                T.copy(b_dh_shared, b_dh_fragment)
            else:
                T.clear(b_dh_fragment)

            for i_s in T.Pipelined(T.ceildiv(S, block_S), num_stages=num_stages):
                # The gradient should be stored in the reverse order
                i_s_inv = T.ceildiv(S, block_S) - i_s - 1  # reverse indices
                # Store the updated dh
                T.copy(b_dh_fragment, b_dh_shared)
                T.copy(b_dh_shared, dh[bb, i_s_inv, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV])

                # Update dv
                T.copy(K[bb, i_s_inv * block_S : (i_s_inv + 1) * block_S, bh, 0:DK], K_shared)
                T.gemm(K_shared, b_dh_shared, dv_fragment, clear_accum=True)
                T.copy(
                    dv[bb, i_s_inv * block_S : (i_s_inv + 1) * block_S, bh, bv * block_DV : (bv + 1) * block_DV], dv_shared
                )  # copy old dv
                T.copy(dv_shared, dv_fragment_2)
                for i_s2, i_v in T.Parallel(block_S, block_DV):
                    dv_fragment[i_s2, i_v] = dv_fragment[i_s2, i_v] + dv_fragment_2[i_s2, i_v]
                # Store the updated dv
                T.copy(dv_fragment, dv_shared)
                T.copy(dv_shared, dv2[bb, i_s_inv * block_S : (i_s_inv + 1) * block_S, bh, bv * block_DV : (bv + 1) * block_DV])

                # Update dh
                T.copy(Q[bb, i_s_inv * block_S : (i_s_inv + 1) * block_S, bh, 0:DK], Q_shared)  # [block_S, DK]
                T.copy(W[bb, i_s_inv * block_S : (i_s_inv + 1) * block_S, bh, 0:DK], W_shared)  # [block_S, DK]
                T.copy(
                    dO[bb, i_s_inv * block_S : (i_s_inv + 1) * block_S, bh, bv * block_DV : (bv + 1) * block_DV], dO_shared
                )  # [block_S, block_DV]

                if use_gk:
                    last_idx = T.min((i_s_inv + 1) * block_S, S) - 1  # chunk last token gk
                    T.copy(GK[bb, last_idx, bh, :], GK_last_shared)
                    for i_k, i_v in T.Parallel(DK, block_DV):
                        b_dh_fragment[i_k, i_v] *= T.exp2(GK_last_shared[i_k])

                T.gemm(Q_shared, dO_shared, b_dh_fragment_1, transpose_A=True, clear_accum=True)  # [DK, block_DV]

                # dv_shared: [block_S, block_DV]
                T.gemm(W_shared, dv_shared, b_dh_fragment_2, transpose_A=True, clear_accum=True)  # [DK, block_DV]
                for i_k, i_v in T.Parallel(DK, block_DV):
                    b_dh_fragment[i_k, i_v] += b_dh_fragment_1[i_k, i_v] * scale - b_dh_fragment_2[i_k, i_v]

            if use_initial_state:
                T.copy(b_dh_fragment, dh0[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV])

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
    scale,
    use_gk=True,
    use_initial_state=True,
    use_final_state_gradient=True,
    block_DV=64,
    threads=256,
    num_stages=0,
):
    Q, K, W, G, h0, dht, dO, dv = prepare_input(
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

    dh_tilelang, dh0_tilelang, dv2_tilelang = prepare_output(
        B, S, H, DK, DV, chunk_size, getattr(torch, output_dtype), getattr(torch, gate_dtype), getattr(torch, state_dtype)
    )

    print("tilelang running...", flush=True)
    kernel = build_kernel(
        tilelang_chunk_gated_delta_rule_bwd_dhu,
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
        scale,
        use_gk,
        use_initial_state,
        use_final_state_gradient,
        block_DV,
        threads,
        num_stages,
    )
    dh_tilelang, dh0_tilelang, dv2_tilelang = kernel(Q, K, W, G, h0, dht, dO, dv)

    # CPU reference (fp32); bf16 inputs, relaxed tolerance
    NT = S // chunk_size
    Kf = K.cpu().float()
    Wf = W.cpu().float()
    Qf = Q.cpu().float()
    Gf = G.cpu().float()
    dOf = dO.cpu().float()
    dvf = dv.cpu().float()
    dhtf = dht.cpu().float()
    dh_ref = torch.empty(B, NT, H, DK, DV)
    dh0_ref = torch.empty(B, H, DK, DV)
    dv2_ref = torch.empty(B, S, H, DV)
    for b in range(B):
        for hh in range(H):
            b_dh = dhtf[b, hh].clone()
            for c in range(NT - 1, -1, -1):
                sl = slice(c * chunk_size, (c + 1) * chunk_size)
                dh_ref[b, c, hh] = b_dh
                Kc = Kf[b, sl, hh]
                Wc = Wf[b, sl, hh]
                Qc = Qf[b, sl, hh]
                dOc = dOf[b, sl, hh]
                dvc = dvf[b, sl, hh]
                b_dv = Kc @ b_dh + dvc
                dv2_ref[b, sl, hh] = b_dv
                gk_last = Gf[b, c * chunk_size + chunk_size - 1, hh]
                b_dh = b_dh * (2.0**gk_last)[:, None]
                b_dh = b_dh + scale * (Qc.T @ dOc) - (Wc.T @ b_dv)
            dh0_ref[b, hh] = b_dh

    assert_similar(dh_tilelang.cpu().float(), dh_ref, eps=5e-2, name="dh", raise_assert=False)
    assert_similar(dh0_tilelang.cpu().float(), dh0_ref, eps=5e-2, name="dh0", raise_assert=False)
    assert_similar(dv2_tilelang.cpu().float(), dv2_ref, eps=5e-2, name="dv2", raise_assert=False)


def main():
    DK = 32
    run_test(
        B=1,
        S=64,
        H=1,
        DK=DK,
        DV=32,
        input_dtype="bfloat16",
        output_dtype="bfloat16",
        accum_dtype="float32",
        gate_dtype="float32",
        state_dtype="float32",
        chunk_size=64,
        scale=DK**-0.5,
        use_gk=True,
        use_initial_state=True,
        use_final_state_gradient=True,
        block_DV=32,
        threads=128,
        num_stages=0,
    )


if __name__ == "__main__":
    main()
