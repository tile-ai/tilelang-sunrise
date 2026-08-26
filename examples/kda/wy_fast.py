import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune
from tilelang.utils.device import get_current_device
import torch
from test_utils_kda import build_kernel

DEVICE = get_current_device()

torch.random.manual_seed(42)


def prepare_input(B, S, H, DK, DV, chunk_size, input_dtype, output_dtype, gate_dtype=torch.float32):
    BS = chunk_size
    K = torch.randn(B, S, H, DK, dtype=input_dtype).to(DEVICE)
    V = torch.randn(B, S, H, DV, dtype=input_dtype).to(DEVICE)
    Beta = torch.randn(B, S, H, dtype=input_dtype).to(DEVICE)
    G = torch.randn(B, S, H, DK, dtype=gate_dtype).to(DEVICE)
    A = torch.randn(B, S, H, BS, dtype=input_dtype).to(DEVICE)
    return K, V, Beta, G, A


def prepare_output(
    B,
    S,
    H,
    DK,
    DV,
    use_qg,
    use_kg,
    output_dtype,
):
    W = torch.empty(B, S, H, DK, dtype=output_dtype).cuda()
    U = torch.empty(B, S, H, DV, dtype=output_dtype).cuda()
    QG = torch.empty(B, S, H, DK, dtype=output_dtype).cuda() if use_qg else None
    KG = torch.empty(B, S, H, DK, dtype=output_dtype).cuda() if use_kg else None
    return W, U, QG, KG


def get_configs():
    import itertools

    block_DK = [32, 64]
    block_DV = [32, 64]
    threads = [64, 128, 256]
    num_stages = [0, 1, 2, 3, 4]
    _configs = list(itertools.product(block_DK, block_DV, threads, num_stages))
    configs = [{"block_DK": c[0], "block_DV": c[1], "threads": c[2], "num_stages": c[3]} for c in _configs]
    return configs


@autotune(configs=get_configs(), warmup=3, rep=5)
@tilelang.jit(out_idx=[-4, -3, -2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def tilelang_recompute_w_u_fwd(
    # task config
    B,
    S,
    H,
    DK,
    DV,
    input_dtype,
    output_dtype,
    gate_dtype,
    accum_dtype,
    chunk_size,
    use_qg,
    use_kg,
    # kernel config
    block_S=64,
    block_DK=32,
    block_DV=32,
    threads=128,
    num_stages=0,
):
    K_shape = (B, S, H, DK)
    V_shape = (B, S, H, DV)
    Beta_shape = (B, S, H)
    assert chunk_size == block_S, "chunk_size must be equal to block_S"
    BS = chunk_size
    G_shape = (B, S, H, DK)
    A_shape = (B, S, H, BS)

    @T.prim_func
    def kernel(
        K: T.Tensor(K_shape, dtype=input_dtype),
        V: T.Tensor(V_shape, dtype=input_dtype),
        Beta: T.Tensor(Beta_shape, dtype=input_dtype),
        G: T.Tensor(G_shape, dtype=gate_dtype),
        A: T.Tensor(A_shape, dtype=output_dtype),
        W: T.Tensor(K_shape, dtype=output_dtype),
        U: T.Tensor(V_shape, dtype=output_dtype),
        QG: T.Tensor(K_shape, dtype=output_dtype),
        KG: T.Tensor(K_shape, dtype=output_dtype),
    ):
        with T.Kernel(T.ceildiv(S, block_S), B * H, threads=threads) as (bs, bbh):
            bb, bh = bbh // H, bbh % H
            Beta_shared = T.alloc_shared((block_S,), dtype=input_dtype, scope="shared")
            K_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            V_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            G_shared = T.alloc_shared((block_S, block_DK), dtype=gate_dtype)
            A_shared = T.alloc_shared((block_S, block_S), dtype=output_dtype)
            W_fragment = T.alloc_fragment((block_S, block_DK), dtype=accum_dtype)
            U_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            W_shared = T.alloc_shared((block_S, block_DK), dtype=output_dtype)
            U_shared = T.alloc_shared((block_S, block_DV), dtype=output_dtype)
            W_Beta_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            U_Beta_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            G_n_shared = T.alloc_shared(block_DK, dtype=gate_dtype)
            KG_shared = T.alloc_shared((block_S, block_DK), dtype=output_dtype)

            T.disable_warp_group_reg_alloc()  # TMA to transfer the last dimension of the data should be 16 times
            for i_s in T.Parallel(block_S):
                Beta_shared[i_s] = Beta[bb, bs * block_S + i_s, bh]

            T.copy(A[bb, bs * block_S : (bs + 1) * block_S, bh, :], A_shared)

            for i_v in T.Pipelined(T.ceildiv(DV, block_DV), num_stages=num_stages):
                T.copy(V[bb, bs * block_S : (bs + 1) * block_S, bh, i_v * block_DV : (i_v + 1) * block_DV], V_shared)
                for i_s, i_v2 in T.Parallel(block_S, block_DV):
                    U_Beta_shared[i_s, i_v2] = V_shared[i_s, i_v2] * Beta_shared[i_s]
                T.gemm(A_shared, U_Beta_shared, U_fragment, clear_accum=True)
                T.copy(U_fragment, U_shared)
                T.copy(U_shared, U[bb, bs * block_S : (bs + 1) * block_S, bh, i_v * block_DV : (i_v + 1) * block_DV])

            for i_k in T.Pipelined(T.ceildiv(DK, block_DK), num_stages=num_stages):
                T.copy(K[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK], K_shared)
                T.copy(G[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK], G_shared)
                for i_s, i_k2 in T.Parallel(block_S, block_DK):
                    W_Beta_shared[i_s, i_k2] = K_shared[i_s, i_k2] * Beta_shared[i_s] * T.exp2(G_shared[i_s, i_k2])
                T.gemm(A_shared, W_Beta_shared, W_fragment, clear_accum=True)
                T.copy(W_fragment, W_shared)
                T.copy(W_shared, W[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK])

                if use_kg:
                    T.copy(G[bb, (bs + 1) * block_S - 1, bh, i_k * block_DK : (i_k + 1) * block_DK], G_n_shared)

                    for i_s3, i_k3 in T.Parallel(block_S, block_DK):
                        KG_shared[i_s3, i_k3] = K_shared[i_s3, i_k3] * T.exp2(G_n_shared[i_k3] - G_shared[i_s3, i_k3])
                    T.copy(KG_shared, KG[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK])

    return kernel


def run_test(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size,
    input_dtype,
    output_dtype,
    gate_dtype,
    accum_dtype,
    block_DK,
    block_DV,
    threads,
    num_stages,
):
    use_qg = False
    use_kg = True
    K, V, Beta, G, A = prepare_input(
        B, S, H, DK, DV, chunk_size, getattr(torch, input_dtype), getattr(torch, output_dtype), gate_dtype=getattr(torch, gate_dtype)
    )
    chunks = S // chunk_size
    k_ref = K.cpu().float().reshape(B, chunks, chunk_size, H, DK).permute(0, 1, 3, 2, 4)
    v_ref = V.cpu().float().reshape(B, chunks, chunk_size, H, DV).permute(0, 1, 3, 2, 4)
    beta_ref = Beta.cpu().float().reshape(B, chunks, chunk_size, H).permute(0, 1, 3, 2)
    g_ref = G.cpu().float().reshape(B, chunks, chunk_size, H, DK).permute(0, 1, 3, 2, 4)
    a_ref = A.cpu().float().reshape(B, chunks, chunk_size, H, chunk_size).permute(0, 1, 3, 2, 4)
    input_dtype_torch = getattr(torch, input_dtype)
    w_input = (k_ref * beta_ref.unsqueeze(-1)).to(input_dtype_torch).float()
    w_input = (w_input * torch.exp2(g_ref)).to(input_dtype_torch).float()
    u_input = (v_ref * beta_ref.unsqueeze(-1)).to(input_dtype_torch).float()
    W_ref = torch.matmul(a_ref, w_input)
    U_ref = torch.matmul(a_ref, u_input)
    KG_ref = k_ref * torch.exp2(g_ref[:, :, :, -1:, :] - g_ref)
    output_dtype_torch = getattr(torch, output_dtype)
    W_ref = W_ref.permute(0, 1, 3, 2, 4).reshape(B, S, H, DK).to(output_dtype_torch)
    U_ref = U_ref.permute(0, 1, 3, 2, 4).reshape(B, S, H, DV).to(output_dtype_torch)
    KG_ref = KG_ref.permute(0, 1, 3, 2, 4).reshape(B, S, H, DK).to(output_dtype_torch)

    block_S = chunk_size
    kernel = build_kernel(
        tilelang_recompute_w_u_fwd,
        DEVICE,
        B,
        S,
        H,
        DK,
        DV,
        input_dtype,
        output_dtype,
        gate_dtype,
        accum_dtype,
        chunk_size,
        use_qg,
        use_kg,
        block_S=block_S,
        block_DK=block_DK,
        block_DV=block_DV,
        threads=threads,
        num_stages=num_stages,
    )
    W_tilelang, U_tilelang, _, KG_tilelang = kernel(K, V, Beta, G, A)

    try:
        torch.testing.assert_close(W_tilelang.cpu(), W_ref, rtol=1e-2, atol=1e-2)
        print("W passed")
    except Exception as e:
        print(f"W failed: {e}")
    try:
        torch.testing.assert_close(U_tilelang.cpu(), U_ref, rtol=1e-2, atol=1e-2)
        print("U passed")
    except Exception as e:
        print(f"U failed: {e}")
    try:
        torch.testing.assert_close(KG_tilelang.cpu(), KG_ref, rtol=1e-2, atol=1e-2)
        print("KG passed")
    except Exception as e:
        print(f"KG failed: {e}")


def main():
    run_test(
        B=1,
        S=64,
        H=1,
        DK=32,
        DV=32,
        chunk_size=64,
        input_dtype=T.bfloat16,
        output_dtype=T.bfloat16,
        gate_dtype=T.float32,
        accum_dtype=T.float32,
        block_DK=32,
        block_DV=32,
        threads=128,
        num_stages=0,
    )


if __name__ == "__main__":
    main()
