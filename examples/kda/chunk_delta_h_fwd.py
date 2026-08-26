import sys  # noqa: F401
import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune
from tilelang.utils.device import get_current_device

import torch
import torch.nn.functional as F

from test_utils_kda import build_kernel

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
):
    K = F.normalize(torch.randn(B, S, H, DK).float(), dim=-1, p=2).to(input_dtype).to(DEVICE)
    W = F.normalize(torch.randn(B, S, H, DK).float(), dim=-1, p=2).to(input_dtype).to(DEVICE)
    U = F.normalize(torch.randn(B, S, H, DV).float(), dim=-1, p=2).to(input_dtype).to(DEVICE)
    G = F.logsigmoid(torch.randn(B, S, H, DK, dtype=gate_dtype))
    G = G.reshape(B, S // chunk_size, chunk_size, H, DK).cumsum(2).reshape(B, S, H, DK).to(DEVICE)
    initial_state = torch.randn(B, H, DK, DV, dtype=input_dtype).to(DEVICE)
    return K, W, U, G, initial_state


def prepare_output(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size,
    output_dtype,
    state_dtype,
):
    BS = (S + chunk_size - 1) // chunk_size  # ceildiv to match kernel iteration
    h = torch.empty(B, BS, H, DK, DV, dtype=output_dtype).cuda()
    final_state = torch.empty(B, H, DK, DV, dtype=state_dtype).cuda()
    V_new = torch.empty(B, S, H, DV, dtype=output_dtype).cuda()
    return h, final_state, V_new


def get_configs():
    import itertools

    block_DK = [32, 64, 128]
    block_DV = [32, 64, 128]
    threads = [128, 256]
    num_stages = [1, 2, 3]
    _configs = list(itertools.product(block_DK, block_DV, threads, num_stages))

    configs = [{"block_DK": c[0], "block_DV": c[1], "threads": c[2], "num_stages": c[3]} for c in _configs]
    return configs


@autotune(configs=get_configs(), warmup=3, rep=5)
@tilelang.jit(out_idx=[-3, -2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def tilelang_chunk_gated_delta_rule_fwd_h(
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
    use_gk,
    use_initial_state,
    store_final_state,
    save_new_value,
    # kernel config
    block_DK=64,
    block_DV=32,
    threads=128,
    num_stages=1,
):
    block_S = chunk_size
    BS = (S + chunk_size - 1) // chunk_size  # ceildiv to match kernel iteration

    K_shape = (B, S, H, DK)
    V_shape = (B, S, H, DV)
    W_shape = (B, S, H, DK)
    U_shape = (B, S, H, DV)
    GK_shape = (B, S, H, DK)
    h_shape = (B, BS, H, DK, DV)
    initial_state_shape = (B, H, DK, DV)
    final_state_shape = (B, H, DK, DV)

    @T.prim_func
    def kernel(
        K: T.Tensor(K_shape, dtype=input_dtype),
        W: T.Tensor(W_shape, dtype=input_dtype),
        U: T.Tensor(U_shape, dtype=input_dtype),
        GK: T.Tensor(GK_shape, dtype=gate_dtype),
        initial_state: T.Tensor(initial_state_shape, dtype=input_dtype),
        h: T.Tensor(h_shape, dtype=output_dtype),
        final_state: T.Tensor(final_state_shape, dtype=state_dtype),
        V_new: T.Tensor(V_shape, dtype=output_dtype),
    ):
        with T.Kernel(T.ceildiv(DV, block_DV), B * H, threads=threads) as (bv, bbh):
            bb, bh = bbh // H, bbh % H

            b_h_shared = T.alloc_shared((DK, block_DV), dtype=input_dtype)
            b_h_fragment = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)

            U_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            U_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            W_shared = T.alloc_shared((block_S, DK), dtype=input_dtype)
            V_new_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            V_new_shared = T.alloc_shared((block_S, block_DV), dtype=output_dtype)
            K_shared = T.alloc_shared((block_S, DK), dtype=input_dtype)
            GK_last_shared = T.alloc_shared((DK), dtype=gate_dtype)

            if use_initial_state:
                T.copy(initial_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV], b_h_shared)
                T.copy(b_h_shared, b_h_fragment)
            else:
                T.clear(b_h_fragment)

            for i_s in T.Pipelined(T.ceildiv(S, block_S), num_stages=num_stages):
                # Store previous result to the hidden tensor, like the epilogue
                T.copy(b_h_shared, h[bb, i_s, bh, :, bv * block_DV : (bv + 1) * block_DV])

                # Recurrence
                T.copy(W[bb, i_s * block_S : (i_s + 1) * block_S, bh, :], W_shared)
                T.gemm(W_shared, b_h_shared, V_new_fragment, clear_accum=True)

                # U - W * S
                T.copy(U[bb, i_s * block_S : (i_s + 1) * block_S, bh, bv * block_DV : (bv + 1) * block_DV], U_shared)
                T.copy(U_shared, U_fragment)
                for i_s2, i_v in T.Parallel(block_S, block_DV):
                    V_new_fragment[i_s2, i_v] = -V_new_fragment[i_s2, i_v] + U_fragment[i_s2, i_v]

                # Save V_new
                if save_new_value:
                    T.copy(V_new_fragment, dst=V_new_shared)
                    T.copy(V_new_shared, V_new[bb, i_s * block_S : (i_s + 1) * block_S, bh, bv * block_DV : (bv + 1) * block_DV])

                T.copy(K[bb, i_s * block_S : (i_s + 1) * block_S, bh, 0:DK], K_shared)
                # use_gk
                if use_gk:
                    T.copy(GK[bb, (i_s + 1) * block_S - 1, bh, :], GK_last_shared)  # block last token
                    for i_k, i_v in T.Parallel(DK, block_DV):
                        b_h_fragment[i_k, i_v] *= T.exp2(GK_last_shared[i_k])

                # Update intermediate results
                T.copy(V_new_fragment, V_new_shared)
                T.gemm(K_shared, V_new_shared, b_h_fragment, transpose_A=True)

                T.copy(b_h_fragment, b_h_shared)

            # Save final state
            if store_final_state:
                T.copy(b_h_fragment, final_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV])

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
    use_gk=True,
    use_initial_state=True,
    store_final_state=True,
    save_new_value=True,
    block_DK=64,
    block_DV=32,
    threads=128,
    num_stages=0,
):
    K, W, U, G, initial_state = prepare_input(
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
    )
    kernel = build_kernel(
        tilelang_chunk_gated_delta_rule_fwd_h,
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
        use_gk,
        use_initial_state,
        store_final_state,
        save_new_value,
        block_DK,
        block_DV,
        threads,
        num_stages,
    )
    h_tilelang, final_state_tilelang, V_new_tilelang = kernel(K, W, U, G, initial_state)

    # CPU reference (fp32)
    NT = (S + chunk_size - 1) // chunk_size
    Kf = K.cpu().float()
    Wf = W.cpu().float()
    Uf = U.cpu().float()
    Gf = G.cpu().float()
    h0f = initial_state.cpu().float()
    h_ref = torch.empty(B, NT, H, DK, DV)
    V_new_ref = torch.empty(B, S, H, DV)
    final_ref = torch.empty(B, H, DK, DV)
    for b in range(B):
        for hh in range(H):
            state = h0f[b, hh].clone()
            for t in range(NT):
                s0, s1 = t * chunk_size, min((t + 1) * chunk_size, S)
                h_ref[b, t, hh] = state
                k_c = Kf[b, s0:s1, hh]
                w_c = Wf[b, s0:s1, hh]
                u_c = Uf[b, s0:s1, hh]
                v_new = u_c - w_c @ state
                V_new_ref[b, s0:s1, hh] = v_new
                gk_last = Gf[b, s1 - 1, hh]
                state = state * torch.exp2(gk_last)[:, None]
                state = state + k_c.transpose(0, 1) @ v_new
            final_ref[b, hh] = state

    try:
        torch.testing.assert_close(h_tilelang.cpu().float(), h_ref, rtol=1e-2, atol=1e-2)
        print("h passed")
    except Exception as e:
        print(f"h failed: {e}")
    try:
        torch.testing.assert_close(V_new_tilelang.cpu().float(), V_new_ref, rtol=1e-2, atol=1e-2)
        print("V_new passed")
    except Exception as e:
        print(f"V_new failed: {e}")
    try:
        torch.testing.assert_close(final_state_tilelang.cpu().float(), final_ref, rtol=1e-2, atol=1e-2)
        print("final_state passed")
    except Exception as e:
        print(f"final_state failed: {e}")


def main():
    run_test(
        B=1,
        S=64,
        H=1,
        DK=32,
        DV=32,
        input_dtype="float16",
        output_dtype="float16",
        accum_dtype="float32",
        gate_dtype="float32",
        state_dtype="float32",
        chunk_size=64,
        use_gk=True,
        use_initial_state=True,
        store_final_state=True,
        save_new_value=True,
        block_DK=32,
        block_DV=32,
        threads=128,
        num_stages=0,
    )


if __name__ == "__main__":
    main()
