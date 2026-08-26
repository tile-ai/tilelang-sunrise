import sys  # noqa: F401
import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune
from test_utils import DEVICE, build_kernel

import torch
import torch.nn.functional as F
from tilelang.engine.callback import register_cuda_postproc_callback  # noqa: F401

# (zhengju) We can slightly modify the generated cuda code from tilelang lowering
# in the debug folder to make the performance better. To enable this callback,
# you can comment out the following function.
# @register_cuda_postproc_callback
# def tilelang_callback_cuda_postproc(code, _):
#     cuda_code = open("../debug/chunk_delta_h_fuse.cu", "r").read()
#     code = cuda_code
#     return code

torch.random.manual_seed(0)


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
    G = F.logsigmoid(torch.randn(B, S, H, dtype=gate_dtype))
    G = G.reshape(B, S // chunk_size, chunk_size, H).cumsum(2).reshape(B, S, H).to(DEVICE)
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
    BS = S // chunk_size
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
    use_g,
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
    BS = S // block_S

    K_shape = (B, S, H, DK)
    V_shape = (B, S, H, DV)
    W_shape = (B, S, H, DK)
    U_shape = (B, S, H, DV)
    G_shape = (B, S, H)
    h_shape = (B, BS, H, DK, DV)
    initial_state_shape = (B, H, DK, DV)
    final_state_shape = (B, H, DK, DV)

    @T.prim_func
    def kernel(
        K: T.Tensor(K_shape, dtype=input_dtype),
        W: T.Tensor(W_shape, dtype=input_dtype),
        U: T.Tensor(U_shape, dtype=input_dtype),
        G: T.Tensor(G_shape, dtype=gate_dtype),
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
            G_last_local = T.alloc_var(T.float32)
            G_shared = T.alloc_shared((block_S, block_DV), dtype=gate_dtype)
            G_fragment = T.alloc_fragment((block_S, block_DV), dtype=gate_dtype)

            T.annotate_layout(
                {
                    U_shared: tilelang.layout.make_swizzled_layout(U_shared),
                    G_shared: tilelang.layout.make_swizzled_layout(G_shared),
                }
            )

            T.use_swizzle(10)

            if use_initial_state:
                T.copy(initial_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV], b_h_shared)
                T.copy(b_h_shared, b_h_fragment)
            else:
                T.clear(b_h_fragment)

            for i_s in T.Pipelined(T.ceildiv(S, block_S), num_stages=num_stages):
                # Store previous result to the hidden tensor, like the epilogue
                T.copy(b_h_shared, h[bb, i_s, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV])

                # Recurrence
                T.copy(W[bb, i_s * block_S : (i_s + 1) * block_S, bh, 0:DK], W_shared)
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
                # use_g
                if use_g:
                    G_last_local = G[bb, (i_s + 1) * block_S - 1, bh]
                    for i_s2, i_v in T.Parallel(block_S, block_DV):
                        G_shared[i_s2, i_v] = G[bb, i_s * block_S + i_s2, bh]
                    T.copy(G_shared, G_fragment)
                    for i_s2, i_v in T.Parallel(block_S, block_DV):
                        V_new_fragment[i_s2, i_v] = (
                            V_new_fragment[i_s2, i_v] * T.exp2((G_last_local - G_fragment[i_s2, i_v]) * 1.442695)
                            if G_last_local - G_fragment[i_s2, i_v] <= 0
                            else 0
                        )
                    G_last_local = T.exp2(G_last_local * 1.442695)
                    for i_k, i_v in T.Parallel(DK, block_DV):
                        b_h_fragment[i_k, i_v] *= G_last_local

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
    use_g=True,
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
    chunks = S // chunk_size
    k_ref = K.cpu().float().reshape(B, chunks, chunk_size, H, DK).permute(0, 1, 3, 2, 4)
    w_ref = W.cpu().float().reshape(B, chunks, chunk_size, H, DK).permute(0, 1, 3, 2, 4)
    u_ref = U.cpu().float().reshape(B, chunks, chunk_size, H, DV).permute(0, 1, 3, 2, 4)
    g_ref = G.cpu().float().reshape(B, chunks, chunk_size, H).permute(0, 1, 3, 2)
    state = initial_state.cpu().float() if use_initial_state else torch.zeros(B, H, DK, DV)
    h_ref = []
    values_ref = []
    for chunk in range(chunks):
        h_ref.append(state.clone())
        value = u_ref[:, chunk] - torch.matmul(w_ref[:, chunk], state)
        values_ref.append(value)
        update = value
        if use_g:
            g_last = g_ref[:, chunk, :, -1]
            update = value * torch.exp(g_last.unsqueeze(-1).unsqueeze(-1) - g_ref[:, chunk].unsqueeze(-1))
            state = state * torch.exp(g_last).unsqueeze(-1).unsqueeze(-1)
        state = state + torch.matmul(k_ref[:, chunk].transpose(-1, -2), update)
    h_ref = torch.stack(h_ref, dim=1).to(getattr(torch, output_dtype))
    V_new_ref = torch.stack(values_ref, dim=1).permute(0, 1, 3, 2, 4).reshape(B, S, H, DV).to(getattr(torch, output_dtype))
    final_state_ref = state.to(getattr(torch, state_dtype))

    # tilelang
    kernel = build_kernel(
        tilelang_chunk_gated_delta_rule_fwd_h,
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
        use_g,
        use_initial_state,
        store_final_state,
        save_new_value,
        block_DK,
        block_DV,
        threads,
        num_stages,
    )
    h_tilelang, final_state_tilelang, V_new_tilelang = kernel(K, W, U, G, initial_state)
    try:
        torch.testing.assert_close(h_tilelang.cpu(), h_ref, rtol=1e-2, atol=1e-2)
        print("chunk_delta_h h passed")
    except Exception as e:
        print(f"chunk_delta_h h failed: {e}")
    if store_final_state:
        try:
            torch.testing.assert_close(final_state_tilelang.cpu(), final_state_ref, rtol=1e-2, atol=1e-2)
            print("chunk_delta_h final_state passed")
        except Exception as e:
            print(f"chunk_delta_h final_state failed: {e}")
    if save_new_value:
        try:
            torch.testing.assert_close(V_new_tilelang.cpu(), V_new_ref, rtol=1e-2, atol=1e-2)
            print("chunk_delta_h V_new passed")
        except Exception as e:
            print(f"chunk_delta_h V_new failed: {e}")


def main():
    run_test(
        B=1,
        S=64,
        H=1,
        DK=32,
        DV=32,
        input_dtype=T.bfloat16,
        output_dtype=T.bfloat16,
        accum_dtype=T.float32,
        gate_dtype=T.float32,
        state_dtype=T.float32,
        chunk_size=64,
        use_g=True,
        use_initial_state=False,
        store_final_state=True,
        save_new_value=True,
        block_DK=32,
        block_DV=32,
        threads=128,
        num_stages=0,
    )


if __name__ == "__main__":
    main()
