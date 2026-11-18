"""TANG gate backward kernels with ordinary loads and compiler-managed reduction synchronization."""
import tilelang
from tilelang import language as T


@tilelang.jit(target='tang', pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def get_engram_gate_bwd_kernel_tang(hidden_size: int, scalar: float, num_partials: int,
                                   clamp_value: float, hc_mult: int, has_image_token_mask: bool):
    num_tokens = T.dynamic('num_tokens')

    @T.prim_func
    def kernel(
        grad_out: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        hidden_states: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        kv: T.Tensor[(num_tokens, hc_mult + 1, hidden_size), T.bfloat16],
        weight_fused: T.Tensor[(hc_mult, hidden_size), T.float32],
        dot: T.Tensor[(num_tokens, hc_mult), T.float32],
        gate_score: T.Tensor[(num_tokens, hc_mult), T.float32],
        rstd_x: T.Tensor[(num_tokens, hc_mult), T.float32],
        rstd_k: T.Tensor[(num_tokens, hc_mult), T.float32],
        image_token_mask: T.Tensor[(num_tokens,), T.bool],
        grad_x: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        grad_kv: T.Tensor[(num_tokens, hc_mult + 1, hidden_size), T.bfloat16],
        grad_w_partial: T.Tensor[(num_partials, hc_mult, hidden_size), T.float32],
    ):
        with T.Kernel(num_partials, hc_mult, threads=128) as (part, head):
            grad_w = T.alloc_fragment((hidden_size,), T.float32)
            go_v = T.alloc_fragment((hidden_size,), T.float32)
            dldg = T.alloc_fragment((1,), T.float32)
            T.clear(grad_w)
            per_block = T.ceildiv(num_tokens, num_partials)
            for token in T.serial(T.min(part * per_block, num_tokens), T.min((part + 1) * per_block, num_tokens)):
                if has_image_token_mask and image_token_mask[token]:
                    for i in T.Parallel(hidden_size):
                        grad_x[token, head, i] = grad_out[token, head, i]
                        grad_kv[token, head, i] = 0
                else:
                    for i in T.Parallel(hidden_size):
                        go_v[i] = T.cast(grad_out[token, head, i], T.float32) * T.cast(kv[token, hc_mult, i], T.float32)
                    T.reduce_sum(go_v, dldg, dim=0)
                    raw_dot = dot[token, head]
                    rx = rstd_x[token, head]
                    rk = rstd_k[token, head]
                    gate = gate_score[token, head]
                    derivative = T.Select(
                        T.abs(raw_dot) * scalar * rx * rk < clamp_value,
                        0.0,
                        dldg[0] * gate * (1.0 - gate) * 0.5 * T.sqrt(scalar * rx * rk / T.abs(raw_dot)),
                    )
                    dot_x = raw_dot * rx * rx / hidden_size
                    dot_k = raw_dot * rk * rk / hidden_size
                    for i in T.Parallel(hidden_size):
                        x = T.cast(hidden_states[token, head, i], T.float32)
                        k = T.cast(kv[token, head, i], T.float32)
                        weight = weight_fused[head, i]
                        grad_x[token, head, i] = T.cast(grad_out[token, head, i], T.float32) + derivative * (k * weight - x * dot_x)
                        grad_kv[token, head, i] = derivative * (x * weight - k * dot_k)
                        grad_w[i] += derivative * x * k
            # Empty partitions and fully masked partitions still write initialized zeros.
            for i in T.Parallel(hidden_size):
                grad_w_partial[part, head, i] = grad_w[i]

    return kernel


@tilelang.jit(target='tang', pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def get_engram_gate_grad_v_kernel_tang(hidden_size: int, hc_mult: int, has_image_token_mask: bool):
    num_tokens = T.dynamic('num_tokens')

    @T.prim_func
    def kernel(
        grad_out: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        gate_score: T.Tensor[(num_tokens, hc_mult), T.float32],
        image_token_mask: T.Tensor[(num_tokens,), T.bool],
        grad_kv: T.Tensor[(num_tokens, hc_mult + 1, hidden_size), T.bfloat16],
    ):
        with T.Kernel(num_tokens, threads=128) as token:
            grad_v = T.alloc_fragment((hidden_size,), T.float32)
            T.clear(grad_v)
            if not has_image_token_mask or not image_token_mask[token]:
                for head in T.serial(hc_mult):
                    for i in T.Parallel(hidden_size):
                        grad_v[i] += T.cast(grad_out[token, head, i], T.float32) * gate_score[token, head]
            for i in T.Parallel(hidden_size):
                grad_kv[token, hc_mult, i] = grad_v[i]

    return kernel
