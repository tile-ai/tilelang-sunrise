import tilelang
from tilelang import language as T

from tile_kernels.config import get_max_smem_per_sm


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def get_engram_gate_fwd_kernel_cuda(
    hidden_size: int,
    eps: float,
    scalar: float,
    num_sms: int,
    clamp_value: float = 1e-6,
    hc_mult: int = 4,
    save_for_backward: bool = True,
    has_image_token_mask: bool = False,
    inplace: bool = False,
    use_pdl: bool = False,
):
    """Forward kernel. When save_for_backward=True, saves dot/gate_score/rstd_x/rstd_k for backward."""
    assert not (inplace and save_for_backward), 'inplace destroys hidden_states, which backward reads'
    num_tokens = T.dynamic('num_tokens')
    threads = 32
    vec_size = 8

    # NOTE Performance only tuned for hidden_size in {4096, 7168}
    def _choose_blk_d(hidden_size):
        for blk in [1024, 768, 512, 256]:
            if hidden_size % blk == 0 and hidden_size >= 2 * blk:
                return blk
        raise ValueError(f'No valid blk_d for hidden_size={hidden_size}')

    def _choose_num_persistent_blocks(hidden_size, blk_d, num_sms, hc_mult):
        """Estimate from SM shared memory occupancy, capped by register pressure."""
        # smem per block: x_smem (bf16) + kv_smem double-buffer (bf16)
        smem_bytes = hidden_size * 2 + blk_d * 4
        blocks_per_sm = min(get_max_smem_per_sm() // smem_bytes, 16)  # 16: register pressure cap
        return num_sms * blocks_per_sm // hc_mult

    blk_d = _choose_blk_d(hidden_size)
    num_persistent_blocks = _choose_num_persistent_blocks(hidden_size, blk_d, num_sms, hc_mult)

    assert hidden_size % blk_d == 0
    assert hidden_size >= 2 * blk_d
    num_blk = hidden_size // blk_d
    reduce_blk = threads * vec_size
    sub_blks = blk_d // reduce_blk
    v_start_phase = num_blk % 2

    @T.prim_func
    def engram_gate_fwd_kernel_cuda(
        hidden_states: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        kv: T.Tensor[(num_tokens, hc_mult + 1, hidden_size), T.bfloat16],
        weight_fused: T.Tensor[(hc_mult, hidden_size), T.float],
        image_token_mask: T.Tensor[(num_tokens,), T.bool],
        output: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        dot_out: T.Tensor[(num_tokens, hc_mult), T.float],
        gate_score: T.Tensor[(num_tokens, hc_mult), T.float],
        rstd_x: T.Tensor[(num_tokens, hc_mult), T.float],
        rstd_k: T.Tensor[(num_tokens, hc_mult), T.float],
    ):
        output_target = hidden_states if inplace else output
        with T.Kernel(hc_mult, num_persistent_blocks, threads=threads) as (pid_h, pid_b):
            if use_pdl:
                T.pdl_sync()
            thread_idx = T.get_thread_binding()
            x_local = T.alloc_local((vec_size,), T.float)
            k_local = T.alloc_local((vec_size,), T.float)
            w_local = T.alloc_local((vec_size,), T.float)
            v_local = T.alloc_local((vec_size,), T.float)
            gate_score_local = T.alloc_local((1,), T.float)
            rstd_x_local = T.alloc_local((1,), T.float)
            rstd_k_local = T.alloc_local((1,), T.float)
            gate_score_reducer = T.alloc_local((1,), T.float)
            rstd_x_reducer = T.alloc_local((1,), T.float)
            rstd_k_reducer = T.alloc_local((1,), T.float)

            x_smem = T.alloc_shared((hidden_size,), T.bfloat16)
            kv_smem = T.alloc_shared((2, blk_d), T.bfloat16)

            per_block = T.ceildiv(num_tokens, num_persistent_blocks)
            t_start = T.min(per_block * pid_b, num_tokens)
            t_end = T.min(per_block * (pid_b + 1), num_tokens)

            for i_s in T.Serial(t_start, t_end):
                # === Pass 1: Reduction with cp.async pipeline ===
                if i_s == t_start:
                    if not has_image_token_mask or not image_token_mask[i_s]:
                        T.async_copy(hidden_states[i_s, pid_h, 0:blk_d], x_smem[0:blk_d])
                        T.async_copy(kv[i_s, pid_h, 0:blk_d], kv_smem[0, :])

                if has_image_token_mask and image_token_mask[i_s]:
                    if not inplace:
                        for i_d in T.Parallel(hidden_size):
                            output_target[i_s, pid_h, i_d] = hidden_states[i_s, pid_h, i_d]
                    if i_s + 1 < t_end:
                        if not image_token_mask[i_s + 1]:
                            T.async_copy(hidden_states[i_s + 1, pid_h, 0:blk_d], x_smem[0:blk_d])
                            T.async_copy(kv[i_s + 1, pid_h, 0:blk_d], kv_smem[0, :])
                else:
                    T.clear(rstd_k_local)
                    T.clear(rstd_x_local)
                    T.clear(gate_score_local)

                    for i_b in T.Serial(1, num_blk):
                        phase = i_b % 2
                        prev_phase = (i_b - 1) % 2
                        T.async_copy(hidden_states[i_s, pid_h, i_b * blk_d : (i_b + 1) * blk_d], x_smem[i_b * blk_d : (i_b + 1) * blk_d])
                        T.async_copy(kv[i_s, pid_h, i_b * blk_d : (i_b + 1) * blk_d], kv_smem[phase, :])
                        T.ptx_wait_group(2)
                        for i_sub in T.Serial(sub_blks):
                            sub_base = (i_b - 1) * blk_d + i_sub * reduce_blk
                            for i_k in T.vectorized(vec_size):
                                x_local[i_k] = x_smem[sub_base + thread_idx * vec_size + i_k]
                                k_local[i_k] = kv_smem[prev_phase, i_sub * reduce_blk + thread_idx * vec_size + i_k]
                            for i_k in T.vectorized(vec_size):
                                w_local[i_k] = weight_fused[pid_h, sub_base + thread_idx * vec_size + i_k]
                            for i_k in T.serial(vec_size):
                                rstd_x_local[0] += x_local[i_k] * x_local[i_k]
                                rstd_k_local[0] += k_local[i_k] * k_local[i_k]
                                gate_score_local[0] += x_local[i_k] * w_local[i_k] * k_local[i_k]

                    # Epilogue: process last tile
                    T.ptx_wait_group(0)

                    # Prefetch v[0] into freed kv_smem bank
                    T.async_copy(kv[i_s, hc_mult, 0:blk_d], kv_smem[v_start_phase, :])

                    for i_sub in T.Serial(sub_blks):
                        sub_base = (num_blk - 1) * blk_d + i_sub * reduce_blk
                        for i_k in T.vectorized(vec_size):
                            x_local[i_k] = x_smem[sub_base + thread_idx * vec_size + i_k]
                            k_local[i_k] = kv_smem[(num_blk - 1) % 2, i_sub * reduce_blk + thread_idx * vec_size + i_k]
                        for i_k in T.vectorized(vec_size):
                            w_local[i_k] = weight_fused[pid_h, sub_base + thread_idx * vec_size + i_k]
                        for i_k in T.serial(vec_size):
                            rstd_x_local[0] += x_local[i_k] * x_local[i_k]
                            rstd_k_local[0] += k_local[i_k] * k_local[i_k]
                            gate_score_local[0] += x_local[i_k] * w_local[i_k] * k_local[i_k]

                    # Prefetch v[1]
                    T.async_copy(kv[i_s, hc_mult, blk_d : 2 * blk_d], kv_smem[1 - v_start_phase, :])

                    rstd_k_reducer[0] = T.warp_reduce_sum(rstd_k_local[0])
                    rstd_x_reducer[0] = T.warp_reduce_sum(rstd_x_local[0])
                    gate_score_reducer[0] = T.warp_reduce_sum(gate_score_local[0])

                    rstd_x_reducer[0] = T.rsqrt(rstd_x_reducer[0] / hidden_size + eps)
                    rstd_k_reducer[0] = T.rsqrt(rstd_k_reducer[0] / hidden_size + eps)

                    if save_for_backward:
                        if thread_idx == 0:
                            dot_out[i_s, pid_h] = gate_score_reducer[0]
                            rstd_x[i_s, pid_h] = rstd_x_reducer[0]
                            rstd_k[i_s, pid_h] = rstd_k_reducer[0]

                    gate_score_reducer[0] = gate_score_reducer[0] * rstd_x_reducer[0] * rstd_k_reducer[0] * scalar

                    gate_score_reducer[0] = T.sigmoid(
                        T.copysign(T.sqrt(T.clamp(T.abs(gate_score_reducer[0]), clamp_value, float('inf'))), gate_score_reducer[0])
                    )

                    if save_for_backward:
                        if thread_idx == 0:
                            gate_score[i_s, pid_h] = gate_score_reducer[0]

                    # === Pass 2: Output — x from smem, v from kv_smem (tiles 0,1 already prefetched) ===
                    for i_b in T.Serial(num_blk):
                        tile_phase = (v_start_phase + i_b) % 2
                        if i_b < num_blk - 1:
                            T.ptx_wait_group(1)
                        else:
                            T.ptx_wait_group(0)
                            # Prefetch next token's k and x
                            if i_s + 1 < t_end:
                                if not has_image_token_mask or not image_token_mask[i_s + 1]:
                                    T.async_copy(kv[i_s + 1, pid_h, 0:blk_d], kv_smem[0, :])
                                    T.async_copy(hidden_states[i_s + 1, pid_h, 0:blk_d], x_smem[0:blk_d])
                        for i_sub in T.Serial(sub_blks):
                            sub_base = i_b * blk_d + i_sub * reduce_blk
                            for i_k in T.vectorized(vec_size):
                                x_local[i_k] = x_smem[sub_base + thread_idx * vec_size + i_k]
                                v_local[i_k] = kv_smem[tile_phase, i_sub * reduce_blk + thread_idx * vec_size + i_k]
                            for i_k in T.vectorized(vec_size):
                                output_target[i_s, pid_h, sub_base + thread_idx * vec_size + i_k] = (
                                    x_local[i_k] + gate_score_reducer[0] * v_local[i_k]
                                )
                        # Prefetch v[i_b+2] into freed kv_smem bank
                        if i_b + 2 < num_blk:
                            T.async_copy(kv[i_s, hc_mult, (i_b + 2) * blk_d : (i_b + 3) * blk_d], kv_smem[tile_phase, :])
            if use_pdl:
                T.pdl_trigger()

    return engram_gate_fwd_kernel_cuda
