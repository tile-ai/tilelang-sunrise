import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S


@tilelang.jit
def get_engram_gate_fwd_kernel_asc(
    hidden_size: int,
    eps: float,
    scalar: float,
    clamp_value: float,
    hc_mult: int,
    save_for_backward: bool,
    has_image_token_mask: bool,
    inplace: bool,
    num_vec_cores: int,
):
    vec_size = 64
    num_accs = 2
    heads_per_task = 2
    num_stages = 2

    assert hc_mult == 4
    assert hidden_size % (vec_size * num_accs) == 0
    assert not (inplace and save_for_backward), 'inplace destroys hidden_states, which backward reads'

    num_tokens = T.dynamic('num_tokens')
    num_head_groups = hc_mult // heads_per_task
    assert num_vec_cores % num_head_groups == 0
    num_chunks = hidden_size // vec_size

    @T.prim_func
    def engram_gate_fwd_kernel_asc(
        hidden_states: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        kv: T.Tensor[(num_tokens, hc_mult + 1, hidden_size), T.bfloat16],
        weight_fused: T.Tensor[(hc_mult, hidden_size), T.float32],
        image_token_mask: T.Tensor[(num_tokens,), T.bool],
        output: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        dot_out: T.Tensor[(num_tokens, hc_mult), T.float32],
        gate_score: T.Tensor[(num_tokens, hc_mult), T.float32],
        rstd_x: T.Tensor[(num_tokens, hc_mult), T.float32],
        rstd_k: T.Tensor[(num_tokens, hc_mult), T.float32],
    ):
        output_target = hidden_states if inplace else output
        with T.Kernel(num_vec_cores) as core_id:
            head_group = core_id % num_head_groups
            head_base = head_group * heads_per_task

            weight_ub = T.alloc_shared((heads_per_task, hidden_size), T.float32)
            x_ub = T.alloc_shared((heads_per_task, hidden_size), T.bfloat16)
            k_output_ub = T.alloc_shared((heads_per_task, hidden_size), T.bfloat16)
            v_ub = T.alloc_shared((hidden_size,), T.bfloat16)

            dot_ub = T.alloc_shared((8,), T.float32)
            gate_ub = T.alloc_shared((8,), T.float32)
            rstd_x_ub = T.alloc_shared((8,), T.float32)
            rstd_k_ub = T.alloc_shared((8,), T.float32)
            buffer_versions = {
                x_ub: num_stages,
                k_output_ub: num_stages,
                v_ub: num_stages,
                gate_ub: num_stages,
            }
            if save_for_backward:
                buffer_versions.update(
                    {
                        dot_ub: num_stages,
                        rstd_x_ub: num_stages,
                        rstd_k_ub: num_stages,
                    }
                )
            T.annotate_buffer_versions(buffer_versions)

            T.copy(weight_fused[head_base : head_base + heads_per_task, :], weight_ub)

            for token_id, task_head_group in T.Persistent(
                [num_tokens, num_head_groups],
                num_vec_cores,
                core_id,
                group_size=num_head_groups,
                num_stages=num_stages,
            ):
                task_head_base = task_head_group * heads_per_task
                if has_image_token_mask and image_token_mask[token_id]:
                    if not inplace:
                        T.copy(
                            hidden_states[
                                token_id,
                                task_head_base : task_head_base + heads_per_task,
                                :,
                            ],
                            x_ub,
                            l2_cache_ctrl='NOTALLOC_KEEP',
                        )
                        T.copy(
                            x_ub,
                            output_target[
                                token_id,
                                task_head_base : task_head_base + heads_per_task,
                                :,
                            ],
                        )
                else:
                    T.copy(
                        hidden_states[
                            token_id,
                            task_head_base : task_head_base + heads_per_task,
                            :,
                        ],
                        x_ub,
                        l2_cache_ctrl='NOTALLOC_KEEP',
                    )
                    T.copy(
                        kv[
                            token_id,
                            task_head_base : task_head_base + heads_per_task,
                            :,
                        ],
                        k_output_ub,
                        l2_cache_ctrl='NOTALLOC_KEEP',
                    )
                    with T.SimdVF():
                        one = S.vdup(1.0, T.float32)
                        zero = S.vdup(0.0, T.float32)
                        accum = S.alloc_local((3 * num_accs,), T.float32)

                        for head_id in T.unroll(heads_per_task, explicit=True):
                            for acc_id in T.unroll(3 * num_accs, explicit=True):
                                accum[acc_id] = zero

                            for chunk_group in T.serial(num_chunks // num_accs):
                                for acc_id in T.unroll(num_accs, explicit=True):
                                    offset = (chunk_group * num_accs + acc_id) * vec_size
                                    x_f32 = S.vcvt(S.vld(x_ub[head_id, offset], dist='UNPK_B16'), T.float32, part=0)
                                    k_f32 = S.vcvt(
                                        S.vld(
                                            k_output_ub[
                                                head_id,
                                                offset,
                                            ],
                                            dist='UNPK_B16',
                                        ),
                                        T.float32,
                                        part=0,
                                    )
                                    weight_f32 = S.vld(weight_ub[head_id, offset])

                                    S.vmula(accum[acc_id], x_f32, x_f32)
                                    S.vmula(accum[num_accs + acc_id], k_f32, k_f32)
                                    weighted_x = S.vmul(x_f32, weight_f32)
                                    S.vmula(accum[2 * num_accs + acc_id], weighted_x, k_f32)

                            for value_id in T.unroll(3, explicit=True):
                                base = value_id * num_accs
                                accum[base] = S.vadd(accum[base], accum[base + 1])

                            sum_x2 = S.vcadd(accum[0])
                            sum_k2 = S.vcadd(accum[num_accs])
                            raw_dot = S.vcadd(accum[2 * num_accs])
                            rms_x = S.vsqrt(S.vadds(S.vmuls(sum_x2, 1.0 / hidden_size), eps))
                            rms_k = S.vsqrt(S.vadds(S.vmuls(sum_k2, 1.0 / hidden_size), eps))
                            inv_rms_x = S.vdiv(one, rms_x)
                            inv_rms_k = S.vdiv(one, rms_k)
                            normalized_dot = S.vmuls(S.vmul(S.vmul(raw_dot, inv_rms_x), inv_rms_k), scalar)
                            signed_root_abs = S.vsqrt(S.vmaxs(S.vabs(normalized_dot), clamp_value))
                            signed_root = S.vsel(
                                signed_root_abs,
                                S.vneg(signed_root_abs),
                                S.vcmps(normalized_dot, 0.0, op='ge'),
                            )
                            gate = S.vdiv(one, S.vadd(S.vexpdif(zero, signed_root), one))

                            S.vsts(gate_ub[head_id], gate, dist='ONEPT_B32')
                            if save_for_backward:
                                S.vsts(dot_ub[head_id], raw_dot, dist='ONEPT_B32')
                                S.vsts(rstd_x_ub[head_id], inv_rms_x, dist='ONEPT_B32')
                                S.vsts(rstd_k_ub[head_id], inv_rms_k, dist='ONEPT_B32')

                    T.copy(
                        kv[
                            token_id,
                            hc_mult,
                            :,
                        ],
                        v_ub,
                    )

                    with T.SimdVF():
                        gates_f32 = S.alloc_local((heads_per_task,), T.float32)
                        for head_id in T.unroll(heads_per_task, explicit=True):
                            gates_f32[head_id] = S.vld(gate_ub[head_id], dist='BRC_B32')

                        out_f32 = S.alloc_local((heads_per_task,), T.float32)
                        for chunk_id in T.serial(num_chunks):
                            offset = chunk_id * vec_size
                            v_f32 = S.vcvt(S.vld(v_ub[offset], dist='UNPK_B16'), T.float32, part=0)
                            for head_id in T.unroll(heads_per_task, explicit=True):
                                out_f32[head_id] = S.vcvt(
                                    S.vld(x_ub[head_id, offset], dist='UNPK_B16'),
                                    T.float32,
                                    part=0,
                                )
                            for head_id in T.unroll(heads_per_task, explicit=True):
                                S.vmula(out_f32[head_id], v_f32, gates_f32[head_id])
                            for head_id in T.unroll(heads_per_task, explicit=True):
                                S.vsts(
                                    k_output_ub[
                                        head_id,
                                        offset,
                                    ],
                                    S.vcvt(out_f32[head_id], T.bfloat16),
                                    dist='PK_B32',
                                )

                    T.copy(
                        k_output_ub,
                        output_target[
                            token_id,
                            task_head_base : task_head_base + heads_per_task,
                            :,
                        ],
                    )
                    if save_for_backward:
                        T.copy(
                            dot_ub[:heads_per_task],
                            dot_out[
                                token_id,
                                task_head_base : task_head_base + heads_per_task,
                            ],
                        )
                        T.copy(
                            gate_ub[:heads_per_task],
                            gate_score[
                                token_id,
                                task_head_base : task_head_base + heads_per_task,
                            ],
                        )
                        T.copy(
                            rstd_x_ub[:heads_per_task],
                            rstd_x[
                                token_id,
                                task_head_base : task_head_base + heads_per_task,
                            ],
                        )
                        T.copy(
                            rstd_k_ub[:heads_per_task],
                            rstd_k[
                                token_id,
                                task_head_base : task_head_base + heads_per_task,
                            ],
                        )

    return engram_gate_fwd_kernel_asc
