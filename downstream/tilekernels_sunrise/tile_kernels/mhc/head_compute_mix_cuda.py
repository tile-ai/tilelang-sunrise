import tilelang
from tilelang import language as T

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def mhc_head_compute_mix_fwd_cuda(
    input_mix,
    mhc_scale,
    mhc_base,
    output_mix,
    mhc_pre_eps: float,
    token_block_size: int = 32,
    use_pdl: bool = False,
):
    num_tokens = T.dynamic('num_tokens')
    mhc_mult = T.const('mhc_mult')

    input_mix: T.Tensor[(num_tokens, mhc_mult), T.float32]
    mhc_scale: T.Tensor[(1,), T.float32]
    mhc_base: T.Tensor[(mhc_mult,), T.float32]
    output_mix: T.Tensor[(num_tokens, mhc_mult), T.float32]

    with T.Kernel(T.ceildiv(num_tokens, token_block_size)) as pid:
        if use_pdl:
            T.pdl_sync()
        for i1, j in T.Parallel(token_block_size, mhc_mult):
            i = pid * token_block_size + i1
            if i < num_tokens:
                output_mix[i, j] = T.sigmoid(input_mix[i, j] * mhc_scale[0] + mhc_base[j]) + mhc_pre_eps
        if use_pdl:
            T.pdl_trigger()


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def mhc_head_compute_mix_bwd_cuda(
    output_mix_grad,
    input_mix,
    mhc_scale,
    mhc_base,
    input_mix_grad,
    mhc_scale_grad_partial,
    mhc_base_grad_partial,
    num_sms: int,
    token_block_size: int = 32,
    use_pdl: bool = False,
):
    num_tokens = T.dynamic('num_tokens')
    mhc_mult = T.const('mhc_mult')

    output_mix_grad: T.Tensor[(num_tokens, mhc_mult), T.float32]
    input_mix: T.Tensor[(num_tokens, mhc_mult), T.float32]
    mhc_scale: T.Tensor[(1,), T.float32]
    mhc_base: T.Tensor[(mhc_mult,), T.float32]
    input_mix_grad: T.Tensor[(num_tokens, mhc_mult), T.float32]
    mhc_scale_grad_partial: T.Tensor[(num_sms, 1), T.float32]
    mhc_base_grad_partial: T.Tensor[(num_sms, mhc_mult), T.float32]

    with T.Kernel(num_sms) as pid:
        if use_pdl:
            T.pdl_sync()
        mhc_scale_grad_reducer = T.alloc_reducer(1, T.float32, op='sum')
        mhc_base_grad_reducer = T.alloc_reducer(mhc_mult, T.float32, op='sum')
        mhc_scale_grad_result = T.alloc_fragment(1, T.float32)
        mhc_base_grad_result = T.alloc_fragment(mhc_mult, T.float32)
        T.reducer_init(mhc_scale_grad_reducer)
        T.reducer_init(mhc_base_grad_reducer)
        for t in T.Persistent(
            [T.ceildiv(num_tokens, token_block_size)],
            num_sms,
            pid,
            group_size=1,
        ):
            grad_frag = T.alloc_fragment((token_block_size, mhc_mult), T.float32)
            input_recompute_frag = T.alloc_fragment((token_block_size, mhc_mult), T.float32)
            for i1, j in T.Parallel(token_block_size, mhc_mult):
                i = t * token_block_size + i1
                if i < num_tokens:
                    input_recompute_frag[i1, j] = T.sigmoid(
                        input_mix[i, j] * mhc_scale[0] + mhc_base[j],
                    )
                    grad_frag[i1, j] = input_recompute_frag[i1, j] * (1 - input_recompute_frag[i1, j]) * output_mix_grad[i, j]
                    input_mix_grad[i, j] = grad_frag[i1, j] * mhc_scale[0]
                    T.reducer_update(mhc_scale_grad_reducer[0], grad_frag[i1, j] * input_mix[i, j])
                    T.reducer_update(mhc_base_grad_reducer[j], grad_frag[i1, j])
        T.finalize_reducer(mhc_scale_grad_reducer, mhc_scale_grad_result)
        T.finalize_reducer(mhc_base_grad_reducer, mhc_base_grad_result)
        T.copy(mhc_scale_grad_result, mhc_scale_grad_partial[pid, :])
        T.copy(mhc_base_grad_result, mhc_base_grad_partial[pid, :])
        if use_pdl:
            T.pdl_trigger()
