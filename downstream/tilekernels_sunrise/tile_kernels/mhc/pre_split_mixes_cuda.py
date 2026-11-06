import tilelang
from tilelang import language as T

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def mhc_pre_split_mixes_fwd_cuda(
    input_mixes,
    mhc_scale,
    mhc_base,
    pre_layer_mix,
    post_layer_mix,
    comb_res_mix,
    mhc_post_mult_value: float,
    mhc_pre_eps: float,
    token_block_size: int = 32,
    dtype: T.dtype = T.float32,
    use_pdl: bool = False,
):
    num_tokens = T.dynamic('num_tokens')
    mhc_mult = T.const('mhc_mult')
    mhc_mult2 = T.const('mhc_mult2')
    mhc_mult3 = T.const('mhc_mult3')

    input_mixes: T.Tensor[(num_tokens, mhc_mult3), dtype]
    mhc_scale: T.Tensor[(3,), dtype]
    mhc_base: T.Tensor[mhc_mult3, dtype]
    pre_layer_mix: T.Tensor[(num_tokens, mhc_mult), dtype]
    post_layer_mix: T.Tensor[(num_tokens, mhc_mult), dtype]
    comb_res_mix: T.Tensor[(num_tokens, mhc_mult2), dtype]

    with T.Kernel(T.ceildiv(num_tokens, token_block_size)) as pid:
        if use_pdl:
            T.pdl_sync()
        input_mixes_frag = T.alloc_fragment((token_block_size, mhc_mult3), dtype)
        pre_layer_mix_frag = T.alloc_fragment((token_block_size, mhc_mult), dtype)
        post_layer_mix_frag = T.alloc_fragment((token_block_size, mhc_mult), dtype)
        comb_res_mix_frag = T.alloc_fragment((token_block_size, mhc_mult2), dtype)

        T.annotate_layout(
            {
                input_mixes_frag: T.Fragment(
                    (token_block_size, mhc_mult3),
                    lambda i, j: (i % 32, i // 32 * mhc_mult3 + j),
                ),
            },
        )

        T.copy(input_mixes[pid * token_block_size, 0], input_mixes_frag)

        for i, j in T.Parallel(token_block_size, mhc_mult):
            pre_layer_mix_frag[i, j] = (
                T.sigmoid(
                    input_mixes_frag[i, j] * mhc_scale[0] + mhc_base[j],
                )
                + mhc_pre_eps
            )
        for i, j in T.Parallel(token_block_size, mhc_mult):
            post_layer_mix_frag[i, j] = T.sigmoid(input_mixes_frag[i, j + mhc_mult] * mhc_scale[1] + mhc_base[j + mhc_mult]) * mhc_post_mult_value
        for i, j in T.Parallel(token_block_size, mhc_mult2):
            comb_res_mix_frag[i, j] = input_mixes_frag[i, j + mhc_mult * 2] * mhc_scale[2] + mhc_base[j + mhc_mult * 2]

        T.copy(pre_layer_mix_frag, pre_layer_mix[pid * token_block_size, 0])
        T.copy(post_layer_mix_frag, post_layer_mix[pid * token_block_size, 0])
        T.copy(comb_res_mix_frag, comb_res_mix[pid * token_block_size, 0])
        if use_pdl:
            T.pdl_trigger()


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def mhc_pre_split_mixes_bwd_cuda(
    pre_layer_mix_grad,
    post_layer_mix_grad,
    comb_res_mix_grad,
    input_mixes,
    post_layer_mix,
    mhc_scale,
    mhc_base,
    input_mixes_grad,
    mhc_scale_grad_partial,
    mhc_base_grad_partial,
    mhc_post_mult_value: float,
    token_block_size: int = 32,
    dtype: T.dtype = T.float32,
    use_pdl: bool = False,
):
    num_tokens = T.dynamic('num_tokens')
    mhc_mult = T.const('mhc_mult')
    mhc_mult2 = T.const('mhc_mult2')
    mhc_mult3 = T.const('mhc_mult3')
    num_sms = T.const('num_sms')
    pre_layer_mix_grad: T.Tensor[(num_tokens, mhc_mult), dtype]
    post_layer_mix_grad: T.Tensor[(num_tokens, mhc_mult), dtype]
    comb_res_mix_grad: T.Tensor[(num_tokens, mhc_mult2), dtype]
    input_mixes: T.Tensor[(num_tokens, mhc_mult3), dtype]
    post_layer_mix: T.Tensor[(num_tokens, mhc_mult), dtype]
    mhc_scale: T.Tensor[(3,), dtype]
    mhc_base: T.Tensor[mhc_mult3, dtype]
    input_mixes_grad: T.Tensor[(num_tokens, mhc_mult3), dtype]
    mhc_scale_grad_partial: T.Tensor[(num_sms, 3), dtype]
    mhc_base_grad_partial: T.Tensor[(num_sms, mhc_mult3), dtype]

    with T.Kernel(num_sms) as pid:
        if use_pdl:
            T.pdl_sync()
        pre_layer_mix_grad_frag = T.alloc_fragment((token_block_size, mhc_mult), dtype)
        post_layer_mix_grad_frag = T.alloc_fragment((token_block_size, mhc_mult), dtype)
        comb_res_mix_grad_frag = T.alloc_fragment((token_block_size, mhc_mult2), dtype)

        pre_layer_mix_frag = T.alloc_fragment((token_block_size, mhc_mult), dtype)
        post_layer_mix_frag = T.alloc_fragment((token_block_size, mhc_mult), dtype)

        input_mixes_frag = T.alloc_fragment((token_block_size, mhc_mult3), dtype)
        input_mixes_grad_frag = T.alloc_fragment((token_block_size, mhc_mult3), dtype)

        T.annotate_layout(
            {
                input_mixes_grad_frag: T.Fragment(
                    (token_block_size, mhc_mult3),
                    lambda i, j: (i % 32, i // 32 * mhc_mult3 + j),
                ),
            },
        )

        mhc_scale_grad_reducer = T.alloc_reducer(3, dtype, op='sum')
        mhc_scale_grad_result = T.alloc_fragment(3, dtype)
        T.reducer_init(mhc_scale_grad_reducer)

        mhc_base_grad_frag = T.alloc_fragment(mhc_mult3, dtype)
        T.clear(mhc_base_grad_frag)

        for t in T.Persistent(
            [T.ceildiv(num_tokens, token_block_size)],
            num_sms,
            pid,
            group_size=1,
        ):
            T.copy(pre_layer_mix_grad[t * token_block_size, 0], pre_layer_mix_grad_frag)
            T.copy(post_layer_mix_grad[t * token_block_size, 0], post_layer_mix_grad_frag)
            T.copy(comb_res_mix_grad[t * token_block_size, 0], comb_res_mix_grad_frag)

            T.copy(post_layer_mix[t * token_block_size, 0], post_layer_mix_frag)
            T.copy(input_mixes[t * token_block_size, 0], input_mixes_frag)

            for i, j in T.Parallel(token_block_size, mhc_mult):
                pre_layer_mix_frag[i, j] = T.sigmoid(
                    input_mixes_frag[i, j] * mhc_scale[0] + mhc_base[j],
                )
                input_mixes_grad_frag[i, j] = pre_layer_mix_grad_frag[i, j] * pre_layer_mix_frag[i, j] * (1 - pre_layer_mix_frag[i, j])
            for i, j in T.Parallel(token_block_size, mhc_mult):
                input_mixes_grad_frag[i, j + mhc_mult] = (
                    post_layer_mix_grad_frag[i, j] * post_layer_mix_frag[i, j] * (1 - post_layer_mix_frag[i, j] / mhc_post_mult_value)
                )
            for i, j in T.Parallel(token_block_size, mhc_mult2):
                input_mixes_grad_frag[i, j + mhc_mult * 2] = comb_res_mix_grad_frag[i, j]

            T.reduce_sum(input_mixes_grad_frag, mhc_base_grad_frag, dim=0, clear=False)

            for i, j in T.Parallel(token_block_size, mhc_mult3):
                T.reducer_update(mhc_scale_grad_reducer[T.min(2, j // mhc_mult)], input_mixes_grad_frag[i, j] * input_mixes_frag[i, j])
                input_mixes_grad_frag[i, j] *= mhc_scale[T.min(2, j // mhc_mult)]

            T.copy(input_mixes_grad_frag, input_mixes_grad[t * token_block_size, 0])

        T.copy(mhc_base_grad_frag, mhc_base_grad_partial[pid, :])

        T.finalize_reducer(mhc_scale_grad_reducer, mhc_scale_grad_result, batch=3)
        T.copy(mhc_scale_grad_result, mhc_scale_grad_partial[pid, :])
        if use_pdl:
            T.pdl_trigger()


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def mhc_pre_split_mixes_bwd_reduce_cuda(
    mhc_scale_grad_partial,
    mhc_base_grad_partial,
    mhc_scale_grad,
    mhc_base_grad,
    num_sms: int,
    dtype: T.dtype = T.float32,
):
    """Reduce one parameter-gradient column per CTA in a single launch."""
    mhc_mult3 = T.const('mhc_mult3')

    mhc_scale_grad_partial: T.Tensor[(num_sms, 3), dtype]
    mhc_base_grad_partial: T.Tensor[(num_sms, mhc_mult3), dtype]
    mhc_scale_grad: T.Tensor[(3,), dtype]
    mhc_base_grad: T.Tensor[(mhc_mult3,), dtype]

    with T.Kernel(mhc_mult3 + 3, threads=128) as pid:
        grad_reducer = T.alloc_reducer((1,), dtype)
        T.reducer_init(grad_reducer)

        for row in T.Parallel(num_sms):
            if pid < mhc_mult3:
                T.reducer_update(grad_reducer[0], mhc_base_grad_partial[row, pid])
            else:
                T.reducer_update(grad_reducer[0], mhc_scale_grad_partial[row, pid - mhc_mult3])

        grad_frag = T.alloc_fragment((1,), dtype)
        T.finalize_reducer(grad_reducer, grad_frag)

        for _ in T.Parallel(1):
            if pid < mhc_mult3:
                mhc_base_grad[pid] = grad_frag[0]
            else:
                mhc_scale_grad[pid - mhc_mult3] = grad_frag[0]
