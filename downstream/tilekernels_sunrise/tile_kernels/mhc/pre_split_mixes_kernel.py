import torch
from tile_kernels.config import get_is_tang, get_num_sms, get_num_vec_cores, get_pdl, is_ascend
from tile_kernels.mhc.pre_split_mixes_cuda import (
    mhc_pre_split_mixes_bwd_cuda,
)
from tile_kernels.mhc.pre_split_mixes_cuda import (
    mhc_pre_split_mixes_bwd_reduce_cuda,
)
from tile_kernels.mhc.pre_split_mixes_cuda import (
    mhc_pre_split_mixes_fwd_cuda,
)


def mhc_pre_split_mixes_fwd(
    input_mixes,
    mhc_scale,
    mhc_base,
    pre_layer_mix,
    post_layer_mix,
    comb_res_mix,
    mhc_post_mult_value: float,
    mhc_pre_eps: float,
):
    if get_is_tang() and input_mixes.shape[0] == 0:
        return
    if is_ascend():
        from tile_kernels.mhc.pre_split_mixes_asc import mhc_pre_split_mixes_fwd_simd_mhc4_asc

        mhc_pre_split_mixes_fwd_simd_mhc4_asc(
            input_mixes,
            mhc_scale,
            mhc_base,
            pre_layer_mix,
            post_layer_mix,
            comb_res_mix,
            mhc_post_mult_value,
            mhc_pre_eps,
            token_block_size=64,
            num_stages=2,
            num_vec_cores=get_num_vec_cores(),
        )
    else:
        mhc_pre_split_mixes_fwd_cuda(
            input_mixes,
            mhc_scale,
            mhc_base,
            pre_layer_mix,
            post_layer_mix,
            comb_res_mix,
            mhc_post_mult_value,
            mhc_pre_eps,
            token_block_size=32,
            use_pdl=get_pdl() and not get_is_tang(),
        )


def mhc_pre_split_mixes_bwd(
    pre_layer_mix_grad,
    post_layer_mix_grad,
    comb_res_mix_grad,
    input_mixes,
    post_layer_mix,
    mhc_scale,
    mhc_base,
    input_mixes_grad,
    mhc_scale_grad,
    mhc_base_grad,
    mhc_post_mult_value: float,
):
    if get_is_tang() and input_mixes.shape[0] == 0:
        mhc_scale_grad.zero_()
        mhc_base_grad.zero_()
        return
    num_sms = get_num_vec_cores() if is_ascend() else get_num_sms()
    mhc_scale_grad_partial = torch.empty(num_sms, *mhc_scale.shape, dtype=mhc_scale.dtype, device=mhc_scale.device)
    mhc_base_grad_partial = torch.empty(num_sms, *mhc_base.shape, dtype=mhc_base.dtype, device=mhc_base.device)

    if is_ascend():
        from tile_kernels.mhc.pre_split_mixes_asc import mhc_pre_split_mixes_bwd_mhc4_asc

        mhc_pre_split_mixes_bwd_mhc4_asc(
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
            mhc_post_mult_value,
            token_block_size=64,
            num_stages=2,
            num_vec_cores=num_sms,
        )
        mhc_scale_grad.copy_(mhc_scale_grad_partial.sum(0))
        mhc_base_grad.copy_(mhc_base_grad_partial.sum(0))
    else:
        mhc_pre_split_mixes_bwd_cuda(
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
            mhc_post_mult_value,
            token_block_size=32,
            use_pdl=get_pdl() and not get_is_tang(),
        )
        mhc_pre_split_mixes_bwd_reduce_cuda(
            mhc_scale_grad_partial,
            mhc_base_grad_partial,
            mhc_scale_grad,
            mhc_base_grad,
            num_sms=num_sms,
        )
