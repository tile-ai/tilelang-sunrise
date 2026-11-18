from tile_kernels.config import get_is_tang, get_num_vec_cores, get_pdl, is_ascend
from tile_kernels.mhc.head_compute_mix_cuda import (
    mhc_head_compute_mix_bwd_cuda,
)
from tile_kernels.mhc.head_compute_mix_cuda import (
    mhc_head_compute_mix_fwd_cuda,
)


def _get_ascend_token_block_size(num_tokens: int, mhc_mult: int) -> int:
    assert mhc_mult == 4, f'Ascend head_compute_mix only supports mHC4, got {mhc_mult}'
    # Target: each core handles >= 2 blocks for pipeline overlap,
    # and each block is as large as possible to amortize launch overhead.
    num_cores = get_num_vec_cores()
    min_blocks_per_core = 2
    # T.SimtVF uses 256 threads. Larger blocks vectorize T.sigmoid and expose
    # an unsupported vector tirx.exp in the current Ascend codegen.
    max_tb = min(64, num_tokens // (num_cores * min_blocks_per_core))
    if max_tb < 1:
        max_tb = 1
    # Round down to power of 2, keep >= 1
    tb = 1
    while tb * 2 <= max_tb:
        tb *= 2
    return tb


def mhc_head_compute_mix_fwd(
    input_mix,
    mhc_scale,
    mhc_base,
    output_mix,
    mhc_pre_eps: float,
    token_block_size: int = 0,
):
    if get_is_tang() and input_mix.shape[0] == 0:
        return
    if is_ascend():
        assert input_mix.shape[1] == 4, f'Ascend head_compute_mix only supports mHC4, got {input_mix.shape[1]}'
        from tile_kernels.mhc.head_compute_mix_asc import mhc_head_compute_mix_fwd_asc

        if token_block_size == 0:
            token_block_size = _get_ascend_token_block_size(input_mix.shape[0], input_mix.shape[1])
        mhc_head_compute_mix_fwd_asc(
            input_mix,
            mhc_scale,
            mhc_base,
            output_mix,
            mhc_pre_eps,
            token_block_size=token_block_size,
            num_vec_cores=get_num_vec_cores(),
        )
    else:
        if token_block_size == 0:
            token_block_size = 32
        mhc_head_compute_mix_fwd_cuda(
            input_mix,
            mhc_scale,
            mhc_base,
            output_mix,
            mhc_pre_eps,
            token_block_size=token_block_size,
            use_pdl=get_pdl() and not get_is_tang(),
        )


def mhc_head_compute_mix_bwd(
    output_mix_grad,
    input_mix,
    mhc_scale,
    mhc_base,
    input_mix_grad,
    mhc_scale_grad_partial,
    mhc_base_grad_partial,
    num_sms: int,
    token_block_size: int = 0,
):
    if get_is_tang() and input_mix.shape[0] == 0:
        mhc_scale_grad_partial.zero_()
        mhc_base_grad_partial.zero_()
        return
    if is_ascend():
        assert input_mix.shape[1] == 4, f'Ascend head_compute_mix only supports mHC4, got {input_mix.shape[1]}'
        from tile_kernels.mhc.head_compute_mix_asc import mhc_head_compute_mix_bwd_asc

        if token_block_size == 0:
            token_block_size = _get_ascend_token_block_size(input_mix.shape[0], input_mix.shape[1])
        mhc_head_compute_mix_bwd_asc(
            output_mix_grad,
            input_mix,
            mhc_scale,
            mhc_base,
            input_mix_grad,
            mhc_scale_grad_partial,
            mhc_base_grad_partial,
            token_block_size=token_block_size,
            num_vec_cores=num_sms,
        )
    else:
        if token_block_size == 0:
            token_block_size = 32
        mhc_head_compute_mix_bwd_cuda(
            output_mix_grad,
            input_mix,
            mhc_scale,
            mhc_base,
            input_mix_grad,
            mhc_scale_grad_partial,
            mhc_base_grad_partial,
            num_sms,
            token_block_size=token_block_size,
            use_pdl=get_pdl() and not get_is_tang(),
        )
