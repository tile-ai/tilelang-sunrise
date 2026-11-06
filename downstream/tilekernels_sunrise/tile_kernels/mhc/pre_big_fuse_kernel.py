import math

from tile_kernels.config import get_is_tang, get_num_vec_cores, get_pdl, is_ascend
from tile_kernels.mhc.pre_big_fuse_cuda import mhc_pre_big_fuse_fwd_cuda

# Ascend fused kernel token block; num_tokens must be a multiple of it.
_ASCEND_TOKEN_BLOCK = 16


def mhc_pre_big_fuse_fwd(
    gemm_out_mul,
    gemm_out_sqrsum,
    mhc_scale,
    mhc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
):
    hidden_size = residual.shape[-1]
    if get_is_tang():
        from tile_kernels.mhc.pre_big_fuse_tang import _mhc_pre_big_fuse

        mhc_mult = residual.shape[-2]
        num_tokens = residual.numel() // (mhc_mult * hidden_size)
        assert gemm_out_mul.shape == (n_splits, num_tokens, mhc_mult * (mhc_mult + 2))
        assert gemm_out_sqrsum.shape == (n_splits, num_tokens)
        if num_tokens:
            _mhc_pre_big_fuse(
                hidden_size, rms_eps, mhc_pre_eps, mhc_sinkhorn_eps,
                mhc_post_mult_value, sinkhorn_repeat, n_splits=n_splits, mhc_mult=mhc_mult,
            )(
                gemm_out_mul, gemm_out_sqrsum, mhc_scale, mhc_base,
                residual.view(num_tokens, mhc_mult, hidden_size),
                post_mix.view(num_tokens, mhc_mult), comb_mix.view(num_tokens, mhc_mult * mhc_mult),
                layer_input.view(num_tokens, hidden_size),
            )
    elif is_ascend():
        mhc_mult = residual.shape[-2]
        assert mhc_mult == 4, f'Ascend pre_big_fuse only supports mHC4, got {mhc_mult}'
        assert gemm_out_mul.shape[0] == 1, 'Ascend pre_big_fuse expects the GEMM to reduce split-K internally (one partial)'
        from tile_kernels.mhc.pre_big_fuse_asc import mhc_pre_big_fuse_fwd_asc

        h_blk = min(math.gcd(512, hidden_size), 256)
        mhc_pre_big_fuse_fwd_asc(
            gemm_out_mul,
            gemm_out_sqrsum,
            mhc_scale,
            mhc_base,
            residual,
            post_mix,
            comb_mix,
            layer_input,
            rms_eps,
            mhc_pre_eps,
            mhc_sinkhorn_eps,
            mhc_post_mult_value,
            sinkhorn_repeat,
            h_blk=h_blk,
            n_splits=1,
            token_block=_ASCEND_TOKEN_BLOCK,
            num_stages=4,
            num_vec_cores=get_num_vec_cores(),
        )
    else:
        hidden_block = math.gcd(512, hidden_size)
        mhc_pre_big_fuse_fwd_cuda(
            gemm_out_mul,
            gemm_out_sqrsum,
            mhc_scale,
            mhc_base,
            residual,
            post_mix,
            comb_mix,
            layer_input,
            rms_eps,
            mhc_pre_eps,
            mhc_sinkhorn_eps,
            mhc_post_mult_value,
            sinkhorn_repeat,
            hidden_block,
            n_splits=n_splits,
            use_pdl=get_pdl(),
        )
