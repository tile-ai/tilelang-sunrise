import torch

from tile_kernels.mhc.norm_fn_kernel import (
    _mhc_pre_norm_fn_fwd_mul_split_channel,
    round_to_tf32,
)
from tile_kernels.mhc.pre_big_fuse_kernel import _mhc_pre_big_fuse
from tile_kernels.mhc.pre_big_fuse_post_fused import _mhc_pre_big_fuse_post_fused


def _run_fwd_mul(residual_flat, fn, hidden_size, mhc_mult, mhc_mult3,
                 n_splits, token_block, gemm_hidden_block):
    """Run the pre-norm GEMM and return gemm intermediates (shared helper)."""
    gemm_out_mul = torch.empty(
        n_splits, residual_flat.shape[0], mhc_mult3,
        dtype=torch.float32, device=residual_flat.device,
    )
    gemm_out_sqrsum = torch.empty(
        n_splits, residual_flat.shape[0],
        dtype=torch.float32, device=residual_flat.device,
    )
    mhc_hidden_size = mhc_mult * hidden_size
    fwd_mul_kernel = _mhc_pre_norm_fn_fwd_mul_split_channel(
        mhc_mult3, mhc_mult, hidden_size, n_splits,
        token_block=token_block, hidden_block=gemm_hidden_block,
    )
    fwd_mul_kernel(
        residual_flat.view(-1, mhc_hidden_size),
        fn,
        gemm_out_mul,
        gemm_out_sqrsum,
    )
    return gemm_out_mul, gemm_out_sqrsum


def _get_splits(num_tokens, hidden_size):
    """Return n_splits for the given (num_tokens, hidden_size)."""
    if num_tokens <= 512:
        return {1280: 40, 2560: 80, 4096: 32}.get(hidden_size, 32)
    elif num_tokens <= 1024:
        return {1280: 40, 2560: 40, 4096: 32}.get(hidden_size, 32)
    else:
        return {1280: 8, 2560: 8, 4096: 32, 8192: 32}.get(hidden_size, 8)


def mhc_pre_big_fuse(
    residual: torch.Tensor,
    fn: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert mhc_scale.dtype == torch.float32
    assert mhc_base.dtype == torch.float32

    mhc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2

    assert fn.shape[0] == mhc_mult3
    assert fn.shape[1] == mhc_mult * hidden_size
    assert mhc_scale.shape == (3,)
    assert mhc_base.shape == (mhc_mult3,)

    outer_shape = residual.shape[:-2]
    residual_flat = residual.view(-1, mhc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    post_mix = torch.empty(num_tokens, mhc_mult, dtype=torch.float32, device=residual.device)
    comb_mix = torch.empty(num_tokens, mhc_mult2, dtype=torch.float32, device=residual.device)
    layer_input = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device)

    token_block = 16 if num_tokens <= 512 else 32
    _GEMM_HIDDEN_BLOCK = {1280: 128, 2560: 128, 4096: 128}
    gemm_hidden_block = _GEMM_HIDDEN_BLOCK.get(hidden_size, 128)

    fn = round_to_tf32(fn)
    n_splits = _get_splits(num_tokens, hidden_size)

    gemm_out_mul, gemm_out_sqrsum = _run_fwd_mul(
        residual_flat, fn, hidden_size, mhc_mult, mhc_mult3,
        n_splits, token_block, gemm_hidden_block,
    )

    _mhc_pre_big_fuse(
        hidden_size, rms_eps, mhc_pre_eps, mhc_sinkhorn_eps,
        mhc_post_mult_value, sinkhorn_repeat,
        n_splits=n_splits, mhc_mult=mhc_mult,
    )(
        gemm_out_mul, gemm_out_sqrsum, mhc_scale, mhc_base,
        residual_flat, post_mix, comb_mix, layer_input,
    )

    post_mix = post_mix.view(*outer_shape, mhc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, mhc_mult, mhc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)

    return post_mix, comb_mix, layer_input


def mhc_pre_big_fuse_post(
    residual: torch.Tensor,
    fn: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused pre_big_fuse + post_fwd — eliminates the layer_input HBM round-trip.

    Returns:
        post_mix:  [..., mhc_mult, 1]         float32
        comb_mix:  [..., mhc_mult, mhc_mult]   float32
        out:       [..., mhc_mult, hidden_size]  bf16   (NOT layer_input!)
    """
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert mhc_scale.dtype == torch.float32
    assert mhc_base.dtype == torch.float32

    mhc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2

    assert fn.shape[0] == mhc_mult3
    assert fn.shape[1] == mhc_mult * hidden_size
    assert mhc_scale.shape == (3,)
    assert mhc_base.shape == (mhc_mult3,)

    outer_shape = residual.shape[:-2]
    residual_flat = residual.view(-1, mhc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    post_mix = torch.empty(num_tokens, mhc_mult, dtype=torch.float32, device=residual.device)
    comb_mix = torch.empty(num_tokens, mhc_mult2, dtype=torch.float32, device=residual.device)
    out = torch.empty(num_tokens, mhc_mult, hidden_size, dtype=torch.bfloat16, device=residual.device)

    token_block = 16 if num_tokens <= 512 else 32
    _GEMM_HIDDEN_BLOCK = {1280: 128, 2560: 128, 4096: 128}
    gemm_hidden_block = _GEMM_HIDDEN_BLOCK.get(hidden_size, 128)

    fn = round_to_tf32(fn)
    n_splits = _get_splits(num_tokens, hidden_size)

    gemm_out_mul, gemm_out_sqrsum = _run_fwd_mul(
        residual_flat, fn, hidden_size, mhc_mult, mhc_mult3,
        n_splits, token_block, gemm_hidden_block,
    )

    _mhc_pre_big_fuse_post_fused(
        hidden_size, rms_eps, mhc_pre_eps, mhc_sinkhorn_eps,
        mhc_post_mult_value, sinkhorn_repeat,
        n_splits=n_splits, mhc_mult=mhc_mult,
    )(
        gemm_out_mul, gemm_out_sqrsum, mhc_scale, mhc_base,
        residual_flat, post_mix, comb_mix, out,
    )

    post_mix = post_mix.view(*outer_shape, mhc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, mhc_mult, mhc_mult)
    out = out.view(*outer_shape, mhc_mult, hidden_size)

    return post_mix, comb_mix, out
