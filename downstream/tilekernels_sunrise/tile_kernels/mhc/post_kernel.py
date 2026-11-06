import math

import torch

from tile_kernels.config import get_is_tang, get_num_vec_cores, get_pdl, is_ascend
from tile_kernels.mhc.post_cuda import mhc_post_bwd_cuda, mhc_post_fwd_cuda
from tile_kernels.utils import align, ceil_div


def mhc_post_fwd(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    num_seqs, num_tokens, mhc, hidden = residual.shape

    assert x.dtype == torch.bfloat16, f'{x.dtype=}'
    assert residual.dtype == torch.bfloat16, f'{residual.dtype=}'
    assert post_layer_mix.dtype == torch.float32, f'{post_layer_mix.dtype=}'
    assert comb_res_mix.dtype == torch.float32, f'{comb_res_mix.dtype=}'
    assert x.shape == (num_seqs, num_tokens, hidden), f'{x.shape=}'
    assert post_layer_mix.shape == (num_seqs, num_tokens, mhc, 1), f'{post_layer_mix.shape=}'
    assert comb_res_mix.shape == (num_seqs, num_tokens, mhc, mhc), f'{comb_res_mix.shape=}'

    residual = residual.contiguous()
    assert x.is_contiguous()
    assert post_layer_mix.is_contiguous()
    assert comb_res_mix.is_contiguous()

    if out is None:
        out = torch.empty_like(residual)

    n = num_seqs * num_tokens
    if get_is_tang() and n == 0:
        return out
    if is_ascend():
        from tile_kernels.mhc.post_asc import mhc_post_fwd_asc

        h_blk = hidden if hidden <= 4096 else min(ceil_div(hidden, 2), 4096)
        h_blk = align(h_blk, 64)
        mhc_post_fwd_asc(
            comb_res_mix.view(n, mhc, mhc),
            residual.view(n, mhc, hidden),
            post_layer_mix.view(n, mhc),
            x.view(n, hidden),
            out.view(n, mhc, hidden),
            h_blk=h_blk,
            num_vec_cores=get_num_vec_cores(),
        )
    else:
        h_blk = math.gcd(hidden, 256 if get_is_tang() else 1024)
        vec = math.gcd(8, h_blk // 128)
        assert h_blk % 128 == 0

        mhc_post_fwd_cuda(
            comb_res_mix.flatten(0, 1),
            residual.flatten(0, 1),
            post_layer_mix.flatten(0, 1).squeeze(-1),
            x.flatten(0, 1),
            out.flatten(0, 1),
            vec=vec,
            n_thr=128,
            h_blk=h_blk,
            use_pdl=get_pdl() and not get_is_tang(),
        )
    return out


def mhc_post_bwd(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    d_o: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = d_o.shape[0] * d_o.shape[1]
    mhc = d_o.shape[2]
    h = d_o.shape[3]
    if get_is_tang() and n == 0:
        return torch.empty_like(x), torch.empty_like(residual), torch.empty_like(post_layer_mix), torch.empty_like(comb_res_mix)

    if is_ascend():
        from tile_kernels.mhc.post_asc import mhc_post_bwd_asc

        h_blk = h if h <= 4096 else min(ceil_div(h, 2), 4096)
        h_blk = align(h_blk, 64)
        num_stages = 3 if h_blk <= 2560 else 2
        d_comb_res_mix, d_residual, d_post_layer_mix, d_x = mhc_post_bwd_asc(
            d_o.contiguous().view(n, mhc, h),
            comb_res_mix.view(n, mhc, mhc),
            residual.view(n, mhc, h),
            post_layer_mix.view(n, mhc),
            x.view(n, h),
            h_blk=h_blk,
            num_stages=num_stages,
            num_vec_cores=get_num_vec_cores(),
        )
    else:
        h_blk = math.gcd(h, 512)
        d_comb_res_mix, d_residual, d_post_layer_mix, d_x = mhc_post_bwd_cuda(
            d_o.contiguous().view(n, mhc, h),
            comb_res_mix.view(n, mhc, mhc),
            residual.view(n, mhc, h),
            post_layer_mix.view(n, mhc),
            x.view(n, h),
            n_thr=64,
            h_blk=h_blk,
            use_pdl=get_pdl() and not get_is_tang(),
        )
    assert isinstance(d_x, torch.Tensor)
    assert isinstance(d_post_layer_mix, torch.Tensor)
    assert isinstance(d_comb_res_mix, torch.Tensor)
    assert isinstance(d_residual, torch.Tensor)

    return (
        d_x.view_as(x),
        d_residual.view_as(residual),
        d_post_layer_mix.view_as(post_layer_mix),
        d_comb_res_mix.view_as(comb_res_mix),
    )
