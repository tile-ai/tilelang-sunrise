import torch


def expand_to_mhc_ref(hidden: torch.Tensor, mhc_mult: int) -> torch.Tensor:
    hidden = hidden.cpu()
    return hidden.unsqueeze(-2).expand(*hidden.shape[:-1], mhc_mult, hidden.shape[-1]).contiguous()


def sinkhorn_normalize_ref(x: torch.Tensor, repeat: int = 10, eps: float = 1e-6) -> torch.Tensor:
    x = x.cpu()
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def mhc_head_compute_mix_ref(
    input_mix: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_pre_eps: float,
) -> torch.Tensor:
    input_mix = input_mix.cpu()
    mhc_scale = mhc_scale.cpu()
    mhc_base = mhc_base.cpu()
    mhc_head_layer_mix = input_mix * mhc_scale + mhc_base
    return torch.sigmoid(mhc_head_layer_mix) + mhc_pre_eps


def mhc_pre_split_mixes_ref(
    input_mixes: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_mult: int,
    mhc_post_mult_value: float,
    mhc_pre_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_mixes = input_mixes.cpu()
    mhc_scale = mhc_scale.cpu()
    mhc_base = mhc_base.cpu()
    a, b = input_mixes.shape[:2]
    mhc_scale = torch.cat(
        [
            mhc_scale[0].expand(mhc_mult),
            mhc_scale[1].expand(mhc_mult),
            mhc_scale[2].expand(mhc_mult * mhc_mult),
        ],
    )
    input_mixes = input_mixes * mhc_scale + mhc_base

    pre_layer_mix = input_mixes[:, :, :mhc_mult].sigmoid().unsqueeze(-1) + mhc_pre_eps
    post_layer_mix = (input_mixes[:, :, mhc_mult : 2 * mhc_mult].sigmoid() * mhc_post_mult_value).unsqueeze(-1)
    comb_res_mix = input_mixes[:, :, 2 * mhc_mult :].view(a, b, mhc_mult, mhc_mult)

    return pre_layer_mix, post_layer_mix, comb_res_mix


def mhc_pre_apply_mix_ref(x: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
    x = x.cpu()
    mix = mix.cpu()
    return (x * mix).sum(-2).bfloat16()


def mhc_post_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    x = x.cpu()
    residual = residual.cpu()
    post_layer_mix = post_layer_mix.cpu()
    comb_res_mix = comb_res_mix.cpu()
    term2 = torch.einsum('abmn,abmc->abnc', comb_res_mix, residual.float())
    return (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()


def mhc_pre_norm_fn_ref(
    residual: torch.Tensor,
    mhc_fn: torch.Tensor,
    mhc_norm_weight: torch.Tensor | None,
    mhc_norm_eps: float,
) -> torch.Tensor:
    residual = residual.cpu()
    mhc_fn = mhc_fn.cpu()
    if mhc_norm_weight is not None:
        mhc_norm_weight = mhc_norm_weight.cpu()
        mhc_fn = mhc_fn * mhc_norm_weight
    residual = residual.flatten(2, 3).float()
    assert mhc_fn.dtype == residual.dtype == torch.float
    mhc_mult = mhc_fn.shape[0]
    rms_group_size = mhc_fn.shape[-1]
    mixes = torch.einsum(
        'mbk,nbk->mbn',
        residual.view(-1, 1, rms_group_size),
        mhc_fn.view(mhc_mult, 1, rms_group_size),
    )
    sqrsum = residual.view(-1, 1, rms_group_size).square().sum(-1)
    mixes = (mixes * (sqrsum.unsqueeze(-1) / rms_group_size + mhc_norm_eps).rsqrt()).sum(-2)
    return mixes.view(*residual.shape[:2], -1)
