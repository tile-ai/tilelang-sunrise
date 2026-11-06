from tile_kernels.config import get_is_tang, get_num_vec_cores, get_pdl, is_ascend
from tile_kernels.mhc.expand_cuda import expand_to_mhc_bwd_cuda, expand_to_mhc_fwd_cuda


def _asc_fwd_h_blk(hidden: int, mhc_mult: int) -> int:
    num_stages = 7
    ub_budget_bytes = 245 * 1024
    max_h_blk = ub_budget_bytes // ((mhc_mult + 1) * 2 * num_stages)
    h_blk = min(hidden, max_h_blk // 128 * 128)
    while h_blk > 128 and hidden % h_blk != 0:
        h_blk -= 128
    return h_blk


def _asc_h_blk(hidden: int) -> int:
    # Keep h_blk small enough that mhc * h_blk fits in UB with multi-stage.
    # For large hidden, split so that h_blk <= 4096.
    h_blk = hidden if hidden <= 4096 else (hidden + 1) // 2
    h_blk = (h_blk + 63) // 64 * 64
    # Cap at 4096 to fit UB budget for mhc * h_blk * num_stages
    while h_blk > 4096 and hidden % (h_blk // 2) == 0:
        h_blk //= 2
    return h_blk


def expand_to_mhc_fwd(x, o):
    if get_is_tang():
        from tile_kernels.mhc.expand_tang import (
            _expand_to_mhc_fwd_fallback,
            choose_expand_blocks,
            expand_to_mhc_fwd_tl,
        )

        hidden = x.shape[-1]
        mhc_mult = o.shape[-2]
        num_tokens = x.numel() // hidden
        if num_tokens == 0:
            return
        blk_n, blk_h = choose_expand_blocks(hidden, mhc_mult, num_tokens)
        if num_tokens % blk_n == 0 and hidden % blk_h == 0:
            expand_to_mhc_fwd_tl(hidden, mhc_mult, blk_n, blk_h)(
                x.view(num_tokens, hidden), o.view(num_tokens, mhc_mult * hidden),
            )
        else:
            _expand_to_mhc_fwd_fallback(hidden, mhc_mult)(
                x.view(num_tokens, hidden), o.view(num_tokens, mhc_mult, hidden),
            )
    elif is_ascend():
        from tile_kernels.mhc.expand_asc import expand_to_mhc_fwd_asc

        hidden = x.shape[-1]
        assert hidden % 128 == 0, f'Ascend expand fwd requires hidden divisible by 128, got {hidden}'
        h_blk = _asc_fwd_h_blk(hidden, o.shape[-2])
        assert hidden % h_blk == 0, f'Ascend expand fwd requires hidden divisible by h_blk, got {hidden=}, {h_blk=}'
        expand_to_mhc_fwd_asc(x, o, h_blk=h_blk, num_stages=7, num_vec_cores=get_num_vec_cores())
    else:
        expand_to_mhc_fwd_cuda(x, o, use_pdl=get_pdl())


def expand_to_mhc_bwd(o_grad, x_grad):
    if get_is_tang():
        from tile_kernels.mhc.expand_tang import expand_to_mhc_bwd_tl

        hidden, mhc_mult = o_grad.shape[-1], o_grad.shape[-2]
        num_tokens = x_grad.numel() // hidden
        if num_tokens:
            expand_to_mhc_bwd_tl(hidden, mhc_mult)(
                o_grad.view(num_tokens, mhc_mult, hidden), x_grad.view(num_tokens, hidden),
            )
    elif is_ascend():
        from tile_kernels.mhc.expand_asc import expand_to_mhc_bwd_asc

        hidden = x_grad.shape[-1]
        assert hidden % 64 == 0, f'Ascend expand bwd requires hidden divisible by 64, got {hidden}'
        h_blk = _asc_h_blk(hidden)
        assert hidden % h_blk == 0, f'Ascend expand bwd requires hidden divisible by h_blk, got {hidden=}, {h_blk=}'
        expand_to_mhc_bwd_asc(o_grad, x_grad, h_blk=h_blk, num_stages=3, num_vec_cores=get_num_vec_cores())
    else:
        expand_to_mhc_bwd_cuda(o_grad, x_grad, use_pdl=get_pdl())
