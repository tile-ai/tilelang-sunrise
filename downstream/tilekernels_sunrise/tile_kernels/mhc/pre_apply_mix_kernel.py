import math

from tile_kernels.config import get_is_tang, get_num_vec_cores, get_pdl, is_ascend
from tile_kernels.mhc.pre_apply_mix_cuda import mhc_pre_apply_mix_bwd_cuda, mhc_pre_apply_mix_fwd_cuda
from tile_kernels.utils import align, ceil_div


def mhc_pre_apply_mix_fwd(x, mix, o):
    if get_is_tang():
        from tile_kernels.mhc.pre_apply_mix_tang import _mhc_pre_apply_mix_fwd

        num_tokens, mhc, hidden = x.shape
        if num_tokens:
            _mhc_pre_apply_mix_fwd(num_tokens, mhc, hidden)(x, mix, o)
    elif is_ascend():
        from tile_kernels.mhc.pre_apply_mix_asc import mhc_pre_apply_mix_fwd_asc

        num_tokens, mhc, hidden = x.shape
        h_blk = hidden if hidden <= 6144 else min(ceil_div(hidden, 2), 6144)
        h_blk = align(h_blk, 64)
        mhc_pre_apply_mix_fwd_asc(x, mix, o, h_blk=h_blk, num_vec_cores=get_num_vec_cores())
    else:
        hidden = x.shape[-1]
        mhc_pre_apply_mix_fwd_cuda(
            x,
            mix,
            o,
            n_thr=64,
            h_blk=math.gcd(1024, hidden),
            use_pdl=get_pdl(),
        )


def mhc_pre_apply_mix_bwd(o_grad, x, mix, x_grad):
    if get_is_tang():
        from tile_kernels.mhc.pre_apply_mix_tang import _mhc_pre_apply_mix_bwd

        num_tokens, mhc, hidden = x.shape
        if num_tokens == 0:
            return mix.new_empty((0, mhc))
        return _mhc_pre_apply_mix_bwd(num_tokens, mhc, hidden)(o_grad, x, mix, x_grad)
    elif is_ascend():
        from tile_kernels.mhc.pre_apply_mix_asc import mhc_pre_apply_mix_bwd_asc

        hidden = x.shape[2]
        if hidden <= 4096:
            h_blk = hidden
            num_stages = 3
        else:
            h_blk = min(ceil_div(hidden, 4), 3072)
            h_blk = align(h_blk, 64)
            num_stages = 4
        return mhc_pre_apply_mix_bwd_asc(o_grad, x, mix, x_grad, h_blk=h_blk, num_stages=num_stages, num_vec_cores=get_num_vec_cores())
    else:
        hidden = x.shape[-1]
        return mhc_pre_apply_mix_bwd_cuda(
            o_grad,
            x,
            mix,
            x_grad,
            n_thr=128,
            h_blk=math.gcd(1024, hidden),
            use_pdl=get_pdl(),
        )
