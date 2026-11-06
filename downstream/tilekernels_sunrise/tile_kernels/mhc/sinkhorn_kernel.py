from tile_kernels.config import get_is_tang, get_num_vec_cores, get_pdl, is_ascend
from tile_kernels.mhc.sinkhorn_cuda import mhc_sinkhorn_bwd_cuda
from tile_kernels.mhc.sinkhorn_cuda import mhc_sinkhorn_fwd_cuda


def mhc_sinkhorn_fwd(comb_res_mix, comb_res_mix_out, repeat: int, eps: float):
    if get_is_tang() and comb_res_mix.shape[0] == 0:
        return
    if is_ascend():
        from tile_kernels.mhc.sinkhorn_asc import mhc_sinkhorn_fwd_asc

        mhc_sinkhorn_fwd_asc(comb_res_mix, comb_res_mix_out, repeat, eps, token_block_size=4, num_vec_cores=get_num_vec_cores())
    else:
        mhc_sinkhorn_fwd_cuda(comb_res_mix, comb_res_mix_out, repeat, eps, token_block_size=8, use_pdl=get_pdl() and not get_is_tang())


def mhc_sinkhorn_bwd(grad_output, x, grad_input, repeat: int, eps: float):
    if get_is_tang() and x.shape[0] == 0:
        return
    if is_ascend():
        from tile_kernels.mhc.sinkhorn_asc import mhc_sinkhorn_bwd_asc

        mhc_sinkhorn_bwd_asc(grad_output, x, grad_input, repeat, eps, token_block_size=8, num_vec_cores=get_num_vec_cores())
    else:
        mhc_sinkhorn_bwd_cuda(grad_output, x, grad_input, repeat, eps, token_block_size=8, use_pdl=get_pdl() and not get_is_tang())
