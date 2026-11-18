import tilelang

from tilelang import language as T
from tile_kernels.config import get_is_tang


_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WGMMA: True,
}


@tilelang.jit
def mhc_fn_normw_merge_fwd_cuda(fn, normw, out_fn, use_pdl: bool = False):
    m = T.const('m')
    n = T.const('n')

    fn: T.Tensor[(m, n), T.float32]
    normw: T.Tensor[n, T.float32]
    out_fn: T.Tensor[(m, n), T.float32]

    n_blk = 1024 if m * n >= 262144 else 256
    with T.Kernel(m, T.ceildiv(n, n_blk)) as (pid_m, pid_n):
        if use_pdl:
            T.pdl_sync()
        for i1_n in T.Parallel(n_blk):
            i_n = pid_n * n_blk + i1_n
            if i_n < n:
                out_fn[pid_m, i_n] = fn[pid_m, i_n] * normw[i_n]
        if use_pdl:
            T.pdl_trigger()


@tilelang.jit
def mhc_fn_normw_merge_bwd_cuda(fn, normw, out_fn_grad, fn_grad, normw_grad, use_pdl: bool = False):
    m = T.const('m')
    n = T.const('n')

    fn: T.Tensor[(m, n), T.float32]
    normw: T.Tensor[n, T.float32]
    out_fn_grad: T.Tensor[(m, n), T.float32]
    fn_grad: T.Tensor[(m, n), T.float32]
    normw_grad: T.Tensor[n, T.float32]

    n_blk = 256
    with T.Kernel(T.ceildiv(n, n_blk)) as pid_n:
        if use_pdl:
            T.pdl_sync()
        normw_frag = T.alloc_fragment(n_blk, T.float32)
        T.copy(normw[pid_n * n_blk], normw_frag)

        normw_grad_frag = T.alloc_fragment(n_blk, T.float32)
        T.clear(normw_grad_frag)

        for i_m in T.serial(m):
            for i1_n in T.Parallel(n_blk):
                i_n = pid_n * n_blk + i1_n
                if i_n < n:
                    fn_grad[i_m, i_n] += out_fn_grad[i_m, i_n] * normw_frag[i1_n]
                    normw_grad_frag[i1_n] += out_fn_grad[i_m, i_n] * fn[i_m, i_n]

        for i1_n in T.Parallel(n_blk):
            normw_grad[pid_n * n_blk + i1_n] += normw_grad_frag[i1_n]
        if use_pdl:
            T.pdl_trigger()


@tilelang.jit(
    pass_configs={**_PASS_CONFIGS, tilelang.PassConfigKey.TL_TANG_DISABLE_WARP_ALU: True} if get_is_tang() else _PASS_CONFIGS
)
def mhc_reduce_partials_and_rmsnorm_fwd_cuda(
    out_mul_splitted,
    sqrsum_splitted,
    out_mul,
    sqrsum,
    out,
    rms_group_size: int,
    rms_eps: float,
    n_splits: int,
    use_pdl: bool = False,
):
    num_tokens = T.dynamic('num_tokens')
    mhc_mult3 = T.const('mhc_mult3')
    n_rms_group = T.const('n_rms_group')
    n_thr = 32

    out_mul_splitted: T.Tensor[(n_splits, num_tokens, n_rms_group, mhc_mult3), T.float32]
    sqrsum_splitted: T.Tensor[(n_splits, num_tokens, n_rms_group), T.float32]
    out_mul: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32]
    sqrsum: T.Tensor[(num_tokens, n_rms_group), T.float32]
    out: T.Tensor[(num_tokens, mhc_mult3), T.float32]

    with T.Kernel(num_tokens, threads=n_thr) as pid:
        if use_pdl:
            T.pdl_sync()
        rms = T.alloc_fragment(1, T.float32)
        out_l = T.alloc_fragment(mhc_mult3, T.float32)
        out_l0 = T.alloc_fragment(mhc_mult3, T.float32)
        T.clear(out_l)
        for k in T.serial(n_rms_group):
            rms[0] = 0
            for i_split in T.serial(n_splits):
                rms[0] += sqrsum_splitted[i_split, pid, k]
            if T.get_thread_binding() == 0:
                sqrsum[pid, k] = rms[0]
            rms[0] = T.rsqrt(rms[0] / rms_group_size + rms_eps)
            for j in T.Parallel(mhc_mult3):
                out_l0[j] = 0
                for i_split in T.serial(n_splits):
                    out_l0[j] += out_mul_splitted[i_split, pid, k, j]
                out_l[j] += out_l0[j] * rms[0]
            T.copy(out_l0, out_mul[pid, k, :])
        T.copy(out_l[:], out[pid, :])
        if use_pdl:
            T.pdl_trigger()


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def mhc_reduce_partials_and_rmsnorm_bwd_cuda(
    out_grad,
    out_mul,
    sqrsum,
    out_mul_grad,
    sqrsum_grad,
    rms_group_size: int,
    rms_eps: float,
    use_pdl: bool = False,
):
    num_tokens = T.dynamic('num_tokens')
    mhc_mult3 = T.const('mhc_mult3')
    n_rms_group = T.const('n_rms_group')
    n_thr = 32

    out_grad: T.Tensor[(num_tokens, mhc_mult3), T.float32]
    out_mul: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32]
    sqrsum: T.Tensor[(num_tokens, n_rms_group), T.float32]
    out_mul_grad: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32]
    sqrsum_grad: T.Tensor[(num_tokens, n_rms_group), T.float32]

    with T.Kernel(num_tokens, n_rms_group, threads=n_thr) as (pid_i, pid_k):
        if use_pdl:
            T.pdl_sync()
        sqrsum_frag = T.alloc_fragment(1, T.float32)
        sqrsum_frag[0] = sqrsum[pid_i, pid_k]
        rms_frag = T.alloc_fragment(1, T.float32)
        rms_frag[0] = T.rsqrt(sqrsum_frag[0] / rms_group_size + rms_eps)

        rms_grad_reducer = T.alloc_reducer(1, T.float32, op='sum')
        rms_grad_result = T.alloc_fragment(1, T.float32)
        T.reducer_init(rms_grad_reducer)
        for j in T.Parallel(mhc_mult3):
            out_mul_grad[pid_i, pid_k, j] = out_grad[pid_i, j] * rms_frag[0]
            T.reducer_update(rms_grad_reducer[0], out_grad[pid_i, j] * out_mul[pid_i, pid_k, j])
        T.finalize_reducer(rms_grad_reducer, rms_grad_result)

        for kk in T.Parallel(1):
            sqrsum_grad[pid_i, pid_k + kk] = rms_grad_result[kk] * rms_frag[kk] / (sqrsum_frag[kk] + rms_eps * rms_group_size) / -2
        if use_pdl:
            T.pdl_trigger()
