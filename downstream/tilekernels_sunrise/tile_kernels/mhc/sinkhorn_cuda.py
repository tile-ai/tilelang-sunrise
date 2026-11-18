import tilelang
from tilelang import language as T

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def mhc_sinkhorn_fwd_cuda(
    comb_res_mix,
    comb_res_mix_out,
    repeat: int,
    eps: float,
    token_block_size: int = 1,
    use_pdl: bool = False,
):
    num_tokens = T.dynamic('num_tokens')
    hidden_size = T.const('hidden_size')

    comb_res_mix: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    comb_res_mix_out: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]

    with T.Kernel(T.ceildiv(num_tokens, token_block_size)) as pid_x:
        if use_pdl:
            T.pdl_sync()
        comb_frag = T.alloc_fragment((token_block_size, hidden_size, hidden_size), T.float32)
        row_sum = T.alloc_fragment((token_block_size, hidden_size), T.float32)
        col_sum = T.alloc_fragment((token_block_size, hidden_size), T.float32)

        T.copy(comb_res_mix[pid_x * token_block_size, 0, 0], comb_frag)

        row_max = T.alloc_fragment((token_block_size, hidden_size), T.float32)
        T.reduce_max(comb_frag, row_max, dim=2)
        for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
            comb_frag[i, j, k] = T.exp(comb_frag[i, j, k] - row_max[i, j])
        T.reduce_sum(comb_frag, row_sum, dim=2)
        for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
            comb_frag[i, j, k] = comb_frag[i, j, k] / row_sum[i, j] + eps

        T.reduce_sum(comb_frag, col_sum, dim=1)
        for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
            comb_frag[i, j, k] = comb_frag[i, j, k] / (col_sum[i, k] + eps)

        for _ in T.serial(repeat - 1):
            T.reduce_sum(comb_frag, row_sum, dim=2)
            for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                comb_frag[i, j, k] = comb_frag[i, j, k] / (row_sum[i, j] + eps)

            T.reduce_sum(comb_frag, col_sum, dim=1)
            for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                comb_frag[i, j, k] = comb_frag[i, j, k] / (col_sum[i, k] + eps)

        T.copy(comb_frag, comb_res_mix_out[pid_x * token_block_size, 0, 0])
        if use_pdl:
            T.pdl_trigger()


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def mhc_sinkhorn_bwd_cuda(
    grad_output,
    x,
    grad_input,
    repeat: int,
    eps: float,
    token_block_size: int = 32,
    use_pdl: bool = False,
):
    num_tokens = T.dynamic('num_tokens')
    hidden_size = T.const('hidden_size')

    grad_output: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    x: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    grad_input: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]

    with T.Kernel(T.ceildiv(num_tokens, token_block_size)) as pid_x:
        if use_pdl:
            T.pdl_sync()
        grad_frag = T.alloc_fragment((token_block_size, hidden_size, hidden_size), T.float32)
        x_frag = T.alloc_fragment((token_block_size, hidden_size, hidden_size), T.float32)

        T.copy(grad_output[pid_x * token_block_size, 0, 0], grad_frag)
        T.copy(x[pid_x * token_block_size, 0, 0], x_frag)

        row_sum = T.alloc_fragment((token_block_size, hidden_size), T.float32)
        row_sum2 = T.alloc_fragment((token_block_size, hidden_size), T.float32)
        col_sum = T.alloc_fragment((token_block_size, hidden_size), T.float32)
        col_sum2 = T.alloc_fragment((token_block_size, hidden_size), T.float32)
        temp = T.alloc_fragment((token_block_size, hidden_size, hidden_size), T.float32)

        xs = T.alloc_shared((repeat * 2, token_block_size, hidden_size, hidden_size), T.float32)
        sums = T.alloc_shared((repeat * 2, token_block_size, hidden_size), T.float32)

        row_max = T.alloc_fragment((token_block_size, hidden_size), T.float32)
        T.reduce_max(x_frag, row_max, dim=2)
        for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
            x_frag[i, j, k] = T.exp(x_frag[i, j, k] - row_max[i, j])
        T.reduce_sum(x_frag, row_sum, dim=2)
        for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
            x_frag[i, j, k] = x_frag[i, j, k] / row_sum[i, j]
        T.copy(x_frag, xs[0, 0, 0, 0])
        for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
            x_frag[i, j, k] = x_frag[i, j, k] + eps
        T.copy(x_frag, xs[1, 0, 0, 0])

        T.reduce_sum(x_frag, col_sum, dim=1)
        T.copy(col_sum, sums[1, 0, 0])
        for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
            x_frag[i, j, k] = x_frag[i, j, k] / (col_sum[i, k] + eps)

        for step in T.unroll(repeat - 1):
            T.reduce_sum(x_frag, row_sum, dim=2)
            T.copy(row_sum, sums[step * 2 + 2, 0, 0])
            T.copy(x_frag, xs[step * 2 + 2, 0, 0, 0])
            for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                x_frag[i, j, k] = x_frag[i, j, k] / (row_sum[i, j] + eps)

            T.reduce_sum(x_frag, col_sum, dim=1)
            T.copy(col_sum, sums[step * 2 + 3, 0, 0])
            T.copy(x_frag, xs[step * 2 + 3, 0, 0, 0])
            for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                x_frag[i, j, k] = x_frag[i, j, k] / (col_sum[i, k] + eps)

        x_inter = T.alloc_fragment((token_block_size, hidden_size, hidden_size), T.float32)
        for inv_step in T.unroll(2 * repeat - 1):
            T.copy(xs[2 * repeat - 1 - inv_step, 0, 0, 0], x_inter)
            if inv_step % 2 == 0:
                T.copy(sums[2 * repeat - 1 - inv_step, 0, 0], col_sum)
                for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                    temp[i, j, k] = grad_frag[i, j, k] * x_inter[i, j, k]
                T.reduce_sum(temp, col_sum2, dim=1)
                for i, k in T.Parallel(token_block_size, hidden_size):
                    col_sum2[i, k] /= col_sum[i, k] + eps
                for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                    grad_frag[i, j, k] = (grad_frag[i, j, k] - col_sum2[i, k]) / (col_sum[i, k] + eps)
            else:
                T.copy(sums[2 * repeat - 1 - inv_step, 0, 0], row_sum)
                for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                    temp[i, j, k] = grad_frag[i, j, k] * x_inter[i, j, k]
                T.reduce_sum(temp, row_sum2, dim=2)
                for i, j in T.Parallel(token_block_size, hidden_size):
                    row_sum2[i, j] /= row_sum[i, j] + eps
                for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                    grad_frag[i, j, k] = (grad_frag[i, j, k] - row_sum2[i, j]) / (row_sum[i, j] + eps)

        T.copy(xs[0, 0, 0, 0], x_inter)
        for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
            temp[i, j, k] = grad_frag[i, j, k] * x_inter[i, j, k]
        T.reduce_sum(temp, row_sum, dim=2)
        for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
            grad_frag[i, j, k] = (grad_frag[i, j, k] - row_sum[i, j]) * x_inter[i, j, k]

        T.copy(grad_frag, grad_input[pid_x * token_block_size, 0, 0])
        if use_pdl:
            T.pdl_trigger()
