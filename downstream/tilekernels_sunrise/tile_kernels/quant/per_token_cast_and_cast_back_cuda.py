import math

import tilelang
from tilelang import language as T

from tile_kernels.quant.common import CastOutputConfig, get_best_vectorize_size, get_sf_and_inv, get_sf_inv_dtype


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_LOWER_LDGSTG_PREDICATED: True,
    },
)
def get_per_token_cast_and_cast_back_kernel_cuda(
    hidden: int,
    token_stride: int,
    dtype: T.dtype,
    out_config: CastOutputConfig,
    use_pdl: bool = False,
):
    num_threads, num_elems_per_thread = 64, 32

    num_elems_per_block = num_threads * num_elems_per_thread
    num_per_channels = out_config.sf_block[1]

    block_k = max(num_per_channels, math.gcd(num_elems_per_block, hidden))
    block_m = 1 if num_elems_per_block % block_k != 0 else num_elems_per_block // block_k
    assert block_k % num_per_channels == 0

    blocks_per_token = T.ceildiv(hidden, block_k)
    num_groups = block_k // num_per_channels
    num_vectorize = min(get_best_vectorize_size(dtype), math.gcd(block_m * block_k // num_threads, 32))
    sf_inv_dtype = get_sf_inv_dtype(dtype, out_config)

    num_tokens = T.dynamic('num_tokens')

    def x_layout_fn(i: int, j: int):
        id = i * block_k + j
        return id // num_vectorize % num_threads, id // (num_vectorize * num_threads) * num_vectorize + id % num_vectorize

    @T.prim_func
    def per_token_cast_and_cast_back_kernel(
        x: T.StridedTensor[(num_tokens, hidden), (token_stride, 1), dtype],
    ):
        with T.Kernel(T.ceildiv(num_tokens, block_m) * blocks_per_token, threads=num_threads) as pid:
            pid_token = pid // blocks_per_token
            pid_hidden = pid % blocks_per_token
            if use_pdl:
                T.pdl_sync()

            x_fragment = T.alloc_fragment((block_m, block_k), dtype)
            sf_inv_fragment = T.alloc_fragment((block_m, num_groups), sf_inv_dtype)
            sf_fragment = T.alloc_fragment((block_m, num_groups), T.float32)
            amax_fragment = T.alloc_fragment((block_m, num_groups), dtype)

            T.annotate_layout(
                {
                    x_fragment: T.Fragment(
                        (block_m, block_k),
                        forward_fn=x_layout_fn,
                    )
                }
            )

            T.copy(x[pid_token * block_m, pid_hidden * block_k], x_fragment, disable_tma=True)

            x_fragment_reshaped = T.reshape(x_fragment, [block_m, num_groups, num_per_channels])
            T.reduce_absmax(x_fragment_reshaped, amax_fragment, dim=2)

            for i, j in T.Parallel(block_m, num_groups):
                amax = T.cast(amax_fragment[i, j], T.float32)
                sf, sf_inv = get_sf_and_inv(amax, out_config, sf_inv_dtype)
                sf_fragment[i, j] = T.cast(sf, T.float32) if out_config.use_e4m3_sf else sf
                sf_inv_fragment[i, j] = sf_inv

            for i, j in T.Parallel(block_m, block_k):
                group_id = j // num_per_channels
                x_quant = T.cast(T.float32(x_fragment[i, j]) * sf_inv_fragment[i, group_id], out_config.dtype)
                x_fragment[i, j] = T.cast(T.float32(x_quant) * sf_fragment[i, group_id], dtype)

            T.copy(x_fragment, x[pid_token * block_m, pid_hidden * block_k], disable_tma=True)

            if use_pdl:
                T.pdl_trigger()

    return per_token_cast_and_cast_back_kernel
