import tilelang
from tilelang import language as T

from tile_kernels.utils import ceil_div
from tile_kernels.quant.common import *


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def get_per_block_cast_kernel_cuda(
    hidden: int,
    in_config: CastInputConfig,
    out_config: CastOutputConfig,
    sf_only: bool = False,
    cast_only: bool = False,
    use_pdl: bool = False,
):
    num_threads = 256
    num_elements_per_block = 8192

    num_vectorize = get_best_vectorize_size(in_config.dtype)

    num_per_tokens, num_per_channels = out_config.sf_block
    assert num_per_tokens in (32, 128) and num_per_channels in (32, 128)

    block_k = max(num_per_channels, num_elements_per_block // num_per_tokens)
    block_m = num_per_tokens
    assert block_m % num_per_tokens == 0
    assert block_k % num_per_channels == 0

    num_repeat_rows = num_threads * num_vectorize // block_k
    assert num_per_tokens % num_repeat_rows == 0, 'scaling factor block size too small to be multiple of the row repeat pattern'
    num_sf_rows_per_block = block_m // num_per_tokens
    num_sf_cols_per_block = block_k // num_per_channels
    num_sf_per_block = num_sf_rows_per_block * num_sf_cols_per_block
    sf_inv_dtype = get_sf_inv_dtype(in_config.dtype, out_config)

    # Runtime symbols
    num_tokens = T.dynamic('num_tokens')
    sf_shape = get_sf_shape((num_tokens, hidden), out_config)
    sf_stride = T.dynamic('sf_stride')

    def amax_forward_fn(i: int, j: int, k: int):
        num_threads_per_token = num_per_channels // num_vectorize
        return j * num_threads_per_token + k // num_threads_per_token * (block_k // num_vectorize) + k % num_threads_per_token, i

    @T.macro
    def transform_fragment(
        x_fragment: T.Fragment,
        out_sf: T.Tensor,
        pid_x: int,
        pid_y: int,
        sf_inv_fragment: T.Fragment,
    ):
        if not cast_only:
            amax_reducer = T.alloc_fragment((num_sf_rows_per_block, num_sf_cols_per_block, num_threads // num_sf_per_block), in_config.dtype)
            amax_fragment = T.alloc_fragment((num_sf_rows_per_block, num_sf_cols_per_block), T.float32)
            T.annotate_layout(
                {
                    amax_reducer: T.Fragment(
                        (num_sf_rows_per_block, num_sf_cols_per_block, num_threads // num_sf_per_block),
                        replicate=1,
                        forward_fn=amax_forward_fn,
                    ),
                    amax_fragment: T.Fragment(
                        (num_sf_rows_per_block, num_sf_cols_per_block),
                        replicate=num_threads // num_sf_per_block,
                        forward_fn=amax_forward_fn,
                    ),
                }
            )
            T.clear(amax_reducer)
            for i, j in T.Parallel(block_m, block_k):
                block_offset = (i % (num_threads * num_vectorize // block_k) * num_per_channels + j % num_per_channels) // num_vectorize
                amax_reducer[i // num_per_tokens, j // num_per_channels, block_offset] = T.max(
                    amax_reducer[i // num_per_tokens, j // num_per_channels, block_offset],
                    T.abs(x_fragment[i, j]),
                )
            T.reduce_max(amax_reducer, amax_fragment, dim=2)
            for i, j in T.Parallel(num_sf_rows_per_block, num_sf_cols_per_block):
                sf, sf_inv = get_sf_and_inv(amax_fragment[i, j], out_config, sf_inv_dtype)
                store_sf(out_sf, sf, pid_x * num_sf_rows_per_block + i, pid_y * num_sf_cols_per_block + j, out_config)
                sf_inv_fragment[i, j] = sf_inv
        else:
            for i, j in T.Parallel(num_sf_rows_per_block, num_sf_cols_per_block):
                sf = load_sf(out_sf, pid_x * num_sf_rows_per_block + i, pid_y * num_sf_cols_per_block + j, out_config)
                sf_inv_fragment[i, j] = 1 / sf

    @T.prim_func
    def per_block_cast_kernel(
        x: T.Tensor[(num_tokens, hidden), in_config.dtype],
        out: T.Tensor[(num_tokens, hidden), out_config.dtype],
        out_sf: T.StridedTensor[sf_shape, (sf_stride, 1), out_config.sf_dtype],
    ):
        with T.Kernel(ceil_div(num_tokens, block_m), ceil_div(hidden, block_k), threads=num_threads) as (pid_x, pid_y):
            if use_pdl:
                T.pdl_sync()
            x_fragment = T.alloc_fragment((block_m, block_k), in_config.dtype)
            sf_inv_fragment = T.alloc_fragment((num_sf_rows_per_block, num_sf_cols_per_block), sf_inv_dtype)

            T.annotate_layout(
                {
                    x_fragment: T.Fragment(
                        (block_m, block_k),
                        forward_fn=lambda i, j: (
                            (i * block_k + j) // num_vectorize % num_threads,
                            i * block_k // (num_threads * num_vectorize) * num_vectorize + j % num_vectorize,
                        ),
                    ),
                    sf_inv_fragment: T.Fragment(
                        (num_sf_rows_per_block, num_sf_cols_per_block),
                        replicate=num_threads // num_sf_per_block,
                        forward_fn=amax_forward_fn,
                    ),
                }
            )

            # Move the `if` statements to the top level to improve SASS code generation.
            if pid_x < ceil_div(num_tokens, block_m) - 1 and pid_y < ceil_div(hidden, block_k) - 1:
                T.copy(x[pid_x * block_m, pid_y * block_k], x_fragment)
                transform_fragment(x_fragment, out_sf, pid_x, pid_y, sf_inv_fragment)

                if not sf_only:
                    for i, j in T.Parallel(block_m, block_k):
                        out[pid_x * block_m + i, pid_y * block_k + j] = x_fragment[i, j] * sf_inv_fragment[i // num_per_tokens, j // num_per_channels]
            else:
                T.copy(x[pid_x * block_m, pid_y * block_k], x_fragment)
                transform_fragment(x_fragment, out_sf, pid_x, pid_y, sf_inv_fragment)

                if not sf_only:
                    for i, j in T.Parallel(block_m, block_k):
                        out[pid_x * block_m + i, pid_y * block_k + j] = x_fragment[i, j] * sf_inv_fragment[i // num_per_tokens, j // num_per_channels]

            if use_pdl:
                T.pdl_trigger()

    return per_block_cast_kernel
