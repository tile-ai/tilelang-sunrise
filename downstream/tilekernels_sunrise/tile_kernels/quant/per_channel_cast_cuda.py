import tilelang
from tilelang import language as T

from tile_kernels.quant.common import *
from tile_kernels.utils import ceil_div


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def get_per_channel_cast_kernel_cuda(
    hidden: int,
    in_config: CastInputConfig,
    out_config: CastOutputConfig,
    use_pdl: bool = False,
):
    num_tokens = T.dynamic('num_tokens')
    num_tokens_out = T.dynamic('num_tokens_out')

    num_per_tokens, _ = out_config.sf_block
    _, num_per_channels = in_config.sf_block
    assert num_per_tokens in (32, 128)
    assert not in_config.with_sf or num_per_channels in (32, 128)

    if in_config.with_sf:
        TILE_K = 256
        TILE_M = num_per_tokens
    else:
        TILE_M, TILE_K = 128, 128
    # Set num_threads_per_token = 32 to avoid bank conflict
    num_threads_per_token = 32
    # Keep at most 16 rows per thread while retaining at least four warps.
    num_threads = max(128, TILE_M * num_threads_per_token // 16)
    assert TILE_K % num_threads_per_token == 0

    VEC_K = TILE_K // num_threads_per_token
    VEC_M = TILE_M * num_threads_per_token // num_threads
    num_sf_rows_per_tile = TILE_M // num_per_tokens
    num_sf_per_tile = num_sf_rows_per_tile * TILE_K
    num_row_slices_per_sf_row = num_per_tokens // VEC_M
    num_sf_iters = ceil_div(num_sf_per_tile, num_threads)
    x_sf_shape = get_sf_shape((num_tokens, hidden), in_config)
    in_sf_stride = T.dynamic('in_sf_stride') if in_config.with_sf else x_sf_shape[1]
    out_sf_shape = get_sf_shape((num_tokens_out, hidden), out_config)
    assert TILE_M % num_per_tokens == 0
    assert num_per_tokens % VEC_M == 0
    assert not in_config.with_sf or TILE_K % num_per_channels == 0
    assert not in_config.with_sf or num_per_channels % VEC_K == 0
    sf_value_dtype = T.float32 if in_config.with_sf else in_config.dtype
    sf_inv_dtype = get_sf_inv_dtype(sf_value_dtype, out_config)

    @T.prim_func
    def per_channel_cast_kernel(
        x: T.Tensor[(num_tokens, hidden), in_config.dtype],
        out: T.Tensor[(num_tokens_out, hidden), out_config.dtype],
        out_sf: T.Tensor[out_sf_shape, out_config.sf_dtype],
        x_sf_invs: T.StridedTensor[x_sf_shape, (in_sf_stride, 1), in_config.sf_dtype],
    ):
        with T.Kernel(T.ceildiv(hidden, TILE_K), T.ceildiv(num_tokens_out, TILE_M), threads=num_threads) as (pid_hidden, pid_token):
            if use_pdl:
                T.pdl_sync()
            x_shared = T.alloc_shared((TILE_M, TILE_K), in_config.dtype)
            sf_invs_local = T.alloc_local((VEC_M,), T.float32)
            amax_local = T.alloc_local((VEC_K,), sf_value_dtype)
            out_sf_invs_local = T.alloc_local((VEC_K,), T.float32)
            amax_shared = T.alloc_shared((VEC_K, num_threads), sf_value_dtype)
            sf_inv_shared = T.alloc_shared((num_sf_rows_per_tile, TILE_K), sf_inv_dtype)
            sf_shared = T.alloc_shared((num_sf_rows_per_tile, TILE_K), out_config.sf_dtype)
            in_local = T.alloc_local((VEC_K,), in_config.dtype)
            out_local = T.alloc_local((VEC_K,), out_config.dtype)
            tid = T.get_thread_binding(0)
            m_id, k_id = tid // num_threads_per_token, tid % num_threads_per_token
            m_offset = pid_token * TILE_M + m_id * VEC_M
            k_offset = pid_hidden * TILE_K + k_id * VEC_K

            T.assume(num_tokens_out % 128 == 0)

            if in_config.with_sf:
                for i in T.serial(VEC_M):
                    pos = i + m_offset
                    T.assume(pos < num_tokens)
                    sf_invs_local[i] = transform_sf(
                        load_sf(x_sf_invs, pos, (pid_hidden * TILE_K + k_id * VEC_K) // num_per_channels, in_config), in_config
                    )

            T.clear(amax_local)
            for i in T.serial(VEC_M):
                pos = i + m_offset
                T.assume(pos < num_tokens)
                for j in T.vectorized(VEC_K):
                    T.assume(pos < num_tokens)
                    in_local[j] = x[pos, j + k_offset]
                    x_shared[i + m_id * VEC_M, j + k_id * VEC_K] = in_local[j]

                for j in T.vectorized(VEC_K):
                    if in_config.with_sf:
                        amax_local[j] = T.max(amax_local[j], T.abs(in_local[j] * sf_invs_local[i]))
                    else:
                        amax_local[j] = T.max(amax_local[j], T.abs(in_local[j]))

            for i in T.unroll(VEC_K):
                amax_shared[i, tid] = amax_local[i]

            for sf_iter in T.unroll(num_sf_iters):
                sf_idx = sf_iter * num_threads + tid
                if sf_idx < num_sf_per_tile:
                    sf_row = sf_idx // TILE_K
                    sf_col = sf_idx % TILE_K
                    owner_tid = sf_col % num_threads_per_token
                    owner_offset = sf_col // num_threads_per_token
                    row_slice_start = sf_row * num_row_slices_per_sf_row

                    amax_var = T.alloc_var(T.float32)
                    amax_var = 0
                    for i in T.serial(num_row_slices_per_sf_row):
                        src_tid = (row_slice_start + i) * num_threads_per_token + owner_tid
                        amax_var = T.max(amax_var, amax_shared[owner_offset, src_tid])

                    sf, sf_inv = get_sf_and_inv(amax_var, out_config, sf_inv_dtype)
                    sf_shared[sf_row, sf_col] = sf
                    sf_inv_shared[sf_row, sf_col] = sf_inv

            T.sync_threads()
            sf_row = m_id * VEC_M // num_per_tokens
            for i in T.serial(VEC_K):
                out_sf_invs_local[i] = sf_inv_shared[sf_row, k_id + i * num_threads_per_token]

            for i in T.serial(VEC_M):
                for j in T.vectorized(VEC_K):
                    in_local[j] = x_shared[i + m_id * VEC_M, j + k_id * VEC_K]
                for j in T.vectorized(VEC_K):
                    if in_config.with_sf:
                        out_local[j] = in_local[j] * sf_invs_local[i] * out_sf_invs_local[j]
                    else:
                        out_local[j] = T.float32(in_local[j]) * out_sf_invs_local[j]
                for j in T.vectorized(VEC_K):
                    out[i + m_offset, j + k_offset] = out_local[j]

            # Preserve scale ownership while coalescing stores across lanes.
            for sf_store_iter in T.unroll(num_sf_iters):
                sf_store_idx = sf_store_iter * num_threads + tid
                if sf_store_idx < num_sf_per_tile:
                    sf_store_row = sf_store_idx // TILE_K
                    sf_store_col = sf_store_idx % TILE_K
                    sf_store_value = sf_shared[sf_store_row, (sf_store_col % VEC_K) * num_threads_per_token + sf_store_col // VEC_K]
                    store_sf(
                        out_sf,
                        sf_store_value,
                        pid_token * num_sf_rows_per_tile + sf_store_row,
                        pid_hidden * TILE_K + sf_store_col,
                        out_config,
                    )

            if use_pdl:
                T.pdl_trigger()

    return per_channel_cast_kernel
