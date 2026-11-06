import tilelang
import torch
from tilelang import language as T


_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WGMMA: True,
}


@tilelang.jit
def _mhc_fn_normw_merge_fwd(m: int, n: int, dtype: T.dtype = T.float32) -> tilelang.JITKernel:
    n_blk = 256

    @T.prim_func
    def _mhc_fn_normw_merge_fwd_(
        fn: T.Tensor[(m, n), dtype],
        normw: T.Tensor[n, dtype],
        out_fn: T.Tensor[(m, n), dtype],
    ) -> None:
        _ = dtype
        with T.Kernel(m, T.ceildiv(n, n_blk)) as (pid_m, pid_n):
            for i1_n in T.Parallel(n_blk):
                i_n = pid_n * n_blk + i1_n
                if i_n < n:
                    out_fn[pid_m, i_n] = fn[pid_m, i_n] * normw[i_n]

    return _mhc_fn_normw_merge_fwd_


@tilelang.jit
def _mhc_fn_normw_merge_bwd(m: int, n: int, dtype: T.dtype = T.float32) -> tilelang.JITKernel:
    n_blk = 256

    @T.prim_func
    def _mhc_fn_normw_merge_bwd_(
        fn: T.Tensor[(m, n), dtype],
        normw: T.Tensor[n, dtype],
        out_fn_grad: T.Tensor[(m, n), dtype],
        fn_grad: T.Tensor[(m, n), dtype],
        normw_grad: T.Tensor[n, dtype],
    ) -> None:
        _ = dtype
        with T.Kernel(T.ceildiv(n, n_blk)) as pid_n:
            normw_frag = T.alloc_fragment(n_blk, dtype)
            T.copy(normw[pid_n * n_blk], normw_frag)

            normw_grad_frag = T.alloc_fragment(n_blk, dtype)
            T.clear(normw_grad_frag)

            for i_m in T.serial(m):
                for i1_n in T.Parallel(n_blk):
                    i_n = pid_n * n_blk + i1_n
                    if i_n < n:
                        fn_grad[i_m, i_n] += out_fn_grad[i_m, i_n] * normw_frag[i1_n]
                        normw_grad_frag[i1_n] += out_fn_grad[i_m, i_n] * fn[i_m, i_n]

            for i1_n in T.Parallel(n_blk):
                normw_grad[pid_n * n_blk + i1_n] += normw_grad_frag[i1_n]

    return _mhc_fn_normw_merge_bwd_


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_pre_norm_fn_fwd_mul(
    mhc_mult3: int,
    n_rms_group: int,
    rms_group_size: int,
    token_block: int = 32,
    hidden_block: int = 128,
) -> tilelang.JITKernel:
    assert mhc_mult3 <= 32
    num_tokens = T.dynamic('num_tokens')
    assert rms_group_size % hidden_block == 0

    @T.prim_func
    def _mhc_pre_norm_fn_fwd_mul_kernel(
        x: T.Tensor[(num_tokens, n_rms_group * rms_group_size), T.bfloat16],
        fn: T.Tensor[(mhc_mult3, n_rms_group * rms_group_size), T.float32],
        out: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32],
        sqrsum: T.Tensor[(num_tokens, n_rms_group), T.float32],
    ) -> None:
        _ = mhc_mult3
        with T.Kernel(T.ceildiv(num_tokens, token_block), n_rms_group) as (pid_x, pid_y):
            out_frag = T.alloc_fragment((token_block, 32), T.float32)
            sqrsum_part = T.alloc_fragment((token_block, 4), T.float32)
            T.clear(out_frag)
            T.clear(sqrsum_part)
            fn_smem = T.alloc_shared((32, hidden_block), T.float32)
            T.use_swizzle(panel_size=4, enable=True)

            for pz in T.Pipelined(rms_group_size // hidden_block, num_stages=2):
                x_smem = T.alloc_shared((token_block, hidden_block), T.float32)
                T.copy(x[pid_x * token_block, pid_y * rms_group_size + pz * hidden_block], x_smem)
                T.copy(fn[0, pid_y * rms_group_size + pz * hidden_block], fn_smem)

                for jj in T.serial(hidden_block // 4):
                    for i, j in T.Parallel(token_block, 4):
                        sqrsum_part[i, j] += x_smem[i, jj * 4 + j] * x_smem[i, jj * 4 + j]

                T.gemm(
                    x_smem,
                    fn_smem,
                    out_frag,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=False,
                    k_step=4,
                    a_local_load_type='load_overlap_mma',
                    b_local_load_type='load_overlap_mma',
                )
            sqrsum_l = T.alloc_fragment(token_block, T.float32)
            T.reduce_sum(sqrsum_part, sqrsum_l)
            T.copy(sqrsum_l, sqrsum[pid_x * token_block, pid_y])

            out_shared = T.alloc_shared((token_block, 32), T.float32)
            T.copy(out_frag, out_shared)
            T.copy(out_shared[:, 0:mhc_mult3], out[pid_x * token_block, pid_y, 0])

    return _mhc_pre_norm_fn_fwd_mul_kernel


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_pre_norm_fn_fwd_mul_split_hidden(
    mhc_mult3: int,
    rms_group_size: int,
    token_block: int = 32,
    hidden_block: int = 128,
) -> tilelang.JITKernel:
    """Variant of _mhc_pre_norm_fn_fwd_mul that parallelizes along the hidden dimension.

    Instead of a serial loop over hidden tiles inside each block, the hidden
    dimension is mapped to the grid y-axis.  Each block processes exactly one
    tile and writes partial results to ``out[pid_y, ...]`` / ``sqrsum[pid_y, ...]``.
    The caller is responsible for reducing across the first dimension (e.g.
    ``_mhc_pre_big_fuse`` already does this via its ``n_splits`` parameter).
    """
    assert mhc_mult3 <= 32
    num_tokens = T.dynamic('num_tokens')
    assert rms_group_size % hidden_block == 0
    n_hidden_splits = rms_group_size // hidden_block

    @T.prim_func
    def kernel(
        x: T.Tensor[(num_tokens, rms_group_size), T.bfloat16],
        fn: T.Tensor[(mhc_mult3, rms_group_size), T.float32],
        out: T.Tensor[(n_hidden_splits, num_tokens, mhc_mult3), T.float32],
        sqrsum: T.Tensor[(n_hidden_splits, num_tokens), T.float32],
    ) -> None:
        _ = mhc_mult3
        with T.Kernel(T.ceildiv(num_tokens, token_block), n_hidden_splits) as (pid_x, pid_y):
            # Each block handles one (token tile, hidden tile)
            out_frag = T.alloc_fragment((token_block, 32), T.float32)
            sqrsum_part = T.alloc_fragment((token_block, 4), T.float32)
            T.clear(out_frag)
            T.clear(sqrsum_part)

            # Async TMA: load x (bf16) and fn (fp32) into shared memory
            x_smem_16 = T.alloc_shared((token_block, hidden_block), T.bfloat16)
            fn_smem = T.alloc_shared((32, hidden_block), T.float32)
            T.annotate_layout({x_smem_16: tilelang.layout.make_swizzled_layout(x_smem_16)})
            T.use_swizzle(panel_size=4, enable=True)
            T.copy(x[pid_x * token_block, pid_y * hidden_block], x_smem_16)
            T.copy(fn[0, pid_y * hidden_block], fn_smem)

            # Fused: tiled bf16→fp32 conversion + sqrsum (small tile fragment avoids register bloat)
            x_smem = T.alloc_shared((token_block, hidden_block), T.float32)
            tile_bf16 = T.alloc_fragment((token_block, 4), T.bfloat16)
            for jj in T.serial(hidden_block // 4):
                T.copy(x_smem_16[:, jj * 4:(jj + 1) * 4], tile_bf16)
                T.copy(tile_bf16, x_smem[:, jj * 4:(jj + 1) * 4])
                for i, j in T.Parallel(token_block, 4):
                    sqrsum_part[i, j] += tile_bf16[i, j] * tile_bf16[i, j]

            # GEMM for this tile
            T.gemm(
                x_smem,
                fn_smem,
                out_frag,
                transpose_A=False,
                transpose_B=True,
                clear_accum=False,
                k_step=4,
                a_local_load_type='load_overlap_mma',
                b_local_load_type='load_overlap_mma',
            )

            # Batched writes: T.copy avoids scalar global stores (core_wr bottleneck)
            sqrsum_l = T.alloc_fragment(token_block, T.float32)
            T.reduce_sum(sqrsum_part, sqrsum_l)
            T.copy(sqrsum_l, sqrsum[pid_y, pid_x * token_block])

            out_shared = T.alloc_shared((token_block, 32), T.float32)
            T.copy(out_frag, out_shared)
            # Slice valid columns: stride-32 in shared mem = no bank conflict
            T.copy(out_shared[:, 0:mhc_mult3], out[pid_y, pid_x * token_block, 0])

    return kernel


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_pre_norm_fn_fwd_mul_split_channel(
    mhc_mult3: int,
    mhc_mult: int,
    hidden_size: int,
    n_splits: int,
    token_block: int = 32,
    hidden_block: int = 128,
) -> tilelang.JITKernel:
    """Split-GEMM with configurable split count.

    Grid y-axis = ``n_splits``.  Each block serially processes
    ``tiles_per_split`` contiguous hidden tiles (all from the same mhc
    channel).  Reducing ``n_splits`` shrinks the intermediate tensor at the
    cost of fewer parallel blocks.

    ``n_splits`` must be a multiple of ``mhc_mult``.
    """
    assert mhc_mult3 <= 32
    assert n_splits % mhc_mult == 0, 'n_splits must be a multiple of mhc_mult'
    num_tokens = T.dynamic('num_tokens')
    mhc_hidden_size = mhc_mult * hidden_size
    n_hidden_splits = mhc_hidden_size // hidden_block
    tiles_per_split = n_hidden_splits // n_splits
    CONV_TILE = 8
    hidden_block_per_conv_tile = hidden_block // CONV_TILE
    # Adaptive swizzle: panel=8 reduces bank conflicts for large hidden_size.
    swizzle_panel = 8 if hidden_size >= 4096 else 4
    # k_step=4 is optimal across all shapes; larger values (8, 16) cause
    # register spilling in the MMA accumulator and degrade performance.
    k_step = 4

    @T.prim_func
    def kernel(
        x: T.Tensor[(num_tokens, mhc_hidden_size), T.bfloat16],
        fn: T.Tensor[(mhc_mult3, mhc_hidden_size), T.float32],
        out: T.Tensor[(n_splits, num_tokens, mhc_mult3), T.float32],
        sqrsum: T.Tensor[(n_splits, num_tokens), T.float32],
    ) -> None:
        _ = mhc_mult3
        with T.Kernel(T.ceildiv(num_tokens, token_block), n_splits) as (pid_x, pid_split):
            out_frag = T.alloc_fragment((token_block, 32), T.float32)
            sqrsum_part = T.alloc_fragment((token_block, 8), T.float32)
            T.clear(out_frag)
            T.clear(sqrsum_part)

            T.use_swizzle(panel_size=swizzle_panel, enable=True)

            split_start = pid_split * tiles_per_split

            # Allocate once outside the loop — reused across tiles
            x_smem_16 = T.alloc_shared((token_block, hidden_block), T.bfloat16)
            x_smem = T.alloc_shared((token_block, hidden_block), T.float32)
            fn_smem = T.alloc_shared((32, hidden_block), T.float32)

            for i_tile in T.serial(tiles_per_split):
                tile_idx = split_start + i_tile
                T.copy(
                    x[pid_x * token_block, tile_idx * hidden_block],
                    x_smem_16,
                )
                T.copy(fn[0, tile_idx * hidden_block], fn_smem)

                # bf16 → fp32 conversion + sqrsum
                tile_bf16 = T.alloc_fragment((token_block, CONV_TILE), T.bfloat16)
                for jj in T.serial(hidden_block_per_conv_tile):
                    T.copy(x_smem_16[:, jj * CONV_TILE:(jj + 1) * CONV_TILE], tile_bf16)
                    T.copy(tile_bf16, x_smem[:, jj * CONV_TILE:(jj + 1) * CONV_TILE])
                    for i, j in T.Parallel(token_block, CONV_TILE):
                        sqrsum_part[i, j] += tile_bf16[i, j] * tile_bf16[i, j]

                T.gemm(
                    x_smem,
                    fn_smem,
                    out_frag,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=False,
                    k_step=k_step,
                )

            sqrsum_l = T.alloc_fragment(token_block, T.float32)
            T.reduce_sum(sqrsum_part, sqrsum_l)

            sqrsum_shared = T.alloc_shared(token_block, T.float32)
            T.copy(sqrsum_l, sqrsum_shared)
            T.copy(sqrsum_shared, sqrsum[pid_split, pid_x * token_block])

            out_shared = T.alloc_shared((token_block, 32), T.float32)
            T.copy(out_frag, out_shared)
            T.copy(out_shared[:, 0:mhc_mult3], out[pid_split, pid_x * token_block, 0])

    return kernel


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_pre_norm_fn_fwd_norm(
    mhc_mult3: int,
    n_rms_group: int,
    rms_group_size: int,
    rms_eps: float,
    n_splits: int,
) -> tilelang.JITKernel:
    num_tokens = T.dynamic('num_tokens')
    n_thr = 32

    @T.prim_func
    def _mhc_pre_norm_fn_fwd_norm_kernel(
        out_mul_splitted: T.Tensor[(n_splits, num_tokens, n_rms_group, mhc_mult3), T.float32],
        sqrsum_splitted: T.Tensor[(n_splits, num_tokens, n_rms_group), T.float32],
        out_mul: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32],
        sqrsum: T.Tensor[(num_tokens, n_rms_group), T.float32],
        out: T.Tensor[(num_tokens, mhc_mult3), T.float32],
    ) -> None:
        with T.Kernel(num_tokens, threads=n_thr) as pid:
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

    return _mhc_pre_norm_fn_fwd_norm_kernel


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_pre_norm_fn_bwd_norm(
    mhc_mult3: int,
    n_rms_group: int,
    rms_group_size: int,
    rms_eps: float,
) -> tilelang.JITKernel:
    num_tokens = T.dynamic('num_tokens')
    n_thr = 32

    @T.prim_func
    def _mhc_pre_norm_fn_bwd_norm_kernel(
        # Gradient of output
        out_grad: T.Tensor[(num_tokens, mhc_mult3), T.float32],
        # Saved inputs
        out_mul: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32],
        sqrsum: T.Tensor[(num_tokens, n_rms_group), T.float32],
        # Computed gradient of inputs
        out_mul_grad: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32],
        sqrsum_grad: T.Tensor[(num_tokens, n_rms_group), T.float32],
    ) -> None:
        with T.Kernel(num_tokens, n_rms_group, threads=n_thr) as (pid_i, pid_k):
            sqrsum_frag = T.alloc_fragment(1, T.float32)
            sqrsum_frag[0] = sqrsum[pid_i, pid_k]
            rms_frag = T.alloc_fragment(1, T.float32)
            rms_frag[0] = T.rsqrt(sqrsum_frag[0] / rms_group_size + rms_eps)

            rms_grad_frag = T.alloc_reducer(1, T.float32, replication='all')
            T.clear(rms_grad_frag)
            for j in T.Parallel(mhc_mult3):
                out_mul_grad[pid_i, pid_k, j] = out_grad[pid_i, j] * rms_frag[0]
                rms_grad_frag[0] += out_grad[pid_i, j] * out_mul[pid_i, pid_k, j]
            T.finalize_reducer(rms_grad_frag)

            for kk in T.Parallel(1):
                sqrsum_grad[pid_i, pid_k + kk] = rms_grad_frag[kk] * rms_frag[kk] / (sqrsum_frag[kk] + rms_eps * rms_group_size) / -2

    return _mhc_pre_norm_fn_bwd_norm_kernel


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_pre_norm_fn_bwd_mul(
    mhc_mult3: int,
    n_rms_group: int,
    rms_group_size: int,
    token_block: int = 128,
    hidden_block: int = 128,
) -> tilelang.JITKernel:
    assert mhc_mult3 <= 32
    num_tokens = T.dynamic('num_tokens')
    assert rms_group_size % hidden_block == 0

    @T.prim_func
    def _mhc_pre_norm_fn_bwd_mul_kernel(
        # Gradient of output
        out_mul_grad: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32],
        sqrsum_grad: T.Tensor[(num_tokens, n_rms_group), T.float32],
        # Saved inputs
        x: T.Tensor[(num_tokens, n_rms_group * rms_group_size), T.bfloat16],
        fn: T.Tensor[(mhc_mult3, n_rms_group * rms_group_size), T.float32],
        # Computed gradient of inputs
        x_grad: T.Tensor[(num_tokens, n_rms_group * rms_group_size), T.bfloat16],
        fn_grad: T.Tensor[(mhc_mult3, n_rms_group * rms_group_size), T.float32],
    ) -> None:
        with T.Kernel(n_rms_group, T.ceildiv(rms_group_size, hidden_block)) as (pid_y, pid_z):
            yz = pid_y * rms_group_size + pid_z * hidden_block

            fn_smem = T.alloc_shared((32, hidden_block), T.float32)
            for i, j in T.Parallel(32, hidden_block):
                if i < mhc_mult3:
                    fn_smem[i, j] = fn[i, yz + j]
                else:
                    fn_smem[i, j] = 0

            fn_grad_frag = T.alloc_fragment((32, hidden_block), T.float32)
            T.fill(fn_grad_frag, 0)

            for px in T.serial(T.ceildiv(num_tokens, token_block)):
                x_smem = T.alloc_shared((token_block, hidden_block), T.float32)
                T.copy(x[px * token_block, yz], x_smem)

                padded_grad = T.alloc_shared((token_block, 32), T.float32)
                for i, j in T.Parallel(token_block, 32):
                    if j < mhc_mult3:
                        padded_grad[i, j] = out_mul_grad[px * token_block + i, pid_y, j]
                    else:
                        padded_grad[i, j] = 0

                x_grad_frag = T.alloc_fragment((token_block, hidden_block), T.float32)
                T.copy(x_grad[px * token_block, yz], x_grad_frag)

                T.gemm(
                    padded_grad,
                    x_smem,
                    fn_grad_frag,
                    transpose_A=True,
                    transpose_B=False,
                    clear_accum=False,
                )
                T.gemm(
                    padded_grad,
                    fn_smem,
                    x_grad_frag,
                    transpose_A=False,
                    transpose_B=False,
                    clear_accum=False,
                )

                sqrsum_grad_frag = T.alloc_fragment((token_block, 1), T.float32)
                T.copy(sqrsum_grad[px * token_block, pid_y], sqrsum_grad_frag)
                for i, j in T.Parallel(token_block, hidden_block):
                    x_grad_frag[i, j] += 2 * x_smem[i, j] * sqrsum_grad_frag[i, 0]

                T.copy(x_grad_frag, x_grad[px * token_block, yz])

            T.copy(fn_grad_frag, fn_grad[0, yz])

    return _mhc_pre_norm_fn_bwd_mul_kernel


def round_to_tf32(x: torch.Tensor) -> torch.Tensor:
    return (x.view(torch.int32) + 0x1000).view(torch.float32)
