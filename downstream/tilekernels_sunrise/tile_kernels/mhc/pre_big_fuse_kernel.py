import math
import tilelang
import torch
from tilelang import language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
        tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
    },
)
def _mhc_pre_big_fuse(
    hidden_size: int,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
    mhc_mult: int = 4,
):
    num_tokens = T.dynamic('num_tokens')
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    # Pick the largest divisor of hidden_size that is ≤ 2048, preferring large
    # tiles for efficient HBM transactions and fewer pipeline sync points.
    # gcd(2048, hidden_size) gives 2048, 512, or 256 depending on the hidden
    # size.  When that yields tiles < 1 KB (e.g. 256 for HS=1280) or too many
    # pipeline iterations (512 for HS=2560 → N=5, 13 syncs), search for a
    # better divisor.  N=1 (hidden_block == hidden_size) is allowed — a single
    # large tile can outperform a shallow pipeline of tiny tiles.
    hidden_block = math.gcd(2048, hidden_size)
    if hidden_block < 1024 or hidden_size // hidden_block > 4:
        for candidate in range(min(2048, hidden_size), 511, -1):
            if hidden_size % candidate == 0:
                hidden_block = candidate
                break
    # When hidden_block == hidden_size (N=1), the double-buffered pipeline
    # has only one stage — no load/compute overlap.  If half-size tiles
    # are still large enough (≥ 512 elements), use them instead to get
    # N=2 with pipelined overlap.  Example: HS=1280 → hidden_block=640
    # instead of 1280, enabling 2-stage overlap with 2.5 KB tiles.
    if hidden_block == hidden_size and hidden_block <= 2048:
        half = hidden_size // 2
        if half >= 512 and hidden_size % half == 0:
            hidden_block = half
    OL_SUB = hidden_block // 4  # 4 sub-blocks per tile, register-safe (64 or 128)
    # Ensure OL_SUB is a multiple of 128 so it maps cleanly to 128 threads.
    # When the gcd fallback picks a non-power-of-2 hidden_block (e.g. 640 or
    # 1280 for HS=1280/2560), OL_SUB may not be a clean multiple; pick the
    # largest of {256, 128, 64} that divides hidden_block.
    if OL_SUB % 128 != 0:
        for sub in (256, 128, 64):
            if hidden_block % sub == 0:
                OL_SUB = sub
                break
    hidden_block_per_ol_sub = hidden_block // OL_SUB

    @T.prim_func
    def mhc_pre_big_fuse(
        gemm_out_mul: T.Tensor[(n_splits, num_tokens, mhc_mult3), T.float32],
        gemm_out_sqrsum: T.Tensor[(n_splits, num_tokens), T.float32],
        mhc_scale: T.Tensor[(3,), T.float32],
        mhc_base: T.Tensor[(mhc_mult3,), T.float32],
        residual: T.Tensor[(num_tokens, mhc_mult, hidden_size), T.bfloat16],
        # outputs
        post_mix: T.Tensor[(num_tokens, mhc_mult), T.float32],
        comb_mix: T.Tensor[(num_tokens, mhc_mult * mhc_mult), T.float32],
        layer_input: T.Tensor[(num_tokens, hidden_size), T.bfloat16],
    ) -> None:
        with T.Kernel(num_tokens, threads=128) as pid:
            ##################################################################
            # Phase 1: reduction + norm + split-mixes + sinkhorn (all threads)
            mixes_shared = T.alloc_shared(mhc_mult3, T.float32)
            rms = T.alloc_fragment(1, T.float32)
            mixes = T.alloc_fragment(mhc_mult3, T.float32)
            rms[0] = 0
            T.clear(mixes)
            for i_split in T.vectorized(n_splits):
                rms[0] += gemm_out_sqrsum[i_split, pid]
                for j in T.Parallel(mhc_mult3):
                    mixes[j] += gemm_out_mul[i_split, pid, j]
            rms[0] = T.rsqrt(rms[0] / (mhc_mult * hidden_size) + rms_eps)
            for j in T.Parallel(mhc_mult3):
                mixes[j] *= rms[0]
            T.copy(mixes, mixes_shared, disable_tma=True)

            # _mhc_pre_split_mixes_fwd (post & comb)
            cm = T.alloc_fragment((mhc_mult, mhc_mult), T.float32)
            ma = mhc_scale[0]
            mb = mhc_scale[1]
            mc = mhc_scale[2]
            for j in T.Parallel(mhc_mult):
                post_mix[pid, j] = T.sigmoid(mixes_shared[j + mhc_mult] * mb + mhc_base[j + mhc_mult]) * mhc_post_mult_value
            for j, k in T.Parallel(mhc_mult, mhc_mult):
                cm[j, k] = mixes_shared[j * mhc_mult + k + mhc_mult * 2] * mc + mhc_base[j * mhc_mult + k + mhc_mult * 2]

            # _mhc_sinkhorn_fwd
            row_sum = T.alloc_fragment(mhc_mult, T.float32)
            col_sum = T.alloc_fragment(mhc_mult, T.float32)
            row_max = T.alloc_fragment(mhc_mult, T.float32)
            T.reduce_max(cm, row_max, dim=1)
            for j, k in T.Parallel(mhc_mult, mhc_mult):
                cm[j, k] = T.exp(cm[j, k] - row_max[j])
            T.reduce_sum(cm, row_sum, dim=1)
            for j, k in T.Parallel(mhc_mult, mhc_mult):
                cm[j, k] = cm[j, k] / row_sum[j] + mhc_sinkhorn_eps
            T.reduce_sum(cm, col_sum, dim=0)
            for j, k in T.Parallel(mhc_mult, mhc_mult):
                cm[j, k] = cm[j, k] / (col_sum[k] + mhc_sinkhorn_eps)
            for _ in T.serial(sinkhorn_repeat - 1):
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = cm[j, k] / (row_sum[j] + mhc_sinkhorn_eps)
                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + mhc_sinkhorn_eps)
            for j, k in T.Parallel(mhc_mult, mhc_mult):
                comb_mix[pid, j * mhc_mult + k] = cm[j, k]

            T.sync_threads()

            ##################################################################
            # Phase 2: pre_mix + apply_mix with async copy pipelining
            pre_mix_shared = T.alloc_shared(mhc_mult, T.float32)
            for j in T.Parallel(mhc_mult):
                pre_mix_shared[j] = (
                    T.sigmoid(mixes_shared[j] * ma + mhc_base[j])
                    + mhc_pre_eps
                )

            N = hidden_size // hidden_block
            ol = T.alloc_fragment(OL_SUB, T.float32)
            layer_shm = T.alloc_shared(hidden_size, T.bfloat16)

            # Double-buffered async copy via T.Pipelined (ref:
            # tilelang/examples/example_gemm.py).  T.copy inside the
            # pipelined loop becomes cp.async; the next tile load overlaps
            # with the current tile compute.
            for i in T.Pipelined(N, num_stages=2):
                xs = T.alloc_shared((mhc_mult, hidden_block), T.bfloat16)
                T.copy(residual[pid, 0, i * hidden_block], xs, disable_tma=True)

                # Process ol in sub-blocks to keep register pressure low.
                for i_sub in T.Unroll(hidden_block_per_ol_sub):
                    T.clear(ol)
                    for i_mhc in T.Unroll(mhc_mult):
                        for i1_h in T.Parallel(OL_SUB):
                            ol[i1_h] += (
                                pre_mix_shared[i_mhc]
                                * xs[i_mhc, i_sub * OL_SUB + i1_h]
                            )
                    T.copy(
                        ol,
                        layer_shm[i * hidden_block + i_sub * OL_SUB],
                        disable_tma=True,
                    )

            T.copy(layer_shm, layer_input[pid, 0], disable_tma=True)

    return mhc_pre_big_fuse
