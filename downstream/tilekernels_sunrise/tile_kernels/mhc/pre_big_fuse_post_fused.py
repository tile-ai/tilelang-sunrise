"""Fused _mhc_pre_big_fuse + _mhc_post_fwd kernel — weight-matrix optimized.

Eliminates the ``layer_input`` HBM round-trip by algebraically fusing
apply_mix and post_fwd into a single 4×4→4×H matrix multiply:

    out[o, h] = Σ_j W[o, j] * residual[j, h]

where  W[o, j] = post_mix[o] · pre_mix[j] + cm[j, o]  is a 4×4 weight
matrix pre-computed in Phase 1, eliminating the intermediate ``layer_input``
buffer from Phase 2.
"""

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
def _mhc_pre_big_fuse_post_fused(
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

    # --- tile selection (same as pre_big_fuse) ---------------------------------
    hidden_block = math.gcd(2048, hidden_size)
    # Search for a larger divisor when the gcd yields tiles that are too small
    # (< 1024) or too many pipeline stages (> 4).  Fewer stages mean larger
    # tiles, better HBM burst efficiency (e.g. HS=7168: 1792→N=4 vs 1024→N=7).
    if hidden_block < 1024 or hidden_size // hidden_block > 4:
        for candidate in range(min(2048, hidden_size), 511, -1):
            if hidden_size % candidate == 0:
                hidden_block = candidate
                break
    if hidden_block == hidden_size and hidden_block <= 2048:
        half = hidden_size // 2
        if half >= 512 and hidden_size % half == 0:
            hidden_block = half

    OL_SUB = hidden_block // 4
    if OL_SUB % 128 != 0:
        for sub in (256, 128, 64):
            if hidden_block % sub == 0:
                OL_SUB = sub
                break
    SPB = hidden_block // OL_SUB  # sub-blocks per tile

    @T.prim_func
    def kernel(
        gemm_out_mul: T.Tensor[(n_splits, num_tokens, mhc_mult3), T.float32],
        gemm_out_sqrsum: T.Tensor[(n_splits, num_tokens), T.float32],
        mhc_scale: T.Tensor[(3,), T.float32],
        mhc_base: T.Tensor[(mhc_mult3,), T.float32],
        residual: T.Tensor[(num_tokens, mhc_mult, hidden_size), T.bfloat16],
        post_mix: T.Tensor[(num_tokens, mhc_mult), T.float32],
        comb_mix: T.Tensor[(num_tokens, mhc_mult * mhc_mult), T.float32],
        out: T.Tensor[(num_tokens, mhc_mult, hidden_size), T.bfloat16],
    ):
        with T.Kernel(num_tokens, threads=128) as pid:
            # ================================================================
            # Phase 1: reduction + norm + mixes + pre-compute weight matrix W
            # ================================================================
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

            ma = mhc_scale[0]
            mb = mhc_scale[1]
            mc = mhc_scale[2]

            # pre_mix[j] = sigmoid(mixes[j]·ma + base[j]) + eps
            pre_mix = T.alloc_shared(mhc_mult, T.float32)
            for j in T.Parallel(mhc_mult):
                pre_mix[j] = (
                    T.sigmoid(mixes_shared[j] * ma + mhc_base[j])
                    + mhc_pre_eps
                )

            # post_mix[o] = sigmoid(mixes[o+M]·mb + base[o+M]) * mult
            post_mix_reg = T.alloc_fragment(mhc_mult, T.float32)
            for j in T.Parallel(mhc_mult):
                post_mix_reg[j] = (
                    T.sigmoid(mixes_shared[j + mhc_mult] * mb
                              + mhc_base[j + mhc_mult])
                    * mhc_post_mult_value
                )

            # cm[i, o] = mixes[i·M+o+M·2]·mc + base[i·M+o+M·2]
            cm = T.alloc_fragment((mhc_mult, mhc_mult), T.float32)
            for j, k in T.Parallel(mhc_mult, mhc_mult):
                cm[j, k] = (
                    mixes_shared[j * mhc_mult + k + mhc_mult * 2] * mc
                    + mhc_base[j * mhc_mult + k + mhc_mult * 2]
                )

            # sinkhorn normalisation (unchanged)
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

            # ---- Pre-compute fused weight matrix ----
            # W[o, j] = post_mix[o] · pre_mix[j] + cm[j, o]
            W = T.alloc_shared((mhc_mult, mhc_mult), T.float32)
            for o_idx, j_idx in T.Parallel(mhc_mult, mhc_mult):
                W[o_idx, j_idx] = (
                    post_mix_reg[o_idx] * pre_mix[j_idx]
                    + cm[j_idx, o_idx]
                )

            T.sync_threads()

            # ================================================================
            # Phase 2: out = W · residual  (single 4×4→4×H matmul)
            # ----------------------------------------------------------------
            # out[o, h] = Σ_j W[o, j] * residual[pid, j, h]
            #
            # Same structure as apply_mix but with 4×4 weights instead of
            # 1×4.  No intermediate layer_input buffer needed.
            # ================================================================
            Nt = hidden_size // hidden_block
            out_local = T.alloc_fragment((mhc_mult, OL_SUB), T.float32)

            for i in T.Pipelined(Nt, num_stages=2):
                xs = T.alloc_shared((mhc_mult, hidden_block), T.bfloat16)
                T.copy(residual[pid, 0, i * hidden_block], xs, disable_tma=True)
                out_shm = T.alloc_shared((mhc_mult, hidden_block), T.bfloat16)

                for i_sub in T.Unroll(SPB):
                    # Init: out_local[o, h] = W[o, 0] * xs[0, h]
                    for i_mhco, i1_h in T.Parallel(mhc_mult, OL_SUB):
                        out_local[i_mhco, i1_h] = (
                            W[i_mhco, 0]
                            * xs[0, i_sub * OL_SUB + i1_h]
                        )
                    # Accumulate: += Σ_{j=1}^{3} W[o, j] * xs[j, h]
                    for i_mhc_in in T.Unroll(mhc_mult - 1):
                        i_j = i_mhc_in + 1
                        for i_mhco, i1_h in T.Parallel(mhc_mult, OL_SUB):
                            out_local[i_mhco, i1_h] += (
                                W[i_mhco, i_j]
                                * xs[i_j, i_sub * OL_SUB + i1_h]
                            )
                    # Write sub-block to shared memory
                    for i_mhco, i1_h in T.Parallel(mhc_mult, OL_SUB):
                        out_shm[i_mhco, i_sub * OL_SUB + i1_h] = T.cast(
                            out_local[i_mhco, i1_h], 'bfloat16'
                        )

                T.copy(out_shm, out[pid, 0, i * hidden_block], disable_tma=True)

            # ---- Deferred output writes (after pipeline) ----
            for j in T.Parallel(mhc_mult):
                post_mix[pid, j] = post_mix_reg[j]
            for j, k in T.Parallel(mhc_mult, mhc_mult):
                comb_mix[pid, j * mhc_mult + k] = cm[j, k]

    return kernel
