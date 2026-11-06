import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S


# Each bf16 chunk is unpacked into a 64-lane fp32 vector register.
_VEC = 64


def _u32_index(index_i32):
    return T.reinterpret(index_i32, 'uint32x64')


def _row_reduce(mat, mask_lo, mask_hi, bcast_idx2):
    """Row reduce: vcgadd ×2 + vselr ×2 + vsel → broadcast row sum."""
    return S.vsel(S.vselr(S.vcgadd(mat, mask_lo), bcast_idx2), S.vselr(S.vcgadd(mat, mask_hi), bcast_idx2), mask_lo)


def _row_max(mat, mask_lo, mask_hi, bcast_idx2):
    """Row max: vcgmax ×2 + vselr ×2 + vsel → broadcast row max."""
    return S.vsel(S.vselr(S.vcgmax(mat, mask_lo), bcast_idx2), S.vselr(S.vcgmax(mat, mask_hi), bcast_idx2), mask_lo)


@tilelang.jit
def mhc_pre_big_fuse_fwd_asc(
    gemm_out_mul,
    gemm_out_sqrsum,
    mhc_scale,
    mhc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    h_blk: int,
    n_splits: int,
    token_block: int,
    num_stages: int,
    num_vec_cores: int,
):
    """Ascend fused forward: reduce partials + RMSNorm, split mixes, Sinkhorn, pre-apply.

    Mirrors the arithmetic of the individual Ascend reference kernels
    (mhc_reduce_partials_and_rmsnorm_fwd_asc, mhc_pre_split_mixes_fwd_simd_mhc4_asc,
    mhc_sinkhorn_fwd_asc, mhc_pre_apply_mix_fwd_asc) op-for-op so the fused result
    is bitwise identical to the composed path.

    The Ascend HC prenorm GEMM already reduces split-K internally, so only
    n_splits=1 is supported here.
    """
    num_tokens = T.dynamic('num_tokens')
    hidden = T.const('hidden')
    mhc = 4
    mhc2 = mhc * mhc
    mhc3 = mhc * 2 + mhc2
    assert mhc3 == 24
    assert n_splits == 1, 'Ascend pre_big_fuse expects the GEMM to reduce split-K internally (n_splits=1)'
    assert h_blk % _VEC == 0 and hidden % h_blk == 0

    gemm_out_mul: T.Tensor[(n_splits, num_tokens, mhc3), T.float32]
    gemm_out_sqrsum: T.Tensor[(n_splits, num_tokens), T.float32]
    mhc_scale: T.Tensor[(3,), T.float32]
    mhc_base: T.Tensor[(mhc3,), T.float32]
    residual: T.Tensor[(num_tokens, mhc, hidden), T.bfloat16]
    post_mix: T.Tensor[(num_tokens, mhc), T.float32]
    comb_mix: T.Tensor[(num_tokens, mhc2), T.float32]
    layer_input: T.Tensor[(num_tokens, hidden), T.bfloat16]

    tb = token_block
    num_hidden_tiles = hidden // h_blk
    num_chunks = h_blk // _VEC
    # Same compile-time fp32 constant as the reference reduce kernel.
    inv_rms_group_size = T.float32(1.0 / (mhc * hidden))

    with T.Kernel(num_vec_cores) as core_id:
        scale_ub = T.alloc_shared((3, 8), T.float32)
        base_ub = T.alloc_shared((mhc3, 8), T.float32)
        mul_flat_ub = T.alloc_shared((tb * mhc3,), T.float32)
        mixes_flat_ub = T.alloc_shared((tb * mhc3,), T.float32)
        sq_ub = T.alloc_shared((tb,), T.float32)
        pre_ub = T.alloc_shared((tb, mhc), T.float32)
        post_ub = T.alloc_shared((tb, mhc), T.float32)
        comb_in_ub = T.alloc_shared((tb, mhc2), T.float32)
        comb_out_ub = T.alloc_shared((tb, mhc2), T.float32)
        x_ub = T.alloc_shared((tb, mhc, h_blk), T.bfloat16)
        layer_input_ub = T.alloc_shared((tb, h_blk), T.bfloat16)
        T.annotate_buffer_versions(
            {
                mul_flat_ub: num_stages,
                mixes_flat_ub: num_stages,
                sq_ub: num_stages,
                pre_ub: num_stages,
                post_ub: num_stages,
                comb_in_ub: num_stages,
                comb_out_ub: num_stages,
                x_ub: num_stages,
                layer_input_ub: num_stages,
            }
        )

        with T.SimtVF(threads=32):
            for i in T.Parallel(3):
                scale_ub[i, 0] = mhc_scale[i]
            for i in T.Parallel(mhc3):
                base_ub[i, 0] = mhc_base[i]

        for pid in T.Persistent(
            [T.ceildiv(num_tokens, tb)],
            num_vec_cores,
            core_id,
            group_size=1,
            num_stages=num_stages,
        ):
            token_base = pid * tb
            T.copy(gemm_out_mul[0:1, token_base : token_base + tb, 0:mhc3], T.view(mul_flat_ub, (tb, mhc3)))
            T.copy(gemm_out_sqrsum[0:1, token_base : token_base + tb], sq_ub)

            ##################################################################
            # _mhc_reduce_partials_and_rmsnorm_fwd + _mhc_pre_split_mixes_fwd
            # + _mhc_sinkhorn_fwd in a single SimdVF phase: the whole latency
            # path is one unit so the scheduler can overlap it with the
            # pre-apply memory stream of the previous token block.
            with T.SimdVF():
                mask = S.pset(32)
                ones = S.vdup(T.float32(1), T.float32, mask)
                zeros = S.vdup(T.float32(0), T.float32, mask)
                inv_group = S.vdup(inv_rms_group_size, T.float32, mask)
                lane_idx = S.vci(T.int32(0), T.int32)

                ##################################################################
                # Phase A: mixes = gemm_out_mul * rms, vectorized with the 6-VR
                # compact token layout (copied from the reduce kernel).
                # 16 tokens x 24 mixes = 384 fp32 = 6 VRs of 64 lanes; each
                # token spans 24 lanes of its VR.
                for vec_idx in T.unroll(tb * mhc3 // 64, explicit=True):
                    m_vec = S.vld(mul_flat_ub[vec_idx * 64])
                    rms_vec = S.alloc_var(T.float32)
                    rms_vec = zeros
                    elem_start = vec_idx * 64
                    elem_end = elem_start + 64
                    token_start = elem_start // mhc3
                    token_end = (elem_end - 1) // mhc3
                    for token_idx in range(token_start, token_end + 1):
                        sq_val = S.vld(sq_ub[token_idx], dist='BRC_B32')
                        rms_val = S.vdiv(
                            ones,
                            S.vsqrt(S.vadds(S.vmul(sq_val, inv_group, mask), T.float32(rms_eps), mask), mask),
                            mask,
                        )
                        lane_start = token_idx * mhc3 - elem_start
                        lane_end = lane_start + mhc3
                        token_mask = S.pand(
                            S.vcmps(lane_idx, T.int32(lane_start), mask, 'ge'),
                            S.vcmps(lane_idx, T.int32(lane_end), mask, 'lt'),
                            mask,
                        )
                        rms_vec = S.vsel(rms_val, rms_vec, token_mask)
                    S.vsts(mixes_flat_ub[vec_idx * 64], S.vmul(m_vec, rms_vec, mask), mask, 'NORM_B32')

                S.mem_bar('VST_VLD')

                ##################################################################
                # Phase B: split mixes into pre/post/comb (copied from
                # pre_split_mixes_fwd_simd_mhc4_asc, adjusted for tb tokens).
                idx = S.vci(T.int32(0), T.int32)
                col3 = S.vand(idx, S.vdup(T.int32(3), T.int32, mask), mask)
                col15 = S.vand(idx, S.vdup(T.int32(15), T.int32, mask), mask)
                scale0 = S.vld(scale_ub[0, 0], dist='BRC_B32')
                scale1 = S.vld(scale_ub[1, 0], dist='BRC_B32')
                scale2 = S.vld(scale_ub[2, 0], dist='BRC_B32')
                pre_base_vec = S.alloc_var(T.float32)
                post_base_vec = S.alloc_var(T.float32)
                comb_base_vec = S.alloc_var(T.float32)
                pre_base_vec = zeros
                post_base_vec = zeros
                comb_base_vec = zeros
                for base_id in range(mhc):
                    base_mask = S.vcmps(col3, T.int32(base_id), mask, 'eq')
                    pre_base_vec = S.vsel(S.vld(base_ub[base_id, 0], dist='BRC_B32'), pre_base_vec, base_mask)
                    post_base_vec = S.vsel(S.vld(base_ub[base_id + mhc, 0], dist='BRC_B32'), post_base_vec, base_mask)
                for base_id in range(mhc2):
                    base_mask = S.vcmps(col15, T.int32(base_id), mask, 'eq')
                    comb_base_vec = S.vsel(
                        S.vld(base_ub[base_id + mhc * 2, 0], dist='BRC_B32'),
                        comb_base_vec,
                        base_mask,
                    )

                # pre/post: 16 tokens per VR (4 lanes each)
                for vec_id in range(tb // 16):
                    token = S.vadd(S.vshrs(idx, T.int32(2), mask), S.vdup(T.int32(vec_id * 16), T.int32, mask), mask)
                    token_offset = S.vadd(S.vshls(token, T.int32(4), mask), S.vshls(token, T.int32(3), mask), mask)
                    pre_index = S.vadd(token_offset, col3, mask)
                    post_index = S.vadds(pre_index, T.int32(4), mask)

                    pre_val = S.vadd(
                        S.vmul(S.vgather2(mixes_flat_ub[0], _u32_index(pre_index), mask), scale0, mask),
                        pre_base_vec,
                        mask,
                    )
                    pre_sig = S.vdiv(ones, S.vadds(S.vexpdif(zeros, pre_val, mask), T.float32(1), mask), mask)
                    S.vsts(pre_ub[vec_id * 16, 0], S.vadds(pre_sig, T.float32(mhc_pre_eps), mask), mask, 'NORM_B32')

                    post_val = S.vadd(
                        S.vmul(S.vgather2(mixes_flat_ub[0], _u32_index(post_index), mask), scale1, mask),
                        post_base_vec,
                        mask,
                    )
                    post_sig = S.vdiv(ones, S.vadds(S.vexpdif(zeros, post_val, mask), T.float32(1), mask), mask)
                    S.vsts(
                        post_ub[vec_id * 16, 0],
                        S.vmuls(post_sig, T.float32(mhc_post_mult_value), mask),
                        mask,
                        'NORM_B32',
                    )

                # comb: 4 tokens per VR (16 lanes each)
                for vec_id in range(tb // 4):
                    token = S.vadd(S.vshrs(idx, T.int32(4), mask), S.vdup(T.int32(vec_id * 4), T.int32, mask), mask)
                    token_offset = S.vadd(S.vshls(token, T.int32(4), mask), S.vshls(token, T.int32(3), mask), mask)
                    comb_index = S.vadd(S.vadds(token_offset, T.int32(8), mask), col15, mask)
                    comb_val = S.vadd(
                        S.vmul(S.vgather2(mixes_flat_ub[0], _u32_index(comb_index), mask), scale2, mask),
                        comb_base_vec,
                        mask,
                    )
                    S.vsts(comb_in_ub[vec_id * 4, 0], comb_val, mask, 'NORM_B32')

                S.mem_bar('VST_VLD')

                ##################################################################
                # _mhc_sinkhorn_fwd (4 tokens packed per VR, copied from sinkhorn_asc)
                mask2 = mask
                idx = S.vci(T.int32(0), T.int32)
                const_7 = S.vdup(T.int32(7), T.int32, mask2)
                mod8 = S.vand(idx, const_7, mask2)
                mask_lo = S.vcmps(mod8, T.int32(4), mask2, 'lt')
                mask_hi = S.vcmps(mod8, T.int32(4), mask2, 'ge')

                two = T.int32(2)
                three = T.int32(3)
                four = T.int32(4)
                const_15 = S.vdup(T.int32(15), T.int32, mask2)
                token2 = S.vshls(S.vshrs(idx, four, mask2), T.int32(1), mask2)
                local = S.vand(idx, const_15, mask2)
                local_div8 = S.vshrs(local, three, mask2)
                bcast_idx2 = S.vadd(token2, local_div8, mask2)

                token_base_tidx = S.vshls(S.vshrs(idx, four, mask2), four, mask2)
                col = S.vand(idx, S.vdup(T.int32(3), T.int32, mask2), mask2)
                col4 = S.vshls(col, two, mask2)
                t_idx = S.vadd(S.vadd(token_base_tidx, col4, mask2), S.vshrs(local, two, mask2), mask2)

                cur_a = S.alloc_var(T.float32)
                cur_b = S.alloc_var(T.float32)
                cur_c = S.alloc_var(T.float32)
                cur_d = S.alloc_var(T.float32)
                for g in range(tb // 16):
                    g0 = 4 * g
                    mat_a_0 = S.vld(comb_in_ub[g0 * 4, 0])
                    mat_b_0 = S.vld(comb_in_ub[(g0 + 1) * 4, 0])
                    mat_c_0 = S.vld(comb_in_ub[(g0 + 2) * 4, 0])
                    mat_d_0 = S.vld(comb_in_ub[(g0 + 3) * 4, 0])

                    rmax_a = _row_max(mat_a_0, mask_lo, mask_hi, bcast_idx2)
                    rmax_b = _row_max(mat_b_0, mask_lo, mask_hi, bcast_idx2)
                    rmax_c = _row_max(mat_c_0, mask_lo, mask_hi, bcast_idx2)
                    rmax_d = _row_max(mat_d_0, mask_lo, mask_hi, bcast_idx2)
                    mat_a_1 = S.vexpdif(mat_a_0, rmax_a, mask2)
                    mat_b_1 = S.vexpdif(mat_b_0, rmax_b, mask2)
                    mat_c_1 = S.vexpdif(mat_c_0, rmax_c, mask2)
                    mat_d_1 = S.vexpdif(mat_d_0, rmax_d, mask2)
                    rsum_a = _row_reduce(mat_a_1, mask_lo, mask_hi, bcast_idx2)
                    rsum_b = _row_reduce(mat_b_1, mask_lo, mask_hi, bcast_idx2)
                    rsum_c = _row_reduce(mat_c_1, mask_lo, mask_hi, bcast_idx2)
                    rsum_d = _row_reduce(mat_d_1, mask_lo, mask_hi, bcast_idx2)
                    mat_a_2 = S.vdiv(mat_a_1, rsum_a, mask2)
                    mat_b_2 = S.vdiv(mat_b_1, rsum_b, mask2)
                    mat_c_2 = S.vdiv(mat_c_1, rsum_c, mask2)
                    mat_d_2 = S.vdiv(mat_d_1, rsum_d, mask2)
                    mat_a_3 = S.vadds(mat_a_2, T.float32(mhc_sinkhorn_eps), mask2)
                    mat_b_3 = S.vadds(mat_b_2, T.float32(mhc_sinkhorn_eps), mask2)
                    mat_c_3 = S.vadds(mat_c_2, T.float32(mhc_sinkhorn_eps), mask2)
                    mat_d_3 = S.vadds(mat_d_2, T.float32(mhc_sinkhorn_eps), mask2)

                    tr_a = S.vselr(mat_a_3, t_idx)
                    tr_b = S.vselr(mat_b_3, t_idx)
                    tr_c = S.vselr(mat_c_3, t_idx)
                    tr_d = S.vselr(mat_d_3, t_idx)
                    csum_a = S.vsel(
                        S.vselr(S.vcgadd(tr_a, mask_lo), bcast_idx2),
                        S.vselr(S.vcgadd(tr_a, mask_hi), bcast_idx2),
                        mask_lo,
                    )
                    csum_b = S.vsel(
                        S.vselr(S.vcgadd(tr_b, mask_lo), bcast_idx2),
                        S.vselr(S.vcgadd(tr_b, mask_hi), bcast_idx2),
                        mask_lo,
                    )
                    csum_c = S.vsel(
                        S.vselr(S.vcgadd(tr_c, mask_lo), bcast_idx2),
                        S.vselr(S.vcgadd(tr_c, mask_hi), bcast_idx2),
                        mask_lo,
                    )
                    csum_d = S.vsel(
                        S.vselr(S.vcgadd(tr_d, mask_lo), bcast_idx2),
                        S.vselr(S.vcgadd(tr_d, mask_hi), bcast_idx2),
                        mask_lo,
                    )
                    tr_a = S.vdiv(tr_a, S.vadds(csum_a, T.float32(mhc_sinkhorn_eps), mask2), mask2)
                    tr_b = S.vdiv(tr_b, S.vadds(csum_b, T.float32(mhc_sinkhorn_eps), mask2), mask2)
                    tr_c = S.vdiv(tr_c, S.vadds(csum_c, T.float32(mhc_sinkhorn_eps), mask2), mask2)
                    tr_d = S.vdiv(tr_d, S.vadds(csum_d, T.float32(mhc_sinkhorn_eps), mask2), mask2)
                    cur_a = S.vselr(tr_a, t_idx)
                    cur_b = S.vselr(tr_b, t_idx)
                    cur_c = S.vselr(tr_c, t_idx)
                    cur_d = S.vselr(tr_d, t_idx)

                    for _step in range(sinkhorn_repeat - 1):
                        rsum_a = S.vsel(
                            S.vselr(S.vcgadd(cur_a, mask_lo), bcast_idx2),
                            S.vselr(S.vcgadd(cur_a, mask_hi), bcast_idx2),
                            mask_lo,
                        )
                        rsum_b = S.vsel(
                            S.vselr(S.vcgadd(cur_b, mask_lo), bcast_idx2),
                            S.vselr(S.vcgadd(cur_b, mask_hi), bcast_idx2),
                            mask_lo,
                        )
                        rsum_c = S.vsel(
                            S.vselr(S.vcgadd(cur_c, mask_lo), bcast_idx2),
                            S.vselr(S.vcgadd(cur_c, mask_hi), bcast_idx2),
                            mask_lo,
                        )
                        rsum_d = S.vsel(
                            S.vselr(S.vcgadd(cur_d, mask_lo), bcast_idx2),
                            S.vselr(S.vcgadd(cur_d, mask_hi), bcast_idx2),
                            mask_lo,
                        )
                        cur_a = S.vdiv(cur_a, S.vadds(rsum_a, T.float32(mhc_sinkhorn_eps), mask2), mask2)
                        cur_b = S.vdiv(cur_b, S.vadds(rsum_b, T.float32(mhc_sinkhorn_eps), mask2), mask2)
                        cur_c = S.vdiv(cur_c, S.vadds(rsum_c, T.float32(mhc_sinkhorn_eps), mask2), mask2)
                        cur_d = S.vdiv(cur_d, S.vadds(rsum_d, T.float32(mhc_sinkhorn_eps), mask2), mask2)

                        tr_a = S.vselr(cur_a, t_idx)
                        tr_b = S.vselr(cur_b, t_idx)
                        tr_c = S.vselr(cur_c, t_idx)
                        tr_d = S.vselr(cur_d, t_idx)
                        csum_a = S.vsel(
                            S.vselr(S.vcgadd(tr_a, mask_lo), bcast_idx2),
                            S.vselr(S.vcgadd(tr_a, mask_hi), bcast_idx2),
                            mask_lo,
                        )
                        csum_b = S.vsel(
                            S.vselr(S.vcgadd(tr_b, mask_lo), bcast_idx2),
                            S.vselr(S.vcgadd(tr_b, mask_hi), bcast_idx2),
                            mask_lo,
                        )
                        csum_c = S.vsel(
                            S.vselr(S.vcgadd(tr_c, mask_lo), bcast_idx2),
                            S.vselr(S.vcgadd(tr_c, mask_hi), bcast_idx2),
                            mask_lo,
                        )
                        csum_d = S.vsel(
                            S.vselr(S.vcgadd(tr_d, mask_lo), bcast_idx2),
                            S.vselr(S.vcgadd(tr_d, mask_hi), bcast_idx2),
                            mask_lo,
                        )
                        tr_a = S.vdiv(tr_a, S.vadds(csum_a, T.float32(mhc_sinkhorn_eps), mask2), mask2)
                        tr_b = S.vdiv(tr_b, S.vadds(csum_b, T.float32(mhc_sinkhorn_eps), mask2), mask2)
                        tr_c = S.vdiv(tr_c, S.vadds(csum_c, T.float32(mhc_sinkhorn_eps), mask2), mask2)
                        tr_d = S.vdiv(tr_d, S.vadds(csum_d, T.float32(mhc_sinkhorn_eps), mask2), mask2)
                        cur_a = S.vselr(tr_a, t_idx)
                        cur_b = S.vselr(tr_b, t_idx)
                        cur_c = S.vselr(tr_c, t_idx)
                        cur_d = S.vselr(tr_d, t_idx)

                    S.vsts(comb_out_ub[g0 * 4, 0], cur_a, mask2)
                    S.vsts(comb_out_ub[(g0 + 1) * 4, 0], cur_b, mask2)
                    S.vsts(comb_out_ub[(g0 + 2) * 4, 0], cur_c, mask2)
                    S.vsts(comb_out_ub[(g0 + 3) * 4, 0], cur_d, mask2)

            T.copy(post_ub[:, 0:mhc], post_mix[token_base : token_base + tb, 0:mhc])
            T.copy(comb_out_ub, comb_mix[token_base : token_base + tb, 0:mhc2])

            ##################################################################
            # _mhc_pre_apply_mix_fwd
            for i0_h in T.Pipelined(num_hidden_tiles, num_stages=2):
                T.copy(
                    residual[
                        token_base : token_base + tb,
                        0:mhc,
                        i0_h * h_blk : (i0_h + 1) * h_blk,
                    ],
                    x_ub,
                )
                with T.SimdVF():
                    mix_reg = S.alloc_local((mhc,), T.float32)
                    xs = S.alloc_local((mhc,), T.float32)
                    acc = S.alloc_var(T.float32)
                    for t in T.serial(tb):
                        for m in T.unroll(mhc, explicit=True):
                            mix_reg[m] = S.vld(pre_ub[t, m], dist='BRC_B32')
                        for c in T.serial(num_chunks):
                            col = c * _VEC
                            for m in T.unroll(mhc, explicit=True):
                                xs[m] = S.vcvt(S.vld(x_ub[t, m, col], dist='UNPK_B16'), T.float32, part=0)
                            acc = S.vmul(xs[0], mix_reg[0])
                            for m in T.unroll(mhc - 1, explicit=True):
                                S.vmula(acc, xs[m + 1], mix_reg[m + 1])
                            S.vsts(layer_input_ub[t, col], S.vcvt(acc, T.bfloat16), dist='PK_B32')

                T.copy(
                    layer_input_ub,
                    layer_input[
                        token_base : token_base + tb,
                        i0_h * h_blk : (i0_h + 1) * h_blk,
                    ],
                )
