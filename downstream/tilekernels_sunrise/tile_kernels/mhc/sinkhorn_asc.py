import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S


def _row_reduce(mat, mask_lo, mask_hi, bcast_idx2):
    """Row reduce: vcgadd ×2 + vselr ×2 + vsel → broadcast row sum."""
    return S.vsel(S.vselr(S.vcgadd(mat, mask_lo), bcast_idx2), S.vselr(S.vcgadd(mat, mask_hi), bcast_idx2), mask_lo)


def _row_max(mat, mask_lo, mask_hi, bcast_idx2):
    """Row max: vcgmax ×2 + vselr ×2 + vsel → broadcast row max."""
    return S.vsel(S.vselr(S.vcgmax(mat, mask_lo), bcast_idx2), S.vselr(S.vcgmax(mat, mask_hi), bcast_idx2), mask_lo)


def _col_reduce(mat, mask_lo, mask_hi, bcast_idx2, t_idx):
    """Col reduce: transpose, vcgadd ×2 + vselr ×2 + vsel → (transposed, broadcast col sum).

    NOTE: The returned broadcast col sum is in TRANSPOSED layout (each lane
    gets its transposed-row sum = original-col sum). To use with original-layout
    data, transpose the broadcast result back via S.vselr(csum_b, t_idx).
    """
    tr = S.vselr(mat, t_idx)
    return tr, S.vsel(S.vselr(S.vcgadd(tr, mask_lo), bcast_idx2), S.vselr(S.vcgadd(tr, mask_hi), bcast_idx2), mask_lo)


@tilelang.jit
def mhc_sinkhorn_fwd_asc(
    comb_res_mix,
    comb_res_mix_out,
    repeat: int,
    eps: float,
    token_block_size: int,
    num_vec_cores: int,
):
    """SimdVF forward: 4 tokens packed per VR (64 fp32 = 4 x 4x4).

    Layout per VR: 4 tokens x 16 elements = 64 fp32.
    Each VLane (8 fp32) spans 2 rows of the same token:
      VLane g: token_g/2, rows 2*(g%2) and 2*(g%2)+1, 4 cols each.

    vcgadd/vcgmax operate per VLane (8 fp32). We run them twice with
    complementary masks ([1,1,1,1,0,0,0,0] and its complement) to
    separate odd/even rows. Results are merged via vintlv.

    Masks are constructed via vci + vand + vcmps, since CCE pset
    patterns don't include per-VLane repeating [11110000].
    """
    num_tokens = T.dynamic('num_tokens')
    hidden_size = T.const('hidden_size')

    comb_res_mix: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    comb_res_mix_out: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]

    num_stages = 4

    with T.Kernel(num_vec_cores) as core_id:
        comb_in_ub = T.alloc_shared((token_block_size, hidden_size, hidden_size), T.float32)
        comb_out_ub = T.alloc_shared((token_block_size, hidden_size, hidden_size), T.float32)
        T.annotate_buffer_versions({comb_in_ub: num_stages, comb_out_ub: num_stages})

        for pid_x in T.Persistent(
            [T.ceildiv(num_tokens, token_block_size)],
            num_vec_cores,
            core_id,
            num_stages=num_stages,
        ):
            token_offset = pid_x * token_block_size
            T.copy(comb_res_mix[token_offset, 0, 0], comb_in_ub)

            with T.SimdVF():
                mask_full = S.pset(32)

                # Construct masks for first/last 4 of each 8-element VLane
                idx = S.vci(T.int32(0), T.int32)
                const_7 = S.vdup(T.int32(7), T.int32, mask_full)
                mod8 = S.vand(idx, const_7, mask_full)
                # mask_lo: mod8 < 4 → [1,1,1,1,0,0,0,0] per VLane
                mask_lo = S.vcmps(mod8, T.int32(4), mask_full, 'lt')
                # mask_hi: mod8 >= 4 → [0,0,0,0,1,1,1,1] per VLane
                mask_hi = S.vcmps(mod8, T.int32(4), mask_full, 'ge')

                # bcast_idx2: broadcast index for the two vcg results.
                # rsum_lo[0..7] = [t0_r0, t0_r2, t1_r0, t1_r2, t2_r0, t2_r2, t3_r0, t3_r2]
                # rsum_hi[0..7] = [t0_r1, t0_r3, t1_r1, t1_r3, t2_r1, t2_r3, t3_r1, t3_r3]
                # For lane in [0..7]: need rsum_lo/hi[0] (t0 r0/r1)
                # For lane in [8..15]: need rsum_lo/hi[1] (t0 r2/r3)
                # For lane in [16..23]: need rsum_lo/hi[2] (t1 r0/r1)
                # bcast_idx2 = (lane // 16) * 2 + (lane % 16) // 8
                two = T.int32(2)
                three = T.int32(3)
                four = T.int32(4)
                const_15 = S.vdup(T.int32(15), T.int32, mask_full)
                token2 = S.vshls(S.vshrs(idx, four, mask_full), T.int32(1), mask_full)
                local = S.vand(idx, const_15, mask_full)
                local_div8 = S.vshrs(local, three, mask_full)
                bcast_idx2 = S.vadd(token2, local_div8, mask_full)

                # t_idx for 4x4 transpose within each 16-element token:
                # [r][c] at r*4+c, transpose → c*4+r
                # t_idx = (lane // 16) * 16 + col*4 + row
                # where col = lane % 4, row = (lane % 16) // 4
                # token_base_tidx = (lane // 16) * 16 = (lane >> 4) << 4
                token_base_tidx = S.vshls(S.vshrs(idx, four, mask_full), four, mask_full)
                col = S.vand(idx, S.vdup(T.int32(3), T.int32, mask_full), mask_full)
                col4 = S.vshls(col, two, mask_full)
                t_idx = S.vadd(S.vadd(token_base_tidx, col4, mask_full), S.vshrs(local, two, mask_full), mask_full)

                # Pure SSA style: no alloc_var, each assignment creates a new
                # Python variable to avoid IR builder CSE issues across loop
                # iterations. Use Python range() for compile-time unrolling.

                # Load 4 tokens x 16 elements = 64 fp32
                mat_0 = S.vld(comb_in_ub[0, 0, 0])

                # Step 1: Row softmax + eps
                rmax_b_0 = S.vsel(
                    S.vselr(S.vcgmax(mat_0, mask_lo), bcast_idx2),
                    S.vselr(S.vcgmax(mat_0, mask_hi), bcast_idx2),
                    mask_lo,
                )
                mat_1 = S.vexpdif(mat_0, rmax_b_0, mask_full)
                rsum_b_0 = S.vsel(
                    S.vselr(S.vcgadd(mat_1, mask_lo), bcast_idx2),
                    S.vselr(S.vcgadd(mat_1, mask_hi), bcast_idx2),
                    mask_lo,
                )
                mat_2 = S.vdiv(mat_1, rsum_b_0, mask_full)
                mat_3 = S.vadds(mat_2, T.float32(eps), mask_full)

                # Step 2: Col normalize (transpose, reduce, normalize, transpose back)
                tr_0 = S.vselr(mat_3, t_idx)
                csum_b_0 = S.vsel(
                    S.vselr(S.vcgadd(tr_0, mask_lo), bcast_idx2),
                    S.vselr(S.vcgadd(tr_0, mask_hi), bcast_idx2),
                    mask_lo,
                )
                tr_1 = S.vdiv(tr_0, S.vadds(csum_b_0, T.float32(eps), mask_full), mask_full)
                mat_4 = S.vselr(tr_1, t_idx)

                # Step 3: repeat-1 iterations (Python range() unroll)
                # Use alloc_var for cur to allow cross-iteration assignment.
                # Pre-loop uses unique variable names (mat_0..mat_4) to avoid
                # CSE matching loop body expressions against pre-loop SSA values.
                cur = S.alloc_var(T.float32)
                cur = mat_4
                for _step in range(repeat - 1):
                    rsum_b = S.vsel(
                        S.vselr(S.vcgadd(cur, mask_lo), bcast_idx2),
                        S.vselr(S.vcgadd(cur, mask_hi), bcast_idx2),
                        mask_lo,
                    )
                    cur = S.vdiv(cur, S.vadds(rsum_b, T.float32(eps), mask_full), mask_full)

                    tr = S.vselr(cur, t_idx)
                    csum_b = S.vsel(
                        S.vselr(S.vcgadd(tr, mask_lo), bcast_idx2),
                        S.vselr(S.vcgadd(tr, mask_hi), bcast_idx2),
                        mask_lo,
                    )
                    tr = S.vdiv(tr, S.vadds(csum_b, T.float32(eps), mask_full), mask_full)
                    cur = S.vselr(tr, t_idx)

                S.vsts(comb_out_ub[0, 0, 0], cur, mask_full)

            T.copy(comb_out_ub, comb_res_mix_out[token_offset, 0, 0])


@tilelang.jit
def mhc_sinkhorn_bwd_asc(
    grad_output,
    x,
    grad_input,
    repeat: int,
    eps: float,
    token_block_size: int,
    num_vec_cores: int,
):
    """SimdVF backward: 8 tokens (2 VRs) per block.

    Two phases in one SimdVF:
    1. Forward recompute: softmax + repeat-1 iterations of row/col normalize.
       Save intermediate x states and sums to UB.
    2. Backward iteration: reverse 2*repeat-1 steps, loading x_inter and sum
       from UB, computing grad updates.

    2 VRs are independent (different tokens), so the compiler can interleave
    their loads and computes to hide vld latency.

    UB layout for intermediate storage (per block of 8 tokens):
      xs_ub:   (2*repeat, 8, 4, 4) fp32 = 2*repeat * 2048B  (e.g. 40KB for repeat=10)
      sums_ub: (2*repeat, 8, 4)   fp32 = 2*repeat *  512B  (e.g. 10KB for repeat=10)

    Each VR holds 4 tokens x 16 elements = 64 fp32.
    VR0 = tokens 0-3 (UB offset 0), VR1 = tokens 4-7 (UB offset 64 fp32 = 256B).
    """
    num_tokens = T.dynamic('num_tokens')
    hidden_size = T.const('hidden_size')

    grad_output: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    x: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    grad_input: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]

    num_stages = 3

    with T.Kernel(num_vec_cores) as core_id:
        grad_ub = T.alloc_shared((token_block_size, hidden_size, hidden_size), T.float32)
        x_ub = T.alloc_shared((token_block_size, hidden_size, hidden_size), T.float32)
        out_ub = T.alloc_shared((token_block_size, hidden_size, hidden_size), T.float32)
        xs_ub = T.alloc_shared((repeat * 2, token_block_size, hidden_size, hidden_size), T.float32)
        sums_ub = T.alloc_shared((repeat * 2, token_block_size, hidden_size, hidden_size), T.float32)
        T.annotate_buffer_versions({grad_ub: num_stages, x_ub: num_stages, out_ub: num_stages})

        for pid_x in T.Persistent(
            [T.ceildiv(num_tokens, token_block_size)],
            num_vec_cores,
            core_id,
            num_stages=num_stages,
        ):
            token_offset = pid_x * token_block_size
            T.copy(grad_output[token_offset, 0, 0], grad_ub)
            T.copy(x[token_offset, 0, 0], x_ub)

            with T.SimdVF():
                mask_full = S.pset(32)

                # --- Precompute index VRs (shared across both VRs) ---
                idx = S.vci(T.int32(0), T.int32)
                const_7 = S.vdup(T.int32(7), T.int32, mask_full)
                mod8 = S.vand(idx, const_7, mask_full)
                mask_lo = S.vcmps(mod8, T.int32(4), mask_full, 'lt')
                mask_hi = S.vcmps(mod8, T.int32(4), mask_full, 'ge')
                two = T.int32(2)
                three = T.int32(3)
                four = T.int32(4)
                const_15 = S.vdup(T.int32(15), T.int32, mask_full)
                token2 = S.vshls(S.vshrs(idx, four, mask_full), T.int32(1), mask_full)
                local = S.vand(idx, const_15, mask_full)
                local_div8 = S.vshrs(local, three, mask_full)
                bcast_idx2 = S.vadd(token2, local_div8, mask_full)
                token_base_tidx = S.vshls(S.vshrs(idx, four, mask_full), four, mask_full)
                col = S.vand(idx, S.vdup(T.int32(3), T.int32, mask_full), mask_full)
                col4 = S.vshls(col, two, mask_full)
                t_idx = S.vadd(S.vadd(token_base_tidx, col4, mask_full), S.vshrs(local, two, mask_full), mask_full)

                # Helpers are module-level _row_reduce / _col_reduce, called
                # with precomputed mask/index VRs as closure args.

                # VR1 starts at token 4 (4 tokens per VR)
                vr1_tok = 4

                # ==================== Phase 1: Forward recompute ====================
                # Load both VRs
                x0_0 = S.vld(x_ub[0, 0, 0])  # VR0: tokens 0-3
                x1_0 = S.vld(x_ub[vr1_tok, 0, 0])  # VR1: tokens 4-7

                # Softmax + eps
                rmax0_b = _row_max(x0_0, mask_lo, mask_hi, bcast_idx2)
                rmax1_b = _row_max(x1_0, mask_lo, mask_hi, bcast_idx2)
                x0_1 = S.alloc_var(T.float32)
                x1_1 = S.alloc_var(T.float32)
                x0_1 = S.vexpdif(x0_0, rmax0_b, mask_full)
                x1_1 = S.vexpdif(x1_0, rmax1_b, mask_full)
                rsum0_b = _row_reduce(x0_1, mask_lo, mask_hi, bcast_idx2)
                rsum1_b = _row_reduce(x1_1, mask_lo, mask_hi, bcast_idx2)
                x0_2 = S.vdiv(x0_1, rsum0_b, mask_full)
                x1_2 = S.vdiv(x1_1, rsum1_b, mask_full)
                x0_3 = S.vadds(x0_2, T.float32(eps), mask_full)
                x1_3 = S.vadds(x1_2, T.float32(eps), mask_full)

                # Save xs[0] = softmax(x) (before +eps)
                S.vsts(xs_ub[0, 0, 0, 0], x0_2, mask_full)
                S.vsts(xs_ub[0, vr1_tok, 0, 0], x1_2, mask_full)
                # Save xs[1] = softmax(x) + eps
                S.vsts(xs_ub[1, 0, 0, 0], x0_3, mask_full)
                S.vsts(xs_ub[1, vr1_tok, 0, 0], x1_3, mask_full)

                # First col normalize
                tr0_0, csum0_b = _col_reduce(x0_3, mask_lo, mask_hi, bcast_idx2, t_idx)
                tr1_0, csum1_b = _col_reduce(x1_3, mask_lo, mask_hi, bcast_idx2, t_idx)
                # Save sums[1] = col_sum (transpose back to original layout for bwd)
                csum0_b_t = S.vselr(csum0_b, t_idx)
                csum1_b_t = S.vselr(csum1_b, t_idx)
                S.vsts(sums_ub[1, 0, 0, 0], csum0_b_t, mask_full)
                S.vsts(sums_ub[1, vr1_tok, 0, 0], csum1_b_t, mask_full)
                tr0_1 = S.vdiv(tr0_0, S.vadds(csum0_b, T.float32(eps), mask_full), mask_full)
                tr1_1 = S.vdiv(tr1_0, S.vadds(csum1_b, T.float32(eps), mask_full), mask_full)
                x0_4 = S.vselr(tr0_1, t_idx)
                x1_4 = S.vselr(tr1_1, t_idx)

                # repeat-1 iterations of row + col normalize
                # Use alloc_var for loop-carried variables + Python range() unroll
                cur0 = S.alloc_var(T.float32)
                cur1 = S.alloc_var(T.float32)
                cur0 = x0_4
                cur1 = x1_4

                for step in range(repeat - 1):
                    s_row = (step + 1) * 2
                    s_col = (step + 1) * 2 + 1

                    # Row normalize
                    rsum0_b = _row_reduce(cur0, mask_lo, mask_hi, bcast_idx2)
                    rsum1_b = _row_reduce(cur1, mask_lo, mask_hi, bcast_idx2)
                    S.vsts(sums_ub[s_row, 0, 0, 0], rsum0_b, mask_full)
                    S.vsts(sums_ub[s_row, vr1_tok, 0, 0], rsum1_b, mask_full)
                    S.vsts(xs_ub[s_row, 0, 0, 0], cur0, mask_full)
                    S.vsts(xs_ub[s_row, vr1_tok, 0, 0], cur1, mask_full)
                    cur0 = S.vdiv(cur0, S.vadds(rsum0_b, T.float32(eps), mask_full), mask_full)
                    cur1 = S.vdiv(cur1, S.vadds(rsum1_b, T.float32(eps), mask_full), mask_full)

                    # Col normalize
                    tr0, csum0_b = _col_reduce(cur0, mask_lo, mask_hi, bcast_idx2, t_idx)
                    tr1, csum1_b = _col_reduce(cur1, mask_lo, mask_hi, bcast_idx2, t_idx)
                    csum0_b_t = S.vselr(csum0_b, t_idx)
                    csum1_b_t = S.vselr(csum1_b, t_idx)
                    S.vsts(sums_ub[s_col, 0, 0, 0], csum0_b_t, mask_full)
                    S.vsts(sums_ub[s_col, vr1_tok, 0, 0], csum1_b_t, mask_full)
                    S.vsts(xs_ub[s_col, 0, 0, 0], cur0, mask_full)
                    S.vsts(xs_ub[s_col, vr1_tok, 0, 0], cur1, mask_full)
                    tr0 = S.vdiv(tr0, S.vadds(csum0_b, T.float32(eps), mask_full), mask_full)
                    tr1 = S.vdiv(tr1, S.vadds(csum1_b, T.float32(eps), mask_full), mask_full)
                    cur0 = S.vselr(tr0, t_idx)
                    cur1 = S.vselr(tr1, t_idx)

                # ==================== Phase 2: Backward iteration ====================
                # Ensure all Phase 1 stores are visible before Phase 2 loads
                S.mem_bar('VST_VLD')

                # Load grad for both VRs
                g0 = S.alloc_var(T.float32)
                g1 = S.alloc_var(T.float32)
                g0 = S.vld(grad_ub[0, 0, 0])
                g1 = S.vld(grad_ub[vr1_tok, 0, 0])

                for inv_step in range(2 * repeat - 1):
                    s_idx = 2 * repeat - 1 - inv_step
                    # Load x_inter and sum from UB
                    xi0 = S.vld(xs_ub[s_idx, 0, 0, 0])
                    xi1 = S.vld(xs_ub[s_idx, vr1_tok, 0, 0])
                    sm0 = S.vld(sums_ub[s_idx, 0, 0, 0])
                    sm1 = S.vld(sums_ub[s_idx, vr1_tok, 0, 0])

                    if inv_step % 2 == 0:
                        # Col normalize backward:
                        # temp = grad * x_inter
                        # col_sum2 = sum(temp, dim=1)  → need col_reduce
                        # col_sum2 /= sum + eps
                        # grad = (grad - col_sum2) / (sum + eps)
                        temp0 = S.vmul(g0, xi0, mask_full)
                        temp1 = S.vmul(g1, xi1, mask_full)
                        _, cs2_0_b = _col_reduce(temp0, mask_lo, mask_hi, bcast_idx2, t_idx)
                        _, cs2_1_b = _col_reduce(temp1, mask_lo, mask_hi, bcast_idx2, t_idx)
                        # cs2 is in transposed layout — transpose back to original
                        cs2_0_b = S.vselr(cs2_0_b, t_idx)
                        cs2_1_b = S.vselr(cs2_1_b, t_idx)
                        cs2_0_b = S.vdiv(cs2_0_b, S.vadds(sm0, T.float32(eps), mask_full), mask_full)
                        cs2_1_b = S.vdiv(cs2_1_b, S.vadds(sm1, T.float32(eps), mask_full), mask_full)
                        g0 = S.vdiv(S.vsub(g0, cs2_0_b, mask_full), S.vadds(sm0, T.float32(eps), mask_full), mask_full)
                        g1 = S.vdiv(S.vsub(g1, cs2_1_b, mask_full), S.vadds(sm1, T.float32(eps), mask_full), mask_full)
                    else:
                        # Row normalize backward:
                        # temp = grad * x_inter
                        # row_sum2 = sum(temp, dim=2)  → row_reduce
                        # row_sum2 /= sum + eps
                        # grad = (grad - row_sum2) / (sum + eps)
                        temp0 = S.vmul(g0, xi0, mask_full)
                        temp1 = S.vmul(g1, xi1, mask_full)
                        rs2_0_b = _row_reduce(temp0, mask_lo, mask_hi, bcast_idx2)
                        rs2_1_b = _row_reduce(temp1, mask_lo, mask_hi, bcast_idx2)
                        rs2_0_b = S.vdiv(rs2_0_b, S.vadds(sm0, T.float32(eps), mask_full), mask_full)
                        rs2_1_b = S.vdiv(rs2_1_b, S.vadds(sm1, T.float32(eps), mask_full), mask_full)
                        g0 = S.vdiv(S.vsub(g0, rs2_0_b, mask_full), S.vadds(sm0, T.float32(eps), mask_full), mask_full)
                        g1 = S.vdiv(S.vsub(g1, rs2_1_b, mask_full), S.vadds(sm1, T.float32(eps), mask_full), mask_full)

                # Final step: grad = (grad - row_sum(grad * x_inter)) * x_inter
                xi0 = S.vld(xs_ub[0, 0, 0, 0])
                xi1 = S.vld(xs_ub[0, vr1_tok, 0, 0])
                temp0 = S.vmul(g0, xi0, mask_full)
                temp1 = S.vmul(g1, xi1, mask_full)
                rs0_b = _row_reduce(temp0, mask_lo, mask_hi, bcast_idx2)
                rs1_b = _row_reduce(temp1, mask_lo, mask_hi, bcast_idx2)
                g0 = S.vmul(S.vsub(g0, rs0_b, mask_full), xi0, mask_full)
                g1 = S.vmul(S.vsub(g1, rs1_b, mask_full), xi1, mask_full)

                S.vsts(out_ub[0, 0, 0], g0, mask_full)
                S.vsts(out_ub[vr1_tok, 0, 0], g1, mask_full)

            T.copy(out_ub, grad_input[token_offset, 0, 0])
