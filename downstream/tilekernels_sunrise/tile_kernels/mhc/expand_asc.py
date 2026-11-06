import tilelang
from tilelang.ascend import language as T


@tilelang.jit
def expand_to_mhc_fwd_asc(x, o, h_blk: int, num_stages: int, num_vec_cores: int):
    num_tokens = T.dynamic('num_tokens')
    h, mhc = T.const('h, mhc')
    x: T.Tensor[(num_tokens, h), T.bfloat16]
    o: T.Tensor[(num_tokens, mhc, h), T.bfloat16]

    h_blk = T.min(h_blk, h)

    with T.Kernel(num_vec_cores) as core_id:
        x_ub = T.alloc_shared((h_blk,), T.bfloat16)
        o_ub = T.alloc_shared((mhc, h_blk), T.bfloat16)
        T.annotate_buffer_versions({x_ub: num_stages, o_ub: num_stages})

        for pid_n, pid_h in T.Persistent(
            [num_tokens, T.ceildiv(h, h_blk)],
            num_vec_cores,
            core_id,
            group_size=1,
            num_stages=num_stages,
        ):
            T.copy(x[pid_n, pid_h * h_blk], x_ub)
            with T.SimdVF():
                mask_16b = T.simd.pset(16)
                num_chunks = h_blk // 128
                for c in T.serial(num_chunks):
                    col = c * 128
                    xb = T.simd.vld(x_ub[col])
                    for m in T.unroll(mhc, explicit=True):
                        T.simd.vsts(o_ub[m, col], xb, mask_16b)

            T.copy(o_ub, o[pid_n, 0, pid_h * h_blk])


@tilelang.jit
def expand_to_mhc_bwd_asc(o_grad, x_grad, h_blk: int, num_stages: int, num_vec_cores: int):
    num_tokens = T.dynamic('num_tokens')
    h, mhc = T.const('h, mhc')
    o_grad: T.Tensor[(num_tokens, mhc, h), T.bfloat16]
    x_grad: T.Tensor[(num_tokens, h), T.bfloat16]

    h_blk = T.min(h_blk, h)

    with T.Kernel(num_vec_cores) as core_id:
        o_grad_ub = T.alloc_shared((mhc, h_blk), T.bfloat16)
        x_grad_ub = T.alloc_shared((h_blk,), T.bfloat16)
        T.annotate_buffer_versions({o_grad_ub: num_stages, x_grad_ub: num_stages})

        for pid_n, pid_h in T.Persistent(
            [num_tokens, T.ceildiv(h, h_blk)],
            num_vec_cores,
            core_id,
            group_size=1,
            num_stages=num_stages,
        ):
            T.copy(o_grad[pid_n, 0, pid_h * h_blk], o_grad_ub)

            with T.SimdVF():
                mask_32b = T.simd.pset(32)
                mask_16b = T.simd.pset(16)
                acc = T.simd.alloc_var(T.float32)
                num_chunks = h_blk // 64
                for c in range(num_chunks):
                    col = c * 64
                    acc = T.simd.vcvt(T.simd.vld(o_grad_ub[0, col], dist='UNPK_B16'), T.float32, mask=mask_16b, part=0)
                    for m in T.unroll(mhc - 1, explicit=True):
                        og = T.simd.vcvt(T.simd.vld(o_grad_ub[m + 1, col], dist='UNPK_B16'), T.float32, mask=mask_16b, part=0)
                        acc = T.simd.vadd(acc, og, mask_32b)
                    T.simd.vsts(x_grad_ub[col], T.simd.vcvt(acc, T.bfloat16, mask=mask_32b), mask_32b, dist='PK_B32')

            T.copy(x_grad_ub, x_grad[pid_n, pid_h * h_blk])
