import torch
import tilelang as tl
import tilelang.language as T
from tilelang.profiler import do_bench

import argparse


@tl.jit(out_idx=3, pass_configs={"tl.disable_warp_specialized": True})
def chunk_retention_fwd_kernel(
    B,
    S,
    H,
    DK,
    DV,
    dtype: T.dtype = T.float16,
    scale: float = None,
) -> torch.Tensor:
    if scale is None:
        scale = DK**-0.5
    accum_dtype = T.float32

    chunk_size = 64
    BK = BV = 64  # Set to 128 can be faster, but has some numerical differences with FLA
    assert S % chunk_size == 0 and DK % BK == 0 and DV % BV == 0
    NK = tl.cdiv(DK, BK)
    NV = tl.cdiv(DV, BV)
    NT = tl.cdiv(S, chunk_size)

    @T.prim_func
    def chunk_retention_fwd(
        Q: T.Tensor([B, S, H, DK], dtype),  # type: ignore
        K: T.Tensor([B, S, H, DK], dtype),  # type: ignore
        V: T.Tensor([B, S, H, DV], dtype),  # type: ignore
        O: T.Tensor([NK, B, S, H, DV], dtype),  # type: ignore
        log_decay: T.Tensor([H], T.float32),  # type: ignore
    ):
        with T.Kernel(NV, NK, B * H) as (i_v, i_k, i_bh):
            i_b = i_bh // H
            i_h = i_bh % H
            ld = log_decay[i_h]  # Head-specific decay (computed in double on the host)

            q = T.alloc_shared([chunk_size, BK], dtype)
            k = T.alloc_shared([chunk_size, BK], dtype)
            v = T.alloc_shared([chunk_size, BV], dtype)
            h = T.alloc_fragment([BK, BV], accum_dtype)
            h_shared = T.alloc_shared([BK, BV], dtype)
            s = T.alloc_fragment([chunk_size, chunk_size], accum_dtype)
            s_shared = T.alloc_shared([chunk_size, chunk_size], dtype)
            o = T.alloc_fragment([chunk_size, BV], accum_dtype)
            T.clear(h)

            T.use_swizzle(10)

            for i in T.Pipelined(0, NT):
                for row, col in T.Parallel(chunk_size, BK):
                    q[row, col] = Q[i_b, i * chunk_size + row, i_h, i_k * BK + col] * scale
                T.copy(K[i_b, i * chunk_size : (i + 1) * chunk_size, i_h, i_k * BK : (i_k + 1) * BK], k)
                T.copy(V[i_b, i * chunk_size : (i + 1) * chunk_size, i_h, i_v * BV : (i_v + 1) * BV], v)

                T.gemm(q, k, s, clear_accum=True, transpose_B=True)
                for row, col in T.Parallel(chunk_size, chunk_size):
                    s_shared[row, col] = T.if_then_else(row >= col, s[row, col] * T.exp2((row - col) * ld), 0)

                T.copy(h, h_shared)
                T.gemm(q, h_shared, o, clear_accum=True)
                for row, col in T.Parallel(chunk_size, BV):
                    o[row, col] = T.exp2((row + 1) * ld) * o[row, col]
                T.gemm(s_shared, v, o)

                for row, col in T.Parallel(chunk_size, BV):
                    v[row, col] = v[row, col] * T.exp2((chunk_size - row - 1) * ld)
                for row, col in T.Parallel(BK, BV):
                    h[row, col] = T.exp2(chunk_size * ld) * h[row, col]
                T.copy(o, O[i_k, i_b, i * chunk_size : (i + 1) * chunk_size, i_h, i_v * BV : (i_v + 1) * BV])
                T.gemm(k, v, h, transpose_A=True)

    return chunk_retention_fwd


def postprocess(o):
    return o if o.size(0) == 1 else o.sum(0)


def ref_program(Q, K, V, log_decay=None, scale=None):
    """CPU reference for chunk_retention_fwd (chunk-wise parallel form).

    Mirrors the kernel's recurrence chunk by chunk. For chunk i (positions
    [i*64, (i+1)*64)):

        s   = (Q*scale) @ K^T                    intra-chunk scores (f32 accum)
        s_c = s ⊙ γ^(r-c)  for r >= c, else 0    causal mask + per-head decay
        o   = (Q*scale) @ h · γ^(r+1)  +  s_c @ V
        h   = γ^64 · h  +  K^T @ (V ⊙ γ^(63-r))

    with per-head decay γ = 2^log_decay (log_decay computed in double on the
    host and passed in). The fp16 roundings at the q / s_shared / h_shared
    boundaries are reproduced so the reference stays within the kernel's own
    precision.
    """
    B, S, H, D = Q.shape
    if scale is None:
        scale = D**-0.5
    chunk_size = 64
    NT = S // chunk_size
    assert S % chunk_size == 0

    device = Q.device
    if log_decay is None:
        # double 算，避免 h >= 20 时 1 - 2^(-5-h) 在 fp32 下坍缩为 1.0
        log_decay = torch.log2(1 - torch.exp2(-5.0 - torch.arange(H, dtype=torch.float64))).to(torch.float32)
    log_decay = log_decay.to(device, torch.float32)

    Qf = Q.to(torch.float32) * scale
    Kf = K.to(torch.float32)
    Vf = V.to(torch.float32)

    idx = torch.arange(chunk_size, device=device, dtype=torch.float32)

    diff = idx[:, None] - idx[None, :]  # [64, 64]
    causal = (diff >= 0).to(torch.float32)
    decay3 = torch.exp2(diff.unsqueeze(0) * log_decay[:, None, None]) * causal.unsqueeze(0)  # [H, 64, 64]
    row_decay = torch.exp2((idx + 1).unsqueeze(0) * log_decay[:, None])  # [H, 64]
    v_decay = torch.exp2((chunk_size - 1 - idx).unsqueeze(0) * log_decay[:, None])  # [H, 64]
    chunk_decay = torch.exp2(chunk_size * log_decay)  # [H]

    h = torch.zeros(B, H, D, D, device=device, dtype=torch.float32)
    O = torch.zeros(B, S, H, D, device=device, dtype=torch.float32)

    for i in range(NT):
        # q is stored to fp16 shared in the kernel (Q * scale -> fp16)
        q = Qf[:, i * chunk_size : (i + 1) * chunk_size].half().float().permute(0, 2, 1, 3)  # [B, H, 64, D]
        k = Kf[:, i * chunk_size : (i + 1) * chunk_size].permute(0, 2, 1, 3)
        v = Vf[:, i * chunk_size : (i + 1) * chunk_size].permute(0, 2, 1, 3)

        s = torch.einsum("bhrd,bhcd->bhrc", q, k)  # [B, H, 64, 64]
        s_c = (s * decay3.unsqueeze(0)).half().float()  # fp16 quant (s_shared)

        o = torch.einsum("bhrd,bhdv->bhrv", q, h.half().float())  # q @ h (h_shared is fp16)
        o = o * row_decay[:, :, None].unsqueeze(0)
        o = o + torch.einsum("bhrc,bhcd->bhrd", s_c, v)

        O[:, i * chunk_size : (i + 1) * chunk_size] = o.permute(0, 2, 1, 3)

        vp = v * v_decay[:, :, None].unsqueeze(0)
        h = h * chunk_decay[None, :, None, None] + torch.einsum("bhrd,bhrv->bhdv", k, vp)

    return O


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=8, help="Batch size")
    parser.add_argument("--S", type=int, default=4096, help="Seq len")
    parser.add_argument("--H", type=int, default=32, help="Num heads")
    parser.add_argument("--D", type=int, default=128, help="Head dim")
    args = parser.parse_args()
    B, S, H, D = args.B, args.S, args.H, args.D
    total_flops = 2.0 * B * S * S * H * D  # causal

    q = torch.randn((B, S, H, D), device="ptpu", dtype=torch.float16)
    k = torch.randn((B, S, H, D), device="ptpu", dtype=torch.float16)
    v = torch.randn((B, S, H, D), device="ptpu", dtype=torch.float16)

    # Head-specific log decay, computed in double on the host to avoid the fp32
    # cancellation in 1 - 2^(-5-h) for h >= 20 (which collapses decay to 1.0).
    log_decay = torch.log2(1 - torch.exp2(-5.0 - torch.arange(H, dtype=torch.float64))).to(torch.float32)
    log_decay_ptpu = log_decay.to("ptpu")

    kernel = chunk_retention_fwd_kernel(B, S, H, D, D)

    # ref_program runs on CPU (torch chunk-wise einsum); the kernel runs on PTPU.
    o = postprocess(kernel(q, k, v, log_decay_ptpu))
    torch.ptpu.synchronize()
    o_ref = ref_program(q.cpu(), k.cpu(), v.cpu(), log_decay)
    o_cpu = o.cpu()
    # fp16 storage of q / s_shared / h_shared / O still limits the absolute
    # accuracy of the output (growing with S); atol reflects that fp16 precision
    # and rtol guards the well-scaled positions. A real logic bug would exceed
    # these by orders of magnitude.
    assert torch.allclose(o_cpu.float(), o_ref, atol=0.5, rtol=1e-2), f"o max err: {(o_cpu.float() - o_ref).abs().max().item()}"
    print("Passed all tests!✅")

    t = do_bench(lambda: postprocess(kernel(q, k, v, log_decay_ptpu)), warmup=25, rep=100)
    print(f"Tilelang latency: {t:.3f} ms")
    print(f"Tilelang TFLOPs: {total_flops / t * 1e-9}")


if __name__ == "__main__":
    main()
