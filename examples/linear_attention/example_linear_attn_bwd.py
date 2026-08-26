import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench
import argparse

try:
    from fla.ops.linear_attn import fused_chunk_linear_attn  # We compare with FLA

    _HAS_FLA = True
except ImportError:
    _HAS_FLA = False

from einops import rearrange
from typing import Optional, Tuple


def l2norm_fwd(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch L2 normalization along the last dimension (replaces FLA's l2norm_fwd)."""
    norm = (x * x).sum(dim=-1, keepdim=True).sqrt()
    return x / norm, norm


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def tl_fused_chunk_bwd_kernel(
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
    NK = tilelang.cdiv(DK, BK)
    NV = tilelang.cdiv(DV, BV)
    NT = tilelang.cdiv(S, chunk_size)

    @T.prim_func
    def fused_chunk_linear_attn_bwd(
        Q: T.Tensor([B, S, H, DK], dtype),  # type: ignore
        K: T.Tensor([B, S, H, DK], dtype),  # type: ignore
        V: T.Tensor([B, S, H, DV], dtype),  # type: ignore
        dO: T.Tensor([B, S, H, DV], dtype),  # type: ignore
        dQ: T.Tensor([B, S, H, DK], accum_dtype),  # type: ignore
        dK: T.Tensor([B, S, H, DK], accum_dtype),  # type: ignore
        dV: T.Tensor([B, S, H, DV], accum_dtype),  # type: ignore
    ):
        with T.Kernel(NV, NK, B * H) as (i_v, i_k, i_bh):
            i_b = i_bh // H
            i_h = i_bh % H

            ds = T.alloc_fragment([chunk_size, chunk_size], accum_dtype)
            ds_shared = T.alloc_shared([chunk_size, chunk_size], dtype)
            dq = T.alloc_fragment([chunk_size, BK], accum_dtype)
            dq_shared = T.alloc_shared([chunk_size, BK], accum_dtype)
            dk = T.alloc_fragment([chunk_size, BK], accum_dtype)
            dk_shared = T.alloc_shared([chunk_size, BK], accum_dtype)
            dv = T.alloc_fragment([chunk_size, BV], accum_dtype)
            dv_shared = T.alloc_shared([chunk_size, BV], accum_dtype)
            q = T.alloc_shared([chunk_size, BK], dtype)
            k = T.alloc_shared([chunk_size, BK], dtype)
            v = T.alloc_shared([chunk_size, BV], dtype)
            do = T.alloc_shared([chunk_size, BV], dtype)
            h = T.alloc_fragment([BV, BK], accum_dtype)
            h_shared = T.alloc_shared([BV, BK], dtype)
            dh = T.alloc_fragment([BK, BV], accum_dtype)
            dh_shared = T.alloc_shared([BK, BV], dtype)

            T.use_swizzle(10)

            T.clear(h)
            T.clear(dh)

            # Calculate dQ
            for i in T.Pipelined(0, NT):
                T.copy(K[i_b, i * chunk_size : (i + 1) * chunk_size, i_h, i_k * BK : (i_k + 1) * BK], k)
                T.copy(V[i_b, i * chunk_size : (i + 1) * chunk_size, i_h, i_v * BV : (i_v + 1) * BV], v)
                T.copy(dO[i_b, i * chunk_size : (i + 1) * chunk_size, i_h, i_v * BV : (i_v + 1) * BV], do)

                T.gemm(do, v, ds, transpose_B=True, clear_accum=True)
                for row, col in T.Parallel(chunk_size, chunk_size):
                    ds_shared[row, col] = T.if_then_else(row >= col, ds[row, col], 0)

                T.gemm(ds_shared, k, dq, clear_accum=True)
                T.copy(h, h_shared)
                T.gemm(do, h_shared, dq)
                T.gemm(v, k, h, transpose_A=True)
                for row, col in T.Parallel(chunk_size, BK):
                    dq[row, col] *= scale
                T.copy(dq, dq_shared)
                T.atomic_add(dQ[i_b, i * chunk_size : (i + 1) * chunk_size, i_h, i_k * BK : (i_k + 1) * BK], dq_shared)

            # Calculate dK, dV (reversely)
            for i in T.Pipelined(1, NT + 1):
                start = NT - i
                for row, col in T.Parallel(chunk_size, BK):
                    q[row, col] = Q[i_b, start * chunk_size + row, i_h, i_k * BK + col] * scale
                T.copy(K[i_b, start * chunk_size : (start + 1) * chunk_size, i_h, i_k * BK : (i_k + 1) * BK], k)
                T.copy(V[i_b, start * chunk_size : (start + 1) * chunk_size, i_h, i_v * BV : (i_v + 1) * BV], v)
                T.copy(dO[i_b, start * chunk_size : (start + 1) * chunk_size, i_h, i_v * BV : (i_v + 1) * BV], do)

                # Calculate dk
                T.gemm(v, do, ds, transpose_B=True, clear_accum=True)  # ds here actually means `s`, but we simply reuse the buffer `ds`
                for row, col in T.Parallel(chunk_size, chunk_size):
                    ds_shared[row, col] = T.if_then_else(row <= col, ds[row, col], 0)
                T.gemm(ds_shared, q, dk, clear_accum=True)
                T.copy(dh, dh_shared)
                T.gemm(v, dh_shared, dk, transpose_B=True)

                # Calculate dv
                T.gemm(k, q, ds, transpose_B=True, clear_accum=True)
                for row, col in T.Parallel(chunk_size, chunk_size):
                    ds_shared[row, col] = T.if_then_else(row <= col, ds[row, col], 0)
                T.gemm(ds_shared, do, dv, clear_accum=True)
                T.gemm(k, dh_shared, dv)

                # Update dh
                T.gemm(q, do, dh, transpose_A=True)

                T.copy(dk, dk_shared)
                T.atomic_add(dK[i_b, start * chunk_size : (start + 1) * chunk_size, i_h, i_k * BK : (i_k + 1) * BK], dk_shared)
                T.copy(dv, dv_shared)
                T.atomic_add(dV[i_b, start * chunk_size : (start + 1) * chunk_size, i_h, i_v * BV : (i_v + 1) * BV], dv_shared)

    return fused_chunk_linear_attn_bwd


def tl_fused_chunk_bwd(Q, K, V, dO):
    B, S, H, D = Q.shape
    kernel = tl_fused_chunk_bwd_kernel(B, S, H, D, D)
    dQ = torch.zeros_like(Q, dtype=torch.float32)
    dK = torch.zeros_like(K, dtype=torch.float32)
    dV = torch.zeros_like(V, dtype=torch.float32)
    kernel(Q, K, V, dO, dQ, dK, dV)
    return dQ.to(torch.float16), dK.to(torch.float16), dV.to(torch.float16)


def ref_bwd_program(q, k, v, do, scale=None):
    """Compute backward reference on CPU. Inputs are already float32 with requires_grad."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    chunk_size = 64
    # Work with scaled q, keep original q/k/v for autograd
    q_scaled = q * scale
    q_r = rearrange(q_scaled, "b (n c) h d -> b h n c d", c=chunk_size)
    k_r = rearrange(k, "b (n c) h d -> b h n c d", c=chunk_size)
    v_r = rearrange(v, "b (n c) h d -> b h n c d", c=chunk_size)
    kv = k_r.transpose(-1, -2) @ v_r
    kv = kv.cumsum(2)
    kv = torch.cat([torch.zeros_like(kv[:, :, :1]), kv[:, :, :-1]], dim=2)
    inter = q_r @ kv
    intra = (
        (q_r @ k_r.transpose(-1, -2)).masked_fill(
            torch.triu(torch.ones(chunk_size, chunk_size, dtype=bool, device=q.device), diagonal=1), 0
        )
    ) @ v_r
    o = inter + intra
    o = rearrange(o, "b h n c d -> b (n c) h d")
    # Use autograd on CPU to get reference gradients
    o.backward(do, retain_graph=True)
    return q.grad, k.grad, v.grad


def ref_program(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    q, k, v = q.float(), k.float(), v.float()
    if scale is None:
        scale = q.shape[-1] ** -0.5
    chunk_size = 64
    q = rearrange(q, "b (n c) h d -> b h n c d", c=chunk_size) * scale
    k = rearrange(k, "b (n c) h d -> b h n c d", c=chunk_size)
    v = rearrange(v, "b (n c) h d -> b h n c d", c=chunk_size)
    kv = k.transpose(-1, -2) @ v
    kv = kv.cumsum(2)
    h = kv[:, :, -1, :, :]
    kv = torch.cat([torch.zeros_like(kv[:, :, :1]), kv[:, :, :-1]], dim=2)
    inter = q @ kv
    intra = (
        (q @ k.transpose(-1, -2)).masked_fill_(torch.triu(torch.ones(chunk_size, chunk_size, dtype=bool, device=q.device), diagonal=1), 0)
    ) @ v
    o = inter + intra
    return rearrange(o, "b h n c d -> b (n c) h d"), h


def main(B=1, S=1024, H=16, D=128):
    q = torch.randn((B, S, H, D), device="ptpu", dtype=torch.float16, requires_grad=True)
    k = torch.randn((B, S, H, D), device="ptpu", dtype=torch.float16, requires_grad=True)
    v = torch.randn((B, S, H, D), device="ptpu", dtype=torch.float16, requires_grad=True)
    do = torch.randn((B, S, H, D), device="ptpu", dtype=torch.float16)

    # qk norm is necessary for linear attn
    q = l2norm_fwd(q)[0].requires_grad_(True)
    k = l2norm_fwd(k)[0].requires_grad_(True)

    dq, dk, dv = tl_fused_chunk_bwd(q, k, v, do)

    # ref_program runs on CPU (torch.triu/cumsum/backward not fully supported on ptpu)
    torch.ptpu.synchronize()
    q_cpu = q.detach().cpu().requires_grad_(True)
    k_cpu = k.detach().cpu().requires_grad_(True)
    v_cpu = v.detach().cpu().requires_grad_(True)
    do_cpu = do.cpu()
    o_ref, _ = ref_program(q_cpu, k_cpu, v_cpu)
    o_ref.backward(do_cpu, retain_graph=True)

    dq_cpu, dk_cpu, dv_cpu = dq.cpu(), dk.cpu(), dv.cpu()
    assert torch.allclose(dq_cpu, q_cpu.grad, atol=1e-2, rtol=1e-2), f"dq max err: {(dq_cpu - q_cpu.grad).abs().max()}"
    assert torch.allclose(dk_cpu, k_cpu.grad, atol=1e-2, rtol=1e-2), f"dk max err: {(dk_cpu - k_cpu.grad).abs().max()}"
    assert torch.allclose(dv_cpu, v_cpu.grad, atol=1e-2, rtol=1e-2), f"dv max err: {(dv_cpu - v_cpu.grad).abs().max()}"
    print("Passed all tests!✅")

    # Benchmark
    t2 = do_bench(lambda: tl_fused_chunk_bwd(q, k, v, do), backend="event")
    if _HAS_FLA:
        q.grad = k.grad = v.grad = None
        o_ref, _ = fused_chunk_linear_attn(q, k, v, output_final_state=True, normalize=False)
        t1 = do_bench(lambda: o_ref.backward(do, retain_graph=True), backend="event")
        print(f"Triton latency: {t1:.3f} ms")
        print(f"TileLang latency: {t2:.3f} ms")
        print(f"Speedup: {t1 / t2:.3f}x")
    else:
        print(f"TileLang latency: {t2:.3f} ms")
        print("(FLA/Triton not available for comparison)")


def run_regression_perf(B=1, S=1024, H=16, D=128):
    q = torch.randn((B, S, H, D), device="ptpu", dtype=torch.float16, requires_grad=True)
    k = torch.randn((B, S, H, D), device="ptpu", dtype=torch.float16, requires_grad=True)
    v = torch.randn((B, S, H, D), device="ptpu", dtype=torch.float16, requires_grad=True)
    do = torch.randn((B, S, H, D), device="ptpu", dtype=torch.float16)
    q = l2norm_fwd(q)[0].requires_grad_(True)
    k = l2norm_fwd(k)[0].requires_grad_(True)
    kernel = tl_fused_chunk_bwd_kernel(B, S, H, D, D)
    dQ = torch.zeros_like(q, dtype=torch.float32)
    dK = torch.zeros_like(k, dtype=torch.float32)
    dV = torch.zeros_like(v, dtype=torch.float32)
    kernel(q, k, v, do, dQ, dK, dV)
    return do_bench(lambda: kernel(q, k, v, do, dQ, dK, dV), backend="event")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=8, help="Batch size")
    parser.add_argument("--S", type=int, default=1024, help="Seq len")
    parser.add_argument("--H", type=int, default=32, help="Num heads")
    parser.add_argument("--D", type=int, default=128, help="Head dim")
    args = parser.parse_args()

    main(args.B, args.S, args.H, args.D)
