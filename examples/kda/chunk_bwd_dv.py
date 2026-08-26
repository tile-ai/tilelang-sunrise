import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune
from tilelang.utils.device import get_current_device
import sys  # noqa: F401

import torch
from test_utils_kda import build_kernel

DEVICE = get_current_device()

torch.random.manual_seed(1)


def prepare_input(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size,
    input_dtype,
    do_dtype,
):
    q = torch.randn(B, S, H, DK, dtype=do_dtype).to(DEVICE)
    k = torch.randn(B, S, H, DK, dtype=do_dtype).to(DEVICE)
    DO = torch.randn(B, S, H, DV, dtype=do_dtype).to(DEVICE)
    A = torch.randn(B, S, H, chunk_size, dtype=input_dtype).to(DEVICE)
    return q, k, DO, A


def prepare_output(
    B,
    S,
    H,
    DV,
    chunk_size,
    output_dtype,
):
    dv = torch.empty(B, S, H, DV, dtype=output_dtype).to(DEVICE)
    return dv


def get_configs():
    import itertools

    block_DV = [32, 64, 128]
    threads = [32, 64, 128]
    num_stages = [0, 1, 2, 3, 4]
    _configs = list(itertools.product(block_DV, threads, num_stages))
    configs = [{"block_DV": c[0], "threads": c[1], "num_stages": c[2]} for c in _configs]
    return configs


@autotune(configs=get_configs(), warmup=10, rep=5)
@tilelang.jit(out_idx=[-1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def tilelang_chunk_bwd_kernel_dv_local(
    B,
    S,
    H,
    DV,
    input_dtype,
    output_dtype,
    do_dtype,
    chunk_size,
    block_DV=128,
    threads=128,
    num_stages=1,
):
    block_S = BS = chunk_size
    DO_shape = (B, S, H, DV)
    A_shape = (B, S, H, BS)

    @T.prim_func
    def kernel(
        DO: T.Tensor(DO_shape, dtype=do_dtype),
        A: T.Tensor(A_shape, dtype=input_dtype),
        dv: T.Tensor(DO_shape, dtype=output_dtype),
    ):
        with T.Kernel(T.ceildiv(S, block_S), B * H, threads=threads) as (bs, bbh):
            bb, bh = bbh // H, bbh % H

            A_shared = T.alloc_shared((BS, BS), dtype=do_dtype)
            DO_shared = T.alloc_shared((BS, block_DV), dtype=do_dtype)
            dv_fragment = T.alloc_fragment((BS, block_DV), dtype=T.float32)
            dv_shared = T.alloc_shared((BS, block_DV), dtype=output_dtype)

            T.copy(A[bb, bs * BS : (bs + 1) * BS, bh, :], A_shared)
            for i_s1, i_s2 in T.Parallel(BS, BS):
                A_shared[i_s1, i_s2] = T.if_then_else(i_s1 >= i_s2, A_shared[i_s1, i_s2], 0.0)
            for i_v in T.Pipelined(T.ceildiv(DV, block_DV), num_stages=num_stages):
                T.copy(DO[bb, bs * BS : (bs + 1) * BS, bh, i_v * block_DV : (i_v + 1) * block_DV], DO_shared)
                T.gemm(A_shared, DO_shared, dv_fragment, transpose_A=True, clear_accum=True)  # transpose_A: A^T
                T.copy(dv_fragment, dv_shared)
                T.copy(dv_shared, dv[bb, bs * BS : (bs + 1) * BS, bh, i_v * block_DV : (i_v + 1) * block_DV])

    return kernel


def run_test(
    B,
    S,
    H,
    DK,
    DV,
    scale,
    input_dtype,
    do_dtype,
    output_dtype,
    chunk_size,
):
    _, _, DO, A = prepare_input(B, S, H, DK, DV, chunk_size, getattr(torch, input_dtype), getattr(torch, do_dtype))
    chunks = S // chunk_size
    do_ref = DO.cpu().float().reshape(B, chunks, chunk_size, H, DV).permute(0, 1, 3, 2, 4)
    a_ref = A.cpu().float().reshape(B, chunks, chunk_size, H, chunk_size).permute(0, 1, 3, 2, 4)
    a_ref *= torch.tril(torch.ones(chunk_size, chunk_size))
    dv_ref = torch.matmul(a_ref.transpose(-1, -2), do_ref)
    dv_ref = dv_ref.permute(0, 1, 3, 2, 4).reshape(B, S, H, DV).to(getattr(torch, output_dtype))

    dv_tilelang = prepare_output(B, S, H, DV, chunk_size, getattr(torch, output_dtype))
    kernel = build_kernel(
        tilelang_chunk_bwd_kernel_dv_local,
        DEVICE,
        B=B,
        S=S,
        H=H,
        DV=DV,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        do_dtype=do_dtype,
        chunk_size=chunk_size,
        block_DV=min(32, DV),
        threads=128,
        num_stages=0,
    )
    dv_tilelang = kernel(DO, A)
    try:
        torch.testing.assert_close(dv_tilelang.cpu(), dv_ref, rtol=1e-2, atol=1e-2)
        print("dv passed")
    except Exception as e:
        print(f"dv failed: {e}")


def main():
    run_test(
        B=1,
        S=64,
        H=1,
        DK=32,
        DV=32,
        scale=1.0,
        input_dtype="bfloat16",
        do_dtype="float32",
        output_dtype="bfloat16",
        chunk_size=64,
    )


if __name__ == "__main__":
    main()
