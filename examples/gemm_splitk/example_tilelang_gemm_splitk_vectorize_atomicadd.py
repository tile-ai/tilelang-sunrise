import tilelang
import tilelang.language as T


@tilelang.jit
def matmul(A, B, C, block_M, block_N, block_K, split_k, dtype=T.float16, accum_dtype=T.float32, out_dtype=T.float32):
    M, N, K = T.const("M, N, K")
    splitK = K // split_k

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C: T.Tensor((M, N), out_dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), split_k, threads=128) as (bx, by, bz):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(splitK, block_K), num_stages=0):
            T.copy(A[by * block_M, bz * splitK + ko * block_K], A_shared)
            T.copy(B[bz * splitK + ko * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.atomic_add(C[by * block_M, bx * block_N], C_local)


def main():
    M = 1024
    N = 1024
    K = 1024
    block_M = 128
    block_N = 128
    block_K = 32
    split_k = 4

    import torch

    device = "ptpu" if hasattr(torch, "ptpu") and torch.ptpu.is_available() else "cuda"

    torch.random.manual_seed(42)
    a = torch.randn(M, K).to(device).half()
    b = torch.randn(K, N).to(device).half()
    c = torch.zeros(M, N).to(device).float()
    matmul(a, b, c, block_M, block_N, block_K, split_k)

    ref_c = a.cpu().float() @ b.cpu().float()

    torch.testing.assert_close(c.cpu(), ref_c.cpu().to(c.dtype), rtol=1e-2, atol=1e-2)


def run_regression_perf():
    M = 4096
    N = 4096
    K = 4096
    block_M = 128
    block_N = 128
    block_K = 32
    split_k = 4

    import torch

    device = "ptpu" if hasattr(torch, "ptpu") and torch.ptpu.is_available() else "cuda"

    torch.random.manual_seed(42)
    a = torch.randn(M, K).to(device).half()
    b = torch.randn(K, N).to(device).half()
    c = torch.zeros(M, N).to(device).float()
    from tilelang.profiler import do_bench

    return do_bench(lambda: matmul(a, b, c, block_M, block_N, block_K, split_k), warmup=5, rep=20, backend="event", return_mode="median")


if __name__ == "__main__":
    main()
