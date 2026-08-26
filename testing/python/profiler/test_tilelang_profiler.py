import tilelang
import tilelang.language as T
import pytest
import tilelang.testing
from tilelang.utils.device import get_current_device


@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


@pytest.mark.perf
@tilelang.testing.requires_cuda
def test_profiler():
    kernel = matmul(1024, 1024, 1024, 128, 128, 32)

    import torch

    a = torch.randn(1024, 1024).cuda().half()
    b = torch.randn(1024, 1024).cuda().half()

    c = kernel(a, b)
    ref_c = a @ b
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    # benchmark
    profiler = kernel.get_profiler()

    # use cupti backend
    cupti_latency = profiler.do_bench(backend="cupti")

    # use event backend
    event_latency = profiler.do_bench(backend="event")
    print(f"cupti Latency: {cupti_latency}ms")
    print(f"event Latency: {event_latency}ms")


def test_stcu_profiler():
    device = get_current_device()
    if device.type != "ptpu":
        pytest.skip("PTPU event profiler test requires a PTPU device")

    kernel = matmul(1024, 1024, 1024, 128, 128, 32)

    import torch

    a = torch.randn(1024, 1024, device=device, dtype=torch.float16)
    b = torch.randn(1024, 1024, device=device, dtype=torch.float16)

    c = kernel(a, b)
    torch.ptpu.synchronize()
    ref_c = a.cpu().float() @ b.cpu().float()
    torch.testing.assert_close(c.cpu().float(), ref_c, rtol=1e-2, atol=1e-2)

    profiler = kernel.get_profiler()
    event_latency = profiler.do_bench(backend="event")
    assert event_latency is not None
    print(f"event Latency: {event_latency}ms")


if __name__ == "__main__":
    tilelang.testing.main()
