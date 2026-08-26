import tilelang
import tilelang.language as T
import torch
import pytest
import tilelang.testing
from tilelang.utils.device import get_current_device


@tilelang.jit
def tilelang_rand_1d(M=1024, seed=42, generator="curandStatePhilox4_32_10_t"):
    num_per_thread = 128
    threads = 1
    blk_M = num_per_thread * threads

    @T.prim_func
    def rand_kernel(
        A: T.Tensor((M,), "uint32"),
        B: T.Tensor((M,), "float32"),
        C: T.Tensor((M,), "float64"),
        D: T.Tensor((M,), "float32"),
        E: T.Tensor((M,), "float64"),
    ):
        with T.Kernel(T.ceildiv(M, threads * num_per_thread), threads=threads) as bx:
            tx = T.get_thread_binding()
            T.rng_init(seed, 0, bx * blk_M + tx * num_per_thread, generator=generator)
            for i, j in T.Parallel(threads, num_per_thread):
                offsets = (bx * threads + i) * num_per_thread
                idx = offsets + j
                if idx < M:
                    A[idx] = T.rng_rand()
            for i, j in T.Parallel(threads, num_per_thread):
                offsets = (bx * threads + i) * num_per_thread
                idx = offsets + j
                if idx < M:
                    B[idx] = T.rng_rand_float()
            for i, j in T.Parallel(threads, num_per_thread):
                offsets = (bx * threads + i) * num_per_thread
                idx = offsets + j
                if idx < M:
                    C[idx] = T.rng_rand_float(bit=64)
            for i, j in T.Parallel(threads, num_per_thread):
                offsets = (bx * threads + i) * num_per_thread
                idx = offsets + j
                if idx < M:
                    D[idx] = T.rng_rand_float(dist="normal")
            for i, j in T.Parallel(threads, num_per_thread):
                offsets = (bx * threads + i) * num_per_thread
                idx = offsets + j
                if idx < M:
                    E[idx] = T.rng_rand_float(bit=64, dist="normal")

    return rand_kernel


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "M, seed, generator", [(1024, 42, "curandStateMRG32k3a_t"), (512, 123, "curandStatePhilox4_32_10_t"), (128, 0, "curandStateXORWOW_t")]
)
def test_rand_1d(M, seed, generator):
    kernel = tilelang_rand_1d(M, seed, generator)
    A = torch.empty(M, dtype=torch.uint32, device="cuda")
    B = torch.empty(M, dtype=torch.float32, device="cuda")
    C = torch.empty(M, dtype=torch.float64, device="cuda")
    D = torch.empty(M, dtype=torch.float32, device="cuda")
    E = torch.empty(M, dtype=torch.float64, device="cuda")
    kernel(A, B, C, D, E)


@tilelang.jit
def tilelang_rand_blockwise(M=64, seed=42, generator="curandStatePhilox4_32_10_t"):
    threads = 32

    @T.prim_func
    def rand_kernel(A: T.Tensor((M,), "uint32")):
        with T.Kernel(M, threads=threads) as bx:
            tx = T.get_thread_binding()
            T.rng_init(seed, 0, bx, generator=generator)
            if tx == 0:
                A[bx] = T.rng_rand()

    return rand_kernel


@tilelang.jit
def tilelang_rand_guarded_cumsum(M=64, seed=42, generator="curandStatePhilox4_32_10_t"):
    threads = 32

    @T.prim_func
    def rand_kernel(
        A: T.Tensor((M,), "uint32"),
        n: T.int32,
    ):
        with T.Kernel(M, threads=threads) as bx:
            tx = T.get_thread_binding()
            s = T.alloc_shared((threads,), "int32")
            # rng_init inside a runtime guard, with a shared-memory cumsum
            # between init and use: sync legalization hoists __syncthreads()
            # out of the guard and splits it into sibling blocks, so the
            # curand state must be declared at function scope to stay visible.
            if bx < n:
                T.rng_init(seed, 0, bx, generator=generator)
                s[tx] = 1
                T.cumsum(s, dim=0)
                if tx == 0:
                    A[bx] = T.rng_rand() + T.cast(s[threads - 1], "uint32") * 0

    return rand_kernel


@pytest.mark.parametrize("generator", ["curandStateMRG32k3a_t", "curandStatePhilox4_32_10_t", "curandStateXORWOW_t"])
def test_rand_init_in_split_guard(generator):
    M, seed, n = 64, 42, 37
    guarded = tilelang_rand_guarded_cumsum(M, seed, generator)
    baseline = tilelang_rand_blockwise(M, seed, generator)

    sentinel = 0xDEADBEEF
    device = get_current_device()
    A = torch.full((M,), sentinel, dtype=torch.uint32).to(device)
    guarded(A, n)
    A_ref = torch.empty(M, dtype=torch.uint32, device=device)
    baseline(A_ref)

    # Compare on host: ptpu has no uint32 fill/equality dispatch path.
    if device.type == "ptpu":
        torch.ptpu.synchronize(device)
    A_cpu = A.cpu()
    A_ref_cpu = A_ref.cpu()
    assert torch.equal(A_cpu[:n], A_ref_cpu[:n]), "guarded rng output differs from unguarded baseline"
    assert (A_cpu[n:] == sentinel).all(), "rows outside the guard must stay untouched"


if __name__ == "__main__":
    tilelang.testing.main()
