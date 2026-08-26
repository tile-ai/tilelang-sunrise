import tilelang
import tilelang.language as T
import torch
import triton
import triton.language as tl
from tilelang.utils.device import get_current_device


@tilelang.jit
def tilelang_rand_1d(M=1024, seed=42):
    num_per_thread = 128
    threads = 1
    blk_M = num_per_thread * threads

    A = T.empty((M,), "uint32")

    with T.Kernel(T.ceildiv(M, threads * num_per_thread), threads=threads) as bx:
        tx = T.get_thread_binding()
        T.rng_init(seed, 0, bx * blk_M + tx * num_per_thread)
        for i, j in T.Parallel(threads, num_per_thread):
            offsets = (bx * threads + i) * num_per_thread
            idx = offsets + j
            if idx < M:
                A[idx] = T.rng_rand()

    return A


@triton.jit
def triton_rand_1d(X, M, elements_per_thread, seed):
    pid = tl.program_id(0)
    offset = pid * elements_per_thread + tl.arange(0, elements_per_thread)

    r0, r1, r2, r3 = tl.randint4x(seed, offset)

    base_idx = offset * 4
    tl.store(X + base_idx, r0, mask=base_idx < M)
    tl.store(X + base_idx + 1, r1, mask=(base_idx + 1) < M)
    tl.store(X + base_idx + 2, r2, mask=(base_idx + 2) < M)
    tl.store(X + base_idx + 3, r3, mask=(base_idx + 3) < M)


def run_tilelang_rand_1d(M, seed):
    """Run one RNG launch and freeze its completed result on the host."""
    result = tilelang_rand_1d(M, seed)
    device = get_current_device()
    if device.type == "ptpu":
        torch.ptpu.synchronize(device)
    return result.cpu().clone()


def test_rand_1d(M, seed):
    device = get_current_device()
    if device.type == "cuda":
        # CUDA: keep the precise Triton oracle.
        tilelang_result = tilelang_rand_1d(M, seed)
        triton_result = torch.empty(M, dtype=torch.uint32, device="cuda")
        grid = (triton.cdiv(M, 128),)
        triton_rand_1d[grid](triton_result, tl.constexpr(M), tl.constexpr(128 // 4), seed)

        torch.testing.assert_close(tilelang_result, triton_result)
    else:
        # TANG: no Triton oracle is available, so validate the RNG
        # contract directly: the same seed must reproduce, a different
        # seed must diverge, and the shape/dtype must match.
        tilelang_result = run_tilelang_rand_1d(M, seed)
        again = run_tilelang_rand_1d(M, seed)
        assert torch.equal(tilelang_result, again), "same seed should reproduce RNG output"

        assert tilelang_result.shape == (M,)
        assert tilelang_result.dtype == torch.uint32

        other_seed = seed + 1 if seed != 1 else 0
        other = run_tilelang_rand_1d(M, other_seed)
        assert not torch.equal(tilelang_result, other), "different seeds should produce different RNG output"


if __name__ == "__main__":
    test_rand_1d(1024, 42)
    test_rand_1d(512, 123)
    test_rand_1d(128, 0)
