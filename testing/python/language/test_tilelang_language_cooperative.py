import tilelang
import tilelang.language as T
import torch
import tilelang.testing
import pytest


@tilelang.jit
def grid_sync(N=1024):
    block = 64

    @T.prim_func
    def kernel(A: T.Tensor((N), T.float32)):
        with T.Kernel(T.ceildiv(N, block), threads=128) as bx:
            A_local = T.alloc_fragment((block), dtype=T.float32)
            n_idx = bx * block
            for i in T.Parallel(block):
                A[n_idx + i] = n_idx + i
            T.sync_grid()
            for i in T.Parallel(block):
                A_local[i] = A[N - n_idx - i - 1]
                T.sync_grid()
                A[n_idx + i] = A[n_idx + i] + A_local[i]

    return kernel


@tilelang.jit
def grid_sync_single(N=1024):
    """Kernel with exactly one grid sync (an odd sync count).

    Stage 1 writes A[i] = i; the single grid sync makes every block's stage-1
    write visible grid-wide; stage 2 reads A reversed across blocks into B, so
    B[i] = A[N-1-i] = N-1-i. Reading A / writing B avoids any in-array race, so
    the result is fully deterministic and depends *only* on the grid sync being
    correct. After each launch the global `bar0` barrier ends in its flipped
    (non-zero) state, which is exactly the residual scenario we want to exercise
    across relaunches.
    """
    block = 64

    @T.prim_func
    def kernel(A: T.Tensor((N), T.float32), B: T.Tensor((N), T.float32)):
        with T.Kernel(T.ceildiv(N, block), threads=128) as bx:
            n_idx = bx * block
            for i in T.Parallel(block):
                A[n_idx + i] = n_idx + i
            T.sync_grid()
            for i in T.Parallel(block):
                B[n_idx + i] = A[N - 1 - n_idx - i]

    return kernel


@tilelang.jit
def grid_sync_oversized(N, block=1):
    """Grid-sync kernel intentionally configured to launch far more blocks than
    a device can co-schedule (``block=1`` makes the grid ~N blocks).

    A cooperative launch requires every block to be simultaneously resident, so
    an oversized grid must make the runtime's ``taLaunchCooperativeKernel`` fail
    (e.g. "too large") rather than hang or silently run.
    """

    @T.prim_func
    def kernel(A: T.Tensor((N), T.float32)):
        with T.Kernel(T.ceildiv(N, block), threads=128) as bx:
            n_idx = bx * block
            for i in T.Parallel(block):
                A[n_idx + i] = n_idx + i
            T.sync_grid()

    return kernel


# Only verifies compilation and generated source
def test_grid_sync_compile():
    N = 1024
    kernel = grid_sync(N)
    kernel.show_source()
    print("=" * 60)
    print("HOST SOURCE CODE:")
    print("=" * 60)
    print(kernel.get_host_source())
    print("=" * 60)
    # Verify the generated kernel source contains the grid sync call
    source = kernel.get_kernel_source()
    assert "#include <cooperative_groups/details/sync.h>" in source
    assert "using cooperative_groups::details::sync_grids" in source
    assert "volatile cooperative_groups::details::barrier_t bar0 = 0" in source


def test_grid_sync_runtime():
    N = 1024
    kernel = grid_sync(N)
    kernel.show_source()
    print(kernel.get_host_source())
    tensor = torch.rand((N), dtype=torch.float32)
    tensor_ptpu = tensor.ptpu()
    kernel(tensor_ptpu)
    torch.ptpu.synchronize()
    # After kernel: A[i] = i + A[N-1-i] = i + (N-1-i) = N-1 for all i
    target = torch.full((N,), N - 1, dtype=torch.float32)
    torch.testing.assert_close(tensor_ptpu.cpu(), target)
    print("PASSED.")


def test_grid_sync_relaunch_residual():
    """Regression: relaunch the SAME cached kernel many times.

    Each launch performs an odd number of grid syncs, so the module-global
    ``bar0`` barrier ends every run at a non-zero (flipped) value. The compiled
    module is cached, so its ``__device__ bar0`` static initializer runs only
    once at load time and is NOT re-run per launch. Without a per-launch reset
    of ``bar0`` (the runtime issues ``taMemsetAsync`` on the kernel's stream
    right before ``taLaunchCooperativeKernel``), a stale/dirty barrier state
    could desync or deadlock subsequent launches.

    This test asserts every relaunch of the cached kernel still produces correct
    results and completes (a true barrier deadlock would hang here and be caught
    by the CI per-case timeout).
    """
    N = 1024

    # Build once so all iterations share the same cached module (hence the same
    # global ``bar0``); this is what makes the residual scenario reproducible.
    kernel = grid_sync_single(N)

    # B[i] = A[N-1-i] = N-1-i  ->  [N-1, N-2, ..., 1, 0]
    expected = torch.arange(N - 1, -1, -1, dtype=torch.float32)

    num_iters = 32
    for it in range(num_iters):
        A = torch.rand((N), dtype=torch.float32).ptpu()
        # Poison the output so we can tell the kernel actually wrote it.
        B = torch.full((N,), -1.0, dtype=torch.float32).ptpu()
        kernel(A, B)
        torch.ptpu.synchronize()
        torch.testing.assert_close(B.cpu(), expected, msg=lambda m, it=it: f"mismatch at relaunch iteration {it}: {m}")
    print(f"PASSED {num_iters} relaunches.")


@pytest.mark.skip(reason="Oversized cooperative launch hangs on TANG instead of returning error")
def test_grid_sync_too_many_blocks_errors():
    """A cooperative launch must fit all blocks co-resident on the device.

    Requesting far more blocks than the device can co-schedule makes the
    runtime's ``taLaunchCooperativeKernel`` return an error, which the runtime
    turns into a fatal/exception that crosses the FFI boundary. This test
    asserts the error propagates to Python (the launch must NOT silently succeed
    or hang).
    """
    # block=1 -> grid ~= N blocks; ~4M blocks is far beyond any device's
    # co-resident capacity, while the tensor stays small (~16 MB).
    N = 1 << 22
    kernel = grid_sync_oversized(N, block=1)
    # Force compilation outside the raises block so the assertion targets the
    # launch failure, not any (unexpected) compile-time error.
    kernel.get_kernel_source()

    A = torch.zeros((N), dtype=torch.float32).ptpu()
    with pytest.raises(Exception) as exc_info:
        kernel(A)
        torch.ptpu.synchronize()
    print(f"PASSED: oversized cooperative launch raised an error: {type(exc_info.value).__name__}: {exc_info.value}")


if __name__ == "__main__":
    tilelang.testing.main()
