# type: ignore
import tilelang
import tilelang.testing
import tilelang.language as T
import pytest
from tilelang.utils.device import get_current_device


@pytest.fixture(
    params=[
        pytest.param("cuda", marks=tilelang.testing.requires_cuda.marks()),
        pytest.param("tang", marks=pytest.mark.skipif(not tilelang.testing.is_tang_target(), reason="Requires TANG target")),
    ]
)
def execution_target(request):
    return request.param


def test_device_assert_no_trigger():
    @T.prim_func
    def program():
        with T.Kernel(threads=128):
            tid = T.get_thread_binding()
            T.device_assert(tid == tid)

    jit_kernel = tilelang.compile(program)
    profiler = jit_kernel.get_profiler()
    profiler.run_once()


def test_kernel_body_assert_compiles_and_runs(execution_target):
    """A plain ``assert`` in a kernel body must lower to a device-legal check.

    The TVMScript parser turns it into a ``tirx.AssertStmt``. The CUDA codegen used
    to fall through to the inherited host visitor for that node, which streams a
    ``TVMFFIErrorSetRaisedFromCStrParts`` call, a host-only symbol, plus
    ``return -1`` into the ``__global__`` kernel, so the kernel failed to build.
    """
    import torch

    @T.prim_func
    def program(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=128):
            tid = T.get_thread_binding()
            assert tid < 128, "oob thread"
            B[tid] = A[tid] + T.float32(1)

    target = "cuda" if execution_target == "cuda" else tilelang.env.get_default_target()
    device = "cuda" if execution_target == "cuda" else get_current_device()
    jit_kernel = tilelang.compile(program, out_idx=[1], target=target)
    a = torch.arange(128, dtype=torch.float32, device=device)
    b = jit_kernel(a)
    assert torch.equal(b.cpu(), a.cpu() + 1)


def test_kernel_body_assert_without_message_compiles(execution_target):
    """The message-less form takes the same ``AssertStmt`` path.

    The parser attaches a default message part, so it does not reach the
    message-less branch of the base visitor either.
    """
    import torch

    @T.prim_func
    def program(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=128):
            tid = T.get_thread_binding()
            assert tid < 128
            B[tid] = A[tid] + T.float32(2)

    target = "cuda" if execution_target == "cuda" else tilelang.env.get_default_target()
    device = "cuda" if execution_target == "cuda" else get_current_device()
    jit_kernel = tilelang.compile(program, out_idx=[1], target=target)
    a = torch.arange(128, dtype=torch.float32, device=device)
    b = jit_kernel(a)
    assert torch.equal(b.cpu(), a.cpu() + 2)


if __name__ == "__main__":
    tilelang.testing.main()
