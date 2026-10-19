import torch
import tilelang.testing
import tilelang.language as T

from tilelang.backend.target import determine_target
from tilelang.intrinsics import make_mma_swizzle_layout
from tilelang.jit import JITKernel
from tilelang.utils.device import get_current_device
import pytest

_FLOAT8_E8M0FNU = getattr(torch, "float8_e8m0fnu", None)

STCUV2 = {"kind": "tang", "arch": "stcuv2"}


def _target_is_tang():
    target = determine_target(tilelang.env.get_default_target(), return_object=True)
    return target.kind.name == "tang"


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True})
def vectorize_test(N, M, stride_A, stride_B):
    @T.prim_func
    def main(
        A: T.StridedTensor[(N, M), (1, stride_A), T.float32],  # noqa: F821
        B: T.StridedTensor[(N, M), (1, stride_B), T.float32],  # noqa: F821
    ):
        with T.Kernel(M // 128, threads=128) as (bx):
            tx = T.get_thread_binding(0)
            col = bx * 128 + tx

            for row in T.vectorized(N):
                B[row, col] = A[row, col]

    return main


def run_vectorize(N, M, stride_A, stride_B):
    assert N % 128 == 0 and M % 128 == 0
    assert stride_A >= N and stride_B >= N

    jit_kernel = vectorize_test(N, M, stride_A, stride_B)

    device = get_current_device()
    base_a = torch.randn(stride_A, M, device=device, dtype=torch.float32)
    base_b = torch.zeros(stride_B, M, device=device, dtype=torch.float32)
    a = torch.as_strided(base_a, size=(N, M), stride=(1, stride_A))
    b = torch.as_strided(base_b, size=(N, M), stride=(1, stride_B))

    jit_kernel(a, b)

    if device.type == "ptpu":
        torch.ptpu.synchronize()
    torch.testing.assert_close(a.cpu(), b.cpu(), atol=1e-8, rtol=1e-8)

    code = jit_kernel.get_kernel_source()

    vectorize_size = 1
    while vectorize_size <= 2 and stride_A % (vectorize_size * 2) == 0 and stride_B % (vectorize_size * 2) == 0:
        vectorize_size *= 2

    if vectorize_size == 4:
        if device.type == "ptpu":
            assert "float4" not in code
        else:
            assert "float4" in code
    elif vectorize_size == 2:
        if device.type == "ptpu":
            assert "float2" not in code
        else:
            assert "float2" in code


def test_vectorize():
    N, M = 128, 128

    run_vectorize(N, M, N, N)
    run_vectorize(N, M, N + 2, N + 4)
    run_vectorize(N, M, N + 4, N + 8)
    run_vectorize(N, M, N + 8, N + 16)


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True})
def vectorize_test_invariant_index(N, M, K):
    @T.prim_func
    def main(
        A: T.Tensor[(N, M), T.float32],  # noqa: F821
        B: T.Tensor[(N, M), T.float32],  # noqa: F821
        C: T.Tensor[(N, M // K), T.float32],  # noqa: F821
    ):
        with T.Kernel(N // 128, threads=128) as (bx):
            tx = T.get_thread_binding(0)
            row = bx * 128 + tx

            for col in T.vectorized(M):
                B[row, col] = A[row, col] * C[row, col // K]

    return main


def run_vectorize_invariant_index(N, M, K):
    assert N % 128 == 0 and M % K == 0

    jit_kernel = vectorize_test_invariant_index(N, M, K)

    device = get_current_device()
    a = torch.randn(N, M, device=device, dtype=torch.float32)
    b = torch.zeros(N, M, device=device, dtype=torch.float32)
    c = torch.randn(N, M // K, device=device, dtype=torch.float32)

    jit_kernel(a, b, c)

    indices = torch.arange(a.size(1)) // K
    ret = a * c[:, indices]
    if device.type == "ptpu":
        torch.ptpu.synchronize()
    torch.testing.assert_close(b.cpu(), ret.cpu(), atol=1e-8, rtol=1e-8)

    code = jit_kernel.get_kernel_source()

    vectorize_size = 1
    while vectorize_size <= 2 and K % (vectorize_size * 2) == 0:
        vectorize_size *= 2

    if vectorize_size == 4:
        if device.type == "ptpu":
            assert "float4" not in code
        else:
            assert "float4" in code
    elif vectorize_size == 2:
        if device.type == "ptpu":
            assert "float2" not in code
        else:
            assert "float2" in code


def test_vectorize_invariant_index():
    N, M = 128, 128

    run_vectorize_invariant_index(N, M, 2)
    run_vectorize_invariant_index(N, M, 4)
    run_vectorize_invariant_index(N, M * 3, 6)
    run_vectorize_invariant_index(N, M * 7, 14)


@tilelang.jit
def vectorize_global_invariant_store_accumulate(M, K):
    @T.prim_func
    def main(
        A: T.Tensor[(M, K), T.float32],  # noqa: F821
        B: T.Tensor[(M,), T.float32],  # noqa: F821
    ):
        with T.Kernel(M // 128, threads=128) as bx:
            row = bx * 128 + T.get_thread_binding(0)
            B[row] = 0.0
            for k in T.vectorized(K):
                B[row] = B[row] + A[row, k]

    return main


@tilelang.jit
def vectorize_local_invariant_store_accumulate(M, K):
    @T.prim_func
    def main(
        A: T.Tensor[(M, K), T.float32],  # noqa: F821
        B: T.Tensor[(M,), T.float32],  # noqa: F821
    ):
        with T.Kernel(M // 128, threads=128) as bx:
            row = bx * 128 + T.get_thread_binding(0)
            acc = T.alloc_local((1,), T.float32)
            acc[0] = 0.0
            for k in T.vectorized(K):
                acc[0] = acc[0] + A[row, k]
            B[row] = acc[0]

    return main


def run_vectorize_invariant_store_accumulate(kernel_factory):
    M, K = 128, 4
    kernel = kernel_factory(M, K)
    a = torch.ones((M, K), device=get_current_device(), dtype=torch.float32)
    b = torch.empty((M,), device=get_current_device(), dtype=torch.float32)

    kernel(a, b)

    torch.testing.assert_close(b.cpu(), torch.full((M,), float(K), dtype=torch.float32), rtol=0, atol=0)


def test_vectorize_global_invariant_store_accumulate():
    run_vectorize_invariant_store_accumulate(vectorize_global_invariant_store_accumulate)


def test_vectorize_local_invariant_store_accumulate():
    run_vectorize_invariant_store_accumulate(vectorize_local_invariant_store_accumulate)


def vectorize_all_dtypes_kernel(dtype, vec_num):
    @T.prim_func
    def main(A: T.Tensor[(64,), dtype]):
        with T.Kernel(1, threads=256):
            for i in T.vectorized(vec_num):
                A[i] = T.cast(i + 1, dtype)

    return main


vectorize_test_all_dtypes = tilelang.jit(vectorize_all_dtypes_kernel)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.uint8,
        torch.uint64,
        torch.int8,
        torch.int64,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        pytest.param(
            _FLOAT8_E8M0FNU,
            marks=pytest.mark.skipif(_FLOAT8_E8M0FNU is None, reason="torch lacks float8_e8m0fnu"),
            id="float8_e8m0fnu",
        ),
    ],
)
@pytest.mark.parametrize("vec_num", [1, 2, 4, 8])
def test_vectorize_all_dtypes(dtype, vec_num):
    device = get_current_device()
    if device.type == "ptpu" and (dtype not in (torch.uint8, torch.int8, torch.int64) or vec_num > 2):
        pytest.skip("S2 supports this regression for uint8/int8/int64 with vector width 1 or 2")
    x = torch.empty((64,), dtype=dtype, device=device)
    kernel = vectorize_test_all_dtypes(dtype, vec_num)
    kernel(x)


def _stcuv2_vectorize_gap(dtype, vec_num):
    """The reason ``dtype`` x ``vec_num`` cannot work on stcuv2 yet, else None."""
    if dtype in (torch.int64, torch.uint64) and vec_num > 4:
        # 8 x 64bit = 512bit exceeds the widest vector the codegen will ramp.
        return "64-bit dtypes cap at 4 lanes (Translate Ramp Node ... lanes <= 4)"
    if dtype is torch.uint64:
        return "simulator HBM decoder has no torch.uint64 case yet"
    if dtype is _FLOAT8_E8M0FNU:
        return "float8_e8m0fnu has no TANG codegen type (codegen_tang.cc GetFP8Type)"
    return None


@pytest.mark.parametrize(
    "dtype",
    [
        torch.uint8,
        torch.uint64,
        torch.int8,
        torch.int64,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        pytest.param(
            _FLOAT8_E8M0FNU,
            marks=pytest.mark.skipif(_FLOAT8_E8M0FNU is None, reason="torch lacks float8_e8m0fnu"),
            id="float8_e8m0fnu",
        ),
    ],
)
@pytest.mark.parametrize("vec_num", [1, 2, 4, 8])
@tilelang.testing.requires_stcuv2
def test_vectorize_all_dtypes_stcuv2(dtype, vec_num, request):
    """Same regression on S3/stcuv2 via the ISS simulator.

    S2 only reaches uint8/int8/int64 at vector width 1-2 (see the skip in
    ``test_vectorize_all_dtypes``); stcuv2 covers those plus widths 4/8 and both
    fp8 formats. The remaining holes are marked non-strict xfail so a fix shows
    up as XPASS instead of silently staying uncovered.
    """
    gap = _stcuv2_vectorize_gap(dtype, vec_num)
    if gap is not None:
        request.applymarker(pytest.mark.xfail(reason=gap, strict=False))

    # out_idx=[0] lets the simulator adapter allocate the buffer itself; it
    # requires ptpu tensors, so a hand-built input would have to be moved there
    # anyway (see testing/python/s3/test_simulator_e2e.py).
    kernel = JITKernel(
        vectorize_all_dtypes_kernel(dtype, vec_num),
        out_idx=[0],
        target=STCUV2,
        execution_backend="simulator",
    )
    out = kernel().cpu()

    # The kernel writes 1..vec_num into the first vec_num slots, so this also
    # pins down the float->dtype conversion rather than only that it compiled.
    expected = torch.arange(1, vec_num + 1, dtype=torch.float32).to(dtype)
    torch.testing.assert_close(out[:vec_num].float(), expected.float(), atol=0, rtol=0)


@tilelang.jit
def vectorize_broadcast_int8(vec_num):
    with T.Kernel(1, threads=128):
        a = T.alloc_local((64,), "int8")
        b = T.alloc_var("int8")

        for i in T.vectorized(vec_num):
            a[i] = b


@pytest.mark.parametrize("vec_num", [4, 32])
def test_vectorize_broadcast_int8(vec_num):
    """Test broadcasting a non-constant int8 value to a vectorized store."""
    vectorize_broadcast_int8.compile(vec_num=vec_num)


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
@pytest.mark.parametrize("vec_num", [4, 8, 16, 32])
def test_tang_register_byte_broadcast(dtype, vec_num):
    device = get_current_device()
    if device.type != "ptpu":
        pytest.skip("TANG packed register broadcast")

    @T.prim_func
    def kernel(A: T.Tensor((1,), dtype), B: T.Tensor((vec_num,), dtype)):
        with T.Kernel(1, threads=1):
            values = T.alloc_local((vec_num,), dtype)
            value = A[0]
            for i in T.vectorized(vec_num):
                values[i] = value
            T.copy(values, B)

    compiled = tilelang.compile(kernel)
    if vec_num >= 16:
        constructor = "make_uint4(" if dtype == "uint8" else "make_int4("
        assert constructor in compiled.get_kernel_source()
    torch_dtype = getattr(torch, dtype)
    output = torch.empty((vec_num,), dtype=torch_dtype, device=device)
    samples = [0, 3, 128, 255] if dtype == "uint8" else [-128, -3, 0, 127]
    for value in samples:
        scalar = torch.tensor([value], dtype=torch_dtype, device=device)
        compiled(scalar, output)
        torch.ptpu.synchronize()
        torch.testing.assert_close(output.cpu(), torch.full((vec_num,), value, dtype=torch_dtype), rtol=0, atol=0)


def vectorize_broadcast_int4(dtype, vec_num):
    @tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_BUFFER_INIT_CHECK: True})
    def kernel():
        with T.Kernel(1, threads=128):
            a = T.alloc_local((64,), dtype)
            b = T.alloc_var(dtype)

            for i in T.vectorized(vec_num):
                a[i] = b

    return kernel


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["int4", "uint4"])
@pytest.mark.parametrize("vec_num", [4, 8, 16, 32])
def test_vectorize_broadcast_int4(dtype, vec_num):
    """Broadcasting a non-constant 4-bit value to a vectorized store must emit
    the packed-nibble carrier rather than aborting (see issue #2997). The int4
    branch used to ICHECK on a non-constant value; the int8 branch already
    handled it. Every packed lane count (16/32-bit carriers and make_(u)int{N})
    is exercised here."""
    vectorize_broadcast_int4(dtype, vec_num).compile()


def fill_int4_buffer(dtype, fill_value):
    M, N = 16, 64

    @T.prim_func
    def func(B: T.Tensor((M, N), dtype)):
        with T.Kernel(1, threads=128):
            sh = T.alloc_shared((M, N), dtype)
            T.fill(sh, fill_value)
            T.copy(sh, B)

    return func


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["int4", "uint4"])
@pytest.mark.parametrize("fill_value", [0, 1, 7])
def test_fill_int4_buffer(dtype, fill_value):
    """End-to-end reproduction of issue #2997: T.fill/T.clear of an int4 buffer.
    HoistBroadcastValues rebinds the constant fill value to a Var, so the CUDA
    Broadcast codegen sees a non-constant value. Verify it both compiles and
    produces the correct packed nibbles."""
    M, N = 16, 64
    kernel = tilelang.compile(fill_int4_buffer(dtype, fill_value))
    # int4 buffer packs two nibbles per byte, so N columns -> N // 2 bytes.
    out = torch.empty(M, N // 2, dtype=torch.int8, device="cuda")
    kernel(out)
    expected = fill_value & 0x0F
    low = out & 0x0F
    high = (out >> 4) & 0x0F
    assert bool((low == expected).all()), f"low nibble mismatch for {dtype}={fill_value}"
    assert bool((high == expected).all()), f"high nibble mismatch for {dtype}={fill_value}"


@tilelang.jit
def vectorize_test_call_infinity():
    A = T.empty((4,), dtype=T.float32)
    with T.Kernel(1, threads=128):
        for i in T.vectorized(4):
            A[i] = T.infinity(T.float32)
    return A


def test_vectorize_call_infinity():
    kernel = vectorize_test_call_infinity.compile()
    if _target_is_tang():
        assert "float4" not in kernel.get_kernel_source()
    else:
        assert "float4" in kernel.get_kernel_source()


@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_VECTORIZE_PLANNER_VERBOSE: True, tilelang.PassConfigKey.TL_ENABLE_ASYNC_COPY: False}
)
def vectorize_test_call_bitwise_logical():
    A = T.empty((128, 32), dtype=T.float32)
    with T.Kernel(1, threads=128):
        A_shared = T.alloc_shared((128, 32), dtype=T.float32)
        T.annotate_layout({A_shared: make_mma_swizzle_layout(A_shared)})
        for i, j in T.Parallel(128, 32):
            A_shared[i, j] = A[i, j]
    return A


def test_vectorize_call_bitwise_logical():
    kernel = vectorize_test_call_bitwise_logical.compile()
    print(kernel.get_kernel_source())
    if _target_is_tang():
        assert "float4" not in kernel.get_kernel_source()
    else:
        assert "float4" in kernel.get_kernel_source()


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True})
def vectorize_test_fp16(N, M, stride_A, stride_B):
    @T.prim_func
    def main(
        A: T.StridedTensor[(N, M), (1, stride_A), T.float16],  # noqa: F821
        B: T.StridedTensor[(N, M), (1, stride_B), T.float16],  # noqa: F821
    ):
        with T.Kernel(M // 128, threads=128) as (bx):
            tx = T.get_thread_binding(0)
            col = bx * 128 + tx

            for row in T.vectorized(N):
                B[row, col] = A[row, col]

    return main


def run_vectorize_fp16(N, M, stride_A, stride_B):
    assert N % 128 == 0 and M % 128 == 0
    assert stride_A >= N and stride_B >= N

    jit_kernel = vectorize_test_fp16(N, M, stride_A, stride_B)
    device = get_current_device()
    base_a = torch.randn(stride_A, M, device=device, dtype=torch.float16)
    base_b = torch.zeros(stride_B, M, device=device, dtype=torch.float16)
    a = torch.as_strided(base_a, size=(N, M), stride=(1, stride_A))
    b = torch.as_strided(base_b, size=(N, M), stride=(1, stride_B))

    jit_kernel(a, b)

    if device.type == "ptpu":
        torch.ptpu.synchronize()
    torch.testing.assert_close(a.cpu(), b.cpu(), atol=1e-3, rtol=1e-3)

    code = jit_kernel.get_kernel_source()
    if device.type == "ptpu" and stride_A % 2 == 0 and stride_B % 2 == 0:
        assert "uint1*" in code, "S2 float16x2 vectorization should generate a 32-bit packed access"
        assert "float4" not in code
        assert "float2" not in code


def test_vectorize_fp16():
    """Cover S2 float16x2 vectorization through 32-bit packed accesses."""
    N, M = 512, 256

    run_vectorize_fp16(N, M, N, N)
    run_vectorize_fp16(N, M, N + 2, N + 4)
    run_vectorize_fp16(N, M, N + 4, N + 8)
    run_vectorize_fp16(N, M, N + 8, N + 16)


if __name__ == "__main__":
    tilelang.testing.main()
