import pytest

import tilelang.language as T
from tilelang.backend.pass_pipeline import resolve_pipeline
from tilelang.engine.lower import device_codegen_without_compile, lower_to_host_device_ir
from tilelang.tang.subtarget import (
    TangSubtarget as S,
    arch_to_subtarget,
    get_tang_capabilities,
    pass_filter,
    subtarget_matches,
)
import tvm
from tvm.target import Target


@pytest.mark.parametrize(
    "name",
    [
        "tl.tang.transform.LowerSharedTmem",
        "tl.tang.transform.LowerTangTmemDrain",
        "tl.tang.transform.InjectPTSAsyncCopy",
    ],
)
def test_tang_transform_is_registered(name):
    assert tvm.ffi.get_global_func(name, allow_missing=True) is not None


def test_tang_pipeline_is_registered():
    pipeline = resolve_pipeline(Target({"kind": "tang", "arch": "stcuv2"}))
    assert pipeline.name == "tang"


@pytest.mark.parametrize("arch", ["stcu", "stcuv2"])
def test_tang_pipeline_lowers_minimal_kernel(arch):
    @T.prim_func
    def main(A: T.Tensor((1,), "float32"), B: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=1):
            B[0] = A[0]

    host_mod, device_mod, _, target, _ = lower_to_host_device_ir(
        main.with_attr("global_symbol", "main"),
        target={"kind": "tang", "arch": arch},
    )

    assert target.attrs["arch"] == arch
    assert len(host_mod.functions) == 1
    assert len(device_mod.functions) == 1


@pytest.mark.parametrize(
    ("order", "expected"),
    [
        ("row", "const dim3 blockIdx = tl::rasterization2DRow<4>();"),
        ("column", "const dim3 blockIdx = tl::rasterization2DColumn<4>();"),
    ],
)
def test_tang_pipeline_lowers_threadblock_swizzle_tuple(order, expected):
    @T.prim_func
    def main(A: T.Tensor((1,), "float32"), B: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=1):
            T.use_swizzle(panel_size=4, order=order)
            B[0] = A[0]

    source = _lower_tang_source(main)

    assert expected in source


def test_tang_pipeline_lowers_synchronized_shuffle():
    @T.prim_func
    def main(A: T.Tensor((32,), "int32"), B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            B[tx] = T.shfl_sync(A[tx], 31)

    source = _lower_tang_source(main)

    assert "__shfl_sync(" in source
    assert "tl.shfl_sync" not in source


def test_tang_pipeline_lowers_synchronized_xor_shuffle():
    @T.prim_func
    def main(A: T.Tensor((32,), "int32"), B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            B[tx] = T.shfl_xor(A[tx], 1)

    source = _lower_tang_source(main)

    assert "__shfl_xor_sync(" in source
    assert "tl.shfl_xor_sync" not in source


def test_tang_pipeline_lowers_shared_cumsum():
    @T.prim_func
    def main(A: T.Tensor((32,), "int32"), B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared((32,), "int32")
            T.copy(A, shared)
            T.cumsum(shared, dim=0)
            T.copy(shared, B)

    source = _lower_tang_source(main)

    assert "tl::CumSum1D<32, false>::run" in source


def _lower_tang_source(func, arch="stcu"):
    requested_target = Target({"kind": "tang", "arch": arch})
    with requested_target:
        _, device_mod, _, target, _ = lower_to_host_device_ir(
            func.with_attr("global_symbol", "main"),
            target=requested_target,
        )
    return device_codegen_without_compile(device_mod, target).inspect_source("tang")


def test_tang_pipeline_lowers_scalar_atomic_builtins():
    @T.prim_func
    def main(
        A: T.Tensor((2,), "int32"),
        B: T.Tensor((2,), "int32"),
        C: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(1, threads=1):
            previous = T.atomic_add(B[0], A[0], return_prev=True)
            T.atomic_max(B[0], A[0])
            T.atomic_min(B[0], A[0])
            T.atomic_or(B[0], 1)
            C[0] = previous

    source = _lower_tang_source(main)

    assert "#include <__clang_tang_builtin_vars.h>" in source
    assert "atomicAdd(" in source
    assert "atomicMax(" in source
    assert "atomicMin(" in source
    assert "atomicOr(" in source


@pytest.mark.parametrize("operation", ["load", "store"])
def test_tang_pipeline_rejects_unsupported_atomic_load_store(operation):
    @T.prim_func
    def main(A: T.Tensor((2,), "int32"), B: T.Tensor((2,), "int32")):
        with T.Kernel(1, threads=1):
            if operation == "load":
                B[0] = T.atomic_load(A[0], memory_order="acquire")
            else:
                T.atomic_store(B[0], A[0], memory_order="release")

    with pytest.raises(tvm.error.InternalError, match=f"does not support atomic_{operation}"):
        _lower_tang_source(main)


def test_tang_pipeline_lowers_tile_atomic_add():
    @T.prim_func
    def main(A: T.Tensor((4,), "float32"), B: T.Tensor((4,), "float32")):
        with T.Kernel(1, threads=4):
            T.atomic_add(B, A)

    source = _lower_tang_source(main)

    assert "atomicAdd(" in source


@pytest.mark.parametrize("arch", ["stcu", "stcuv2"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_tang_pipeline_rejects_unimplemented_16bit_atomic_add(arch, dtype):
    @T.prim_func
    def main(A: T.Tensor((8,), dtype), B: T.Tensor((8,), dtype)):
        with T.Kernel(1, threads=4):
            T.atomic_add(B, A)

    with pytest.raises(
        tvm.error.InternalError,
        match="atomicAdd only supports float32, int32, or uint32",
    ):
        _lower_tang_source(main, arch=arch)


def test_tang_pipeline_caps_memory_vectorization_at_32_bits():
    @T.prim_func
    def main(A: T.Tensor((8,), "float32"), B: T.Tensor((8,), "float32")):
        with T.Kernel(1, threads=1):
            for i in T.vectorized(8):
                B[i] = A[i]

    source = _lower_tang_source(main)

    assert "float2" not in source
    assert "float4" not in source


def test_tang_pipeline_preserves_extended_scalar_atomic_contract():
    @T.prim_func
    def main(
        A: T.Tensor((8,), "int32"),
        U: T.Tensor((4,), "uint32"),
        Out: T.Tensor((8,), "int32"),
    ):
        with T.Kernel(1, threads=1):
            Out[0] = T.atomic_sub(A[0], 1, return_prev=True)
            Out[1] = T.atomic_exch(A[1], 2)
            Out[2] = T.atomic_cas(A[2], 2, 3)
            Out[3] = T.atomic_xor(A[3], 1, return_prev=True, uint_atomic=False)
            Out[4] = T.atomic_and(A[4], 7, return_prev=True, uint_atomic=False)
            U[1] = T.atomic_inc(U[0], 7, uint_atomic=True)
            U[3] = T.atomic_dec(U[2], 7, uint_atomic=True)
            T.atomic_add(U[0], 1, uint_atomic=True)
            T.atomic_max(U[0], 1, uint_atomic=True)
            T.atomic_min(U[0], 1, uint_atomic=True)
            T.atomic_or(U[0], 1, uint_atomic=True)

    source = _lower_tang_source(main)

    for name in (
        "atomicSub(",
        "atomicExch(",
        "atomicCAS(",
        "atomicXor(",
        "atomicAnd(",
        "atomicInc(",
        "atomicDec(",
    ):
        assert name in source
    assert source.count("(unsigned int*)") >= 6


def test_tang_pipeline_lowers_extended_tile_atomics():
    @T.prim_func
    def main(
        A: T.Tensor((8,), "int32"),
        B: T.Tensor((8,), "int32"),
        U: T.Tensor((8,), "uint32"),
    ):
        with T.Kernel(1, threads=8):
            T.atomic_sub(B, A)
            T.atomic_exch(B, 2, return_prev=False)
            T.atomic_cas(B, 2, 3, return_prev=False)
            T.atomic_and(U, 7)
            T.atomic_or(U, 1, uint_atomic=True)
            T.atomic_xor(U, 3)
            T.atomic_inc(U, 7, return_prev=False, uint_atomic=True)
            T.atomic_dec(U, 7, return_prev=False, uint_atomic=True)

    source = _lower_tang_source(main)

    for name in (
        "atomicSub(",
        "atomicExch(",
        "atomicCAS(",
        "atomicAnd(",
        "atomicOr(",
        "atomicXor(",
        "atomicInc(",
        "atomicDec(",
    ):
        assert name in source


def test_tang_pipeline_lowers_stcu_tmma_gemm():
    @T.prim_func
    def main(
        A: T.Tensor((64, 32), "float16"),
        B: T.Tensor((32, 64), "float16"),
        C: T.Tensor((64, 64), "float32"),
    ):
        with T.Kernel(1, threads=128):
            a_shared = T.alloc_shared((64, 32), "float16")
            b_shared = T.alloc_shared((32, 64), "float16")
            c_local = T.alloc_fragment((64, 64), "float32")
            T.copy(A, a_shared)
            T.copy(B, b_shared)
            T.clear(c_local)
            T.gemm(a_shared, b_shared, c_local, clear_accum=False)
            T.copy(c_local, C)

    source = _lower_tang_source(main)

    assert "tl::gemm_tang<64, 64, 32, 2, 2" in source


def test_tang_pipeline_lowers_strided_stcu_tmma_from_region_base():
    @T.prim_func
    def main(
        A: T.Tensor((32, 32), "float16"),
        B: T.Tensor((32, 32), "float16"),
        C: T.Tensor((32, 32), "float32"),
    ):
        with T.Kernel(1, threads=128):
            a_shared = T.alloc_shared((32, 64), "float16")
            b_shared = T.alloc_shared((32, 64), "float16")
            c_local = T.alloc_fragment((32, 32), "float32")
            T.clear(a_shared)
            T.clear(b_shared)
            T.copy(A, a_shared[:, 32:])
            T.copy(B, b_shared[:, :32])
            T.clear(c_local)
            T.gemm(a_shared[:, 32:], b_shared[:, :32], c_local, clear_accum=False)
            T.copy(c_local, C)

    source = _lower_tang_source(main)

    assert "tl::gemm_tang<32, 32, 32, 2, 2, 64, 64, 32, 0" in source
    assert "A_shared[-" not in source
    assert " - 480" not in source


def test_tang_pipeline_preserves_stcu_tmma_load_controls():
    @T.prim_func
    def main(
        A: T.Tensor((64, 32), "float16"),
        B: T.Tensor((32, 64), "float16"),
        C: T.Tensor((64, 64), "float32"),
    ):
        with T.Kernel(1, threads=128):
            a_shared = T.alloc_shared((64, 32), "float16")
            b_shared = T.alloc_shared((32, 64), "float16")
            c_local = T.alloc_fragment((64, 64), "float32")
            T.copy(A, a_shared)
            T.copy(B, b_shared)
            T.clear(c_local)
            T.gemm(
                a_shared,
                b_shared,
                c_local,
                clear_accum=False,
                k_step=4,
                a_local_load_type="load_overlap_mma",
                b_local_load_type="load_before_mma",
            )
            T.copy(c_local, C)

    source = _lower_tang_source(main)

    # Verify template prefix with correct dimensions and warp config
    assert "tl::gemm_tang<64, 64, 32, 2, 2, " in source
    # k_step is still accepted by T.gemm but was removed from the gemm_tang
    # template (871cae7d); trans_A/trans_B default to 0 and the trailing 0 is
    # clear_accum=False.
    assert "tl::gemm_tang<64, 64, 32, 2, 2, 32, 64, 0, 0, 0, 0, 0>" in source
    # Verify that template parameters removed in the refactor (a7134ed3) are absent
    assert "is_a_local_load_overlap_mma" not in source
    assert "is_b_local_load_overlap_mma" not in source


def test_tang_pipeline_lowers_stcu_tmma_warp_rows_gt_one():
    """M_Tile=128, threads=64 → num_warp_m=2, num_warp_n=1 → warp_rows=8.
    Exercises the multi-row warp loop and contiguous acc_ptr layout with a
    warp_rows count that exercises the row-index computation thoroughly."""

    @T.prim_func
    def main(
        A: T.Tensor((128, 32), "float16"),
        B: T.Tensor((32, 64), "float16"),
        C: T.Tensor((128, 64), "float32"),
    ):
        with T.Kernel(1, threads=64):
            a_shared = T.alloc_shared((128, 32), "float16")
            b_shared = T.alloc_shared((32, 64), "float16")
            c_local = T.alloc_fragment((128, 64), "float32")
            T.copy(A, a_shared)
            T.copy(B, b_shared)
            T.clear(c_local)
            T.gemm(a_shared, b_shared, c_local, clear_accum=False)
            T.copy(c_local, C)

    source = _lower_tang_source(main)

    # Full template: M=128, N=64, K=32, mwarp=2, nwarp=1, stride_a=32, stride_b=64,
    # offset_a=0, offset_b=0, trans_A=0, trans_B=0, clear_accum=0.
    # warp_rows = 128/(2*8) = 8, warp_cols = 64/(1*8) = 8: both > 1.
    assert "tl::gemm_tang<128, 64, 32, 2, 1, 32, 64, 0, 0, 0, 0, 0>" in source
    # call_mma is a template function inlined by the compiler, so source-level
    # counting is not possible.  The template params above confirm the layout.


def test_tang_pipeline_lowers_pipelined_stcu_tmma_gemm():
    @T.prim_func
    def main(
        A: T.Tensor((64, 64), "float16"),
        B: T.Tensor((64, 64), "float16"),
        C: T.Tensor((64, 64), "float32"),
    ):
        with T.Kernel(1, threads=128):
            a_shared = T.alloc_shared((64, 32), "float16")
            b_shared = T.alloc_shared((32, 64), "float16")
            c_local = T.alloc_fragment((64, 64), "float32")
            T.clear(c_local)
            for ko in T.Pipelined(2, num_stages=2):
                T.copy(A[:, ko * 32 : (ko + 1) * 32], a_shared)
                T.copy(B[ko * 32 : (ko + 1) * 32, :], b_shared)
                T.gemm(a_shared, b_shared, c_local, clear_accum=False)
            T.copy(c_local, C)

    source = _lower_tang_source(main)

    assert "tl::gemm_tang<64, 64, 32, 2, 2" in source


def test_tang_pipeline_lowers_stcu_tmma_clear_accum():
    """clear_accum=True is supported and reaches GemmTensorOp as the last
    template argument, letting the caller drop T.clear on the accumulator: the
    first MMA of each chain lowers to a `tensor.mul` that defines C instead of
    reading it."""

    @T.prim_func
    def main(
        A: T.Tensor((64, 32), "float16"),
        B: T.Tensor((32, 64), "float16"),
        C: T.Tensor((64, 64), "float32"),
    ):
        with T.Kernel(1, threads=128):
            a_shared = T.alloc_shared((64, 32), "float16")
            b_shared = T.alloc_shared((32, 64), "float16")
            c_local = T.alloc_fragment((64, 64), "float32")
            T.copy(A, a_shared)
            T.copy(B, b_shared)
            T.gemm(a_shared, b_shared, c_local, clear_accum=True)
            T.copy(c_local, C)

    source = _lower_tang_source(main)

    # Trailing "1>" is clear_accum=1.
    assert "tl::gemm_tang<64, 64, 32, 2, 2, 32, 64, 0, 0, 0, 0, 1>" in source


@pytest.mark.skip(reason="stcuv2 gemm claim test is S3-specific")
def test_tang_pipeline_does_not_claim_stcuv2_gemm():
    @T.prim_func
    def main(
        A: T.Tensor((64, 32), "float16"),
        B: T.Tensor((32, 64), "float16"),
        C: T.Tensor((64, 64), "float32"),
    ):
        with T.Kernel(1, threads=128):
            a_shared = T.alloc_shared((64, 32), "float16")
            b_shared = T.alloc_shared((32, 64), "float16")
            c_local = T.alloc_fragment((64, 64), "float32")
            T.copy(A, a_shared)
            T.copy(B, b_shared)
            T.clear(c_local)
            T.gemm(a_shared, b_shared, c_local, clear_accum=False)
            T.copy(c_local, C)

    with pytest.raises(tvm.error.InternalError, match="no gemm implementation"):
        _lower_tang_source(main, arch="stcuv2")


def test_tile_atomic_rejects_return_previous_value():
    with pytest.raises(NotImplementedError, match="return_prev"):

        @T.prim_func
        def main(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
            with T.Kernel(1, threads=4):
                T.atomic_exch(B, A, return_prev=True)


@pytest.mark.parametrize(("width", "expected_count"), [(2, 2), (4, 4)])
def test_tang_vector_atomic_add_is_scalarized(width, expected_count):
    @T.prim_func
    def main(A: T.Tensor((4,), "float32"), B: T.Tensor((4,), "float32")):
        with T.Kernel(1, threads=1):
            if width == 2:
                T.atomic_addx2(B[0], A[0])
            else:
                T.atomic_addx4(B[0], A[0])

    source = _lower_tang_source(main)

    assert source.count("atomicAdd(") == expected_count
    assert "AtomicAddx" not in source


@pytest.mark.parametrize("width", [2, 4])
def test_tang_vector_atomic_add_preserves_return_values(width):
    if width == 2:

        @T.prim_func
        def main(
            A: T.Tensor((4,), "float32"),
            B: T.Tensor((4,), "float32"),
            Out: T.Tensor((4,), "float32"),
        ):
            with T.Kernel(1, threads=1):
                previous = T.atomic_addx2(B[0], A[0], return_prev=True)
                Out[0] = previous[0]
                Out[1] = previous[1]
    else:

        @T.prim_func
        def main(
            A: T.Tensor((4,), "float32"),
            B: T.Tensor((4,), "float32"),
            Out: T.Tensor((4,), "float32"),
        ):
            with T.Kernel(1, threads=1):
                previous = T.atomic_addx4(B[0], A[0], return_prev=True)
                Out[0] = previous[0]
                Out[1] = previous[1]
                Out[2] = previous[2]
                Out[3] = previous[3]

    source = _lower_tang_source(main)

    assert source.count(" = atomicAdd(") == width


@pytest.mark.parametrize("width", [2, 4])
def test_tang_vector_atomic_add_preserves_unsigned_contract(width):
    @T.prim_func
    def main(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=1):
            if width == 2:
                T.atomic_addx2(B[0], A[0], uint_atomic=True)
            else:
                T.atomic_addx4(B[0], A[0], uint_atomic=True)

    source = _lower_tang_source(main)

    assert source.count("atomicAdd(") == width
    assert source.count("(unsigned int*)") == width


def test_tang_vector_atomic_add_region_uses_common_tile_lowering():
    @T.prim_func
    def main(A: T.Tensor((8,), "float32"), B: T.Tensor((8,), "float32")):
        with T.Kernel(1, threads=1):
            T.atomic_addx4(B[0:8], A[0:8])

    source = _lower_tang_source(main)

    assert "#pragma unroll 8" in source
    assert "atomicAdd(&(B[i]), A[i])" in source


@pytest.mark.parametrize(
    ("dtype", "constant", "header"),
    [
        ("float16", "TANGRT_INF_FP16", "__clang_tang_fp16.h"),
        ("bfloat16", "TANGRT_INF_BF16", "__clang_tang_bf16.h"),
        ("float32", "TANG_RT_INF_F", "__clang_tang_builtin_vars.h"),
    ],
)
def test_tang_infinity_lowering(dtype, constant, header):
    @T.prim_func
    def main(A: T.Tensor((1,), dtype)):
        with T.Kernel(1, threads=1):
            A[0] = T.infinity(dtype)

    source = _lower_tang_source(main)

    assert constant in source
    assert f"#include <{header}>" in source


@pytest.mark.parametrize(
    ("generator", "state", "suffix"),
    [
        ("curandStatePhilox4_32_10_t", "TangRNGStatePhilox", "philox"),
        ("curandStateMRG32k3a_t", "TangRNGStateMRG32k3a", "mrg32k3a"),
        ("curandStateXORWOW_t", "TangRNGStateXORWOW", "xorwow"),
    ],
)
def test_tang_rng_algorithm_selection(generator, state, suffix):
    @T.prim_func
    def main(A: T.Tensor((2,), "float32")):
        with T.Kernel(1, threads=1):
            T.rng_init(42, 0, 0, generator=generator)
            A[0] = T.cast(T.rng_rand(), "float32")
            A[1] = T.rng_rand_float(dist="normal")

    source = _lower_tang_source(main)

    assert state in source
    assert f"rng_init_{suffix}" in source
    assert f"rng_rand_{suffix}" in source
    assert f"rng_normal_float_{suffix}" in source


def test_tang_rng_rejects_float64():
    @T.prim_func
    def main(A: T.Tensor((1,), "float64")):
        with T.Kernel(1, threads=1):
            T.rng_init(42, 0, 0)
            A[0] = T.rng_rand_float(bit=64)

    with pytest.raises(tvm.error.InternalError, match="does not support float64"):
        _lower_tang_source(main)


@pytest.mark.parametrize(("random_kind", "dtype"), [("integer", "uint32"), ("float", "float32")])
def test_tang_rng_rejects_use_before_init(random_kind, dtype):
    @T.prim_func
    def main(A: T.Tensor((1,), dtype)):
        with T.Kernel(1, threads=1):
            if random_kind == "integer":
                A[0] = T.rng_rand()
            else:
                A[0] = T.rng_rand_float()
            T.rng_init(42, 0, 0)

    with pytest.raises(tvm.error.InternalError, match="called before rng_init"):
        _lower_tang_source(main)


def test_tang_rng_rejects_multiple_init():
    @T.prim_func
    def main():
        with T.Kernel(1, threads=1):
            T.rng_init(42, 0, 0)
            T.rng_init(43, 0, 0)

    with pytest.raises(tvm.error.InternalError, match="only one rng_init"):
        _lower_tang_source(main)


def test_tang_subtarget_capability_table():
    stcu = Target({"kind": "tang", "arch": "stcu"})
    stcuv2 = Target({"kind": "tang", "arch": "stcuv2"})

    assert arch_to_subtarget("stcu") == S.STCU
    assert arch_to_subtarget("stcuv2") == S.STCUV2
    assert subtarget_matches(stcu, S.STCU)
    assert not subtarget_matches(stcu, S.STCUV2)
    assert subtarget_matches(stcuv2, S.STCUV2)

    assert get_tang_capabilities(stcu).max_vector_load_bits == 32
    assert get_tang_capabilities(stcuv2).max_vector_load_bits == 32
    assert get_tang_capabilities(stcu).supports_pts_async_copy
    assert get_tang_capabilities(stcuv2).supports_tmem_drain


def test_tang_pass_filter_skips_non_matching_arch():
    calls = []

    class FakeModule:
        functions = {}

        def __init__(self, target):
            self.target = target

        def get_attr(self, name):
            return self.target if name == "target" else None

    def factory():
        def run(mod):
            calls.append(mod.target.attrs["arch"])
            return mod

        return run

    filtered = pass_filter(factory, S.STCU)()
    stcu_mod = FakeModule(Target({"kind": "tang", "arch": "stcu"}))
    stcuv2_mod = FakeModule(Target({"kind": "tang", "arch": "stcuv2"}))

    assert filtered(stcu_mod) is stcu_mod
    assert filtered(stcuv2_mod) is stcuv2_mod
    assert [str(arch) for arch in calls] == ["stcu"]


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (None, "requires target metadata"),
        (Target("llvm"), "requires a valid TANG target"),
    ],
)
def test_tang_pass_filter_rejects_malformed_target_metadata(target, message):
    class FakeModule:
        functions = {}

        def get_attr(self, name):
            return target if name == "target" else None

    filtered = pass_filter(lambda: lambda mod: mod, S.STCU)()

    with pytest.raises(ValueError, match=message):
        filtered(FakeModule())
