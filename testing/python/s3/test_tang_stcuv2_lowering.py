"""S3 (stcuv2) lowering assertions, one kernel at a time.

These cases lower a TileLang kernel to TANG C source (no device, no ISS) and
assert what the stcuv2 path emits.  They used to live in
``testing/python/target/test_tilelang_tang_pipeline.py``; that file keeps the
arch-agnostic pipeline coverage, so S3-only cases belong here.
"""

import pytest

import tilelang.tang.language as T
from tilelang.backend import create_backend_context
from tilelang.engine.lower import (
    device_codegen_without_compile,
    lower_to_host_device_ir,
)
import tvm
from tvm.target import Target


def _lower_tang_source(func, arch="stcu"):
    requested_target = Target({"kind": "tang", "arch": arch})
    context = create_backend_context(requested_target)
    with requested_target:
        _, device_mod, _, target, _ = lower_to_host_device_ir(
            func.with_attr("global_symbol", "main"),
            context,
        )
    return device_codegen_without_compile(device_mod, context).inspect_source("tang")


_PACKED_SUFFIX = {"float32": "f32", "float16": "f16", "bfloat16": "bf16"}


@pytest.mark.parametrize(
    ("builder", "expected"),
    [
        (lambda: T.any_sync(1), "__any_sync("),
        (lambda: T.all_sync(1), "__all_sync("),
        (lambda: T.ballot_sync(1), "(unsigned long long)__ballot_sync("),
        (lambda: T.ballot(1), "__ballot_sync(0xFFFFFFFFu,"),
        (lambda: T.activemask(), "(unsigned long long)__activemask()"),
        (lambda: T.match_all_sync(1), "__match_all_sync("),
    ],
)
def test_tang_stcuv2_lowers_warp_votes(builder, expected):
    @T.prim_func
    def main(Out: T.Tensor((1,), "uint64")):
        with T.Kernel(1, threads=32):
            Out[0] = builder()

    source = _lower_tang_source(main, arch="stcuv2")

    assert expected in source
    assert "#include <__clang_tang_builtin_vars.h>" in source


def test_tang_stcuv2_hides_match_all_predicate():
    @T.prim_func
    def main(Values: T.Tensor((32,), "int32"), Out: T.Tensor((32,), "uint32")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            Out[tx] = T.match_all_sync(Values[tx])

    source = _lower_tang_source(main, arch="stcuv2")

    assert "&_tl_pred" in source


@pytest.mark.parametrize("operation", ["any_sync", "all_sync", "ballot", "activemask", "match_all_sync"])
def test_tang_stcu_rejects_warp_votes(operation):
    @T.prim_func
    def main(Out: T.Tensor((1,), "uint64")):
        with T.Kernel(1, threads=32):
            if operation == "any_sync":
                Out[0] = T.any_sync(1)
            elif operation == "all_sync":
                Out[0] = T.all_sync(1)
            elif operation == "ballot":
                Out[0] = T.ballot(1)
            elif operation == "activemask":
                Out[0] = T.activemask()
            else:
                Out[0] = T.match_all_sync(1)

    with pytest.raises(tvm.error.InternalError, match="warp vote intrinsics require arch=stcuv2"):
        _lower_tang_source(main, arch="stcu")


def _make_tang_pair_math_kernel(op, dtype, arity):
    @T.prim_func
    def main(
        A: T.Tensor((2,), dtype),
        B: T.Tensor((2,), dtype),
        C: T.Tensor((2,), dtype),
        Out: T.Tensor((2,), dtype),
    ):
        with T.Kernel(1, threads=1):
            lanes = T.Ramp(0, 1, 2)
            if arity == 1:
                Out[lanes] = op(A[lanes])
            elif arity == 2:
                Out[lanes] = op(A[lanes], B[lanes])
            else:
                Out[lanes] = op(A[lanes], B[lanes], C[lanes])

    return main


@pytest.mark.parametrize(
    ("name", "op", "arity"),
    [
        ("add2", T.add2, 2),
        ("sub2", T.sub2, 2),
        ("mul2", T.mul2, 2),
        ("fma2", T.fma2, 3),
        ("max2", T.max2, 2),
        ("min2", T.min2, 2),
        ("abs2", T.abs2, 1),
    ],
)
@pytest.mark.parametrize(
    ("dtype", "header"),
    [
        ("float32", None),
        ("float16", "__clang_tang_fp16.h"),
        ("bfloat16", "__clang_tang_bf16.h"),
    ],
)
def test_tang_stcuv2_lowers_explicit_pair_math(name, op, arity, dtype, header):
    source = _lower_tang_source(_make_tang_pair_math_kernel(op, dtype, arity), arch="stcuv2")

    # One call into the tl::packed_<op>_<dtype> helper; the scalar
    # spelling of the same op must be gone.
    assert "_packed_pair" in source
    assert f"tl::packed_{name}_{_PACKED_SUFFIX[dtype]}(" in source
    assert f"tl.{name}" not in source
    if header is not None:
        assert f"#include <{header}>" in source


@pytest.mark.parametrize(
    ("operation", "memory_order", "memory_order_id"),
    [
        ("load", "relaxed", 0),
        ("load", "consume", 1),
        ("load", "acquire", 2),
        ("load", "seq_cst", 5),
        ("store", "relaxed", 0),
        ("store", "release", 3),
        ("store", "seq_cst", 5),
    ],
)
def test_tang_stcuv2_lowers_atomic_load_store(
    operation,
    memory_order,
    memory_order_id,
):
    @T.prim_func
    def main(A: T.Tensor((2,), "int32"), B: T.Tensor((2,), "int32")):
        with T.Kernel(1, threads=1):
            if operation == "load":
                B[0] = T.atomic_load(A[0], memory_order=memory_order)
            else:
                T.atomic_store(B[0], A[0], memory_order=memory_order)

    source = _lower_tang_source(main, arch="stcuv2")

    assert "#include <__clang_tang_builtin_vars.h>" in source
    assert f"tl::atomic_{operation}(" in source
    assert f", {memory_order_id})" in source


@pytest.mark.parametrize("operation", ["load", "store"])
def test_tang_stcu_rejects_atomic_load_store(operation):
    @T.prim_func
    def main(A: T.Tensor((2,), "int32"), B: T.Tensor((2,), "int32")):
        with T.Kernel(1, threads=1):
            if operation == "load":
                B[0] = T.atomic_load(A[0], memory_order="acquire")
            else:
                T.atomic_store(B[0], A[0], memory_order="release")

    with pytest.raises(
        tvm.error.InternalError,
        match=f"atomic_{operation} requires arch=stcuv2",
    ):
        _lower_tang_source(main, arch="stcu")


def test_tang_stcuv2_lowers_shared_atomic_load_store():
    @T.prim_func
    def main(A: T.Tensor((1,), "int32"), B: T.Tensor((1,), "int32")):
        with T.Kernel(1, threads=1):
            shared = T.alloc_shared((1,), "int32")
            T.atomic_store(shared[0], A[0], memory_order="release")
            B[0] = T.atomic_load(shared[0], memory_order="acquire")

    source = _lower_tang_source(main, arch="stcuv2")

    assert "tl::atomic_store(" in source
    assert "tl::atomic_load(" in source


@pytest.mark.parametrize("operation", ["load", "store"])
def test_tang_stcuv2_rejects_local_atomic_load_store(operation):
    @T.prim_func
    def main(A: T.Tensor((1,), "int32"), B: T.Tensor((1,), "int32")):
        with T.Kernel(1, threads=1):
            local = T.alloc_local((1,), "int32")
            local[0] = A[0]
            if operation == "load":
                B[0] = T.atomic_load(local[0], memory_order="relaxed")
            else:
                T.atomic_store(local[0], A[0], memory_order="relaxed")

    with pytest.raises(
        tvm.error.InternalError,
        match="only global/shared/generic memory",
    ):
        _lower_tang_source(main, arch="stcuv2")


@pytest.mark.parametrize(
    ("width", "dtype", "suffix"),
    [
        (2, "float32", "f32"),
        (4, "float32", "f32"),
        (2, "float16", "f16"),
        (4, "float16", "f16"),
        (2, "bfloat16", "bf16"),
        (4, "bfloat16", "bf16"),
    ],
)
def test_tang_stcuv2_vector_atomic_add_uses_native_builtin(width, dtype, suffix):
    @T.prim_func
    def main(A: T.Tensor((4,), dtype), B: T.Tensor((4,), dtype)):
        with T.Kernel(1, threads=1):
            if width == 2:
                T.atomic_addx2(B[0], A[0])
            else:
                T.atomic_addx4(B[0], A[0])

    source = _lower_tang_source(main, arch="stcuv2")

    assert f"tl::atomic_addx{width}_{suffix}(" in source
    assert source.count("atomicAdd(") == 0


@pytest.mark.parametrize(
    ("width", "dtype", "suffix"),
    [
        (2, "float32", "f32"),
        (4, "float32", "f32"),
        (2, "float16", "f16"),
        (4, "float16", "f16"),
        (2, "bfloat16", "bf16"),
        (4, "bfloat16", "bf16"),
    ],
)
def test_tang_stcuv2_vector_atomic_add_returns_previous_vector(width, dtype, suffix):
    @T.prim_func
    def main(
        A: T.Tensor((4,), dtype),
        B: T.Tensor((4,), dtype),
        Out: T.Tensor((4,), dtype),
    ):
        with T.Kernel(1, threads=1):
            if width == 2:
                Out[0:2] = T.atomic_addx2(B[0:2], A[0:2], return_prev=True)
            else:
                Out[0:4] = T.atomic_addx4(B[0:4], A[0:4], return_prev=True)

    source = _lower_tang_source(main, arch="stcuv2")

    helper = f"tl::atomic_addx{width}_{suffix}(" if suffix == "f32" else f"tl::atomic_addx{width}_ret_{suffix}("
    assert helper in source
    assert source.count("atomicAdd(") == 0
