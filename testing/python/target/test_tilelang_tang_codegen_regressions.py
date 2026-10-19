"""Device-only TANG source checks for generic IR operations."""

import re

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from tvm import tirx


def _source(func):
    build = tvm.get_global_func("target.build.tilelang_tang_without_compile", allow_missing=True)
    if build is None:
        pytest.skip("TileLang was built without the TANG code generator")
    func = func.with_attr("global_symbol", "regression")
    func = func.with_attr("calling_conv", tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    return build(tvm.IRModule({"regression": func}), tvm.target.Target("tang")).inspect_source()


@pytest.mark.parametrize("lanes", [1, 2, 4])
def test_tang_boolean_bitwise_not_uses_logical_negation(lanes):
    value = tirx.Var("value", "bool" if lanes == 1 else f"boolx{lanes}")
    source = _source(tirx.PrimFunc([value], tirx.Evaluate(~value)))
    # Check the operation on this operand, not unrelated operators in the translation unit.
    assert re.search(r"!\s*(?:bool\s*)?\(*\s*value\b", source)
    assert not re.search(r"~\s*(?:bool\s*)?\(*\s*value\b", source)


@pytest.mark.parametrize("message", [None, 'bounds "check"\\nfailed'])
def test_tang_assert_stmt_uses_device_helper(message):
    value = tirx.Var("value", "int32")
    parts = [] if message is None else [tirx.StringImm(message)]
    func = tirx.PrimFunc([value], tirx.AssertStmt(value >= 0, tirx.StringImm("RuntimeError"), parts))
    source = _source(func)
    assert "TVMFFIErrorSetRaisedFromCStrParts" not in source
    assert "return -1" not in source
    helper = "device_assert(" if message is None else "device_assert_with_msg("
    assert helper in source
    if message is not None:
        assert '\\"check\\"' in source


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("lanes", [2, 4, 8, 16])
def test_tang_clamp_broadcast_bounds_compiles(dtype, lanes):
    if tvm.get_global_func("target.build.tilelang_tang_without_compile", allow_missing=True) is None:
        pytest.skip("TileLang was built without the TANG code generator")

    @T.prim_func
    def main(A: T.Tensor((32 * lanes,), dtype), B: T.Tensor((32 * lanes,), dtype)):
        with T.Kernel(1, threads=32):
            indices = T.Ramp(T.get_thread_binding() * lanes, 1, lanes)
            B[indices] = T.clamp(A[indices], T.cast(-1.0, dtype), T.cast(1.0, dtype))

    source = tilelang.lower(main, target="tang").kernel_source
    # Compile device source only: no runtime module, launch or device access.
    from tilelang.contrib import ptcc
    from tilelang.tang.codegen import tilelang_callback_tang_compile

    try:
        ptcc.get_ptcc_compiler(arch="stcu")
    except (FileNotFoundError, RuntimeError):
        pytest.skip("The STCU PTCC compiler is unavailable")
    assert tilelang_callback_tang_compile(source, tvm.target.Target({"kind": "tang", "arch": "stcu"}))


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("write_between", [False, True])
def test_tang_packed_shuffle_source_is_evaluated_once_per_read(dtype, write_between):
    source_buffer = tirx.decl_buffer((4,), dtype, name="A")
    output_buffer = tirx.decl_buffer((2,), dtype, name="Out")
    vector = tirx.BufferLoad(source_buffer, [tirx.Ramp(0, 1, 4)])
    element = tirx.Shuffle([vector], [0])
    statements = [tirx.BufferStore(output_buffer, element, [0])]
    if write_between:
        statements += [
            tirx.BufferStore(source_buffer, tirx.const(2, dtype), [0]),
            tirx.BufferStore(output_buffer, element, [1]),
        ]
    body = statements[0] if len(statements) == 1 else tirx.SeqStmt(statements)
    body = tirx.AttrStmt(source_buffer.data, "volatile_scope", 1, body)
    func = tirx.PrimFunc(
        [source_buffer.data, output_buffer.data],
        body,
        buffer_map={source_buffer.data: source_buffer, output_buffer.data: output_buffer},
    )
    source = _source(func)
    # A vector source is a single observable read, even when only one lane is selected.
    reads = [match.start() for match in re.finditer(r"=\s*[^;]*\bA\b[^;]*;", source)]
    assert len(reads) == (2 if write_between else 1)
    if write_between:
        store = re.search(r"\bA\s*\[\s*0\s*\]\s*=", source).start()
        assert reads[0] < store < reads[1]
