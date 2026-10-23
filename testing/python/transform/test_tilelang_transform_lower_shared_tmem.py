# ruff: noqa
import pytest

from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang.backend.target import determine_target
from tilelang.tang.target import target_is_tang
import pytest


TARGET = tvm.target.Target({"kind": "tang", "arch": "stcuv2"})

# stcuv2 allows a single live TMEM reservation, so tang's LowerSharedTmem merges
# every shared.tmem buffer in a block into one allocation and emits a single
# collective init/dealloc pair sized to the merged column count (rounded up to a
# power of two). Explicit T.deallocate_tmem therefore releases the whole
# reservation rather than one buffer, and the automatic tail dealloc is dropped
# entirely once the user takes manual control -- otherwise the reservation would
# be freed twice.
_INIT = "tl.tang_init_tensor_memory"
_DEALLOC = "tl.tang_deallocate_tensor_memory"


def _apply(func):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tirx.transform.BindTarget(TARGET)(mod)
    mod = tl.transform.MaterializeKernelLaunch()(mod)
    mod = tl.tang.transform.LowerSharedTmem()(mod)
    return mod


def _collect_calls(stmt, op_name: str):
    calls = []

    def visitor(node):
        if isinstance(node, tvm.tirx.Call) and hasattr(node, "op") and hasattr(node.op, "name") and node.op.name == op_name:
            calls.append(node)

    tvm.tirx.stmt_functor.post_order_visit(stmt, visitor)
    return calls


def test_explicit_deallocate_tmem_suppresses_auto_dealloc():
    """Explicit T.deallocate_tmem on fallthrough suppresses auto-dealloc."""

    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            C_tmem = T.alloc_tmem([128, 128], T.float32)
            T.deallocate_tmem(C_tmem)

    mod = _apply(func)
    body = mod["main"].body
    assert len(_collect_calls(body, _INIT)) == 1
    assert len(_collect_calls(body, _DEALLOC)) == 1
    # The frontend marker must be fully rewritten, or codegen dies with
    # "Unresolved call ... tl.deallocate_tmem".
    assert len(_collect_calls(body, "tl.deallocate_tmem")) == 0

    dealloc_call = _collect_calls(body, _DEALLOC)[0]
    assert dealloc_call.args[1].value == 128


def test_explicit_deallocate_releases_whole_merged_reservation():
    """An explicit dealloc of any buffer releases the whole merged reservation."""

    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            A_tmem = T.alloc_tmem([128, 128], T.float32)
            B_tmem = T.alloc_tmem([128, 64], T.float32)
            T.deallocate_tmem(A_tmem)

    mod = _apply(func)
    body = mod["main"].body
    assert len(_collect_calls(body, "tl.deallocate_tmem")) == 0

    # Both buffers live in one reservation, so deallocating A_tmem frees the
    # whole region: exactly one dealloc, and the auto one is dropped.
    dealloc_calls = _collect_calls(body, _DEALLOC)
    assert len(dealloc_calls) == 1

    # 128 + 64 columns merged, rounded up to the next power of two.
    init_call = _collect_calls(body, _INIT)[0]
    assert init_call.args[1].value == 256
    assert dealloc_calls[0].args[1].value == 256


def test_dealloc_before_thread_return_keeps_auto_dealloc():
    """Dealloc on non-fallthrough path (before thread_return) does NOT suppress auto-dealloc."""

    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            C_tmem = T.alloc_tmem([128, 128], T.float32)
            tx = T.get_thread_binding()

            if tx < 32:
                T.deallocate_tmem(C_tmem)
                T.thread_return()

    mod = _apply(func)
    body = mod["main"].body
    assert len(_collect_calls(body, "tl.deallocate_tmem")) == 0

    dealloc_calls = _collect_calls(body, _DEALLOC)
    # 1 explicit (non-fallthrough) + 1 auto (block end) = 2
    assert len(dealloc_calls) == 2
    assert [call.args[1].value for call in dealloc_calls] == [128, 128]


def test_untyped_buffer_reports_internal_error():
    target = determine_target(tl.env.get_default_target(), return_object=True)
    valid_buffer = tvm.tirx.decl_buffer((1,), name="malformed_buffer")
    untyped_data = tvm.tirx.Var("malformed_buffer", "handle")
    malformed_buffer = tvm.ir.make_node(
        "tirx.Buffer",
        data=untyped_data,
        dtype=valid_buffer.dtype,
        shape=valid_buffer.shape,
        strides=valid_buffer.strides,
        axis_separators=valid_buffer.axis_separators,
        elem_offset=valid_buffer.elem_offset,
        name=valid_buffer.name,
        data_alignment=valid_buffer.data_alignment,
        offset_factor=valid_buffer.offset_factor,
        buffer_type=valid_buffer.buffer_type,
        span=None,
    )
    block = tvm.tirx.SBlock(
        iter_vars=[],
        reads=[],
        writes=[],
        name_hint="root",
        body=tvm.tirx.Evaluate(0),
        alloc_buffers=[malformed_buffer],
    )
    func = tvm.tirx.PrimFunc([], block).with_attr("target", target)
    mod = tvm.IRModule.from_expr(func)
    lower_shared_tmem = tl.tang.transform.LowerSharedTmem() if target_is_tang(target) else tl.cuda.transform.LowerSharedTmem()

    with pytest.raises(
        tvm.error.InternalError,
        match=r"LowerSharedTmem requires buffer malformed_buffer's data Var "
        r"to have a PointerType annotation",
    ):
        lower_shared_tmem(mod)


if __name__ == "__main__":
    test_explicit_deallocate_tmem_suppresses_auto_dealloc()
    test_explicit_deallocate_releases_whole_merged_reservation()
    test_dealloc_before_thread_return_keeps_auto_dealloc()
    test_untyped_buffer_reports_internal_error()
