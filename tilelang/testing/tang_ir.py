"""TANG/TANG-IR test utilities: read tcgen05 MMA descriptor parameters from IR.

The tcgen05 shared-memory and instruction descriptors are IR *values* emitted by
the TANG lowering (``tl.tang_mma_data_desc`` / ``tl.tang_mma_instr_desc`` /
``tl.tang_mma_scale_instr_desc``), and the MMA op consumes them as operands, with
fixed operand roles (A descriptor, B descriptor, instruction descriptor).

Tests that want to pin what the lowering computed -- operand element codes, the
a_major/b_major layout flags, subM/subN, K, LBO/SBO/swizzle -- can therefore read
those values straight off the device IR.  Doing so is more precise than grepping
the generated C++ (the operand roles come from the MMA call instead of the
textual order of the marker statements) and it stops at the last pass before
codegen, so it does not depend on how the codegen names, spaces or materialises
the values.

Usage::

    from tilelang.testing.tang_ir import mma_descriptors

    desc = mma_descriptors(gemm_kernel)
    assert desc.instr == (1, 2, 2, 0, 1, 32, 128, 64)
    assert desc.data == ((128, 512, 14), (256, 256, 14))

The one thing this cannot check is *emission* (how many ``mma`` instantiations
are printed, which template is used, that the descriptor marker survives next to
the MMA): those stay source-text assertions on purpose, since they are the
contract the STCU backend's TCProcess pass reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tilelang.backend.module import create_backend_context
from tilelang.engine.lower import lower_to_host_device_ir
from tvm import IRModule, tirx
from tvm.target import Target

#: The two TANG arch targets, in the dict form the target resolver accepts.
STCUV2_TARGET: dict[str, str] = {"kind": "tang", "arch": "stcuv2"}
STCU_TARGET: dict[str, str] = {"kind": "tang", "arch": "stcu"}

#: Every tcgen05 MMA op that consumes descriptors.
MMA_OPS: tuple[str, ...] = (
    "tl.tang_tcgen05_mma_ss",
    "tl.tang_tcgen05_mma_ts",
    "tl.tang_tcgen05_mma_scale",
)

#: The three descriptor value ops.
DATA_DESC_OP = "tl.tang_mma_data_desc"
INSTR_DESC_OP = "tl.tang_mma_instr_desc"
SCALE_INSTR_DESC_OP = "tl.tang_mma_scale_instr_desc"


def lower_device_ir(
    kernel: tirx.PrimFunc | IRModule,
    target: str | dict[str, object] | Target = STCUV2_TARGET,
) -> IRModule:
    """Lower ``kernel`` up to (but not including) target codegen.

    This is the last stage of the TANG pipeline, so the returned module is what
    the codegen consumes -- including the descriptor value ops and the MMA ops
    that read them.
    """
    context = create_backend_context(target)
    _, device_mod, _, _, _ = lower_to_host_device_ir(kernel, context)
    return device_mod


def device_func(device_mod: IRModule) -> tirx.PrimFunc:
    """The (single) device PrimFunc of a ``lower_device_ir`` result."""
    for _, func in device_mod.functions.items():
        if isinstance(func, tirx.PrimFunc):
            return func
    raise AssertionError("the lowered device module has no PrimFunc")


def find_calls(stmt: tirx.Stmt, op_names: str | tuple[str, ...]) -> list[tirx.Call]:
    """Every ``Call`` of the given op name(s) in ``stmt``, in post-order."""
    names = (op_names,) if isinstance(op_names, str) else tuple(op_names)
    found: list[tirx.Call] = []

    def visit(node):
        if isinstance(node, tirx.Call) and getattr(node.op, "name", None) in names:
            found.append(node)

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return found


def _const_ints(args) -> tuple[int, ...]:
    """The integer values of constant-only call arguments."""
    values = []
    for arg in args:
        value = getattr(arg, "value", None)
        if value is None:
            raise AssertionError(f"descriptor parameter is not a compile-time constant: {arg}")
        values.append(int(value))
    return tuple(values)


@dataclass(frozen=True)
class MmaDescriptors:
    """Descriptor parameters of one tcgen05 MMA, in operand order.

    ``data`` holds one ``(LBO, SBO, SW)`` triple per shared-memory operand (A
    first, then B); ``instr`` is the instruction descriptor
    ``(D_FMT, A_FMT, B_FMT, a_major, b_major, M, N, K)``; ``scale_instr`` is the
    block-scaled variant, which appends ``(scale_vec, scale_format)``.
    """

    data: tuple[tuple[int, int, int], ...] = ()
    instr: tuple[int, ...] | None = None
    scale_instr: tuple[int, ...] | None = None
    call: tirx.Call | None = field(default=None, compare=False)

    @property
    def a_major(self) -> int:
        """``a_m_major``: 0 = A is K-major, 1 = A is M-major."""
        assert self.instr is not None, "no tcgen05 MMA instruction descriptor found"
        return self.instr[3]

    @property
    def b_major(self) -> int:
        """``b_n_major``: 0 = B is K-major, 1 = B is N-major."""
        assert self.instr is not None, "no tcgen05 MMA instruction descriptor found"
        return self.instr[4]

    @property
    def sub_m(self) -> int:
        """M of one MMA issue (the per-warp M tile)."""
        assert self.instr is not None, "no tcgen05 MMA instruction descriptor found"
        return self.instr[5]

    @property
    def sub_n(self) -> int:
        """N of one MMA issue (the per-warp N tile)."""
        assert self.instr is not None, "no tcgen05 MMA instruction descriptor found"
        return self.instr[6]

    def partition(self, full_m: int = 128, full_n: int = 128) -> tuple[int, int]:
        """The warp partition ``(warp_m, warp_n)`` implied by subM/subN."""
        return full_m // self.sub_m, full_n // self.sub_n


def mma_descriptors(
    kernel: tirx.PrimFunc | IRModule,
    target: str | dict[str, object] | Target = STCUV2_TARGET,
    index: int = 0,
) -> MmaDescriptors:
    """The descriptors consumed by the ``index``-th tcgen05 MMA of ``kernel``.

    ``kernel`` may be a PrimFunc (it is lowered here) or an already lowered
    device IRModule.  Raises ``AssertionError`` if the kernel emits no tcgen05
    MMA at all -- a lowering regression that would otherwise silently pass a
    test asserting on a stale source string.
    """
    device_mod = kernel if isinstance(kernel, IRModule) else lower_device_ir(kernel, target)
    calls = find_calls(device_func(device_mod).body, MMA_OPS)
    assert calls, f"no tcgen05 MMA ({', '.join(MMA_OPS)}) in the lowered device IR"
    call = calls[index]

    data: list[tuple[int, int, int]] = []
    instr: tuple[int, ...] | None = None
    scale_instr: tuple[int, ...] | None = None
    for arg in call.args:
        if not isinstance(arg, tirx.Call):
            continue
        name = getattr(arg.op, "name", None)
        if name == DATA_DESC_OP:
            # args: [ptr, LBO, SBO, SW] or [ptr, offset, LBO, SBO, SW]
            data.append(_const_ints(arg.args[-3:]))
        elif name == INSTR_DESC_OP:
            instr = _const_ints(arg.args)
        elif name == SCALE_INSTR_DESC_OP:
            scale_instr = _const_ints(arg.args)

    return MmaDescriptors(data=tuple(data), instr=instr, scale_instr=scale_instr, call=call)


__all__ = [
    "STCU_TARGET",
    "STCUV2_TARGET",
    "DATA_DESC_OP",
    "INSTR_DESC_OP",
    "SCALE_INSTR_DESC_OP",
    "MMA_OPS",
    "MmaDescriptors",
    "device_func",
    "find_calls",
    "lower_device_ir",
    "mma_descriptors",
]
