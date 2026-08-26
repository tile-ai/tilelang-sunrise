from __future__ import annotations

import warnings

from tvm import tirx

import tilelang.language as T
from tilelang.language.eager.builder import Builder, macro
from tilelang.utils.device import is_device_assert_supported

_IS_DEVICE_ASSERT_SUPPORTED = is_device_assert_supported()


def get_stack_str(msg, stacklevel=1):
    stack = Builder.current().get_fileline_stack(stacklevel)
    msg = msg + "\n"
    for fileline, lineno, macro_name in stack:
        msg += f"  at {fileline}:{lineno} in {macro_name}\n"
    return msg


@macro
def device_assert(condition: tirx.PrimExpr, msg: str = "", no_stack_info=False):
    """
    Device-side assert emulation for targets that lower ``tl.device_assert``.

    ``tl.device_assert`` / ``tl.device_assert_with_msg`` are backend-neutral
    intrinsics lowered by both the CUDA and TANG codegens; this macro only
    degrades to a no-op when no such backend is present on the host.
    """
    if _IS_DEVICE_ASSERT_SUPPORTED:
        if no_stack_info:
            if msg == "":
                T.call_intrin("void", tirx.op.Op.get("tl.device_assert"), condition)
            else:
                warnings.warn("Non-empty msg may slightly slow down the kernel", stacklevel=2)
                T.call_intrin("void", tirx.op.Op.get("tl.device_assert_with_msg"), condition, msg)
        else:
            T.call_intrin("void", tirx.op.Op.get("tl.device_assert_with_msg"), condition, get_stack_str(msg, stacklevel=2))
