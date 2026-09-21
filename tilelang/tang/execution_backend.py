from tilelang.backend.execution_backend import ExecutionBackendSpec
from tilelang.tang.target import target_is_stcuv2


def _is_simulator_auto_selectable() -> bool:
    """Defer the adapter import until TileLang package initialization finishes."""
    from tilelang.jit.adapter.simulator import _is_simulator_enabled

    return _is_simulator_enabled()


EXECUTION_BACKENDS = (
    ExecutionBackendSpec(
        "simulator",
        auto_selectable=_is_simulator_auto_selectable,
        supports_target=target_is_stcuv2,
    ),
    ExecutionBackendSpec("tvm_ffi", enable_host_codegen=True, enable_device_compile=True),
    ExecutionBackendSpec("cython"),
)
