from tilelang.backend.execution_backend import ExecutionBackendSpec, register_execution_backend
from tilelang.tang.target import target_is_stcuv2


def _is_simulator_auto_selectable() -> bool:
    """Defer the adapter import until TileLang package initialization finishes."""
    from tilelang.jit.adapter.simulator import _is_simulator_enabled

    return _is_simulator_enabled()


register_execution_backend(
    "tang",
    ExecutionBackendSpec(
        "simulator",
        auto_selectable=_is_simulator_auto_selectable,
        supports_target=target_is_stcuv2,
    ),
    override=True,
)
register_execution_backend(
    "tang",
    ExecutionBackendSpec("tvm_ffi", enable_host_codegen=True, enable_device_compile=True),
    override=True,
)
register_execution_backend("tang", ExecutionBackendSpec("cython"), override=True)
