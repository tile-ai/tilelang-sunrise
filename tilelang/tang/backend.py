"""TANG backend manifest."""

from tilelang.backend.host_codegen import STANDARD_HOST_CODEGENS
from tilelang.backend.module import BackendModule, register_backend

from . import codegen, execution_backend, pipeline

BACKEND = register_backend(
    BackendModule(
        name="tang",
        target_kinds=("tang",),
        pipelines={"tang": pipeline.tang_pipeline},
        device_codegens={"tang": codegen.DEVICE_CODEGEN},
        execution_backends=execution_backend.EXECUTION_BACKENDS,
        host_codegens=STANDARD_HOST_CODEGENS,
        callbacks={"tilelang_callback_tang_compile": codegen.tilelang_callback_tang_compile},
    )
)
