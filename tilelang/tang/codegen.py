from __future__ import annotations

import os
import shlex

from tilelang.backend.device_codegen import DeviceCodegen, global_func_device_codegen
from tilelang.contrib import ptcc
from tilelang._ptcc import default_jit_options
from tilelang.env import TANG_HOME, TILELANG_TEMPLATE_PATH
from tilelang.transform import PassConfigKey


def tilelang_callback_tang_compile(code, target, pass_config=None):
    config = pass_config or {}
    arch = str(target.attrs.get("arch", "stcu"))
    options = default_jit_options(arch) + [
        "-Wall",
        "-Wno-parentheses-equality",
        "-Wno-deprecated-declarations",
        f"-I{TILELANG_TEMPLATE_PATH}",
    ]
    if TANG_HOME:
        options.append(f"-I{os.path.join(TANG_HOME, 'include')}")
    if arch == "stcuv2":
        options.append("-DTANG_STCUV2")
    if bool(config.get(PassConfigKey.TL_ENABLE_FAST_MATH, False)):
        options.append("-ffast-math")
    if bool(config.get(PassConfigKey.TL_TANG_DISABLE_WARP_ALU, False)):
        options = [option for option in options if option not in ("-fstpu-warp-alu", "-fno-stpu-warp-alu")]
        options.append("-fno-stpu-warp-alu")

    extra_flags = config.get(PassConfigKey.TL_DEVICE_COMPILE_FLAGS, None)
    if extra_flags:
        flags = [extra_flags] if isinstance(extra_flags, str) else extra_flags
        tokens = [token for flag in flags for token in shlex.split(str(flag))]
        if any(token.startswith("-O") for token in tokens):
            options = [option for option in options if not option.startswith("-O")]
        options.extend(tokens)
    return ptcc.compile_tang(code, options=options, verbose=True)


DEVICE_CODEGEN = DeviceCodegen(
    "tang",
    build=global_func_device_codegen("target.build.tilelang_tang"),
    build_without_compile=global_func_device_codegen("target.build.tilelang_tang_without_compile"),
)
