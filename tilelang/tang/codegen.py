from __future__ import annotations

import os
import shlex

import tvm_ffi

from tilelang.backend.device_codegen import DeviceCodegen, global_func_device_codegen, register_device_codegen
from tilelang.contrib import ptcc
from tilelang.env import TANG_HOME, TILELANG_TEMPLATE_PATH
from tilelang.transform import PassConfigKey


@tvm_ffi.register_global_func("tilelang_callback_tang_compile", override=True)
def tilelang_callback_tang_compile(code, target, pass_config=None):
    config = pass_config or {}
    arch = str(target.attrs.get("arch", "stcu"))
    options = [
        "-xtang",
        "-Wall",
        "-Wno-parentheses-equality",
        "-Wno-deprecated-declarations",
        "-std=c++17",
        "-DTANG",
        "-fstpu-warp-alu",
        "-stpu-loop",
        "-use-load-const",
        "-O3",
        "-c",
        "--tang-device-only",
        f"--tang-gpu-arch={arch}",
        f"-I{TILELANG_TEMPLATE_PATH}",
    ]
    if TANG_HOME:
        options.append(f"-I{os.path.join(TANG_HOME, 'include')}")
    if arch == "stcuv2":
        options.append("-DTANG_STCUV2")
    if bool(config.get(PassConfigKey.TL_ENABLE_FAST_MATH, False)):
        options.append("-ffast-math")
    if bool(config.get(PassConfigKey.TL_TANG_DISABLE_WARP_ALU, False)):
        options.remove("-fstpu-warp-alu")

    extra_flags = config.get(PassConfigKey.TL_DEVICE_COMPILE_FLAGS, None)
    if extra_flags:
        flags = [extra_flags] if isinstance(extra_flags, str) else extra_flags
        tokens = [token for flag in flags for token in shlex.split(str(flag))]
        if any(token.startswith("-O") for token in tokens):
            options = [option for option in options if not option.startswith("-O")]
        options.extend(tokens)
    return ptcc.compile_tang(code, options=options, verbose=True)


register_device_codegen(
    "tang",
    DeviceCodegen(
        "tang",
        build=global_func_device_codegen("target.build.tilelang_tang"),
        build_without_compile=global_func_device_codegen("target.build.tilelang_tang_without_compile"),
    ),
    override=True,
)
