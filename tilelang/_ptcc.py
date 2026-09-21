"""Internal PTCC selection, JIT defaults, and cache identity."""

from __future__ import annotations

import functools
import hashlib
import os
import platform
import shutil
import sys


_COMMON_OPTIMIZATION_FLAGS = ("-stpu-loop", "-use-load-const", "-O3")
_JIT_OPTIMIZATION_FLAGS = {
    "llvm20": ("-fstpu-warp-alu", *_COMMON_OPTIMIZATION_FLAGS),
    # Use the LLVM22 compatibility configuration.
    "llvm22": ("-fno-stpu-warp-alu", *_COMMON_OPTIMIZATION_FLAGS),
}


def default_jit_options(arch: str = "stcu") -> list[str]:
    """Select explicit JIT defaults, independent of PTCC's target defaults."""
    profile = os.environ.get("PTCC_JIT_PROFILE") or "llvm20"
    if profile not in _JIT_OPTIMIZATION_FLAGS:
        raise ValueError(f"Unknown PTCC_JIT_PROFILE={profile!r}; expected llvm20 or llvm22")
    return [
        "-xtang",
        "-std=c++17",
        "-DTANG",
        *_JIT_OPTIMIZATION_FLAGS[profile],
        "-c",
        "--tang-device-only",
        f"--tang-gpu-arch={arch}",
    ]


def resolve_ptcc(tang_home: str) -> str:
    configured = os.environ.get("PTCC_PATH")
    if configured:
        compiler = os.path.abspath(configured)
        if not os.path.isfile(compiler) or not os.access(compiler, os.X_OK):
            raise RuntimeError(f"PTCC_PATH={configured!r} must point to an executable file")
        return compiler
    compiler = shutil.which("ptcc")
    if compiler:
        return os.path.abspath(compiler)
    if tang_home:
        for relative in ("bin/ptcc", f"toolchains/llvm/prebuilt/linux-{platform.machine()}/bin/ptcc"):
            compiler = os.path.join(tang_home, relative)
            if os.path.isfile(compiler) and os.access(compiler, os.X_OK):
                return os.path.abspath(compiler)
    raise RuntimeError("Cannot find ptcc. Set PTCC_PATH, add ptcc to PATH, or configure a TANG toolkit root.")


@functools.cache
def compiler_identity(compiler: str) -> tuple[str, str]:
    """Fingerprint once per process; restart Python after replacing a toolchain."""
    compiler = os.path.realpath(compiler)
    digest = hashlib.sha256()
    with open(compiler, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return compiler, digest.hexdigest()


if __name__ == "__main__":
    print(resolve_ptcc(sys.argv[1]))
