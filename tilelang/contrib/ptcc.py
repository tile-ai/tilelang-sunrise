# pylint: disable=invalid-name
"""Utilities for invoking the TANG ``ptcc`` compiler."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

from tilelang.env import TANG_HOME, env
from tvm.base import py_str


_kernel_compile_counter = 0


def compile_tang(code, options=None, path_target=None, verbose=False):
    """Compile TANG source code to a device object with ``ptcc``.

    Parameters
    ----------
    code : str
        TANG source code.
    options : str or list[str], optional
        Additional compiler options.
    path_target : str, optional
        Explicit output path. Otherwise an automatically cleaned temporary
        directory below ``TILELANG_TMP_DIR`` is used.
    verbose : bool
        Print compiler output when true.

    Returns
    -------
    bytearray
        Compiled device object bytes.
    """
    temp_root = env.TILELANG_TMP_DIR
    os.makedirs(temp_root, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ptcc-", dir=temp_root) as temp_dir:
        kernels_output_name = os.environ.get("TL_KERNEL_NAME")
        base_file_name = kernels_output_name or "tvm_kernels"

        global _kernel_compile_counter
        if kernels_output_name:
            _kernel_compile_counter += 1
            file_name = f"{base_file_name}_{_kernel_compile_counter:02d}"
        else:
            file_name = base_file_name

        temp_code = os.path.join(temp_dir, f"{file_name}.t")
        temp_target = os.path.join(temp_dir, f"{file_name}.o")

        kernels_output_dir = os.environ.get("TL_KERNEL_OUTPUT_DIR")
        if kernels_output_dir is not None:
            os.makedirs(kernels_output_dir, exist_ok=True)
            temp_code = os.path.join(kernels_output_dir, f"{file_name}.t")
            temp_target = os.path.join(kernels_output_dir, f"{file_name}.o")

        with open(temp_code, "w", encoding="utf-8") as out_file:
            out_file.write(code)

        file_target = path_target or temp_target
        cmd = [get_ptcc_compiler()]
        if options:
            if isinstance(options, str):
                cmd.append(options)
            elif isinstance(options, list):
                cmd.extend(options)
            else:
                raise ValueError("options must be str or list of str")
        cmd.extend(["-o", file_target, temp_code])

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out, _ = proc.communicate()
        if verbose:
            print(py_str(out))
        if proc.returncode != 0:
            raise RuntimeError(f"{code}\nCompilation error:\n{py_str(out)}\nCommand: {' '.join(cmd)}\n")

        with open(file_target, "rb") as target_file:
            data = bytearray(target_file.read())
        if not data:
            raise RuntimeError("Compilation error: empty result is generated")
        return data


def find_tang_path() -> str:
    """Return the detected TANG toolkit root."""
    if TANG_HOME:
        return TANG_HOME
    raise RuntimeError("Cannot find a TANG installation. Set TANG_HOME or TANG_PATH to the toolkit root.")


def get_ptcc_compiler() -> str:
    """Return the ``ptcc`` executable path."""
    compiler = shutil.which("ptcc")
    if compiler:
        return compiler
    if TANG_HOME:
        candidate = os.path.join(TANG_HOME, "bin", "ptcc")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError("Cannot find ptcc. Add it to PATH or set TANG_HOME to a toolkit containing bin/ptcc.")
