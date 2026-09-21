from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tilelang.contrib import ptcc
from tilelang.env import TANG_HOME, TILELANG_TEMPLATE_PATH
from tilelang.tang import codegen
from tilelang.transform import PassConfigKey


def test_tang_codegen_uses_toolkit_and_user_flags(monkeypatch):
    captured = {}

    def fake_compile(source, options, verbose):
        captured.update(source=source, options=options, verbose=verbose)
        return bytearray(b"object")

    monkeypatch.setattr(codegen.ptcc, "compile_tang", fake_compile)
    result = codegen.tilelang_callback_tang_compile(
        "kernel",
        SimpleNamespace(attrs={"arch": "stcuv2"}),
        {
            PassConfigKey.TL_TANG_DISABLE_WARP_ALU: True,
            PassConfigKey.TL_DEVICE_COMPILE_FLAGS: "-O1 -g",
        },
    )

    assert result == bytearray(b"object")
    assert captured["source"] == "kernel"
    assert "-DTANG_STCUV2" in captured["options"]
    assert "--tang-gpu-arch=stcuv2" in captured["options"]
    assert "-fstpu-warp-alu" not in captured["options"]
    assert "-fno-stpu-warp-alu" in captured["options"]
    assert "-O3" not in captured["options"]
    assert "-O1" in captured["options"]
    assert f"-I{TILELANG_TEMPLATE_PATH}" in captured["options"]
    if TANG_HOME:
        assert f"-I{os.path.join(TANG_HOME, 'include')}" in captured["options"]


@pytest.mark.parametrize("profile", ["llvm20", "llvm22"])
def test_tang_codegen_passes_stcu_arch(monkeypatch, profile):
    monkeypatch.setenv("PTCC_JIT_PROFILE", profile)
    captured = {}

    def fake_compile(_source, options, **_kwargs):
        captured["options"] = options
        return bytearray(b"object")

    monkeypatch.setattr(codegen.ptcc, "compile_tang", fake_compile)
    codegen.tilelang_callback_tang_compile("kernel", SimpleNamespace(attrs={"arch": "stcu"}))

    assert "--tang-gpu-arch=stcu" in captured["options"]
    assert ("-fstpu-warp-alu" in captured["options"]) == (profile == "llvm20")
    assert ("-fno-stpu-warp-alu" in captured["options"]) == (profile == "llvm22")
    assert "-DTANG_STCUV2" not in captured["options"]


def test_ptcc_temporary_files_use_tilelang_tmp_dir(monkeypatch, tmp_path):
    temp_root = tmp_path / "tilelang-tmp"
    observed = {}

    class FakePopen:
        returncode = 0

        def __init__(self, cmd, **_kwargs):
            observed["cmd"] = cmd
            output = Path(cmd[cmd.index("-o") + 1])
            source = Path(cmd[-1])
            observed["source"] = source
            assert source.is_file()
            output.write_bytes(b"object")

        def communicate(self):
            return b"", None

    monkeypatch.setenv("TILELANG_TMP_DIR", str(temp_root))
    monkeypatch.delenv("TL_KERNEL_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("TL_KERNEL_NAME", raising=False)
    # Signature must track ptcc.get_ptcc_compiler(arch=None); compile_tang calls
    # it positionally with the arch parsed out of the options.
    monkeypatch.setattr(ptcc, "get_ptcc_compiler", lambda arch=None: "ptcc")
    monkeypatch.setattr(ptcc.subprocess, "Popen", FakePopen)

    assert ptcc.compile_tang("kernel") == bytearray(b"object")
    assert observed["source"].is_relative_to(temp_root)
    assert not observed["source"].parent.exists()


def test_ptcc_forwards_target_arch_to_compiler_lookup(monkeypatch, tmp_path):
    """compile_tang must hand the --tang-gpu-arch= value to get_ptcc_compiler.

    That argument is what selects the separately-shipped S3 toolchain via
    TANG_S3_PTCC_PATH. If it stops being forwarded, stcuv2 kernels are quietly
    built with whichever ptcc is on PATH, which lacks the S3 headers -- the
    symptom is a missing-include error that points nowhere near the compiler
    choice.
    """
    observed = {}

    class FakePopen:
        returncode = 0

        def __init__(self, cmd, **_kwargs):
            observed["compiler"] = cmd[0]
            Path(cmd[cmd.index("-o") + 1]).write_bytes(b"object")

        def communicate(self):
            return b"", None

    def fake_lookup(arch=None):
        observed["arch"] = arch
        return "ptcc-for-" + str(arch)

    monkeypatch.setenv("TILELANG_TMP_DIR", str(tmp_path / "tilelang-tmp"))
    monkeypatch.delenv("TL_KERNEL_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("TL_KERNEL_NAME", raising=False)
    monkeypatch.setattr(ptcc, "get_ptcc_compiler", fake_lookup)
    monkeypatch.setattr(ptcc.subprocess, "Popen", FakePopen)

    ptcc.compile_tang("kernel", options=["--tang-gpu-arch=stcuv2", "-O1"])
    assert observed["arch"] == "stcuv2"
    assert observed["compiler"] == "ptcc-for-stcuv2"

    observed.clear()
    ptcc.compile_tang("kernel")
    assert observed["arch"] is None
