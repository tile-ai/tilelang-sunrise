#!/usr/bin/env python3
"""Compile representative STCU GEMM template instantiations without running a GPU."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def translation_unit():
    lines = [
        "#include <tang.h>",
        "#include <stdint.h>",
        "#include <type_traits>",
        "#include <tl_templates/tang/common.h>",
        "#include <tl_templates/tang/gemm.h>",
    ]
    cases = []
    for dtype, accumulator in [("__fp16", "float"), ("__bf16", "float"), ("float", "float"), ("int8_t", "int32_t")]:
        for size in (32, 64):
            for transpose_a in (False, True):
                for transpose_b in (False, True):
                    for clear in (False, True):
                        name = f"gemm_header_check_{len(cases)}"
                        values = [size, size, size, 2, 2, size, size, 0, 0, int(transpose_a), int(transpose_b), int(clear)]
                        args = ", ".join(map(str, values))
                        lines.extend(
                            [
                                f'extern "C" __global__ void {name}({dtype}* a, {dtype}* b, {accumulator}* c) {{',
                                f"  tl::gemm_tang<{args}>(a, b, c);",
                                "}",
                            ]
                        )
                        cases.append(
                            {
                                "dtype": dtype,
                                "accumulator": accumulator,
                                "size": size,
                                "transpose_a": transpose_a,
                                "transpose_b": transpose_b,
                                "clear_accum": clear,
                            }
                        )
    return "\n".join(lines) + "\n", cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-dir", type=Path, required=True)
    parser.add_argument("--ptcc", default=os.environ.get("PTCC_PATH"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.ptcc or not Path(args.ptcc).is_file():
        parser.error("Set PTCC_PATH or --ptcc to the configured compiler")
    include_dir = args.include_dir.resolve()
    header = include_dir / "tl_templates/tang/gemm_tmma.h"
    if not header.is_file():
        parser.error("GEMM header is missing from --include-dir")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source, cases = translation_unit()
    unit = output / "gemm_header.t"
    unit.write_text(source)
    compiler = str(Path(args.ptcc).resolve())
    command = [
        compiler,
        "-std=c++17",
        "--tang-gpu-arch=stcu",
        "--tang-device-only",
        "-c",
        "-O2",
        "-I",
        str(include_dir),
        str(unit),
        "-o",
        str(output / "gemm_header.o"),
    ]
    version = subprocess.run([compiler, "--version"], capture_output=True, text=True, check=True)
    result = subprocess.run(command, capture_output=True, text=True)
    (output / "compile.log").write_text(result.stdout + result.stderr)
    report = {
        "command": command,
        "compiler_version": version.stdout.strip(),
        "header_sha256": hashlib.sha256(header.read_bytes()).hexdigest(),
        "cases": cases,
        "exit_code": result.returncode,
        "scope": "device_compilation_only_no_gpu_execution",
    }
    artifact = output / "gemm_header.o"
    if result.returncode == 0 and artifact.is_file() and artifact.stat().st_size:
        report["object_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    else:
        report["exit_code"] = result.returncode or 1
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"GEMM header compilation: {len(cases)} instantiations, exit={report['exit_code']}")
    if report["exit_code"]:
        print(result.stdout + result.stderr)
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
