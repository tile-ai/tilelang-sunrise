"""Simulator kernel adapter — runs TANG kernels through S3 ISS instead of hardware.

Activated either by passing ``execution_backend="simulator"`` to
:class:`~tilelang.jit.JITKernel` or by setting the environment variable
``TANG_SIMULATOR=1``.

The adapter follows this pipeline on each invocation:

    torch tensors
      → hbm*.in (hex-format input data files)
      → user_config.json (auto-generated from kernel params)
      → ptcc source.t → source.o
      → STCU_loader.py → simulator input files
      → stcu_rm (ISS simulation)
      → hbm*.iss.out → torch tensors (results)
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from collections.abc import Callable

import numpy as np
import torch
from tvm.target import Target

from tilelang.engine.param import KernelParam
from tilelang.jit.adapter.base import BaseKernelAdapter
from tilelang.transform import PassConfigKey
from tilelang.backend.target import determine_target
from tilelang.tang.target import target_is_stcuv2

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
#  Environment variables
# ──────────────────────────────────────────────────────────────────────
_ENV_SIMULATOR = "TANG_SIMULATOR"  # set to "1" to enable
_ENV_PTCC_PATH = "TANG_S3_PTCC_PATH"  # path to the ptcc compiler binary
_ENV_ISS_PATH = "TANG_S3_ISS_PATH"  # path to the stcu_rm ISS executable
_ENV_LOADER_PATH = "TANG_S3_LOADER_PATH"  # path to STCU_loader.py


def _is_simulator_enabled() -> bool:
    """Check whether the simulator backend should be used."""
    val = os.environ.get(_ENV_SIMULATOR, "0")
    return val.lower() in ("1", "true", "yes", "on")


def _simulator_work_root() -> Path:
    """Return a non-/tmp work root for simulator intermediates."""
    configured = os.environ.get("TANG_SIMULATOR_WORK_ROOT") or os.environ.get("TILELANG_CACHE_DIR")
    root = Path(configured).expanduser() if configured else Path.home() / ".tilelang" / "cache"
    root = root / "simulator"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _ptcc_path() -> Path:
    """Locate the ptcc compiler binary from the ``TANG_S3_PTCC_PATH`` env var."""
    env = os.environ.get(_ENV_PTCC_PATH)
    if not env:
        raise FileNotFoundError(
            f"{_ENV_PTCC_PATH} is not set. Point it at the ptcc binary, e.g. "
            "`export TANG_S3_PTCC_PATH=/path/to/ptcc` "
            "(this is exported by env_s3_*.sh)."
        )
    p = Path(env)
    if not p.is_file():
        raise FileNotFoundError(f"{_ENV_PTCC_PATH}='{env}' does not point to an existing file.")
    return p.resolve()


def _loader_path() -> Path:
    """Locate STCU_loader.py from the ``TANG_S3_LOADER_PATH`` env var."""
    env = os.environ.get(_ENV_LOADER_PATH)
    if not env:
        raise FileNotFoundError(
            f"{_ENV_LOADER_PATH} is not set. Point it at STCU_loader.py, e.g. "
            "`export TANG_S3_LOADER_PATH=/path/to/loader/STCU_loader.py` "
            "(this is exported by env_s3_*.sh)."
        )
    p = Path(env)
    if not p.is_file():
        raise FileNotFoundError(f"{_ENV_LOADER_PATH}='{env}' does not point to an existing file.")
    return p.resolve()


def _iss_path() -> Path:
    """Locate the ``stcu_rm`` ISS executable from the ``TANG_S3_ISS_PATH`` env var."""
    env = os.environ.get(_ENV_ISS_PATH)
    if not env:
        raise FileNotFoundError(
            f"{_ENV_ISS_PATH} is not set. Point it at the stcu_rm binary, e.g. "
            "`export TANG_S3_ISS_PATH=/path/to/stcu_rm` "
            "(this is exported by env_s3_*.sh)."
        )
    p = Path(env)
    if not p.is_file():
        raise FileNotFoundError(f"{_ENV_ISS_PATH}='{env}' does not point to an existing file.")
    return p.resolve()


# ──────────────────────────────────────────────────────────────────────
#  Data format helpers
# ──────────────────────────────────────────────────────────────────────

# torch dtypes that map directly to a numpy dtype of the same byte width.
# bfloat16 is handled separately (numpy has no native bf16).
_TORCH_TO_NUMPY = {
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.float64: np.float64,
    torch.int64: np.int64,
    torch.int32: np.int32,
    torch.int16: np.int16,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
    torch.bool: np.bool_,
}


# fp8 dtypes have no numpy equivalent; reinterpret as uint8 (same 1-byte layout).
_TORCH_FP8 = tuple(d for d in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz") if hasattr(torch, d))


# Sub-byte float types (fp4/fp6) are physically packed into 1-byte storage units
# (e.g. float4_e2m1fn_x2 stores two fp4 per byte). torch has no numpy equivalent
# and the device kernel treats these buffers as raw ``uchar`` bytes, so we
# reinterpret them as uint8 (same 1-byte storage layout) for the simulator.
_TORCH_SUBBYTE = tuple(
    d
    for d in ("float4_e2m1fn_x2", "float4_e2m1fn", "float6_e2m3fn", "float6_e3m2fn", "float6_e2m3fn_x2", "float6_e3m2fn_x2")
    if hasattr(torch, d)
)


def _is_byte_view_dtype(dtype: torch.dtype) -> bool:
    """Whether ``dtype`` should be reinterpreted as uint8 for the simulator."""
    if dtype in (getattr(torch, n) for n in _TORCH_FP8):
        return True
    if dtype in (getattr(torch, n) for n in _TORCH_SUBBYTE):
        return True
    # Generic fallback: any unsupported 1-byte-storage dtype -> uint8.
    if dtype not in _TORCH_TO_NUMPY and dtype != torch.bfloat16:
        try:
            return torch.empty(0, dtype=dtype).element_size() == 1
        except Exception:  # pragma: no cover - defensive
            return False
    return False


def _tensor_raw_bytes(tensor: torch.Tensor) -> bytes:
    """Return the little-endian raw byte stream of a tensor's elements.

    bfloat16 has no numpy equivalent, so we reinterpret it as int16 first
    (same 2-byte little-endian layout); fp8 dtypes are reinterpreted as uint8.
    """
    t = tensor.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().tobytes()
    if _is_byte_view_dtype(t.dtype):
        return t.view(torch.uint8).numpy().tobytes()
    if t.dtype not in _TORCH_TO_NUMPY:
        raise TypeError(
            f"Simulator backend does not support dtype {t.dtype} yet. Supported: {sorted(str(d) for d in _TORCH_TO_NUMPY)} + bfloat16."
        )
    return t.numpy().tobytes()


def _tensor_to_hex_lines(tensor: torch.Tensor) -> list[str]:
    """Convert a torch tensor of any supported dtype to big-endian word hex.

    The HBM ``.in`` format is a sequence of 32-bit words, one per line, each
    written as 8 big-endian hex chars (see ``docs/tang_simulator_plugin.md``).
    HBM stores data little-endian, so we take the element byte stream (LE),
    pad to a multiple of 4 bytes, and reverse each 4-byte word to obtain its
    big-endian representation. For ``float32`` this is exactly
    ``struct.pack(">f", val)`` per element; for narrower dtypes it correctly
    packs multiple elements into one 32-bit word.
    """
    raw = bytearray(_tensor_raw_bytes(tensor))
    if len(raw) % 4:
        raw.extend(b"\x00" * (4 - (len(raw) % 4)))
    lines: list[str] = []
    for i in range(0, len(raw), 4):
        lines.append(raw[i : i + 4][::-1].hex())  # LE word → BE hex
    return lines


def _hex_lines_to_tensor(lines: list[str], dtype: torch.dtype = torch.float32, numel: int | None = None) -> torch.Tensor:
    """Parse big-endian word hex lines back into a flat torch tensor.

    Inverse of :func:`_tensor_to_hex_lines`: each 8-char line is a big-endian
    32-bit word; we reverse it back to the little-endian byte stream and
    reinterpret as ``dtype``. ``numel`` (if given) truncates trailing padding
    words so narrow/unaligned outputs decode to the exact element count.
    """
    raw = bytearray()
    for line in lines:
        raw.extend(bytes.fromhex(line.strip())[::-1])  # BE word → LE bytes

    if dtype == torch.bfloat16:
        arr = np.frombuffer(bytes(raw), dtype=np.int16).copy()
        flat = torch.from_numpy(arr).view(torch.bfloat16)
    elif _is_byte_view_dtype(dtype):
        arr = np.frombuffer(bytes(raw), dtype=np.uint8).copy()
        flat = torch.from_numpy(arr).view(dtype)
    else:
        if dtype not in _TORCH_TO_NUMPY:
            raise TypeError(f"Simulator backend cannot decode dtype {dtype} yet.")
        item = torch.empty(0, dtype=dtype).element_size()
        usable = (len(raw) // item) * item
        arr = np.frombuffer(bytes(raw[:usable]), dtype=_TORCH_TO_NUMPY[dtype]).copy()
        flat = torch.from_numpy(arr)

    if numel is not None and flat.numel() >= numel:
        flat = flat[:numel]
    return flat


def _align4(size: int) -> int:
    """Round up to the nearest multiple of 4."""
    return (size + 3) // 4 * 4


# ──────────────────────────────────────────────────────────────────────
#  user_config.json generator
# ──────────────────────────────────────────────────────────────────────


def _extract_launch_dims(
    device_mod,
    kernel_source: str = "",
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Derive (thread_dims, block_dims) for the kernel launch.

    The simulator's ``user_config.json`` needs the real ``threadIdx`` extents
    (``thread_x/y/z``) and ``blockIdx`` extents (``block_x/y/z``). These come
    from the lowered device module's ``thread_extent`` attribute (the same
    source ``wrapper.py`` uses for the host launcher).

    Falls back to parsing ``__launch_bounds__(N, ...)`` from the generated
    source for the thread count (grid defaults to 1) when the module is
    unavailable.
    """
    thread = [1, 1, 1]
    block = [1, 1, 1]

    found = False
    if device_mod is not None:
        try:
            for _gv, func in device_mod.functions.items():
                attrs = func.attrs or {}
                thread_extent = attrs.get("thread_extent", None) if attrs else None
                if not thread_extent:
                    continue
                for tag, extent in thread_extent.items():
                    idx = "xyz".index(str(tag)[-1])
                    if "threadIdx" in str(tag):
                        thread[idx] = int(extent)
                    elif "blockIdx" in str(tag):
                        block[idx] = int(extent)
                found = True
                break  # simulator handles a single device kernel
        except Exception:  # pragma: no cover - defensive
            found = False

    if not found and kernel_source:
        m = re.search(r"__launch_bounds__\((\d+)", kernel_source)
        if m:
            thread = [int(m.group(1)), 1, 1]

    return tuple(thread), tuple(block)


def _estimate_tcgen5_shared_mem(kernel_source: str) -> int:
    """Parse tcgen5 GEMM template parameters from source and estimate shared mem.

    The generated kernel contains a call like:
      tl::gemm_tang_tcgen5<BM, BN, BK, ...>(...)

    Estimated shared memory = A_shared(BM*BK*2) + B_shared(BK*BN*2) +
                              C_shared(BM*BN*2) + alignment(~4KB).
    """
    import re

    m = re.search(r"gemm_tang_tcgen5<(\d+),\s*(\d+),\s*(\d+)", kernel_source)
    if not m:
        return 0
    BM, BN, BK = int(m.group(1)), int(m.group(2)), int(m.group(3))

    def _shared(rows, cols, elem_bytes):
        # Align each allocation to 128 bytes
        sz = rows * cols * elem_bytes
        return (sz + 127) // 128 * 128

    total = (
        _shared(BM, BK, 2)  # A_shared: float16
        + _shared(BK, BN, 2)  # B_shared: float16
        + _shared(BM, BN, 2)
    )  # C_shared: float16
    return total


def _compute_device_param_order(func_or_mod, device_mod, n_params: int) -> list[int]:
    """Map GM argument slots to host parameter indices.

    The ISS loader assigns global-memory base addresses to kernel arguments in
    the order they appear in ``user_config.json`` (slot ``n`` -> GM
    ``0x(n+1)00000``), and the compiled device kernel binds its ``n``-th
    signature parameter to that same slot. The device kernel's parameter order,
    however, is **not** guaranteed to match the host ``prim_func`` parameter
    order: lowering may reorder them (e.g. a scaled GEMM yields a device
    signature ``(A, B, C, SFA, SFB)`` while the ``prim_func`` declares
    ``(A, B, SFA, SFB, C)``).

    This helper returns ``slot_to_host``: a list where ``slot_to_host[n]`` is the
    host parameter index whose data must be placed in GM slot ``n`` so the device
    kernel reads the correct buffer. On any failure (or when the orders already
    match) it returns the identity permutation, leaving existing kernels
    unaffected.
    """
    identity = list(range(n_params))
    try:
        host_func = func_or_mod
        if not hasattr(host_func, "buffer_map"):
            # IRModule -> pick its (single) function
            host_func = next(iter(host_func.functions.values()))
        bmap = dict(host_func.buffer_map) if host_func.buffer_map else {}
        host_names = [(bmap[p].name if p in bmap else p.name) for p in host_func.params]

        dev_names: list[str] | None = None
        if device_mod is not None:
            for _gv, f in device_mod.functions.items():
                attrs = f.attrs or {}
                is_kernel = bool(attrs.get("tir.is_global_func", None)) or bool(attrs.get("thread_extent", None))
                if is_kernel:
                    dev_names = [p.name for p in f.params]
                    break
        if not dev_names:
            return identity

        # Keep only device params that correspond to real host params (drop any
        # synthetic trailing params such as dynamic-shared-memory handles).
        slot_to_host = [host_names.index(dn) for dn in dev_names if dn in host_names]
        if sorted(slot_to_host) != list(range(n_params)):
            return identity
        return slot_to_host
    except Exception:  # pragma: no cover - defensive
        return identity


def _build_user_config(
    params: list[KernelParam],
    result_idx: list[int],
    n_inputs: int,
    thread_dims: tuple[int, int, int] = (32, 1, 1),
    block_dims: tuple[int, int, int] = (1, 1, 1),
    blk_split_mode: int = 0,
    shared_mem_size: int = 0,
    slot_to_host: list[int] | None = None,
) -> dict:
    """Build a user_config.json dict from kernel parameter info.

    Parameters
    ----------
    params : list[KernelParam]
        Kernel parameters from ``CompiledArtifact.params``.
    result_idx : list[int]
        Indices of output (dump) parameters.
    n_inputs : int
        Number of actual input tensors (excludes outputs).
    thread_dims : tuple[int, int, int]
        Threads per block as (x, y, z) — the kernel's ``threadIdx`` extents.
    block_dims : tuple[int, int, int]
        Grid size as (x, y, z) — the kernel's ``blockIdx`` extents.
    blk_split_mode : int
        Block split mode (default 0).
    shared_mem_size : int
        Dynamic shared memory size in bytes (default 0).
        Used to generate captured_kernel.json for STCU_loader.
    """
    argument: list[dict] = []
    in_idx = 0  # global-memory slot / file counter

    if slot_to_host is None:
        slot_to_host = list(range(len(params)))

    # Iterate in device-kernel signature order so the GM base the loader assigns
    # to slot ``n`` matches the buffer the device kernel reads as its n-th param.
    for slot, host_idx in enumerate(slot_to_host):
        p = params[host_idx]
        is_output = host_idx in result_idx
        is_input = host_idx < n_inputs

        # Compute size in bytes (4-byte aligned). Use bit-width arithmetic so
        # sub-byte element types (fp4=4b, fp6=6b) are sized correctly: integer
        # ``bits // 8`` truncates them to 0, which would collapse the loader's
        # cumulative GM base-address assignment and alias distinct operands.
        elems = 1
        for d in p.shape:
            if isinstance(d, int):
                elems *= d
        raw_size = (elems * p.dtype.bits + 7) // 8
        size = _align4(raw_size)

        entry: dict[str, Any] = {
            "name": f"arg{slot}",
            "size": size,
            "format": "hex",
        }

        if is_input:
            entry["input_file_name"] = f"hbm{in_idx}.in"
            entry["initial"] = True
            entry["dump"] = False
            in_idx += 1
        elif is_output:
            entry["output_file_name"] = f"hbm{in_idx}.iss.out"
            entry["initial"] = False
            entry["dump"] = True
            in_idx += 1
        else:
            # Scalar / variable parameter — skip for now
            continue

        argument.append(entry)

    return {
        "argument": argument,
        "thread": {
            "thread_x": thread_dims[0],
            "thread_y": thread_dims[1],
            "thread_z": thread_dims[2],
        },
        "block": {
            "block_x": block_dims[0],
            "block_y": block_dims[1],
            "block_z": block_dims[2],
        },
        "blk_split_mode": blk_split_mode,
    }


# ──────────────────────────────────────────────────────────────────────
#  SimulatorKernelAdapter
# ──────────────────────────────────────────────────────────────────────


class SimulatorKernelAdapter(BaseKernelAdapter):
    """Kernel adapter that runs TANG kernels through the S3 ISS simulator.

    Parameters
    ----------
    params : list[KernelParam]
        Kernel parameter descriptors (dtype, shape).
    result_idx : list[int]
        Indices of output tensors.
    target : str | Target
        Compilation target (must be a TANG target).
    func_or_mod :
        The original TIR function or module (used for kernel source).
    device_kernel_source : str
        The generated ``.t`` source code of the kernel.
    verbose : bool
        Enable verbose logging.
    """

    def __init__(
        self,
        params: list[KernelParam],
        result_idx: list[int],
        target: str | Target,
        func_or_mod=None,
        device_kernel_source: str | None = None,
        verbose: bool = False,
        sim_build_dir: str | None = None,
        device_mod=None,
        pass_configs: dict[str, Any] | None = None,
    ):
        self.params = params
        self.result_idx = self._legalize_result_idx(result_idx)
        self.kernel_source = device_kernel_source or ""
        self.pass_configs = pass_configs or {}
        self.target = Target.canon_target(determine_target(target))
        # The ISS simulator backend is only supported on the STCUV2 subtarget.
        # Guard here (in addition to backend resolution) so that any direct
        # construction with an unsupported target fails fast with a clear error.
        if not target_is_stcuv2(self.target):
            raise ValueError(
                "The simulator execution backend is only supported on the "
                f"TANG 'stcuv2' subtarget, but got target '{self.target}'. "
                "Use a target such as `tang -arch=stcuv2`."
            )
        self.verbose = verbose
        self.sim_build_dir = sim_build_dir  # dir with pre-compiled .t/.o from callback

        # Real launch configuration (threadIdx / blockIdx extents) extracted
        # from the lowered device module — required so the simulator launches
        # the correct number of threads/blocks instead of a hardcoded default.
        self.thread_dims, self.block_dims = _extract_launch_dims(device_mod, self.kernel_source)

        # The device kernel may reorder its parameters relative to the host
        # prim_func (e.g. scaled GEMM). The ISS loader assigns GM bases by slot
        # order, so map each slot to the host parameter it must carry.
        self.device_param_order = _compute_device_param_order(func_or_mod, device_mod, len(self.params))

        # Cache compiled object path (None = not yet compiled)
        self._o_path: Path | None = None

        self._post_init()

    def _convert_torch_func(self) -> Callable[..., Any]:
        params = self.params
        result_idx = self.result_idx
        n_outputs = len(result_idx)
        n_inputs = len(params) - n_outputs
        kernel_source = self.kernel_source
        verbose = self.verbose
        sim_build_dir = self.sim_build_dir
        pass_configs = self.pass_configs
        thread_dims = self.thread_dims
        block_dims = self.block_dims
        slot_to_host = self.device_param_order

        # Pre-compute dtype/shape info for output creation
        param_dtypes = [p.torch_dtype() for p in params]
        param_shapes: list[list[int]] = []
        for p in params:
            shape = [int(d) for d in p.shape if isinstance(d, (int))]
            param_shapes.append(shape)

        # If we have a pre-compiled .o from the callback, use it directly
        _prebuilt_o = None
        _prebuilt_src = None
        if sim_build_dir:
            _prebuilt_o = Path(sim_build_dir) / "sim_kernel_01.o"
            _prebuilt_src = Path(sim_build_dir) / "sim_kernel_01.t"
            if not _prebuilt_o.exists():
                # Try alternative naming (without counter suffix)
                _prebuilt_o = Path(sim_build_dir) / "sim_kernel.o"
                _prebuilt_src = Path(sim_build_dir) / "sim_kernel.t"
                if not _prebuilt_o.exists():
                    _prebuilt_o = None

        def func(*inputs: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
            # ── 1. Validate ────────────────────────────────────────
            if len(inputs) != n_inputs:
                raise ValueError(f"Expected {n_inputs} input tensors, got {len(inputs)}.")

            # ── 1b. Device handling ────────────────────────────────
            # The simulator must behave like real hardware from the caller's
            # point of view: kernel usage is identical except (possibly) the
            # `execution_backend` argument. On real hardware inputs live on the
            # `ptpu` device, so we *require* `ptpu` inputs here too, then copy
            # them to CPU for the ISS run and finally place the results back on
            # `ptpu` so the output device matches what real hardware returns.
            non_ptpu = [(idx, str(t.device)) for idx, t in enumerate(inputs) if t.device.type != "ptpu"]
            if non_ptpu:
                details = ", ".join(f"arg{i} on '{d}'" for i, d in non_ptpu)
                raise ValueError(
                    "Simulator inputs must be on the 'ptpu' device to mirror "
                    "real-hardware usage (move tensors with `.ptpu()`); got: "
                    f"{details}."
                )
            input_devices = {t.device for t in inputs}
            if len(input_devices) > 1:
                raise ValueError(f"All simulator inputs must be on the same device, got {sorted(str(d) for d in input_devices)}.")
            result_device = next(iter(input_devices)) if input_devices else torch.device("ptpu")
            if verbose:
                logger.info("Simulator inputs on device '%s'; running on CPU and returning results on '%s'", result_device, result_device)
            inputs = tuple(t.detach().cpu() for t in inputs)

            # ── 2. Create working directory ─────────────────────────
            _persist_dir = os.environ.get("TANG_SIM_BUILD_DIR", "")
            if _persist_dir:
                work_dir = Path(_persist_dir)
                work_dir.mkdir(parents=True, exist_ok=True)
                _cleanup = False
            else:
                work_dir = Path(tempfile.mkdtemp(prefix="tilelang_sim_", dir=_simulator_work_root()))
                _cleanup = True
            try:
                # ── 3. Get compiled .o (from pre-built or compile) ──
                if _prebuilt_o and _prebuilt_o.exists():
                    o_path = _prebuilt_o
                    if verbose:
                        logger.info("Using pre-compiled kernel: %s", o_path)
                else:
                    # Write source and compile with ptcc
                    src_path = work_dir / "source.t"
                    src_path.write_text(kernel_source)
                    o_path = _compile_with_ptcc(src_path, work_dir, verbose, pass_configs=pass_configs)

                # ── 5. Write input data files ───────────────────────
                # Walk slots in device-kernel order; the file index for each
                # slot must match the ``hbm{idx}`` name emitted in user_config.
                in_file_idx = 0
                for host_idx in slot_to_host:
                    is_output = host_idx in result_idx
                    if is_output:
                        in_file_idx += 1
                        continue
                    fname = f"hbm{in_file_idx}.in"
                    lines = _tensor_to_hex_lines(inputs[host_idx])
                    (work_dir / fname).write_text("\n".join(lines) + "\n")
                    in_file_idx += 1

                # ── 6. Build & write user_config.json ───────────────
                # Estimate dynamic shared memory from kernel source
                _shm = _estimate_tcgen5_shared_mem(kernel_source)
                if _shm > 0:
                    # Write captured_kernel.json for STCU_loader
                    _cap = {"selected_launch": {"shared": _shm}}
                    (work_dir / "captured_kernel.json").write_text(json.dumps(_cap))

                cfg = _build_user_config(
                    params,
                    result_idx,
                    n_inputs,
                    thread_dims=thread_dims,
                    block_dims=block_dims,
                    blk_split_mode=0,
                    shared_mem_size=_shm,
                    slot_to_host=slot_to_host,
                )
                cfg_path = work_dir / "user_config.json"
                cfg_path.write_text(json.dumps(cfg, indent=4))

                # ── 7. Run STCU_loader ──────────────────────────────
                _run_loader(o_path, cfg_path, work_dir, verbose)

                # ── 8. Run ISS simulator ────────────────────────────
                _run_iss(work_dir, verbose)

                # ── 9. Read output files ────────────────────────────
                # Walk slots in device order (matching the write/config loops)
                # and collect each output by its host parameter index, then emit
                # outputs in ascending host-parameter order.
                outputs_by_host: dict[int, torch.Tensor] = {}
                in_file_idx = 0
                for host_idx in slot_to_host:
                    if host_idx not in result_idx:
                        in_file_idx += 1
                        continue
                    out_name = f"hbm{in_file_idx}.iss.out"
                    out_path = work_dir / "input" / out_name
                    if out_path.exists():
                        lines = out_path.read_text().strip().splitlines()
                        _numel = 1
                        for d in param_shapes[host_idx]:
                            _numel *= d
                        tensor = _hex_lines_to_tensor(lines, param_dtypes[host_idx], numel=_numel)
                        # The simulator writes a flat HBM word stream; restore
                        # the parameter's N-D shape so callers don't need a
                        # manual reshape.
                        # numel mismatch: leave flat
                        with contextlib.suppress(RuntimeError):
                            tensor = tensor.reshape(param_shapes[host_idx])
                        outputs_by_host[host_idx] = tensor
                    else:
                        # Create empty tensor with expected shape
                        outputs_by_host[host_idx] = torch.empty(*param_shapes[host_idx], dtype=param_dtypes[host_idx])
                    in_file_idx += 1

                outputs: list[torch.Tensor] = [outputs_by_host[h] for h in sorted(outputs_by_host)]

            finally:
                if _cleanup:
                    shutil.rmtree(work_dir, ignore_errors=True)

            # Place results back on the device the inputs came from.
            outputs = [o.to(result_device) for o in outputs]

            if len(outputs) == 1:
                return outputs[0]
            return outputs

        return func


# ──────────────────────────────────────────────────────────────────────
#  Pipeline helpers (standalone functions for testability)
# ──────────────────────────────────────────────────────────────────────


def _compile_with_ptcc(src_path: Path, work_dir: Path, verbose: bool, pass_configs: dict[str, Any] | None = None) -> Path:
    """Compile a ``.t`` source file with ptcc, return path to ``.o``."""
    ptcc = _ptcc_path()
    o_path = work_dir / "source.o"

    # Resolve TileLang template include path (tl_templates/tang/*.h).
    # Wheel:  site-packages/tilelang/src/tl_templates/     → parents[2]/src
    # Editable: <repo>/tilelang/src/tl_templates/           → parents[3]/src
    _pkg = Path(__file__).resolve()
    _tl_source = _pkg.parents[2] / "src"  # wheel install
    if not (_tl_source / "tl_templates").is_dir():
        _tl_source = _pkg.parents[3] / "src"  # editable install

    # Use system TANG include paths (matching tilelang_callback_tang_compile)
    _tang_include = Path("/usr/local/tangrt/include")

    # Optimization level defaults to -O2, but a user-provided -O<x> in the
    # kernel's TL_DEVICE_COMPILE_FLAGS pass config takes precedence (mirrors
    # tilelang_callback_tang_compile in engine/lower.py). Any other tokens in
    # that config are forwarded to ptcc as extra flags.
    opt_flag = "-O2"
    extra_tokens: list[str] = []
    cfg = pass_configs or {}
    extra_flags = cfg.get(PassConfigKey.TL_DEVICE_COMPILE_FLAGS, None)
    if extra_flags:
        import shlex

        if isinstance(extra_flags, str):
            tokens = shlex.split(extra_flags)
        else:
            tokens = []
            for flag in extra_flags:
                if isinstance(flag, str):
                    tokens.extend(shlex.split(flag))
                else:
                    tokens.append(str(flag))
        for token in tokens:
            if token.startswith("-O") and len(token) > 2:
                opt_flag = token
            else:
                extra_tokens.append(token)

    cmd = [
        str(ptcc),
        "-x",
        "tang",
        opt_flag,
        "-Wall",
        "--tang-gpu-arch=stcuv2",
        "-DTANG",
        "-DTANG_STCUV2",
        "--tang-device-only",
        "-std=c++17",
        "-I",
        str(_tang_include),
        "-I",
        str(_tl_source),
        *extra_tokens,
        *os.environ.get("TL_PTCC_EXTRA", "").split(),
        "-c",
        str(src_path),
        "-o",
        str(o_path),
    ]

    if verbose:
        logger.info("ptcc: %s", " ".join(str(c) for c in cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    # NOTE: ptcc may report a non-zero exit code on failure, but it has also
    # been observed to return exit code 0 while still failing to compile (e.g.
    # "error: Unsupported A matrix swizzle layout."). Treat a missing output
    # file as a failure regardless of the return code, and always surface the
    # captured diagnostics so the real error is visible.
    if result.returncode != 0 or not o_path.exists():
        diagnostics = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
        raise RuntimeError(
            "ptcc compilation failed "
            f"(returncode={result.returncode}, "
            f"output_produced={o_path.exists()}):\n"
            f"command: {' '.join(str(c) for c in cmd)}\n"
            f"{diagnostics}"
        )

    if verbose:
        for line in result.stdout.splitlines():
            logger.info("ptcc: %s", line)
        for line in result.stderr.splitlines():
            logger.info("ptcc: %s", line)

    return o_path


def _run_loader(o_path: Path, cfg_path: Path, work_dir: Path, verbose: bool) -> None:
    """Run STCU_loader.py to generate simulator input files."""
    loader = _loader_path()
    input_dir = work_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    # Copy config and data files into input/
    shutil.copy2(str(cfg_path), str(input_dir / "user_config.json"))
    for f in work_dir.glob("hbm*.in"):
        shutil.copy2(str(f), str(input_dir / f.name))

    # Generate captured_kernel.json if the kernel uses dynamic shared memory
    # (extern __shared__). The STCU_loader reads this to determine the
    # dynamic shared memory size for kernel_cfg.in.
    _captured_json = work_dir / "captured_kernel.json"
    if _captured_json.exists():
        shutil.copy2(str(_captured_json), str(input_dir / "captured_kernel.json"))

    cmd = [
        "python3",
        str(loader),
        "--target",
        "s3",
        "--elf",
        str(o_path),
        "--output",
        str(input_dir),
        "--config",
        str(input_dir / "user_config.json"),
        "--align",
    ]
    if verbose:
        cmd.append("--debug")

    if verbose:
        logger.info("STCU_loader: %s", " ".join(str(c) for c in cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"STCU_loader failed:\n{result.stderr}")

    if verbose:
        for line in result.stdout.splitlines():
            logger.info("loader: %s", line)
        for line in result.stderr.splitlines():
            logger.info("loader: %s", line)


def _run_iss(work_dir: Path, verbose: bool) -> None:
    """Run the stcu_rm ISS simulator."""
    iss = _iss_path()
    input_dir = work_dir / "input"

    if not iss.exists():
        raise FileNotFoundError(f"ISS simulator not found: {iss}")

    # The simulator must be run from the directory that contains input/
    iss_log = work_dir / "simulator.log"
    cmd = [str(iss)]

    if verbose:
        logger.info("stcu_rm: %s (cwd=%s)", " ".join(str(c) for c in cmd), input_dir.parent)

    # ISS run timeout (seconds). Configurable so test suites can cap
    # hang-prone kernels; defaults to a 10-minute safety limit.
    try:
        _iss_timeout = int(os.environ.get("TANG_SIM_ISS_TIMEOUT", "600"))
    except ValueError:
        _iss_timeout = 600
    result = subprocess.run(
        cmd,
        cwd=str(input_dir.parent),  # work_dir (contains input/)
        capture_output=True,
        text=True,
        timeout=_iss_timeout,
    )

    # Write log regardless of success
    log_text = result.stdout + "\n" + result.stderr
    iss_log.write_text(log_text)

    if result.returncode != 0:
        # Check for common error patterns
        if "ESL_PATH_ERROR" in result.stderr:
            raise RuntimeError("ISS simulator could not find input/. Ensure the simulator binary supports the current execution mode.")
        raise RuntimeError(f"ISS simulator failed (exit={result.returncode}).\nLog: {iss_log}\n{result.stderr[:2000]}")

    if verbose:
        # Print summary lines
        for line in result.stdout.splitlines():
            if "Execution Summary" in line or "Status:" in line or "SUCCESS" in line:
                logger.info("iss: %s", line)
