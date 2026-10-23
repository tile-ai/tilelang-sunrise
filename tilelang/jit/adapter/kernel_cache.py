import functools
import os
import subprocess
from collections.abc import Callable
from typing import Literal

from tvm.target import Target

from tilelang.cache.kernel_cache import KernelCache
from tilelang.jit import JITKernel


class TVMFFIKernelCache(KernelCache):
    kernel_lib_path = "executable.so"

    @staticmethod
    def _get_export_kwargs(kernel: JITKernel) -> dict:
        artifact = getattr(kernel, "artifact", None)
        target_host = getattr(artifact, "target_host", None)
        target_host_kind = getattr(getattr(target_host, "kind", None), "name", None)
        if target_host_kind == "c":
            return KernelCache._get_source_compile_args()
        return KernelCache._get_export_link_args()

    def _save_wrapper_kernel_code_to_disk(self, kernel: JITKernel, cache_path: str, verbose: bool = False):
        host_kernel_path = os.path.join(cache_path, self.host_kernel_path)
        if verbose:
            self.logger.debug(f"Saving wrapped kernel source code to file: {host_kernel_path}")
        KernelCache._safe_write_file(host_kernel_path, "w", lambda file: file.write(kernel.adapter.get_host_source()))

    def _save_so_cubin_to_disk(self, kernel: JITKernel, cache_path: str, verbose: bool = False):
        kernel_lib_path = os.path.join(cache_path, self.kernel_lib_path)
        executable = kernel.adapter.get_exportable_executable()
        if verbose:
            self.logger.debug(f"Saving kernel executable to file: {executable}")
        KernelCache._safe_write_executable(executable, kernel_lib_path, export_kwargs=self._get_export_kwargs(kernel))


class SimulatorKernelCache(KernelCache):
    """Kernel cache for the S3 ISS simulator backend.

    Differs from the compiled backends in two ways:

    - The cache key carries a stamp of the ISS binary. Swapping the simulator
      changes kernel behaviour without changing the input TIR, which is the same
      stale-hit hazard that ``_get_tilelang_lib_stamp`` guards against for the
      native libraries.
    - Disk persistence is disabled. A simulator kernel is not a loadable shared
      object but a work directory the ISS is re-run over, so there is no
      artifact for the base class to export and reload; only the in-process
      memory cache applies.
    """

    @staticmethod
    @functools.cache
    def _get_iss_stamp() -> str | None:
        """Identify the ISS build behind ``TANG_S3_ISS_PATH``.

        ``stcu_rm --version`` reports a version plus commit hash, which pins the
        build far more cheaply than hashing the 160MB+ binary. Falls back to
        path/size/mtime when the probe is unavailable.
        """
        iss_path = os.environ.get("TANG_S3_ISS_PATH")
        if not iss_path or not os.path.exists(iss_path):
            return None
        try:
            probe = subprocess.run(
                [iss_path, "--version"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            for line in (probe.stdout or "").splitlines() + (probe.stderr or "").splitlines():
                if line.strip():
                    return line.strip()
        except (OSError, subprocess.SubprocessError):
            pass
        stat = os.stat(iss_path)
        return f"{os.path.realpath(iss_path)}:{stat.st_size}:{int(stat.st_mtime)}"

    @staticmethod
    def _get_base_key() -> dict:
        base = KernelCache._get_base_key()
        iss_stamp = SimulatorKernelCache._get_iss_stamp()
        if iss_stamp:
            base["tang_iss"] = iss_stamp
        return base

    def _save_kernel_to_disk(self, key: str, kernel: JITKernel, func: Callable = None, verbose: bool = False):
        return

    def _load_kernel_from_disk(
        self,
        key: str,
        target: str | Target = "auto",
        target_host: str | Target | None = None,
        out_idx: list[int] | None = None,
        execution_backend: Literal["tvm_ffi", "cython", "nvrtc", "torch", "cutedsl", "simulator"] = "simulator",
        pass_configs: dict | None = None,
        compile_flags: list[str] | str | None = None,
        func: Callable | None = None,
        verbose: bool = False,
    ) -> JITKernel | None:
        return None
