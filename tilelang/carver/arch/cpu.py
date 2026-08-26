import tvm
from tvm.target import Target
from .arch_base import TileDevice


def is_cpu_arch(arch: TileDevice) -> bool:
    return isinstance(arch, CPU)


# For LLVM Backend, we do not provide the detailed information of the CPU
# As the LLVM backend do not required tuning, just maintain the consistency
class CPU(TileDevice):
    def __init__(self, target: Target):
        self.target = target
        device = tvm.runtime.cpu(0)
        if not device.exist:
            raise RuntimeError("Cannot find cpu device 0.")
        self.device: tvm.runtime.Device = device
        self.platform: str = "CPU"
        self.bandwidth: list[int] = [0, 0]
        self.transaction_size: list[int] = [0, 0]
        self.compute_max_core: int = 1
        self.warp_size: int = 1
        self.smem_cap: int = 0
        self.sm_partition: int = 1
        self.reg_cap: int = 0
        self.max_smem_usage: int = 0
        self.compute_capability: str = "unknown"
        self.l2_cache_size_bytes: int = 0


__all__ = [
    "is_cpu_arch",
    "CPU",
]
