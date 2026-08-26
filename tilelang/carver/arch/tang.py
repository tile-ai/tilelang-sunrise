from __future__ import annotations
import tvm
from tvm.target import Target
from .arch_base import TileDevice
from .driver import tang_driver


def is_tang_arch(arch: TileDevice) -> bool:
    return isinstance(arch, TANG)


def has_mma_support(arch: TileDevice) -> bool:
    return True


stcu_s2_tensorcore_supported = [
    ("bfloat16", "bfloat16"),
    ("bfloat16", "float32"),
    ("float16", "float32"),
    ("float16", "float16"),
    ("float32", "float32"),
    ("int8", "int32"),
    ("uint8", "int32"),
]


def is_tensorcore_supported_precision_tang(in_dtype: str, accum_dtype: str, arch: TileDevice) -> bool:
    if is_tang_arch(arch):
        # for now, we only supports s2
        return (in_dtype, accum_dtype) in stcu_s2_tensorcore_supported
    else:
        raise ValueError(f"Unsupported architecture: {arch}")


class TensorInstruction:
    def __init__(
        self,
        name: str,
        shape: list[int],
    ):
        self.name: str = name
        # only hold the shape of M and N
        self.shape: list[int] = shape


class TANG(TileDevice):
    def __init__(self, target: Target | str):
        if isinstance(target, str):
            target = tvm.target.Target(target)
        self.target = target
        self.sm_version = None
        device = tvm.runtime.device("tang", 0)
        if not device.exist:
            raise RuntimeError("Cannot find tang device 0.")
        self.name = tang_driver.get_device_name()
        self.device: tvm.runtime.Device = device
        self.platform: str = "TANG"
        # TODO(lei): maybe static shared memory, can be improved in future
        self.smem_cap = tang_driver.get_shared_memory_per_block()
        self.compute_max_core = device.multi_processor_count
        self.warp_size = device.warp_size
        self.compute_capability = device.compute_version.replace(".", "")
        self.reg_cap: int = 65536
        self.max_smem_usage: int = 2 * self.smem_cap
        self.sm_partition: int = 4
        self.l2_cache_size_bytes: int = getattr(target, "l2_cache_size_bytes", 0)
        # the number of transaction size in bytes
        self.transaction_size: list[int] = [32, 128]  # in bytes
        # bandwidth in MB/s, will be used for recommend basic tile size
        # TODO(lei): find some way to get the real bandwidth
        # However, the ratio of bandwidth between different devices can
        # be similar. The bandwidth can work for another devices as well.
        self.bandwidth: list[int] = [750, 12080]
        # get the available tensor instructions during runtime to avoid
        # the dependency of the tensor intrinsics registration
        self.available_tensor_instructions: list[TensorInstruction] = None

    def get_avaliable_tensorintrin_shapes(self):
        self.available_tensor_instructions = (
            TensorInstruction("mma", [16, 16]),
            TensorInstruction("wmma", [16, 16]),
        )
        return [t.shape for t in self.available_tensor_instructions]

    def __repr__(self):
        return f"TANG({self.target})"


__all__ = ["is_tang_arch", "is_tensorcore_supported_precision_tang", "has_mma_support", "TANG"]
