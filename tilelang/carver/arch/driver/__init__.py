from tilelang.utils.device import is_ptpu_available

from .cuda_driver import (
    get_cuda_device_properties,  # noqa: F401
)
from .tang_driver import get_tang_device_properties  # noqa: F401

from . import cuda_driver
from . import tang_driver


def _get_impl(name):
    """Return the platform-appropriate implementation of a device-query function.

    Checks ``is_ptpu_available()`` at call time so that late binding (e.g. a
    PTPU backend loaded after the initial import) is handled correctly.
    """
    if is_ptpu_available():
        return getattr(tang_driver, name)
    return getattr(cuda_driver, name)


def get_device_name(device_id: int = 0):
    return _get_impl("get_device_name")(device_id)


def get_shared_memory_per_block(device_id: int = 0, format: str = "bytes"):
    return _get_impl("get_shared_memory_per_block")(device_id, format=format)


def get_device_attribute(attr: int, device_id: int = 0):
    return _get_impl("get_device_attribute")(attr, device_id)


def get_max_dynamic_shared_size_bytes(device_id: int = 0, format: str = "bytes"):
    return _get_impl("get_max_dynamic_shared_size_bytes")(device_id, format=format)


def get_persisting_l2_cache_max_size(device_id: int = 0):
    return _get_impl("get_persisting_l2_cache_max_size")(device_id)


def get_num_sms(device_id: int = 0):
    return _get_impl("get_num_sms")(device_id)


def get_registers_per_block(device_id: int = 0):
    return _get_impl("get_registers_per_block")(device_id)


__all__ = [
    "get_cuda_device_properties",
    "get_tang_device_properties",
    "get_device_name",
    "get_shared_memory_per_block",
    "get_device_attribute",
    "get_max_dynamic_shared_size_bytes",
    "get_persisting_l2_cache_max_size",
    "get_num_sms",
    "get_registers_per_block",
]
