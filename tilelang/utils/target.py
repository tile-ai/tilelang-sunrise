"""Compatibility target API.

Use :mod:`tilelang.backend.target` for new code. This module remains available
for compatibility with existing TileLang-Sunrise applications.
"""

from tilelang.backend.target import (  # noqa: F401
    TargetConfig,
    TargetInput,
    TargetLike,
    auto_detect_target,
    determine_target,
    list_target_detectors,
    register_target_detector,
    register_target_normalizer,
)
from tilelang.tang.target import (  # noqa: F401
    DEFAULT_TANG_ARCH,
    TANG_ARCHES,
    normalize_tang_arch,
    normalize_tang_target,
    target_get_arch,
    target_is_stcu,
    target_is_stcuv2,
    target_is_tang,
)


def target_tang_is_stcu(target):
    from tilelang import _ffi_api

    return bool(_ffi_api.TargetTangIsSTCU(target))


def target_tang_is_stcuv2(target):
    from tilelang import _ffi_api

    return bool(_ffi_api.TargetTangIsSTCUV2(target))


def target_tang_has_tmem(target):
    from tilelang import _ffi_api

    return bool(_ffi_api.TargetTangHasTmem(target))


def target_tang_has_bulk_copy(target):
    from tilelang import _ffi_api

    return bool(_ffi_api.TargetTangHasBulkCopy(target))
