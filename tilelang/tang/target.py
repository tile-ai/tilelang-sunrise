from __future__ import annotations

import os

from tvm.target import Target

from tilelang.backend.target import TargetLike, register_target_detector, register_target_normalizer

TANG_ARCHES = ("stcu", "stcuv2")
DEFAULT_TANG_ARCH = "stcu"


def normalize_tang_arch(arch: object | None) -> str | None:
    if arch is None:
        return DEFAULT_TANG_ARCH
    normalized = str(arch).strip().lower()
    return normalized if normalized in TANG_ARCHES else None


def target_get_arch(target: str | Target | None) -> str | None:
    if target is None:
        return None
    if isinstance(target, str):
        target = Target(target)
    if target.kind.name != "tang":
        return None
    return normalize_tang_arch(target.attrs.get("arch"))


def target_is_tang(target: Target) -> bool:
    return target.kind.name == "tang"


def target_is_stcu(target: Target) -> bool:
    return target_get_arch(target) == "stcu"


def target_is_stcuv2(target: Target) -> bool:
    return target_get_arch(target) == "stcuv2"


def _ptpu_is_available() -> bool:
    try:
        import torch
        import torch_ptpu  # noqa: F401

        ptpu = getattr(torch, "ptpu", None)
        if ptpu is None:
            return False
        is_available = getattr(ptpu, "is_available", None)
        if callable(is_available):
            return bool(is_available())
        device_count = getattr(ptpu, "device_count", None)
        return bool(device_count()) if callable(device_count) else False
    except Exception:
        return False


def _detect_tang_target() -> Target | None:
    if not _ptpu_is_available():
        return None
    arch = normalize_tang_arch(os.environ.get("TANG_ARCH", DEFAULT_TANG_ARCH))
    if arch is None:
        raise ValueError(f"Unsupported TANG_ARCH. Expected one of: {', '.join(TANG_ARCHES)}")
    return Target({"kind": "tang", "arch": arch})


def normalize_tang_target(target: TargetLike) -> Target | None:
    if isinstance(target, Target):
        parsed_target = target
    elif isinstance(target, dict):
        if target.get("kind") != "tang":
            return None
        arch = normalize_tang_arch(target.get("arch"))
        if arch is None:
            raise ValueError(f"Unsupported TANG arch {target.get('arch')!r}. Expected one of: {', '.join(TANG_ARCHES)}")
        config = dict(target)
        config["arch"] = arch
        return Target(config)
    elif isinstance(target, str):
        try:
            parsed_target = Target(target)
        except Exception:
            return None
    else:
        return None

    if parsed_target.kind.name != "tang":
        return None
    arch = normalize_tang_arch(parsed_target.attrs.get("arch"))
    if arch is None:
        raise ValueError(f"Unsupported TANG arch {parsed_target.attrs.get('arch')!r}. Expected one of: {', '.join(TANG_ARCHES)}")
    config = dict(parsed_target.export())
    config["arch"] = arch
    return Target(config)


register_target_detector("tang", _detect_tang_target, override=True)
register_target_normalizer("tang", normalize_tang_target, override=True)
