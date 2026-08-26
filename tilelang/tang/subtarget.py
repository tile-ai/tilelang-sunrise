"""Subtarget capabilities and pass filtering for the TANG backend."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from enum import IntFlag, auto

from tvm import IRModule
from tvm.target import Target

from .target import target_get_arch


class TangSubtarget(IntFlag):
    """Bitmask used to constrain a pass to one or more TANG architectures."""

    DEFAULT = 0
    STCU = auto()
    STCUV2 = auto()


@dataclass(frozen=True, slots=True)
class TangCapabilities:
    """Backend capabilities that must not leak into generic planners."""

    max_vector_load_bits: int
    supports_pts_async_copy: bool
    supports_tmem_drain: bool


_CAPABILITIES = {
    "stcu": TangCapabilities(32, True, False),
    "stcuv2": TangCapabilities(32, False, True),
}


def get_tang_capabilities(target: Target | str) -> TangCapabilities:
    arch = target if isinstance(target, str) else target_get_arch(target)
    try:
        return _CAPABILITIES[arch]
    except KeyError as err:
        raise ValueError(f"Unsupported TANG arch: {arch!r}") from err


def arch_to_subtarget(arch: str | None) -> TangSubtarget:
    if arch == "stcu":
        return TangSubtarget.STCU
    if arch == "stcuv2":
        return TangSubtarget.STCUV2
    return TangSubtarget.DEFAULT


def subtarget_matches(
    target: Target | None,
    required: TangSubtarget = TangSubtarget.DEFAULT,
) -> bool:
    if required == TangSubtarget.DEFAULT:
        return True
    actual = arch_to_subtarget(target_get_arch(target))
    return actual != TangSubtarget.DEFAULT and bool(actual & required)


def _module_target(mod: IRModule) -> Target | None:
    target = mod.get_attr("target") if hasattr(mod, "get_attr") else None
    if target is not None:
        return target
    for func in mod.functions.values():
        if func.attrs and "target" in func.attrs:
            return func.attrs["target"]
    return None


def pass_filter(
    pass_factory,
    subtarget: TangSubtarget = TangSubtarget.DEFAULT,
):
    """Return a pass factory that skips modules for non-matching TANG arches.

    Architecture-specific passes must never silently disappear because target
    metadata was lost.  The TANG pipeline binds a target before reaching these
    filters, so missing or non-TANG metadata indicates a malformed module.
    """

    @functools.wraps(pass_factory)
    def filtered_pass(*args, **kwargs):
        inner_pass = pass_factory(*args, **kwargs)

        def check_and_run(mod: IRModule, *pass_args, **pass_kwargs):
            target = _module_target(mod)
            if subtarget != TangSubtarget.DEFAULT:
                if target is None:
                    raise ValueError("TANG architecture-specific pass requires target metadata on the IRModule or a function")
                actual = arch_to_subtarget(target_get_arch(target))
                if actual == TangSubtarget.DEFAULT:
                    raise ValueError(f"TANG architecture-specific pass requires a valid TANG target, but received {target}")
            if not subtarget_matches(target, subtarget):
                return mod
            return inner_pass(mod, *pass_args, **pass_kwargs)

        return check_and_run

    return filtered_pass
