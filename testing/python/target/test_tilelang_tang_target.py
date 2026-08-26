from __future__ import annotations

import pytest

from tilelang.backend.target import determine_target
from tilelang.tang import target as tang_target
from tvm.target import Target


@pytest.mark.parametrize("arch", tang_target.TANG_ARCHES)
def test_determine_tang_target_arch(arch):
    target = determine_target({"kind": "tang", "arch": arch}, return_object=True)
    assert isinstance(target, Target)
    assert target.kind.name == "tang"
    assert tang_target.target_get_arch(target) == arch


def test_determine_tang_target_defaults_to_stcu():
    target = determine_target({"kind": "tang"}, return_object=True)
    assert tang_target.target_is_stcu(target)


def test_determine_tang_target_rejects_unknown_arch():
    with pytest.raises(ValueError, match="Unsupported TANG arch"):
        determine_target({"kind": "tang", "arch": "unknown"}, return_object=True)


def test_detect_tang_target_uses_tang_arch(monkeypatch):
    monkeypatch.setattr(tang_target, "_ptpu_is_available", lambda: True)
    monkeypatch.setenv("TANG_ARCH", "stcuv2")
    target = tang_target._detect_tang_target()
    assert target is not None
    assert tang_target.target_is_stcuv2(target)


def test_legacy_target_module_reexports_canonical_api():
    from tilelang.utils.target import determine_target as legacy_determine_target

    assert legacy_determine_target is determine_target
