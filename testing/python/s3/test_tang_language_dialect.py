"""TANG (stcuv2) language dialect surface tests.

The TANG backend is built on top of the CUDA code-generation infrastructure, so
unlike the cpu/metal/rocm/webgpu dialects the TANG dialect is a strict *superset*
of the CUDA facade instead of a disjoint surface. That makes the generic
"non-cuda dialects must not export cuda symbols" invariant inapplicable here, so
TANG gets its own set of invariants:

* the dialect is exactly ``cuda facade + TANG_ONLY_NAMES``;
* none of the TANG-only names leak into the default facade or the other dialects;
* the two ``tcgen05_*_thread_sync`` fences are deliberately shadowed -- same name
  as CUDA, but bound to a TANG-only op with a wider signature.

These are pure import-surface / IR-construction checks: no target, no codegen and
no device are involved.
"""

import importlib

import pytest

import tilelang
import tilelang.testing
import tilelang.language as T_default
import tilelang.language.common as T_comm


# Symbols the TANG dialect adds on top of the CUDA facade. ``tcgen05_cp`` counts as
# TANG-only because the CUDA facade exposes only the narrower ``tcgen05_cp_warpx4``;
# it moves to TANG_SHADOWED_NAMES if CUDA ever grows a generic ``tcgen05_cp``.
TANG_ONLY_NAMES = {
    "tang_cp_tmem_to_shared",
    "tang_ldmatrix",
    "tang_stmatrix",
    "tcgen05_cp",
    "tcgen05_ld",
    "tcgen05_st",
    "tcgen05_sync_arrive",
    "tcgen05_sync_wait",
}

# Names present in both facades, but re-bound by the TANG dialect. They do not
# show up as a set difference, so they need to be pinned by identity.
TANG_SHADOWED_NAMES = {
    "tcgen05_after_thread_sync",
    "tcgen05_before_thread_sync",
    # TANG adds a keyword-only warp_group_size (single-warp 32) that the CUDA
    # binding does not have; see tilelang/tang/language/warpgroup.py.
    "WarpSpecialize",
    "ws",
}

# The two fence re-bindings are the ones with fence-specific semantics; the
# fence parametrized tests below must not pick up the warpgroup shadow.
_FENCE_SHADOWED_NAMES = sorted(name for name in TANG_SHADOWED_NAMES if name.startswith("tcgen05_"))

OTHER_DIALECT_MODULES = [
    "tilelang.language",
    "tilelang.cpu.language",
    "tilelang.cuda.language",
    "tilelang.metal.language",
    "tilelang.rocm.language",
    "tilelang.webgpu.language",
]


def test_tang_dialect_marker():
    from tilelang.tang import language as T

    assert T.__tilelang_dialect__ == "tang"
    assert importlib.import_module("tilelang.tang.language") is T


def test_tang_dialect_is_strict_cuda_superset():
    from tilelang.cuda import language as cuda_language
    from tilelang.tang import language as T

    cuda_names = set(cuda_language.__all__)
    tang_names = set(T.__all__)

    assert cuda_names <= tang_names
    assert tang_names - cuda_names == TANG_ONLY_NAMES
    assert len(T.__all__) == len(set(T.__all__)), "__all__ must not contain duplicates"


def test_tang_dialect_shares_the_common_surface():
    from tilelang.cuda import language as cuda_language
    from tilelang.tang import language as T

    assert set(T.__all__) >= set(T_comm.__all__)
    assert T.copy is T_comm.copy
    # Everything TANG does not override is the very same object as in CUDA.
    for name in set(cuda_language.__all__) - TANG_SHADOWED_NAMES:
        assert getattr(T, name) is getattr(cuda_language, name), name


@pytest.mark.parametrize("module_name", OTHER_DIALECT_MODULES)
def test_tang_only_names_do_not_leak_into_other_dialects(module_name):
    module = importlib.import_module(module_name)

    assert TANG_ONLY_NAMES.isdisjoint(module.__all__)
    for name in TANG_ONLY_NAMES:
        assert not hasattr(module, name), f"{module_name} unexpectedly exposes {name}"


def test_default_facade_stays_the_cuda_dialect():
    from tilelang.cuda import language as cuda_language

    assert T_default.__tilelang_dialect__ == "cuda"
    assert set(T_default.__all__) == set(cuda_language.__all__)


def test_tang_extensions_live_in_their_owning_modules():
    from tilelang.tang import language as T

    for name in ("tcgen05_cp", "tang_cp_tmem_to_shared", "tcgen05_ld", "tcgen05_st", "tcgen05_sync_arrive", "tcgen05_sync_wait"):
        assert getattr(T, name).__module__ == "tilelang.language.tang_tcgen05", name
    for name in ("tang_stmatrix", "tang_ldmatrix"):
        assert getattr(T, name).__module__ == "tilelang.language.builtin", name


@pytest.mark.parametrize("name", _FENCE_SHADOWED_NAMES)
def test_tang_shadows_the_cuda_thread_sync_fences(name):
    from tilelang.cuda import language as cuda_language
    from tilelang.tang import language as T

    tang_fence = getattr(T, name)
    cuda_fence = getattr(cuda_language, name)

    assert tang_fence is not cuda_fence
    assert tang_fence.__module__ == "tilelang.language.tang_tcgen05"
    assert cuda_fence.__module__ == "tilelang.language.builtin"

    # Both spellings are call-compatible with no arguments; they lower differently.
    assert tang_fence().op.name == "tl.tang_fence_tc"
    assert cuda_fence().op.name == f"tl.{name}"


@pytest.mark.parametrize("name", _FENCE_SHADOWED_NAMES)
def test_tang_fence_takes_an_optional_fence_group(name):
    from tilelang.cuda import language as cuda_language
    from tilelang.tang import language as T

    tang_fence = getattr(T, name)

    for fence_group in (0, 1):
        call = tang_fence(fence_group)
        assert call.op.name == "tl.tang_fence_tc"
        assert [arg.value for arg in call.args] == [fence_group]

    assert [arg.value for arg in tang_fence().args] == [0]
    assert tang_fence(fence_group=1).args[0].value == 1

    with pytest.raises(AssertionError, match="fence_group must be 0 or 1"):
        tang_fence(2)

    # The CUDA op is arg-less, which is why TANG cannot simply widen it in place.
    with pytest.raises(TypeError):
        getattr(cuda_language, name)(0)


def test_tang_named_barrier_handshake():
    from tilelang.tang import language as T

    arrive = T.tcgen05_sync_arrive(3)
    assert arrive.op.name == "tl.tang_sync_arrive"
    assert [arg.value for arg in arrive.args] == [3]
    assert [arg.value for arg in T.tcgen05_sync_arrive().args] == [0]

    wait = T.tcgen05_sync_wait(1, 2, 3)
    assert wait.op.name == "tl.tang_sync_wait"
    assert [arg.value for arg in wait.args] == [1, 2, 3]
    assert [arg.value for arg in T.tcgen05_sync_wait().args] == [0, 1, 1]


if __name__ == "__main__":
    tilelang.testing.main()
