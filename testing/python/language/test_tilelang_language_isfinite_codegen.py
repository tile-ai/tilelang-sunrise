import re

import pytest
import tilelang
import tilelang.testing
import tvm
from tilelang import language as TL
from tvm.script.parser import ir_module, tir as T

# Codegen-only tests: an explicit target keeps them independent of whatever the
# box's default target happens to be.
STCUV2 = {"kind": "tang", "arch": "stcuv2"}


@ir_module
class _IsFiniteModule:
    @T.prim_func
    def main(x: T.Buffer((1,), "float32"), y: T.Buffer((1,), "int32")):
        T.func_attr({"global_symbol": "main_kernel", "tir.noalias": True})
        for _bx in T.thread_binding(1, thread="blockIdx.x"):
            for _tx in T.thread_binding(1, thread="threadIdx.x"):
                pred = T.isfinite(x[0])
                y[0] = T.if_then_else(pred, 1, 0)


def _get_isfinite_source(target: str) -> str:
    rt_mod = tvm.tirx.build(_IsFiniteModule, target=target)
    return rt_mod.imports[0].inspect_source()


def _get_isfinite_expr(code: str) -> str:
    pattern = r"\b\w+\s*=\s*(.*\bisfinite\s*\(.*\));"
    for line in code.splitlines():
        match = re.search(pattern, line)
        if match:
            return match.group(1)
    raise AssertionError("Failed to find CUDA isfinite call in generated source")


def test_isfinite_codegen_uses_cuda_intrinsic():
    """Check T.isfinite lowers to CUDA's isfinite for float32."""
    src = _get_isfinite_source("cuda")
    expr = _get_isfinite_expr(src)

    print("=== isfinite codegen ===")
    print(src)
    print("=== extracted expression ===")
    print(expr)

    assert "isfinite(" in expr
    assert "fabsf(" not in expr
    assert "CUDART_INF_F" not in expr
    assert "!= x[0]" not in expr
    assert "x[0] != x[0]" not in expr


@pytest.mark.skipif(
    not tilelang.testing.is_tang_target(),
    reason="Requires TANG target",
)
def test_isfinite_codegen_uses_tang_semantics():
    src = _get_isfinite_source("tang")
    predicate = next(line for line in src.splitlines() if "pred =" in line and "TANG_RT_INF_F" in line)

    assert "fabsf(x[0]) == TANG_RT_INF_F" in predicate
    assert predicate.count("!(x[0] != x[0])") == 2
    assert "isfinite(" not in predicate
    assert "CUDART_INF_F" not in predicate


def _non_finite_source(dtype: str, value: float) -> str:
    """Lower a kernel that stores a non-finite ``value`` into a ``dtype`` buffer."""

    @TL.prim_func
    def main(out: TL.Tensor((4,), dtype)):
        with TL.Kernel(1, threads=32):
            for i in TL.Parallel(4):
                out[i] = TL.cast(value, dtype)

    return tilelang.lower(main, target=STCUV2).kernel_source


@pytest.mark.skipif(
    not tilelang.testing.is_tang_target(),
    reason="Requires TANG target",
)
@pytest.mark.parametrize(
    ("dtype", "value", "expected"),
    [
        ("float8_e4m3fn", float("inf"), "fp8_e4_t::from_float_nosat(TANG_RT_INF_F)"),
        ("float8_e4m3fn", float("-inf"), "fp8_e4_t::from_float_nosat(-TANG_RT_INF_F)"),
        ("float8_e4m3fn", float("nan"), "fp8_e4_t::from_float_nosat(TANG_RT_NAN_F)"),
        ("float8_e5m2", float("inf"), "fp8_e5_t::from_float_nosat(TANG_RT_INF_F)"),
        ("float8_e5m2", float("-inf"), "fp8_e5_t::from_float_nosat(-TANG_RT_INF_F)"),
        ("float8_e5m2", float("nan"), "fp8_e5_t::from_float_nosat(TANG_RT_NAN_F)"),
    ],
)
def test_non_finite_fp8_constant_uses_the_vendor_conversion(dtype, value, expected):
    """fp8 has no hexfloat spelling for inf/nan, so the constant is materialised
    through ``from_float_nosat`` with the vendor's float macros -- and the header
    that defines them has to come along.

    ``from_float_nosat`` is ``__tang_cvt_float_to_fp8(val, __TANG_NOSAT, ...)``
    (see ``src/tl_templates/tang/tang_fp8.h``): e5m2 keeps its inf/nan encodings,
    e4m3 has none and converts an infinity to NaN, which is the expected
    IEEE-e4m3fn NOSAT behaviour.
    """
    src = _non_finite_source(dtype, value)
    assert expected in src
    assert "#include <__clang_tang_builtin_vars.h>" in src
    # The saturating conversion would clamp an infinity down to MAXNORM.
    assert "::from_float(" not in src


@pytest.mark.skipif(
    not tilelang.testing.is_tang_target(),
    reason="Requires TANG target",
)
@pytest.mark.parametrize(
    ("dtype", "value", "expected"),
    [
        ("float64", float("inf"), "((double)TANG_RT_INF_F)"),
        ("float64", float("-inf"), "-((double)TANG_RT_INF_F)"),
        ("float64", float("nan"), "((double)TANG_RT_NAN_F)"),
    ],
)
def test_non_finite_float64_constant_widens_the_float_macro(dtype, value, expected):
    """The vendor header only defines the float spellings of the non-finite
    macros; a double constant must widen them instead of naming a 64-bit macro
    that does not exist (that used to emit an undeclared ``TANG_RT_INF``)."""
    src = _non_finite_source(dtype, value)
    assert expected in src
    assert "TANG_RT_INF;" not in src
    assert "TANG_RT_NAN;" not in src


@pytest.mark.skipif(
    not tilelang.testing.is_tang_target(),
    reason="Requires TANG target",
)
@pytest.mark.parametrize("dtype", ["float4_e2m1fn", "custom[float4_e2m1_unpacked]8"])
@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_float4_non_finite_constant_is_rejected(dtype, value):
    """float4_e2m1 has no infinity or NaN encoding at all, so the code generator
    must refuse instead of emitting a literal."""
    with pytest.raises(Exception, match="has no infinity or NaN encoding"):
        _non_finite_source(dtype, value)


if __name__ == "__main__":
    tilelang.testing.main()
