import re

import pytest
import tilelang
import tilelang.testing
import tvm
from tvm.script.parser import ir_module, tir as T


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


@tilelang.testing.requires_cuda
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
    not str(tilelang.env.get_default_target()).startswith("tang"),
    reason="Requires TANG target",
)
def test_isfinite_codegen_uses_tang_semantics():
    src = _get_isfinite_source("tang")
    predicate = next(line for line in src.splitlines() if "pred =" in line and "TANG_RT_INF_F" in line)

    assert "fabsf(x[0]) == TANG_RT_INF_F" in predicate
    assert predicate.count("!(x[0] != x[0])") == 2
    assert "isfinite(" not in predicate
    assert "CUDART_INF_F" not in predicate


if __name__ == "__main__":
    tilelang.testing.main()
