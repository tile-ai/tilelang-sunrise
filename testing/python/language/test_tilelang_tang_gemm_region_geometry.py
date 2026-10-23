"""TANG descriptor geometry retains region origins without retired node fields."""

from types import SimpleNamespace

import pytest
from tilelang.tang.op.gemm.gemm_tmma import GemmTMMA
from tilelang.tang.op.gemm.gemm_tcgen5 import GemmTangTCGEN5
from tvm import tirx
from tvm.ir import Range


@pytest.mark.parametrize("implementation", [GemmTMMA, GemmTangTCGEN5])
def test_tang_gemm_derives_strides_and_offsets_from_regions(implementation):
    stage = tirx.Var("stage", "int32")
    a = tirx.decl_buffer((2, 64, 64), "float16", strides=(6144, 96, 1), scope="shared")
    b = tirx.decl_buffer((64, 80), "float16", scope="shared")
    c = tirx.decl_buffer((32, 32), "float32", scope="local.fragment")
    node = SimpleNamespace(
        a=a,
        b=b,
        c=c,
        aRegion=tirx.BufferRegion(a, [Range.from_min_extent(stage, 1), Range.from_min_extent(8, 32), Range.from_min_extent(16, 32)]),
        bRegion=tirx.BufferRegion(b, [Range.from_min_extent(4, 32), Range.from_min_extent(32, 32)]),
    )
    gemm = implementation(node)
    assert int(gemm.stride_A) == 96
    assert int(gemm.stride_B) == 80
    assert int(gemm.offset_A) == 16
    assert int(gemm.offset_B) == 32
    assert gemm.A_base_offsets[0].same_as(stage)
    assert int(gemm.A_base_offsets[1]) == 8
    assert int(gemm.B_base_offsets[0]) == 4
