"""Construct legacy TANG GEMM calls on the CPU without compiling a kernel."""

import pytest

import tilelang.language as T
import tilelang.language.common as Tcommon
from tvm import ir, tirx


def _operands(c_scope="local.fragment"):
    a = tirx.decl_buffer((32, 32), "float16", name="A", scope="shared")
    b = tirx.decl_buffer((32, 32), "float16", name="B", scope="shared")
    c = tirx.decl_buffer((32, 32), "float32", name="C", scope=c_scope)
    return a, b, c


@pytest.mark.parametrize("gemm", [T.gemm, Tcommon.gemm])
def test_generic_gemm_preserves_tang_k_step(gemm):
    call = gemm(*_operands(), k_step=4, wc_interleave=4)
    assert call.op.name == "tl.tileop.gemm"
    assert len(call.args) == 13
    assert int(call.annotations["tang_k_step"]) == 4


@pytest.mark.parametrize("gemm", [T.gemm, Tcommon.gemm])
def test_default_gemm_preserves_upstream_dense_protocol(gemm):
    call = gemm(*_operands())
    assert call.op.name == "tl.tileop.gemm"
    assert len(call.args) == 13
    assert not any(str(key).startswith("tang_") for key in call.annotations)


@pytest.mark.parametrize("gemm", [T.gemm, Tcommon.gemm])
@pytest.mark.parametrize("k_step", [0, -1, True])
def test_generic_gemm_rejects_invalid_tang_k_step(gemm, k_step):
    with pytest.raises(ValueError, match="k_step must be a positive integer"):
        gemm(*_operands(), k_step=k_step)


@pytest.mark.parametrize("gemm", [T.gemm, Tcommon.gemm])
def test_generic_gemm_preserves_tang_scale_operands(gemm):
    sfa = tirx.decl_buffer((32, 4), "uint8", name="SFA", scope="shared")
    sfb = tirx.decl_buffer((32, 4), "uint8", name="SFB", scope="shared")
    workspace = tirx.decl_buffer((32, 4), "uint32", name="SF", scope="shared.tmem")
    call = gemm(*_operands("shared.tmem"), scale_a=sfa, scale_b=sfb, scale_tmem=workspace)
    assert call.op.name == "tl.tileop.gemm_blockscaled"
    assert len(call.args) == 17
    assert call.args[13].args[0].buffer.same_as(sfa)
    assert call.args[14].args[0].buffer.same_as(sfb)
    assert call.args[16].args[0].buffer.same_as(workspace)


@pytest.mark.parametrize("gemm", [T.gemm, Tcommon.gemm])
@pytest.mark.parametrize("keyword", ["a_local_load_type", "b_local_load_type"])
def test_generic_gemm_preserves_legacy_local_load_keywords(gemm, keyword):
    operands = _operands()
    default = gemm(*operands, k_step=1)
    # Both accepted legacy policies leave the emitted scheduling contract unchanged.
    for policy in ("load_overlap_mma", "load_before_mma"):
        call = gemm(*operands, **{keyword: policy})
        ir.assert_structural_equal(call, default)
    with pytest.raises(ValueError, match=keyword):
        gemm(*operands, **{keyword: "invalid"})
