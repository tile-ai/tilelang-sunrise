import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.norm.fused_add_layer_norm import FusedAddLayerNormFwdOp
from workloads.normalization import (
    FusedAddLayerNormWorkload,
)


class FusedAddLayerNormTest(FusedAddLayerNormWorkload, TestBase):
    pass


class FusedAddLayerNormFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype, tune",
            [
                # Standard aligned shapes -- fp32
                pytest.param(1024, 4096, torch.float32, False, marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.float16, False, marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.bfloat16, False, marks=pytest.mark.smoke),
                pytest.param(4096, 4096, torch.float32, False, marks=pytest.mark.full),
                # Standard aligned shapes -- fp16
                pytest.param(4096, 4096, torch.float16, False, marks=pytest.mark.full),
                # Standard aligned shapes -- bf16
                pytest.param(4096, 4096, torch.bfloat16, False, marks=pytest.mark.full),
                # Non-power-of-two hidden dims
                pytest.param(1024, 3000, torch.float32, False, marks=pytest.mark.full),
                pytest.param(1024, 3000, torch.float16, False, marks=pytest.mark.full),
                pytest.param(1024, 3000, torch.bfloat16, False, marks=pytest.mark.full),
                # Tail-M: M not divisible by block_m
                pytest.param(1025, 4096, torch.float16, False, marks=pytest.mark.full),
                pytest.param(1025, 4096, torch.bfloat16, False, marks=pytest.mark.full),
            ],
        ),
    ]


def _get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    elif dtype == torch.float16:
        return 1e-3, 1e-3
    else:  # bfloat16
        return 1.6e-2, 1.6e-2


@FusedAddLayerNormFixture
def test_fused_add_layer_norm_op(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = FusedAddLayerNormTest(m, n, dtype)
    op = FusedAddLayerNormFwdOp(tune=tune)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


class FusedAddLayerNormNonContigFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(1024, 4096, torch.float32, marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.float16, marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


@FusedAddLayerNormNonContigFixture
def test_fused_add_layer_norm_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    """Test with non-contiguous input (sliced tensor)."""
    x_full = torch.randn(m, n * 2, dtype=dtype).ptpu()
    r_full = torch.randn(m, n * 2, dtype=dtype).ptpu()
    x = x_full[:, :n]  # non-contiguous slice
    residual = r_full[:, :n]
    weight = torch.randn(n, dtype=dtype).ptpu()
    bias = torch.randn(n, dtype=dtype).ptpu()

    op = FusedAddLayerNormFwdOp()

    # Reference on contiguous copies
    test = FusedAddLayerNormTest(m, n, dtype)
    y_ref, add_ref = test.ref_program(
        x.cpu().contiguous(), residual.cpu().contiguous(), weight.cpu(), bias.cpu()
    )

    y, residual_out = op(x, residual, weight, bias)
    torch.ptpu.synchronize()
    y_cpu = y.cpu()
    y_ref_cpu = y_ref.cpu()
    residual_out_cpu = residual_out.cpu()
    add_ref_cpu = add_ref.cpu()
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y_cpu, y_ref_cpu, atol=atol, rtol=rtol), (
        f"Non-contiguous y test failed, max err: {(y_cpu - y_ref_cpu).abs().max()}"
    )
    assert torch.allclose(residual_out_cpu, add_ref_cpu, atol=atol, rtol=rtol), (
        f"Non-contiguous residual_out test failed, max err: {(residual_out_cpu - add_ref_cpu).abs().max()}"
    )


class FusedAddLayerNorm3DFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq, hidden, dtype",
            [
                pytest.param(2, 512, 4096, torch.float32, marks=pytest.mark.smoke),
                pytest.param(2, 512, 4096, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 512, 4096, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


@FusedAddLayerNorm3DFixture
def test_fused_add_layer_norm_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Test with 3D input (batch, seq, hidden)."""
    x = torch.randn(batch, seq, hidden, dtype=dtype).ptpu()
    residual = torch.randn(batch, seq, hidden, dtype=dtype).ptpu()
    weight = torch.randn(hidden, dtype=dtype).ptpu()
    bias = torch.randn(hidden, dtype=dtype).ptpu()

    M = batch * seq
    op = FusedAddLayerNormFwdOp()

    test = FusedAddLayerNormTest(M, hidden, dtype)
    y_ref, add_ref = test.ref_program(x.cpu(), residual.cpu(), weight.cpu(), bias.cpu())

    y, residual_out = op(x, residual, weight, bias)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y.cpu(), y_ref.cpu(), atol=atol, rtol=rtol), (
        f"3D y test failed, max err: {(y.cpu() - y_ref.cpu()).abs().max()}"
    )
    assert torch.allclose(residual_out.cpu(), add_ref.cpu(), atol=atol, rtol=rtol), (
        f"3D residual_out test failed, max err: {(residual_out.cpu() - add_ref.cpu()).abs().max()}"
    )
