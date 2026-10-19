import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.norm.ada_layer_norm import AdaLayerNormKernel
from tileops.ops.norm.ada_layer_norm_zero import AdaLayerNormZeroFwdOp
from workloads.normalization import AdaLayerNormZeroWorkload


class AdaLayerNormZeroTest(AdaLayerNormZeroWorkload, TestBase):
    pass


class AdaLayerNormZeroFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                # Standard aligned shapes -- fp32
                pytest.param(1024, 4096, torch.float32, marks=pytest.mark.smoke),
                # Standard aligned shapes -- fp16
                pytest.param(1024, 4096, torch.float16, marks=pytest.mark.smoke),
                # Standard aligned shapes -- bf16
                pytest.param(1024, 4096, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(4096, 4096, torch.float32, marks=pytest.mark.full),
                pytest.param(4096, 4096, torch.float16, marks=pytest.mark.full),
                pytest.param(4096, 4096, torch.bfloat16, marks=pytest.mark.full),
                # Non-power-of-two hidden dims
                pytest.param(1024, 3000, torch.float32, marks=pytest.mark.full),
                pytest.param(1024, 3000, torch.float16, marks=pytest.mark.full),
                pytest.param(1024, 3000, torch.bfloat16, marks=pytest.mark.full),
                # Tail-M: M not divisible by block_m
                pytest.param(1025, 4096, torch.float16, marks=pytest.mark.full),
                pytest.param(1025, 4096, torch.bfloat16, marks=pytest.mark.full),
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


@AdaLayerNormZeroFixture
def test_ada_layer_norm_zero_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = AdaLayerNormZeroTest(m, n, dtype)
    op = AdaLayerNormZeroFwdOp()
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_ada_layer_norm_zero_kernel_handles_natural_unaligned_shape(
    dtype: torch.dtype,
) -> None:
    m, n = 16, 1152
    test = AdaLayerNormZeroTest(m, n, dtype)
    inputs = test.gen_inputs()
    kernel = AdaLayerNormKernel(n, test.eps, dtype, has_gate=True)
    actual = kernel(*inputs)
    expected = test.ref_program(*(x.cpu() for x in inputs))
    assert actual.shape == (m, n)
    atol, rtol = _get_tolerances(dtype)
    torch.testing.assert_close(actual.cpu(), expected, atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_ada_layer_norm_zero_copy_handles_row_tail() -> None:
    """Regression: the selected copy path must support block_m > 1 and tail rows."""
    m, n, block_m = 17, 514, 4
    dtype = torch.float16
    test = AdaLayerNormZeroTest(m, n, dtype)
    inputs = test.gen_inputs()
    kernel = AdaLayerNormKernel(
        n,
        test.eps,
        dtype,
        has_gate=True,
        config={"block_m": block_m, "threads": 128},
    )
    assert kernel.config["block_m"] == block_m
    actual = kernel(*inputs)
    expected = test.ref_program(*(x.cpu() for x in inputs))
    atol, rtol = _get_tolerances(dtype)
    torch.testing.assert_close(actual.cpu(), expected, atol=atol, rtol=rtol)


class AdaLayerNormZero3DFixture(FixtureBase):
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


@AdaLayerNormZero3DFixture
def test_ada_layer_norm_zero_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Test with 3D input (batch, seq, hidden)."""
    x = torch.randn(batch, seq, hidden, dtype=dtype).ptpu()
    scale = torch.randn(batch, seq, hidden, dtype=dtype).ptpu()
    shift = torch.randn(batch, seq, hidden, dtype=dtype).ptpu()
    gate = torch.randn(batch, seq, hidden, dtype=dtype).ptpu()

    op = AdaLayerNormZeroFwdOp()

    # Reference: gate * (scale * LayerNorm(x) + shift)
    eps = 1e-5
    x_cpu = x.cpu().float()
    scale_cpu = scale.cpu().float()
    shift_cpu = shift.cpu().float()
    gate_cpu = gate.cpu().float()
    normed_cpu = F.layer_norm(x_cpu, (hidden,), weight=None, bias=None, eps=eps)
    y_ref = (gate_cpu * (scale_cpu * normed_cpu + shift_cpu)).to(dtype).to(x.device)

    y = op(x, scale, shift, gate)
    # PTPU kernel launches are asynchronous; wait for completion before
    # reading y, otherwise the comparison races the kernel.
    torch.ptpu.synchronize()
    atol, rtol = _get_tolerances(dtype)
    y_cpu = y.cpu()
    y_ref_cpu = y_ref.cpu()
    assert torch.allclose(y_cpu, y_ref_cpu, atol=atol, rtol=rtol), (
        f"3D test failed, max err: {(y_cpu - y_ref_cpu).abs().max()}"
    )
