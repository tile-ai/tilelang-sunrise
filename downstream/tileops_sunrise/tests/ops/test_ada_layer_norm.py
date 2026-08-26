import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.ops.norm.ada_layer_norm import AdaLayerNormFwdOp
from workloads.normalization import AdaLayerNormTest as _AdaLayerNormTestWorkload


class AdaLayerNormTest(_AdaLayerNormTestWorkload, TestBase):
    def ref_program(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        # AdaLN: y = scale * LayerNorm(x) + shift
        normed = F.layer_norm(
            x.float(),
            (self.n,),
            weight=None,
            bias=None,
            eps=self.eps,
        )
        y = scale.float() * normed + shift.float()
        return y.to(x.dtype)


class AdaLayerNormFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype", [
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
        ]),
    ]


def _get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    elif dtype == torch.float16:
        return 1e-3, 1e-3
    else:  # bfloat16
        return 1.6e-2, 1.6e-2


@AdaLayerNormFixture
def test_ada_layer_norm_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = AdaLayerNormTest(m, n, dtype)
    op = AdaLayerNormFwdOp(dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


class AdaLayerNorm3DFixture(FixtureBase):
    PARAMS = [
        ("batch, seq, hidden, dtype", [
            pytest.param(2, 512, 4096, torch.float32, marks=pytest.mark.smoke),
            pytest.param(2, 512, 4096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(2, 512, 4096, torch.bfloat16, marks=pytest.mark.smoke),
        ]),
    ]


@AdaLayerNorm3DFixture
def test_ada_layer_norm_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Test with 3D input (batch, seq, hidden)."""
    x = torch.randn(batch, seq, hidden, dtype=dtype).ptpu()
    scale = torch.randn(batch, seq, hidden, dtype=dtype).ptpu()
    shift = torch.randn(batch, seq, hidden, dtype=dtype).ptpu()

    op = AdaLayerNormFwdOp(dtype=dtype)

    # Reference computed on CPU: F.layer_norm is not available on PTPU.
    eps = 1e-5
    x_cpu = x.cpu().float()
    scale_cpu = scale.cpu().float()
    shift_cpu = shift.cpu().float()
    normed_cpu = F.layer_norm(x_cpu, (hidden,), weight=None, bias=None, eps=eps)
    y_ref = (scale_cpu * normed_cpu + shift_cpu).to(dtype).to(x.device)

    y = op(x, scale, shift)
    # PTPU kernel launches are asynchronous; wait for completion before
    # reading y, otherwise the comparison races the kernel.
    torch.ptpu.synchronize()
    atol, rtol = _get_tolerances(dtype)
    y_cpu = y.cpu()
    y_ref_cpu = y_ref.cpu()
    assert torch.allclose(y_cpu, y_ref_cpu, atol=atol, rtol=rtol), \
        f"3D test failed, max err: {(y_cpu - y_ref_cpu).abs().max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
