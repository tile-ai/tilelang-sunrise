"""Tests for BatchNormFwdOp and BatchNormBwdOp.

Correctness is validated against torch.nn.functional.batch_norm and the
analytical gradient via torch.autograd.

Run:
    conda run -n tileops python -m pytest tests/ops/test_batch_norm.py -vvs
"""


import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.norm.batch_norm import BatchNormBwdOp, BatchNormFwdOp
from workloads.normalization import (
    BatchNormBwdTest as _BatchNormBwdTestWorkload,
)
from workloads.normalization import (
    BatchNormFwdTest as _BatchNormFwdTestWorkload,
)


def _ref_fwd(x, weight, bias, running_mean, running_var, training, momentum=0.1, eps=1e-5):
    """Reference: torch.nn.functional.batch_norm (float32 upcast)."""
    x32 = x.float()
    rm = running_mean.clone()
    rv = running_var.clone()
    y32 = torch.nn.functional.batch_norm(
        x32, rm, rv, weight.float(), bias.float(),
        training=training, momentum=momentum, eps=eps)
    return y32.to(x.dtype), rm, rv


class BatchNormBwdTest(_BatchNormBwdTestWorkload, TestBase):
    def ref_program(self, grad_out, x, weight, mean, rstd):
        """Reference via torch.autograd on a float32 graph."""
        x32 = x.float().requires_grad_(True)
        w32 = weight.float().requires_grad_(True)
        b32 = torch.zeros(self.C, device=x.device, dtype=torch.float32, requires_grad=True)
        rm = torch.zeros(self.C, device=x.device, dtype=torch.float32)
        rv = torch.ones(self.C, device=x.device, dtype=torch.float32)
        y32 = torch.nn.functional.batch_norm(
            x32, rm, rv, w32, b32, training=True, momentum=0.1, eps=1e-5)
        y32.backward(grad_out.float())
        return x32.grad.to(x.dtype), w32.grad, b32.grad

class BatchNormFwdTest(_BatchNormFwdTestWorkload, TestBase):
    def ref_program(self, x, weight, bias, running_mean, running_var):
        y, rm, rv = _ref_fwd(x, weight, bias, running_mean, running_var,
                             training=self.training)
        return (y,)


# Fixtures

class BatchNormFwdFixture(FixtureBase):
    """(N, C, *spatial, dtype, training)"""
    PARAMS = [
        ("N, C, spatial, dtype, training", [
            # BatchNorm1d – (N, C)
            pytest.param(32, 64, (), torch.float16, True, marks=pytest.mark.smoke),
            pytest.param(32, 64, (), torch.bfloat16, True, marks=pytest.mark.smoke),
            pytest.param(32, 64, (), torch.float16, False, marks=pytest.mark.full),
            pytest.param(32, 256, (), torch.bfloat16, True, marks=pytest.mark.full),
            # BatchNorm1d – (N, C, L)
            pytest.param(16, 64, (512,), torch.float16, True, marks=pytest.mark.full),
            # Non-persistent path (L > 8192): smallest representative case L=16384.
            pytest.param(4, 64, (64, 64), torch.float16, True, marks=pytest.mark.full),
            # BatchNorm2d – (N, C, H, W)
            pytest.param(8, 64, (1024, 1024), torch.float16, True, marks=pytest.mark.full),
            # Keep the 2048x2048 spatial path while bounding the CPU reference
            # memory footprint for 32 GiB CI hosts.
            pytest.param(2, 64, (2048, 2048), torch.float16, False, marks=pytest.mark.full),
            pytest.param(4, 128, (32, 32), torch.bfloat16, True, marks=pytest.mark.full),
            # Non-aligned spatial: H*W=900, exercises partial-tile path
            pytest.param(8, 64, (30, 30), torch.float16, True, marks=pytest.mark.full),
            pytest.param(8, 64, (30, 30), torch.bfloat16, True, marks=pytest.mark.full),
            # High channel count oversubscribes the SMs, exposing the running-stat update race.
            pytest.param(16, 1024, (512,), torch.float16, True, marks=pytest.mark.full),
        ]),
    ]


class BatchNormBwdFixture(FixtureBase):
    """(N, C, *spatial, dtype)"""
    PARAMS = [
        ("N, C, spatial, dtype", [
            pytest.param(32, 64, (), torch.float16, marks=pytest.mark.smoke),
            pytest.param(32, 64, (), torch.bfloat16, marks=pytest.mark.smoke),
            pytest.param(8, 64, (32, 32), torch.float16, marks=pytest.mark.full),
            pytest.param(4, 128, (32, 32), torch.bfloat16, marks=pytest.mark.full),
            # Non-persistent backward path (L=16384 > 8192).
            pytest.param(4, 64, (64, 64), torch.float16, marks=pytest.mark.full),
            # Non-aligned spatial: H*W=900, exercises partial-tile path
            pytest.param(8, 64, (30, 30), torch.float16, marks=pytest.mark.full),
            pytest.param(8, 64, (30, 30), torch.bfloat16, marks=pytest.mark.full),
        ]),
    ]


# Test helpers


# Test functions

@BatchNormFwdFixture
def test_batch_norm_fwd(N, C, spatial, dtype, training):
    test = BatchNormFwdTest(N, C, spatial, dtype, training)
    x, weight, bias, running_mean, running_var = test.gen_inputs()

    # Clone before op call so reference sees the same initial state.
    running_mean_ref = running_mean.clone()
    running_var_ref = running_var.clone()

    op = BatchNormFwdOp(training=training)
    # Manifest input order: (x, running_mean, running_var, weight, bias).
    y = op(x, running_mean, running_var, weight, bias)

    torch.ptpu.synchronize()
    ref_y, ref_rm, ref_rv = _ref_fwd(
        x.cpu(),
        weight.cpu(),
        bias.cpu(),
        running_mean_ref.cpu(),
        running_var_ref.cpu(),
        training=training,
    )
    y_cpu = y.cpu()

    # float16 accumulates more error; use loose tolerances.
    atol, rtol = (1e-2, 1e-2) if dtype == torch.float16 else (2e-2, 2e-2)
    max_err = (y_cpu.float() - ref_y.float()).abs().max()
    assert torch.allclose(y_cpu.float(), ref_y.float(), atol=atol, rtol=rtol), \
        f"fwd mismatch (training={training}): max_err={max_err:.4e}"

    if training:
        # allclose is masked when running_mean starts near the batch mean; check determinism.
        rm2, rv2 = running_mean_ref.clone(), running_var_ref.clone()
        op(x, rm2, rv2, weight, bias)
        torch.ptpu.synchronize()
        running_mean_cpu = running_mean.cpu()
        running_var_cpu = running_var.cpu()
        rm2_cpu = rm2.cpu()
        rv2_cpu = rv2.cpu()
        det_err = (running_mean_cpu.float() - rm2_cpu.float()).abs().max()
        assert torch.equal(running_mean_cpu, rm2_cpu) and torch.equal(
            running_var_cpu, rv2_cpu,
        ), \
            f"running stats non-deterministic across runs: max_err={det_err:.4e}"

        rm_err = (running_mean_cpu.float() - ref_rm.float()).abs().max()
        assert torch.allclose(
            running_mean_cpu.float(), ref_rm.float(), atol=atol, rtol=rtol,
        ), \
            f"running_mean mismatch: max_err={rm_err:.4e}"
        rv_err = (running_var_cpu.float() - ref_rv.float()).abs().max()
        assert torch.allclose(
            running_var_cpu.float(), ref_rv.float(), atol=atol, rtol=rtol,
        ), \
            f"running_var mismatch: max_err={rv_err:.4e}"



@BatchNormBwdFixture
def test_batch_norm_bwd(N, C, spatial, dtype):
    test = BatchNormBwdTest(N, C, spatial, dtype)
    grad_out, x, weight, mean, rstd = test.gen_inputs()

    op = BatchNormBwdOp()
    grad_x, grad_weight, grad_bias = op(grad_out, x, weight, mean, rstd)

    torch.ptpu.synchronize()
    ref_gx, ref_gw, ref_gb = test.ref_program(
        grad_out.cpu(), x.cpu(), weight.cpu(), mean.cpu(), rstd.cpu(),
    )
    grad_x_cpu = grad_x.cpu()
    grad_weight_cpu = grad_weight.cpu()
    grad_bias_cpu = grad_bias.cpu()

    atol, rtol = (1e-2, 1e-2) if dtype == torch.float16 else (2e-2, 2e-2)

    for name, got, ref in [
        ("grad_x", grad_x_cpu.float(), ref_gx.float()),
        ("grad_weight", grad_weight_cpu.float(), ref_gw.float()),
        ("grad_bias", grad_bias_cpu.float(), ref_gb.float()),
    ]:
        max_err = (got - ref).abs().max()
        assert torch.allclose(got, ref, atol=atol, rtol=rtol), \
            f"bwd {name} mismatch: max_err={max_err:.4e}"


@pytest.mark.smoke
def test_batch_norm_fwd_returns_single_tensor() -> None:
    """BatchNormFwdOp forward must produce one tensor — manifest declares
    a single output. ``training`` is bound at ctor; the runtime kwarg is
    no longer accepted."""
    if not torch.ptpu.is_available():
        pytest.skip("PTPU required for forward call")

    N, C, H, W = 4, 8, 4, 4
    op = BatchNormFwdOp(training=False)
    x = torch.randn(N, C, H, W, device="ptpu", dtype=torch.float16)
    weight = torch.randn(C, device="ptpu", dtype=torch.float32)
    bias = torch.randn(C, device="ptpu", dtype=torch.float32)
    rm = torch.zeros(C, device="ptpu", dtype=torch.float32)
    rv = torch.ones(C, device="ptpu", dtype=torch.float32)

    y = op(x, rm, rv, weight, bias)
    assert isinstance(y, torch.Tensor)
    assert y.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
