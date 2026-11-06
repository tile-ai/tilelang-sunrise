"""Autograd composition contracts; CPU stubs test the wrapper, not device kernels."""

import pytest
import torch

from tileops.backend import BUILTIN
from tileops.linear_attention import DeltaNetAutogradOp
from workloads.linear_attention.deltanet import deltanet_differentiable_fwd_torch

pytestmark = pytest.mark.smoke


def _inputs(dtype=torch.float32):
    g = torch.Generator().manual_seed(187)
    return tuple(
        (torch.randn(shape, generator=g, dtype=dtype) * 0.1).requires_grad_()
        for shape in ((1, 1, 8, 4), (1, 1, 8, 4), (1, 1, 8, 3), (1, 1, 8))
    )


def _cpu_protocol_op():
    op = DeltaNetAutogradOp(4, target=BUILTIN)
    saved = {}

    def fwd(q, k, v, beta):
        o = deltanet_differentiable_fwd_torch(q, k, v, beta, 4)
        state = tuple(torch.zeros(1) for _ in range(5))
        saved["tensors"] = (q, k, v, beta, *state)
        return (o, *state)

    def bwd(do, *tensors):
        assert do.is_contiguous()
        assert all(a is b for a, b in zip(tensors, saved["tensors"], strict=True))
        with torch.enable_grad():
            inputs = tuple(t.detach().requires_grad_() for t in tensors[:4])
            o = deltanet_differentiable_fwd_torch(*inputs, 4)
            return torch.autograd.grad(o, inputs, do)

    op._fwd, op._bwd = fwd, bwd
    return op


def test_deltanet_autograd_composes_gradients_and_reuses_saved_buffers():
    inputs = _inputs()
    refs = tuple(t.detach().clone().requires_grad_() for t in inputs)
    op = _cpu_protocol_op()
    got = op(*inputs)
    expected = deltanet_differentiable_fwd_torch(*refs, 4)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    # Autograd supplies a zero-stride gradient to sum(); the BWD entry receives contiguous.
    grads = torch.autograd.grad(got.sum(), inputs, retain_graph=True)
    gold = torch.autograd.grad(expected.sum(), refs)
    for a, b in zip(grads, gold, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    repeat = torch.autograd.grad(got.sum(), inputs)
    for a, b in zip(repeat, gold, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_deltanet_autograd_preserves_saved_tensor_version_checks():
    inputs = _inputs()
    got = _cpu_protocol_op()(*inputs)
    with torch.no_grad():
        inputs[1].add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        got.sum().backward()


def test_deltanet_autograd_exports_current_signature_and_delegates():
    from tileops._ops.linear_attention.deltanet import DeltaNetAutogradOp as Family
    from tileops.linear_attention import DeltaNetAutogradOp as Public

    assert Public is Family is DeltaNetAutogradOp
    op = DeltaNetAutogradOp(4, target=BUILTIN)
    assert tuple(op._signature.sig.inputs) == ("q", "k", "v", "beta")
    assert tuple(op._signature.sig.outputs) == ("o",)
    assert [type(x).__name__ for x in op._held_delegates()] == [
        "DeltaNetChunkFwdOp",
        "DeltaNetChunkBwdOp",
    ]
    with pytest.raises(ValueError, match="chunk_size"):
        op(*(x[:, :, :7] for x in _inputs()))
