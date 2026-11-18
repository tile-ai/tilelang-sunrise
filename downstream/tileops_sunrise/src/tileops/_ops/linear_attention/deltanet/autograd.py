"""PyTorch autograd composition of the chunked DeltaNet entries."""

from typing import ClassVar, Mapping

import torch
from torch.autograd.function import once_differentiable

from tileops._ops.linear_attention.deltanet.chunk import DeltaNetChunkBwdOp, DeltaNetChunkFwdOp
from tileops._ops.op_base import Op
from tileops.backend import Target

__all__ = ["DeltaNetAutogradOp"]


class _DeltaNetFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, beta, fwd, bwd):
        """Run the chunk forward entry and save tensors for its backward entry.

        Args:
            q: Queries, BHSD.
            k: Keys, BHSD.
            v: Values, BHSD.
            beta: Per-token update strength.
            fwd: The held chunk forward op.
            bwd: The held chunk backward op.

        Returns:
            The output, BHSD.
        """
        o, state, aw, au, w, u = fwd(q, k, v, beta)
        ctx.save_for_backward(q, k, v, beta, state, aw, au, w, u)
        ctx.bwd = bwd
        return o

    @staticmethod
    @once_differentiable
    def backward(ctx, do):
        # The chunk entry expects its gradient's BHSD storage to be contiguous;
        # autograd can supply a zero-stride expansion for a scalar loss.
        gradients = ctx.bwd(do.contiguous(), *ctx.saved_tensors)
        return (*gradients, None, None)


class DeltaNetAutogradOp(Op):
    """BHSD DeltaNet with first-order gradients for q, k, v and beta.

    Shapes are taken from each call. The held forward/backward ops use the current
    Entry and CallSpec caches, including the normal ``request_tune`` policy.
    """

    delegate_types: ClassVar[Mapping[str, type[Op]]] = {
        "chunk_fwd": DeltaNetChunkFwdOp,
        "chunk_bwd": DeltaNetChunkBwdOp,
    }

    def __init__(self, chunk_size: int = 64, *, target: Target = None) -> None:
        """Construct the paired chunk forward and backward delegates.

        Args:
            chunk_size: Tokens per chunk.
            target: The backend target, ``None`` for the default.
        """
        self.chunk_size = chunk_size
        super().__init__(target=target)
        self._fwd = self.delegate_for("chunk_fwd", None, chunk_size=chunk_size)
        self._bwd = self.delegate_for("chunk_bwd", None, chunk_size=chunk_size)

    def forward(self, q, k, v, beta):
        """Evaluate DeltaNet with first-order autograd support.

        Args:
            q: Queries, BHSD.
            k: Keys, BHSD.
            v: Values, BHSD.
            beta: Per-token update strength.

        Returns:
            The output, BHSD.
        """
        return _DeltaNetFunction.apply(q, k, v, beta, self._fwd, self._bwd)
