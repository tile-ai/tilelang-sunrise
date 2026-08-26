import torch

from tile_kernels.mhc.expand_kernel import (
    _expand_to_mhc_fwd_fallback,
    choose_expand_blocks,
    expand_to_mhc_bwd_tl,
    expand_to_mhc_fwd_tl,
)


class ExpandToMHCFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: 'ExpandToMHCFn',
        hidden: torch.Tensor,
        mhc_mult: int,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        if out is None:
            out = hidden.new_empty(*hidden.shape[:-1], mhc_mult, hidden.shape[-1])
        assert hidden.is_contiguous()
        h_dim = hidden.shape[-1]
        n_tokens = hidden.flatten(0, -2).shape[0]

        blk_n, blk_h = choose_expand_blocks(h_dim, mhc_mult, n_tokens)

        if n_tokens > 0 and n_tokens % blk_n == 0 and h_dim % blk_h == 0:
            # Fast path: aligned dimensions → vectorized T.copy kernel
            kernel = expand_to_mhc_fwd_tl(h_dim, mhc_mult, blk_n, blk_h)
            kernel(
                hidden.flatten(0, -2),
                out.flatten(0, -3).reshape(-1, mhc_mult * h_dim),
            )
        else:
            # Fallback: unaligned dimensions → original element-wise kernel
            kernel = _expand_to_mhc_fwd_fallback(h_dim, mhc_mult)
            kernel(hidden.flatten(0, -2), out.flatten(0, -3))

        return out

    @staticmethod
    def backward(ctx: 'ExpandToMHCFn', out_grad: torch.Tensor) -> torch.Tensor:
        hidden_grad = out_grad.new_empty(*out_grad.shape[:-2], out_grad.shape[-1])
        kernel = expand_to_mhc_bwd_tl(out_grad.shape[-1], out_grad.shape[-2])
        kernel(out_grad.flatten(0, -3), hidden_grad.flatten(0, -2))
        return hidden_grad, None, None


def expand_to_mhc(
    hidden: torch.Tensor,
    mhc_mult: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    return ExpandToMHCFn.apply(hidden, mhc_mult, out)
