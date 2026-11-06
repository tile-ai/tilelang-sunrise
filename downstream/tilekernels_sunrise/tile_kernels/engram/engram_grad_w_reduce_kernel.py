import torch

from tile_kernels.config import get_is_tang, get_num_vec_cores, get_pdl, is_ascend
from tile_kernels.engram.engram_grad_w_reduce_cuda import (
    get_engram_grad_w_reduce_kernel_cuda,
)


def grad_w_reduce(
    grad_w_partial: torch.Tensor,
    weight_hidden: torch.Tensor,
    weight_embed: torch.Tensor,
    grad_weight_hidden: torch.Tensor,
    grad_weight_embed: torch.Tensor,
) -> None:
    """Reduce grad_w_partial over persistent blocks, fused with weight multiply, accumulating into grad_weight tensors.

    Args:
        grad_w_partial: Partial weight gradients, shape (num_persistent_blocks, hc_mult, hidden_size), float32.
        weight_hidden: RMSNorm weight for hidden states, shape (hc_mult, hidden_size), bfloat16.
        weight_embed: RMSNorm weight for key embeddings, shape (hc_mult, hidden_size), bfloat16.
        grad_weight_hidden: Accumulated gradient for weight_hidden, shape (hc_mult, hidden_size), float32. Modified in-place.
        grad_weight_embed: Accumulated gradient for weight_embed, shape (hc_mult, hidden_size), float32. Modified in-place.
    """
    num_persistent_blocks, hc_mult, hidden_size = grad_w_partial.shape

    if get_is_tang():
        from tile_kernels.engram.engram_grad_w_reduce_tang import get_engram_grad_w_reduce_kernel_tang

        kernel = get_engram_grad_w_reduce_kernel_tang(hidden_size, num_persistent_blocks, hc_mult)
    elif is_ascend():
        from tile_kernels.engram.engram_grad_w_reduce_asc import (
            get_engram_grad_w_reduce_kernel_asc,
        )

        kernel = get_engram_grad_w_reduce_kernel_asc(
            hidden_size,
            num_persistent_blocks,
            hc_mult,
            get_num_vec_cores(),
        )
    else:
        kernel = get_engram_grad_w_reduce_kernel_cuda(
            hidden_size,
            num_persistent_blocks,
            hc_mult,
            use_pdl=get_pdl(),
        )

    kernel(grad_w_partial, weight_hidden, weight_embed, grad_weight_hidden, grad_weight_embed)
