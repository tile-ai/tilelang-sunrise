import torch

from tile_kernels.config import (
    get_num_sms,
    get_is_tang,
    get_num_vec_cores,
    get_pdl,
    is_ascend,
)
from tile_kernels.engram.engram_gate_bwd_cuda import get_engram_gate_bwd_kernel_cuda


def engram_gate_bwd(
    grad_out: torch.Tensor,
    hidden_states: torch.Tensor,
    kv: torch.Tensor,
    weight_fused: torch.Tensor,
    dot: torch.Tensor,
    gate_score: torch.Tensor,
    rstd_x: torch.Tensor,
    rstd_k: torch.Tensor,
    clamp_value: float,
    image_token_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Engram gate backward: computes grad_hidden_states, grad_kv, grad_w_partial.

    Args:
        grad_out: Gradient of output, shape (num_tokens, hc_mult, hidden_size), bfloat16.
        hidden_states: Original input from forward, shape (num_tokens, hc_mult, hidden_size), bfloat16.
        kv: Key/value embeddings from forward, shape (num_tokens, hc_mult + 1, hidden_size), bfloat16.
        weight_fused: Fused RMSNorm weight (weight_hidden * weight_embed), shape (hc_mult, hidden_size), float32.
        dot: Saved raw dot product from forward, shape (num_tokens, hc_mult), float32.
        gate_score: Saved gate scores from forward, shape (num_tokens, hc_mult), float32.
        rstd_x: Saved reciprocal std of x from forward, shape (num_tokens, hc_mult), float32.
        rstd_k: Saved reciprocal std of k from forward, shape (num_tokens, hc_mult), float32.
        clamp_value: Clamp threshold for signed-sqrt gate activation.
        image_token_mask: Optional, ``True`` stands for image tokens, shape (num_tokens,), bool.

    Returns:
        tuple: (grad_hidden_states, grad_kv, grad_w_partial) where
        grad_w_partial needs further reduction over its first dimension.
    """
    _, hc_mult, hidden_size = hidden_states.shape
    scalar = hidden_size**-0.5
    if get_is_tang():
        from tile_kernels.engram.engram_gate_tang import get_engram_gate_bwd_kernel_tang

        num_partials = get_num_sms()
        kernel = get_engram_gate_bwd_kernel_tang(
            hidden_size, scalar, num_partials, clamp_value, hc_mult, image_token_mask is not None,
        )
    elif is_ascend():
        from tile_kernels.engram.engram_gate_bwd_asc import (
            get_engram_gate_bwd_kernel_asc,
        )

        num_vec_cores = get_num_vec_cores()
        kernel = get_engram_gate_bwd_kernel_asc(
            hidden_size,
            scalar,
            clamp_value,
            hc_mult,
            has_image_token_mask=image_token_mask is not None,
            num_vec_cores=num_vec_cores,
        )
        num_partials = num_vec_cores // hc_mult
    else:
        kernel = get_engram_gate_bwd_kernel_cuda(
            hidden_size,
            scalar,
            get_num_sms(),
            clamp_value,
            hc_mult,
            has_image_token_mask=image_token_mask is not None,
            use_pdl=get_pdl(),
        )
        num_partials = get_num_sms()

    grad_hidden_states = torch.empty_like(hidden_states)
    grad_kv = torch.empty_like(kv)

    grad_w_partial = torch.empty(
        (num_partials, hc_mult, hidden_size),
        dtype=torch.float32,
        device=hidden_states.device,
    )

    kernel_mask = image_token_mask
    if get_is_tang() and kernel_mask is None:
        kernel_mask = torch.zeros((hidden_states.shape[0],), dtype=torch.bool, device=hidden_states.device)
    kernel(
        grad_out, hidden_states, kv, weight_fused,
        dot, gate_score, rstd_x, rstd_k,
        kernel_mask,
        grad_hidden_states, grad_kv, grad_w_partial,
    )  # fmt: off
    if get_is_tang() and hidden_states.shape[0] > 0:
        from tile_kernels.engram.engram_gate_tang import get_engram_gate_grad_v_kernel_tang

        # Ordinary launches in the caller's stream; no PDL or cross-CTA partial ownership.
        # Keep the validated mask-aware V schedule for the unmasked case too.
        get_engram_gate_grad_v_kernel_tang(hidden_size, hc_mult, True)(
            grad_out, gate_score, kernel_mask, grad_kv,
        )
    return grad_hidden_states, grad_kv, grad_w_partial
