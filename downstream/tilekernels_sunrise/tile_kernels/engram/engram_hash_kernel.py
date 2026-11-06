import torch

from tile_kernels.config import get_is_tang, get_num_vec_cores, get_pdl, is_ascend
from tile_kernels.engram.engram_hash_cuda import get_engram_hash_kernel_cuda


def engram_hash(
    ngram_token_ids: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
    image_token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute n-gram hash embedding indices.

    Args:
        ngram_token_ids: N-gram token IDs, shape (num_tokens, max_ngram_size), int32.
        multipliers: Per-layer hash multipliers, shape (num_ngram_layers, max_ngram_size), int64.
        vocab_sizes: Per-layer embedding table sizes,
            shape (num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram), int32.
        offsets: Per-layer embedding table offsets,
            shape (num_ngram_layers, (max_ngram_size - 1) * num_embed_table_per_ngram), int32.
        image_token_mask: Optional, ``True`` stands for image tokens, shape (num_tokens,), bool.

    Returns:
        Embedding indices, shape (num_ngram_layers, num_tokens, (max_ngram_size - 1) * num_embed_table_per_ngram), int32.
    """
    num_tokens, max_ngram_size = ngram_token_ids.shape
    num_ngram_layers, _, num_embed_table_per_ngram = vocab_sizes.shape
    num_out_cols = (max_ngram_size - 1) * num_embed_table_per_ngram

    output = torch.empty((num_ngram_layers, num_tokens, num_out_cols), dtype=torch.int32, device=ngram_token_ids.device)

    if is_ascend():
        from tile_kernels.engram.engram_hash_asc import get_engram_hash_kernel_asc

        kernel = get_engram_hash_kernel_asc(
            max_ngram_size,
            num_ngram_layers,
            num_embed_table_per_ngram,
            has_image_token_mask=image_token_mask is not None,
            num_vec_cores=get_num_vec_cores(),
        )
    else:
        kernel = get_engram_hash_kernel_cuda(
            max_ngram_size,
            num_ngram_layers,
            num_embed_table_per_ngram,
            has_image_token_mask=image_token_mask is not None and not get_is_tang(),
            use_pdl=get_pdl(),
        )

    if num_tokens > 0:
        kernel_mask = image_token_mask
        if get_is_tang() and kernel_mask is None:
            kernel_mask = torch.empty((num_tokens,), dtype=torch.bool, device=ngram_token_ids.device)
        kernel(ngram_token_ids, multipliers, vocab_sizes, offsets, kernel_mask, output)
        if get_is_tang() and image_token_mask is not None:
            from tile_kernels.engram.engram_hash_tang import get_engram_hash_mask_kernel_tang

            mask_kernel = get_engram_hash_mask_kernel_tang(num_ngram_layers, num_out_cols)
            mask_kernel(image_token_mask, output)

    return output
