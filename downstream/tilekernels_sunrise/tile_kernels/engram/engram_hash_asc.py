import tilelang
from tilelang.ascend import language as T


@tilelang.jit
def get_engram_hash_kernel_asc(
    max_ngram_size: int,
    num_ngram_layers: int,
    num_embed_table_per_ngram: int,
    has_image_token_mask: bool,
    num_vec_cores: int,
):
    num_tokens = T.dynamic('num_tokens')
    num_out_cols = (max_ngram_size - 1) * num_embed_table_per_ngram

    tile = 128
    threads = tile

    @T.prim_func
    def engram_hash_kernel_asc(
        ngram_token_ids: T.Tensor[(num_tokens, max_ngram_size), T.int32],
        multipliers: T.Tensor[(num_ngram_layers, max_ngram_size), T.int64],
        vocab_sizes: T.Tensor[(num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram), T.int32],
        offsets: T.Tensor[(num_ngram_layers, num_out_cols), T.int32],
        image_token_mask: T.Tensor[(num_tokens,), T.bool],
        output: T.Tensor[(num_ngram_layers, num_tokens, num_out_cols), T.int32],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            multipliers_ub = T.alloc_shared((num_ngram_layers, max_ngram_size), T.int64)
            vocab_sizes_ub = T.alloc_shared((num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram), T.int32)
            offsets_ub = T.alloc_shared((num_ngram_layers, num_out_cols), T.int32)
            token_ids_ub = T.alloc_shared((tile, max_ngram_size), T.int32)
            image_token_mask_ub = T.alloc_shared((tile,), T.bool)
            output_ub = T.alloc_shared((num_ngram_layers, tile, num_out_cols), T.int32)
            T.annotate_buffer_versions(
                {
                    token_ids_ub: 1,
                    image_token_mask_ub: 1,
                    output_ub: 1,
                }
            )

            T.copy(multipliers, multipliers_ub)
            T.copy(vocab_sizes, vocab_sizes_ub)
            T.copy(offsets, offsets_ub)

            for tile_id in T.Persistent(
                [T.ceildiv(num_tokens, tile)],
                num_vec_cores,
                core_id,
                group_size=1,
                num_stages=0,
            ):
                row = tile_id * tile
                valid_rows = T.min(tile, num_tokens - row)

                T.copy(ngram_token_ids[row : row + valid_rows, :], token_ids_ub[:valid_rows, :])
                if has_image_token_mask:
                    T.copy(image_token_mask[row : row + valid_rows], image_token_mask_ub[:valid_rows])

                with T.SimtVF(threads=threads):
                    for token_in_tile in T.Parallel(tile):
                        hash_local = T.alloc_var(T.int64)

                        if token_in_tile < valid_rows:
                            if has_image_token_mask and image_token_mask_ub[token_in_tile]:
                                for layer_idx in T.unroll(num_ngram_layers):
                                    for col in T.unroll(num_out_cols):
                                        output_ub[layer_idx, token_in_tile, col] = -1
                            else:
                                for layer_idx in T.unroll(num_ngram_layers):
                                    hash_local = 0
                                    for ngram_idx in T.unroll(max_ngram_size):
                                        hash_local = T.bitwise_xor(
                                            hash_local,
                                            T.cast(token_ids_ub[token_in_tile, ngram_idx], T.int64) * multipliers_ub[layer_idx, ngram_idx],
                                        )
                                        if ngram_idx > 0:
                                            for table_idx in T.unroll(num_embed_table_per_ngram):
                                                col = (ngram_idx - 1) * num_embed_table_per_ngram + table_idx
                                                output_ub[layer_idx, token_in_tile, col] = (
                                                    hash_local % T.cast(vocab_sizes_ub[layer_idx, ngram_idx - 1, table_idx], T.int64)
                                                    + offsets_ub[layer_idx, col]
                                                )

                T.copy(output_ub[:, :valid_rows, :], output[:, row : row + valid_rows, :])

    return engram_hash_kernel_asc
