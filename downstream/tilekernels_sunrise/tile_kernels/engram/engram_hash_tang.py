import tilelang
from tilelang import language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def get_engram_hash_mask_kernel_tang(num_ngram_layers: int, num_out_cols: int):
    num_tokens = T.dynamic('num_tokens', dtype='int64')
    threads = 128

    @T.prim_func
    def engram_hash_mask_kernel_tang(
        image_token_mask: T.Tensor[(num_tokens,), T.bool],
        output: T.Tensor[(num_ngram_layers, num_tokens, num_out_cols), T.int32],
    ):
        with T.Kernel(T.ceildiv(T.cast(num_tokens, T.int64) * num_out_cols, threads), num_ngram_layers, threads=threads) as (pid, layer):
            for i in T.Parallel(threads):
                index = T.cast(pid, T.int64) * threads + i
                if index < T.cast(num_tokens, T.int64) * num_out_cols:
                    token = index // num_out_cols
                    col = index % num_out_cols
                    if image_token_mask[token]:
                        output[T.cast(layer, T.int64), token, col] = -1

    return engram_hash_mask_kernel_tang
