"""TANG weight-gradient accumulation with automatic storage synchronization enabled."""
import tilelang
from tilelang import language as T


@tilelang.jit(target='tang', pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def get_engram_grad_w_reduce_kernel_tang(hidden_size: int, num_persistent_blocks: int, hc_mult: int):
    block = 256
    assert hidden_size % block == 0

    @T.prim_func
    def kernel(
        grad_w_partial: T.Tensor[(num_persistent_blocks, hc_mult, hidden_size), T.float32],
        weight_hidden: T.Tensor[(hc_mult, hidden_size), T.bfloat16],
        weight_embed: T.Tensor[(hc_mult, hidden_size), T.bfloat16],
        grad_weight_hidden: T.Tensor[(hc_mult, hidden_size), T.float32],
        grad_weight_embed: T.Tensor[(hc_mult, hidden_size), T.float32],
    ):
        with T.Kernel(hc_mult, hidden_size // block, threads=128) as (head, tile):
            grad = T.alloc_fragment((block,), T.float32)
            T.clear(grad)
            for part in T.serial(num_persistent_blocks):
                for i in T.Parallel(block):
                    grad[i] += grad_w_partial[part, head, tile * block + i]
            for i in T.Parallel(block):
                grad_weight_hidden[head, tile * block + i] += grad[i] * T.cast(weight_embed[head, tile * block + i], T.float32)
                grad_weight_embed[head, tile * block + i] += grad[i] * T.cast(weight_hidden[head, tile * block + i], T.float32)

    return kernel
