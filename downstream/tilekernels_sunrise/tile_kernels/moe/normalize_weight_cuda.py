import tilelang
from tilelang import language as T

from tile_kernels.config import get_is_tang


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def get_normalize_weight_kernel_cuda(num_topk: int, use_pdl: bool = False):
    num_threads = 128

    num_tokens = T.dynamic('num_tokens')
    num_blocks = T.ceildiv(num_tokens, 128)

    @T.prim_func
    def normalize_weight_kernel(
        topk_weights: T.Tensor[(num_tokens, num_topk), T.float32],
        denominator: T.Tensor[(num_tokens,), T.float32],
        normalized_weights: T.Tensor[(num_tokens, num_topk), T.float32],
    ):
        with T.Kernel(num_blocks, threads=num_threads) as (pid,):
            if use_pdl:
                T.pdl_sync()
            tid = T.get_thread_binding()
            weights_local = T.alloc_local((num_topk,), T.float32)
            sum_local = T.alloc_local((num_topk,), T.float32)
            row = pid * num_threads + tid

            if row < num_tokens:
                # NOTE: Align with moe_topk_gate_forward kernel implementation
                sum = T.alloc_var(T.float32, init=1e-20)
                for i in T.vectorized(num_topk):
                    weights_local[i] = topk_weights[row, i]
                    sum_local[i] = weights_local[i]

                for i in T.unroll(0, 5, explicit=get_is_tang()):
                    mask = 1 << (4 - i)
                    for j in T.unroll(0, num_topk, explicit=get_is_tang()):
                        if (j & mask) == 0 and (j ^ mask) < num_topk:
                            sum_local[j] += sum_local[j ^ mask]
                sum += sum_local[0]

                denominator[row] = sum
                for i in T.vectorized(num_topk):
                    normalized_weights[row, i] = weights_local[i] / sum
            if use_pdl:
                T.pdl_trigger()

    return normalize_weight_kernel
