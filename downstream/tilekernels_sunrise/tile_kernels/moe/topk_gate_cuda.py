import tilelang
from tilelang import language as T

from tile_kernels.utils import align


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def get_topk_gate_kernel_cuda(num_experts: int, num_topk: int, use_pdl: bool = False):
    num_tokens = T.dynamic('num_tokens')
    num_threads = 32
    num_aligned_experts = align(num_experts, num_threads)

    @T.prim_func
    def topk_gate_kernel(
        scores: T.Tensor[(num_tokens, num_experts), T.float32],
        topk_idx: T.Tensor[(num_tokens, num_topk), T.int64],
    ):
        with T.Kernel(num_tokens, threads=num_threads) as pid:
            if use_pdl:
                T.pdl_sync()
            scores_fragment = T.alloc_fragment((num_aligned_experts,), T.float32)
            amax_fragment = T.alloc_fragment((1,), T.float32)
            idx_fragment = T.alloc_fragment((num_aligned_experts,), T.int32)
            idx_reducer = T.alloc_reducer((1,), T.int32, op='min')
            idx_result = T.alloc_fragment((1,), T.int32)
            nonfinite_reducer = T.alloc_reducer((1,), T.int32, op='sum')
            nonfinite_result = T.alloc_fragment((1,), T.int32)
            topk_idx_shared = T.alloc_shared((num_topk,), T.int32)

            for i in T.Parallel(num_aligned_experts):
                if i < num_experts:
                    scores_fragment[i] = scores[pid, i]
                else:
                    scores_fragment[i] = -T.infinity(T.float32)

            # Invalid input check, single deferred assert to keep the load loop vectorized
            T.reducer_init(nonfinite_reducer)
            for i in T.Parallel(num_aligned_experts):
                if i < num_experts and not T.isfinite(scores_fragment[i]):
                    T.reducer_update(nonfinite_reducer[0], 1)
            T.finalize_reducer(nonfinite_reducer, nonfinite_result)
            T.device_assert(nonfinite_result[0] == 0, msg='Score input for topk gate must be finite')

            for i in T.Parallel(num_aligned_experts):
                idx_fragment[i] = i

            # Get topk via repeatedly finding max
            for k in T.unroll(num_topk):
                T.reduce_max(scores_fragment, amax_fragment)
                T.reducer_init(idx_reducer)
                for i in T.Parallel(num_aligned_experts):
                    if scores_fragment[i] == amax_fragment[0]:
                        T.reducer_update(idx_reducer[0], idx_fragment[i])
                T.finalize_reducer(idx_reducer, idx_result)
                topk_idx_shared[k] = idx_result[0]
                for i in T.Parallel(num_aligned_experts):
                    if idx_fragment[i] == idx_result[0]:
                        scores_fragment[i] = -T.infinity(T.float32)

            T.copy(topk_idx_shared, topk_idx[pid, 0], disable_tma=True)
            if use_pdl:
                T.pdl_trigger()

    return topk_gate_kernel
