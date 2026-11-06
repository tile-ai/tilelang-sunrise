import tilelang
from tilelang import language as T

from tile_kernels.moe.sqrt_softplus import sqrt_softplus_backward_cuda
from tile_kernels.utils import align


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_OUT_OF_BOUND_WARNING: True,
    },
)
def get_moe_topk_gate_backward_kernel_cuda(
    num_topk: int,
    num_routed_experts: int,
    aux_exists: bool,
    use_pdl: bool = False,
):
    # Kernel config
    warp_size = 32
    num_threads = 32
    assert num_topk <= warp_size, f'num_topk must be less than or equal to {warp_size}'

    # Each warp handles one token; the expert dimension is padded to the warp size
    num_aligned_experts = align(num_routed_experts, warp_size)

    # Runtime symbols
    num_tokens = T.dynamic('num_tokens')
    num_seqs = T.dynamic('num_seqs')

    @T.prim_func
    def moe_topk_gate_backward_kernel(
        scores: T.Tensor[(num_tokens, num_routed_experts), T.float32],
        topk_idx: T.Tensor[(num_tokens, num_topk), T.int64],
        topk_weights: T.Tensor[(num_tokens, num_topk), T.float32],
        grad_topk_weights: T.Tensor[(num_tokens, num_topk), T.float32],
        mask: T.Tensor[(num_tokens,), T.bool],
        grad_scores_sum: T.Tensor[(num_seqs, num_routed_experts), T.float32],
        grad_logits: T.Tensor[(num_tokens, num_routed_experts), T.float32],
        routed_scaling_factor: T.float32,
    ):
        T.assume(num_tokens > 0)
        if aux_exists:
            T.assume(num_seqs > 0)
        with T.Kernel(num_tokens, threads=num_threads) as pid:
            if use_pdl:
                T.pdl_sync()

            scores_fragment = T.alloc_fragment((num_aligned_experts,), T.float32)
            grad_scores_sum_fragment = T.alloc_fragment((num_aligned_experts,), T.float32)
            grad_logits_fragment = T.alloc_fragment((num_aligned_experts,), T.float32)
            route_grad_shared = T.alloc_shared((num_aligned_experts,), T.float32)

            topk_idx_fragment = T.alloc_fragment((num_topk,), T.int64)
            topk_scores_fragment = T.alloc_fragment((num_topk,), T.float32)
            topk_weights_fragment = T.alloc_fragment((num_topk,), T.float32)
            grad_topk_weights_fragment = T.alloc_fragment((num_topk,), T.float32)
            topk_sum_reducer = T.alloc_reducer((1,), T.float32, op='sum')
            dot_reducer = T.alloc_reducer((1,), T.float32, op='sum')
            topk_sum_fragment = T.alloc_fragment((1,), T.float32)
            dot_fragment = T.alloc_fragment((1,), T.float32)

            # Load scores `u = sqrt(softplus(logits))` from global memory
            T.clear(route_grad_shared)
            for i in T.Parallel(num_aligned_experts):
                scores_fragment[i] = T.if_then_else(i < num_routed_experts, scores[pid, i], T.float32(0.0))

            # Gather the top-k selections. Masked (PAD) positions carry `-1`: their score, weight and incoming
            # gradient must not contribute.
            T.copy(topk_idx[pid, :], topk_idx_fragment)
            T.copy(topk_weights[pid, :], topk_weights_fragment)
            T.copy(grad_topk_weights[pid, :], grad_topk_weights_fragment)
            T.reducer_init(topk_sum_reducer)
            T.reducer_init(dot_reducer)
            for k in T.Parallel(num_topk):
                is_valid = topk_idx_fragment[k] >= 0
                topk_scores_fragment[k] = T.if_then_else(is_valid, scores[pid, T.max(topk_idx_fragment[k], 0)], T.float32(0.0))
                grad_topk_weights_fragment[k] = T.if_then_else(is_valid, grad_topk_weights_fragment[k], T.float32(0.0))
                T.reducer_update(topk_sum_reducer[0], topk_scores_fragment[k])
                T.reducer_update(dot_reducer[0], grad_topk_weights_fragment[k] * topk_weights_fragment[k])
            T.finalize_reducer(topk_sum_reducer, topk_sum_fragment)
            T.finalize_reducer(dot_reducer, dot_fragment)

            # Route term: w_j = s * u_j / topk_sum  =>  dL/du_j = (s * g_j - <g, w>) / topk_sum, scattered back to
            # the dense expert dimension. Duplicated experts (fixed routing) accumulate.
            # NOTE: `topk_sum = sum_j u_j + 1e-20` aligns with the forward normalization
            for k in T.Parallel(num_topk):
                if topk_idx_fragment[k] >= 0:
                    T.atomic_add(
                        route_grad_shared[topk_idx_fragment[k]],
                        (routed_scaling_factor * grad_topk_weights_fragment[k] - dot_fragment[0]) / (topk_sum_fragment[0] + 1e-20),
                    )

            if aux_exists:
                # Aux term: the aux loss consumes `u / sum(u)` per token, whose upstream gradient `h` is shared by the
                # whole sequence  =>  dL/du = (h - <h, u> * inv) * inv with inv = 1 / sum(u).
                scores_sum_reducer = T.alloc_reducer((1,), T.float32, op='sum')
                aux_dot_reducer = T.alloc_reducer((1,), T.float32, op='sum')
                scores_sum_fragment = T.alloc_fragment((1,), T.float32)
                aux_dot_fragment = T.alloc_fragment((1,), T.float32)
                T.reducer_init(scores_sum_reducer)
                T.reducer_init(aux_dot_reducer)
                for i in T.Parallel(num_aligned_experts):
                    grad_scores_sum_fragment[i] = T.if_then_else(
                        i < num_routed_experts, grad_scores_sum[pid // (num_tokens // num_seqs), i], T.float32(0.0)
                    )
                for i in T.Parallel(num_aligned_experts):
                    T.reducer_update(scores_sum_reducer[0], scores_fragment[i])
                    T.reducer_update(aux_dot_reducer[0], grad_scores_sum_fragment[i] * scores_fragment[i])
                T.finalize_reducer(scores_sum_reducer, scores_sum_fragment)
                T.finalize_reducer(aux_dot_reducer, aux_dot_fragment)
                inv_sum = T.if_then_else(
                    mask[pid],
                    1.0 / scores_sum_fragment[0],
                    T.float32(0.0),
                )
                for i in T.Parallel(num_aligned_experts):
                    grad_logits_fragment[i] = sqrt_softplus_backward_cuda(scores_fragment[i]) * (
                        route_grad_shared[i] + (grad_scores_sum_fragment[i] - aux_dot_fragment[0] * inv_sum) * inv_sum
                    )
            else:
                for i in T.Parallel(num_aligned_experts):
                    grad_logits_fragment[i] = sqrt_softplus_backward_cuda(scores_fragment[i]) * route_grad_shared[i]

            # Store to global memory
            for i in T.Parallel(num_aligned_experts):
                if i < num_routed_experts:
                    grad_logits[pid, i] = grad_logits_fragment[i]

            if use_pdl:
                T.pdl_trigger()

    return moe_topk_gate_backward_kernel
