import tilelang
from tilelang import language as T

from tile_kernels.utils import align, ceil_div
from tile_kernels.moe.sqrt_softplus import sqrt_softplus_forward_cuda


@T.macro
def warp_reduce_sum(x: T.Ref):
    # Keep the same with the old implementation
    for i in T.unroll(0, 5):
        x += T.shfl_xor(x, 1 << (4 - i))


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_OUT_OF_BOUND_WARNING: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def get_moe_topk_gate_forward_kernel_cuda(
    num_topk: int,
    num_routed_experts: int,
    mask_exists: bool, fix_routing_mask_exists: bool,
    unmapped_topk_idx_exists: bool, to_physical_map_exists: bool,
    bias_exists: bool, image_token_mask_exists: bool,
    force_random_exists: bool, scores_exists: bool,
    use_pdl: bool = False,
    num_physical_topk_static: int | None = None,
):  # fmt: off
    # Kernel config
    warp_size = 32
    # Avoid mixing random and regular routing warps in the same TANG block.
    num_warps = 1 if num_physical_topk_static is not None else 2
    num_threads = warp_size * num_warps
    assert num_topk <= warp_size, f'num_topk must be less than or equal to {warp_size}'

    # Each warp handles one token
    num_tokens_per_block = num_threads // warp_size

    # Keep the same with the old implementation
    large_prime_number = 23333
    num_vectorize = 4
    assert num_routed_experts % num_vectorize == 0, f'`num_routed_experts` must be divisible by {num_vectorize}, but got {num_routed_experts}'

    # Runtime symbols
    num_tokens = T.dynamic('num_tokens')
    unmapped_topk_idx_stride = T.dynamic('unmapped_topk_idx_stride')
    # num_physical_topk = num_topk + num_shared_experts
    num_physical_topk = T.dynamic('num_physical_topk') if num_physical_topk_static is None else num_physical_topk_static
    # num_logical_experts = num_routed_experts + num_shared_experts
    num_logical_experts = T.dynamic('num_logical_experts')
    # num_duplicate_experts is the duplicate count per logical expert.
    num_duplicate_experts = T.dynamic('num_duplicate_experts')

    # Number of routed experts each thread needs to handle in top-k selection
    num_routed_experts_per_thread = align(ceil_div(num_routed_experts, warp_size), num_vectorize)
    assert num_routed_experts_per_thread % num_vectorize == 0

    @T.prim_func
    def moe_topk_gate_forward_kernel(
        logits: T.Tensor[(num_tokens, num_routed_experts), T.float32],
        bias: T.Tensor[(num_routed_experts), T.float32],
        image_bias: T.Tensor[(num_routed_experts), T.float32],
        image_token_mask: T.Tensor[(num_tokens,), T.bool],
        mask: T.Tensor[(num_tokens,), T.bool],
        fix_routing_mask: T.Tensor[(num_tokens,), T.bool],
        to_physical_map: T.Tensor[(num_logical_experts, num_duplicate_experts), T.int32],
        logical_count: T.Tensor[(num_logical_experts), T.int32],
        topk_idx: T.Tensor[(num_tokens, num_physical_topk), T.int64],
        unmapped_topk_idx: T.StridedTensor[(num_tokens, num_topk), (unmapped_topk_idx_stride, 1), T.int64],
        topk_weights: T.Tensor[(num_tokens, num_physical_topk), T.float32],
        scores: T.Tensor[(num_tokens, num_routed_experts), T.float32],
        force_random: T.Tensor[(num_tokens,), T.bool],
        num_shared_experts: T.int32,
        routed_scaling_factor: T.float32,
        ep_rank: T.int32,
    ):
        with T.Kernel(T.ceildiv(num_tokens, num_tokens_per_block), threads=num_threads) as pid:
            if use_pdl:
                T.pdl_sync()

            thread_idx = T.get_thread_binding()
            token_idx = thread_idx // 32
            global_token_idx = token_idx + pid * num_tokens_per_block
            lane_idx = thread_idx % 32

            if global_token_idx >= num_tokens:
                if use_pdl:
                    T.pdl_trigger()
                T.thread_return()

            scores_wo_bias_shared = T.alloc_shared((num_tokens_per_block, num_routed_experts), dtype=T.float32)

            bias_local = T.alloc_local((num_routed_experts_per_thread,), dtype=T.float32)
            scores_local = T.alloc_local((num_routed_experts_per_thread,), dtype=T.float32)
            idx_local = T.alloc_local((num_routed_experts_per_thread,), dtype=T.int32)
            winner_score = T.alloc_var(dtype=T.float32)
            winner_idx = T.alloc_var(dtype=T.int32, init=-1)
            previous_winner_idx = T.alloc_var(dtype=T.int32, init=-1)

            is_image_token = T.alloc_var(dtype=T.bool)
            other_idx = T.alloc_var(dtype=T.int32)
            topk_score_var = T.alloc_var(dtype=T.float32)
            topk_idx_var = T.alloc_var(dtype=T.int64)
            topk_sum_var = T.alloc_var(dtype=T.float32)

            # Tokens with mask = 0 does not participate in routing.
            is_masked_token = mask_exists and not mask[global_token_idx]
            if is_masked_token:
                if lane_idx < num_topk and unmapped_topk_idx_exists:
                    unmapped_topk_idx[global_token_idx, lane_idx] = -1
                if lane_idx < num_physical_topk:
                    topk_idx[global_token_idx, lane_idx] = -1
                    topk_weights[global_token_idx, lane_idx] = 0.0
                if scores_exists:
                    for i in T.serial(lane_idx, num_routed_experts, warp_size):
                        scores[global_token_idx, i] = 0.0
                if use_pdl:
                    T.pdl_trigger()
                T.thread_return()

            # Load image tokens mask
            is_image_token = image_token_mask[global_token_idx] if image_token_mask_exists else False

            # Load and do activation functions
            # NOTES: Must be initialized before use to avoid undefined behavior
            T.fill(idx_local, -1)
            T.fill(scores_local, -T.infinity(T.float32))
            T.fill(bias_local, 0.0)

            # Ensure one warp can handle one token
            T.device_assert(num_physical_topk <= warp_size)

            # Force random, generate topk idx from all physical experts
            is_force_random_token = force_random_exists and force_random[global_token_idx]
            if is_force_random_token:
                if scores_exists and num_physical_topk_static is not None:
                    # Clear scores before narrowing the active lanes to top-k on TANG.
                    for i in T.unroll(T.ceildiv(num_routed_experts, warp_size)):
                        expert_idx = i * warp_size + lane_idx
                        if expert_idx < num_routed_experts:
                            scores[global_token_idx, expert_idx] = 0.0

                rand_topk_idx_var = T.alloc_var(T.int32)
                rand_topk_idx_tmp_var = T.alloc_var(T.int32)
                rand_topk_weight_var = T.alloc_var(T.float32)
                min_lane_idx = T.alloc_var(T.int32)

                seed = ep_rank
                seq_offset = global_token_idx
                num_physical_experts = T.alloc_var(T.int32, init=0)
                if to_physical_map_exists:
                    for i in T.serial(lane_idx, num_logical_experts, warp_size):
                        num_physical_experts += logical_count[i]
                    num_physical_experts = T.warp_reduce_sum(num_physical_experts)
                else:
                    num_physical_experts = num_routed_experts + num_shared_experts

                T.rng_init(seed, seq_offset * warp_size + lane_idx, 0)
                rand_topk_weight_var = T.rng_rand_float()

                # Generate unique topk idx sequence by generate from all possibilities and then mapping
                if lane_idx < num_physical_topk:
                    rand_topk_idx_tmp_var = T.rng_rand() % (num_physical_experts - lane_idx)
                    rand_topk_idx_var = -1
                else:
                    rand_topk_idx_tmp_var = T.max_value(T.int32)
                    rand_topk_idx_var = T.max_value(T.int32)
                for i in T.serial(num_physical_topk):
                    min_topk_idx = T.warp_reduce_min(rand_topk_idx_tmp_var)
                    min_lane_idx = T.Select(min_topk_idx == rand_topk_idx_tmp_var, lane_idx, T.max_value(T.int32))
                    min_lane_idx = T.warp_reduce_min(min_lane_idx)
                    if rand_topk_idx_var == -1:
                        if rand_topk_idx_tmp_var >= min_topk_idx and min_lane_idx < lane_idx:
                            rand_topk_idx_tmp_var += 1
                        elif min_lane_idx == lane_idx:
                            rand_topk_idx_var = rand_topk_idx_tmp_var
                            rand_topk_idx_tmp_var = T.max_value(T.int32)

                if num_physical_topk_static is not None:
                    # Use explicit lane comparisons for the TANG uniqueness assertion.
                    match_count = T.alloc_var(T.int32, init=0)
                    for other_lane in T.unroll(num_physical_topk_static):
                        other_expert = T.shfl_sync(rand_topk_idx_var, other_lane)
                        match_count += T.Cast(T.int32, rand_topk_idx_var == other_expert)
                    T.device_assert(lane_idx >= num_physical_topk or match_count == 1)
                else:
                    # All lanes in the match mask must participate in the collective.
                    matches = T.alloc_var(T.uint32)
                    matches = T.match_any_sync(rand_topk_idx_var)
                    T.device_assert(lane_idx >= num_physical_topk or matches == (T.Cast(T.uint32, 1) << lane_idx))

                # Store to global memory
                if lane_idx < num_physical_topk:
                    topk_idx[global_token_idx, lane_idx] = rand_topk_idx_var
                    topk_weights[global_token_idx, lane_idx] = rand_topk_weight_var
                    if unmapped_topk_idx_exists and lane_idx < num_topk:
                        unmapped_topk_idx[global_token_idx, lane_idx] = -1
                if scores_exists and num_physical_topk_static is None:
                    for i in T.serial(lane_idx, num_routed_experts, warp_size):
                        scores[global_token_idx, i] = 0.0
                if use_pdl:
                    T.pdl_trigger()
                T.thread_return()

            # Load logits from global memory
            for i in T.unroll(0, num_routed_experts_per_thread // num_vectorize):
                start_expert_idx = i * num_vectorize * warp_size + lane_idx * num_vectorize
                if start_expert_idx < num_routed_experts:
                    for j in T.vectorized(num_vectorize):
                        expert_idx = start_expert_idx + j
                        scores_local[i * num_vectorize + j] = logits[global_token_idx, expert_idx]
                        bias_local[i * num_vectorize + j] = (
                            image_bias[expert_idx] if (image_token_mask_exists and is_image_token) else bias[expert_idx] if bias_exists else 0
                        )

                    # Test nan or inf for each element
                    for j in T.unroll(num_vectorize):
                        is_finite = T.isfinite(scores_local[i * num_vectorize + j])
                        T.device_assert(is_finite, msg='moe_topk_gate_forward input contains nan or inf!')

            for i in T.unroll(0, T.ceildiv(num_routed_experts, num_vectorize * warp_size)):
                start_expert_idx = i * num_vectorize * warp_size + lane_idx * num_vectorize
                if start_expert_idx < num_routed_experts:
                    for j in T.unroll(num_vectorize):
                        local_idx = i * num_vectorize + j
                        scores_local[local_idx] = sqrt_softplus_forward_cuda(scores_local[local_idx])

                    for j in T.vectorized(num_vectorize):
                        expert_idx = start_expert_idx + j
                        local_idx = i * num_vectorize + j
                        scores_wo_bias_shared[token_idx, expert_idx] = scores_local[local_idx]
                        if scores_exists:
                            scores[global_token_idx, expert_idx] = scores_local[local_idx]
                        scores_local[local_idx] += bias_local[local_idx]
                        idx_local[local_idx] = expert_idx

            # Each warp exchanges scores only within its own shared-memory row.
            T.sync_warp()

            if not fix_routing_mask_exists or not fix_routing_mask[global_token_idx]:
                # Get topk via repeatly finding max
                for k in T.unroll(num_topk):
                    # Get local max score
                    winner_score = -T.infinity(T.float32)
                    winner_idx = -1
                    for i in T.unroll(0, num_routed_experts_per_thread):
                        if k != 0 and previous_winner_idx == idx_local[i]:
                            scores_local[i] = -T.infinity(T.float32)
                        # If j > i, then idx_local[j] > idx_local[i]
                        elif scores_local[i] > winner_score:
                            winner_score = scores_local[i]
                            winner_idx = idx_local[i]

                    # Get max score across all threads
                    for i in T.unroll(5):
                        other_score = T.shfl_xor(winner_score, 1 << i)
                        other_idx = T.shfl_xor(winner_idx, 1 << i)
                        if other_score > winner_score or (other_score == winner_score and other_idx < winner_idx):
                            winner_score = other_score
                            winner_idx = other_idx

                    previous_winner_idx = winner_idx
                    # Each output lane retains only its own winner, avoiding a
                    # dynamically indexed per-thread array in local memory.
                    if lane_idx == k:
                        topk_idx_var = winner_idx

                topk_score_var = 0.0
                if lane_idx < num_topk:
                    topk_score_var = scores_wo_bias_shared[token_idx, topk_idx_var]

            else:
                topk_score_var = 0.0
                if lane_idx < num_topk:
                    topk_idx_var = unmapped_topk_idx[global_token_idx, lane_idx]
                    topk_score_var = scores_wo_bias_shared[token_idx, topk_idx_var]

            # Get topk sum
            # NOTE: Align with normalize weight
            topk_sum_var = topk_score_var
            warp_reduce_sum(topk_sum_var)
            topk_sum_var += 1e-20

            # Normalize top-k weights
            if lane_idx < num_topk:
                # NOTES: If this fails, there may be some NaN values in logits input or internal error in the kernel
                T.device_assert(topk_idx_var >= 0)
                topk_score_var = topk_score_var / topk_sum_var * routed_scaling_factor
                if unmapped_topk_idx_exists:
                    unmapped_topk_idx[global_token_idx, lane_idx] = topk_idx_var
            elif lane_idx < num_physical_topk:
                topk_score_var = 1.0
                topk_idx_var = lane_idx + (num_routed_experts - num_topk)

            # Map to physical experts
            if to_physical_map_exists and lane_idx < num_physical_topk:
                logical_expert_idx = topk_idx_var
                num_duplicates = logical_count[logical_expert_idx]
                duplicate_idx = (ep_rank + global_token_idx * large_prime_number) % num_duplicates
                topk_idx_var = to_physical_map[logical_expert_idx, duplicate_idx]

            if lane_idx < num_physical_topk:
                topk_idx[global_token_idx, lane_idx] = topk_idx_var
                topk_weights[global_token_idx, lane_idx] = topk_score_var

            if use_pdl:
                T.pdl_trigger()

    return moe_topk_gate_forward_kernel
