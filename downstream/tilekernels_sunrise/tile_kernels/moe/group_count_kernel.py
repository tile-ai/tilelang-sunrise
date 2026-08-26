import os
import torch
import tilelang
from tilelang import language as T

from tile_kernels.config import get_num_sms
from tile_kernels.utils import align


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def get_group_count_kernel(num_topk: int, num_groups: int, num_sms: int):
    num_threads = 128
    num_tokens = T.dynamic('num_tokens')

    if num_topk >= 6:
        # Large topk: the per-token row read has a num_topk*8-byte stride
        # between lanes (one cache line per lane, uncoalesced), capping
        # effective bandwidth. Flatten (num_tokens, num_topk) and stride
        # threads by 1 over the flat index so a warp's 32 lanes read 32
        # consecutive int64 (256 B = 2 cache lines) — coalesced.
        #
        # Use num_sms blocks (not 2*num_sms): for large topk the per-thread
        # work is ample, and halving the block count halves the global
        # atomic-add contention in the final reduction as well as launch
        # overhead — the kernel is overhead-bound for these shapes.
        num_blocks_flat = num_sms

        @T.prim_func
        def group_count_kernel(
            group_idx: T.Tensor[(num_tokens, num_topk), T.int64],
            out: T.Tensor[(num_groups, ), T.int32],
        ):
            with T.Kernel(num_blocks_flat, threads=num_threads) as (pid, ):
                thread_idx = T.get_thread_binding()
                global_thread_idx = pid * num_threads + thread_idx
                total = num_tokens * num_topk

                out_shared = T.alloc_shared((align(num_groups, num_threads), ), T.int32)
                T.clear(out_shared)
                T.sync_threads()

                for k in T.serial(global_thread_idx, total, num_blocks_flat * num_threads):
                    expert_idx = T.int32(group_idx[k // num_topk, k % num_topk])
                    T.device_assert(-1 <= expert_idx < num_groups)
                    T.assume(expert_idx < num_groups)
                    if expert_idx >= 0:
                        T.atomic_add(out_shared[expert_idx], 1)

                T.sync_threads()
                for g in T.serial(thread_idx, num_groups, num_threads):
                    if out_shared[g] > 0:
                        T.atomic_add(out[g], out_shared[g])
        return group_count_kernel

    # Small topk: data is small and the kernel is launch-overhead-bound, so
    # the original per-token structure (unrolled topk loads + shared atomic)
    # amortizes better than the flat traversal.
    num_blocks = num_sms * 2

    @T.prim_func
    def group_count_kernel(
        group_idx: T.Tensor[(num_tokens, num_topk), T.int64],
        out: T.Tensor[(num_groups, ), T.int32],
    ):
        with T.Kernel(num_blocks, threads=num_threads) as (pid, ):
            thread_idx = T.get_thread_binding()
            global_thread_idx = pid * num_threads + thread_idx

            out_shared = T.alloc_shared((align(num_groups, num_threads), ), T.int32)
            T.clear(out_shared)
            T.sync_threads()

            for i in T.serial(global_thread_idx, num_tokens, num_blocks * num_threads):
                for j in T.unroll(num_topk):
                    expert_idx = T.int32(group_idx[i, j])
                    T.device_assert(-1 <= expert_idx < num_groups)
                    T.assume(expert_idx < num_groups)
                    if expert_idx >= 0:
                        T.atomic_add(out_shared[expert_idx], 1)

            T.sync_threads()
            for i in T.serial(thread_idx, num_groups, num_threads):
                if out_shared[i] > 0:
                    T.atomic_add(out[i], out_shared[i])

    return group_count_kernel


def group_count(group_idx: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Count the number of tokens assigned to each expert.

    Args:
        group_idx: Int64 expert index tensor of shape (num_tokens, num_topk).
        num_groups: Total number of experts.

    Returns:
        Int32 tensor of shape (num_groups,) with per-expert token counts.
    """
    assert group_idx.dim() == 2 and group_idx.is_contiguous()

    kernel = get_group_count_kernel(group_idx.shape[1], num_groups, get_num_sms())

    if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
        print(kernel.get_kernel_source())

    out = torch.zeros(num_groups, dtype=torch.int32, device='ptpu')
    kernel(group_idx, out)

    return out
