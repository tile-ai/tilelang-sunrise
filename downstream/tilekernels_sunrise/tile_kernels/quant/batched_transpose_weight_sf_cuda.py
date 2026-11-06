import tilelang
from tilelang import language as T

from tile_kernels.quant.common import get_best_vectorize_size, get_packed_ue8m0_pack_factor


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_LOWER_LDGSTG_PREDICATED: True,
    },
)
def get_batched_transpose_weight_sf_kernel_cuda(block_size: int, use_pdl: bool = False):
    assert block_size in (32, 128)
    num_sf_per_block, num_threads = 16, 64
    pack_factor = get_packed_ue8m0_pack_factor()
    weight_block_size = block_size * num_sf_per_block
    num_pack_per_block = num_sf_per_block // pack_factor
    sf_out_cols_per_sf = block_size * pack_factor
    sf_out_cols_per_block = num_sf_per_block * sf_out_cols_per_sf
    num_vec = get_best_vectorize_size(T.uint8)
    num_weights = T.dynamic('num_weights')

    def sf_layout_fn(i: int, j: int):
        offset = i * sf_out_cols_per_block + j
        return offset // num_vec % num_threads, offset // (num_vec * num_threads) * num_vec + offset % num_vec

    @T.macro
    def store_sf_block(sf_shared: T.Tensor, sf_store: T.Fragment, sf: T.Buffer, group: int, pid_pos: int, pid_pack: int):
        for p, q in T.Parallel(num_pack_per_block, sf_out_cols_per_block):
            sf_store[p, q] = sf_shared[q // sf_out_cols_per_sf, p * pack_factor + q % pack_factor]
        T.copy(sf_store, sf[group, pid_pack * num_pack_per_block, pid_pos * sf_out_cols_per_block], disable_tma=True)

    @T.prim_func
    def batched_transpose_weight_sf_kernel(
        num_blocks: T.Tensor[(num_weights,), T.int32],
        weight_infos: T.Tensor[(num_weights, 3), T.int32],
        ptrs: T.Tensor[(num_weights, 3), T.ptr],
        num_tasks: T.int32,
    ):
        with T.Kernel(num_tasks, threads=num_threads) as task_id:
            if use_pdl:
                T.pdl_sync()
            sf_shared = T.alloc_shared((num_sf_per_block, num_sf_per_block), T.float32)
            sf0_out_shared = T.alloc_shared((num_sf_per_block, num_sf_per_block), T.uint8)
            sf1_out_shared = T.alloc_shared((num_sf_per_block, num_sf_per_block), T.uint8)
            sf_store = T.alloc_fragment((num_pack_per_block, sf_out_cols_per_block), T.uint8)
            T.annotate_layout({sf_store: T.Fragment((num_pack_per_block, sf_out_cols_per_block), forward_fn=sf_layout_fn)})
            invalid_sf_bits = T.alloc_var(T.uint32)

            weight_lo = T.alloc_var(T.int32, init=0)
            weight_hi = T.alloc_var(T.int32, init=num_weights)
            while weight_lo + 1 < weight_hi:
                weight_mid = (weight_lo + weight_hi) // 2
                if num_blocks[weight_mid] <= task_id:
                    weight_lo = weight_mid
                else:
                    weight_hi = weight_mid

            weight_idx = weight_lo
            block_idx = task_id - num_blocks[weight_idx]
            T.assume(0 <= block_idx)
            n, k, num_groups = weight_infos[weight_idx, 0], weight_infos[weight_idx, 1], weight_infos[weight_idx, 2]
            T.assume(n > 0 and k > 0)
            num_block_n, num_block_k = T.ceildiv(n, weight_block_size), T.ceildiv(k, weight_block_size)
            num_block_per_group = num_block_n * num_block_k
            group, block_idx_in_group = block_idx // num_block_per_group, block_idx % num_block_per_group
            pid_n, pid_k = block_idx_in_group // num_block_k, block_idx_in_group % num_block_k
            num_sf_n, num_sf_k = n // block_size, k // block_size
            num_packed_sf0, num_packed_sf1 = T.ceildiv(num_sf_k, pack_factor), T.ceildiv(num_sf_n, pack_factor)

            sf = T.make_tensor(ptrs[weight_idx, 0], (num_groups, num_sf_n, num_sf_k), T.float32)
            sf0_out = T.make_tensor(ptrs[weight_idx, 1], (num_groups, num_packed_sf0, num_sf_n * sf_out_cols_per_sf), T.uint8)
            sf1_out = T.make_tensor(ptrs[weight_idx, 2], (num_groups, num_packed_sf1, num_sf_k * sf_out_cols_per_sf), T.uint8)

            T.copy(sf[group, pid_n * num_sf_per_block, pid_k * num_sf_per_block], sf_shared)
            invalid_sf_bits = 0
            for i, j in T.Parallel(num_sf_per_block, num_sf_per_block):
                sf_bits = T.reinterpret(sf_shared[i, j], T.uint32)
                invalid_sf_bits |= sf_bits & T.uint32(0x807FFFFF)
                sf0_out_shared[i, j] = T.uint8(sf_bits >> 23)
                sf1_out_shared[j, i] = T.uint8(sf_bits >> 23)
            T.device_assert(invalid_sf_bits == 0, no_stack_info=True)
            store_sf_block(sf0_out_shared, sf_store, sf0_out, group, pid_n, pid_k)
            store_sf_block(sf1_out_shared, sf_store, sf1_out, group, pid_k, pid_n)

            if use_pdl:
                T.pdl_trigger()

    return batched_transpose_weight_sf_kernel
