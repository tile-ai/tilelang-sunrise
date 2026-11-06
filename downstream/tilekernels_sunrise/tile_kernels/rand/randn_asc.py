from typing import Any

import tilelang
from tilelang.ascend import language as T


@tilelang.jit(out_idx=None, target='ascend')
def get_randn_kernel_asc(dtype: Any, num_vec_cores: int, block_size: int, num_stages: int, tile_size: int, unroll_factor: int) -> Any:
    n = T.dynamic('n', T.int64)
    B = block_size
    threads = 512

    @T.prim_func
    def randn_kernel_asc(
        output: T.Tensor[(n,), dtype],
        seed: T.int64,
    ) -> None:
        num_blocks = T.ceildiv(n, B)

        with T.Kernel(num_vec_cores) as core_id:
            # using T.int64(B) is to prevent a tilelang bug which casts n to int32 too early in the copyout part.
            output_ub = T.alloc_shared((T.int64(B),), dtype)
            T.annotate_buffer_versions({output_ub: num_stages})

            for pid in T.Persistent(
                [num_blocks],
                num_vec_cores,
                core_id,
                group_size=1,
                num_stages=num_stages,
            ):
                offset = T.int64(pid) * B

                with T.SimtVF(threads=threads):
                    T.rng_init(seed + pid)
                    for serial in T.unroll(B // tile_size, unroll_factor=unroll_factor):
                        for tx in T.Parallel(tile_size):
                            output_ub[serial * tile_size + tx] = T.rng_rand_float(dist='normal')

                T.copy(output_ub, output[offset])

    return randn_kernel_asc
