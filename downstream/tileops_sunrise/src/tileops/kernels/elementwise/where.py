"""The where kernel."""

import functools

import tilelang.language as T

import tilelang

from ._base import MultiInputElementwiseKernel

__all__ = [
    "WhereFwdKernel",
]


@functools.lru_cache(maxsize=32)
def _make_where_kernel(N, dtype, threads=256, npt=8):
    """Build where kernel: out = cond ? x : y.

    ``WhereFwdKernel.forward`` packs the bool condition as uint8 so that T.copy
    can perform vectorized loads (TileLang does not vectorize bool tensors).
    Each uint8 element is 0 or 1; the kernel compares it against 0 in the compute
    loop.

    The x/y value paths load into register fragments via ``T.copy`` (vectorized
    loads) and write back via ``T.copy``. The 1-byte ``cond`` mask is read
    straight from global memory in the compute loop: loading it through a register
    fragment (``T.alloc_fragment`` + ``T.copy``) scalarizes the 1-byte accesses on
    TANG/PT200 and collapses effective bandwidth, so the fragment variant is
    avoided for the mask only.
    """

    @tilelang.jit(out_idx=[3])
    def kernel(threads_arg, npt_arg):
        block_size = threads_arg * npt_arg

        @T.prim_func
        def main(
            cond: T.Tensor((N,), "uint8"),
            x: T.Tensor((N,), dtype),
            y_in: T.Tensor((N,), dtype),
            out: T.Tensor((N,), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                x_reg = T.alloc_fragment((block_size,), dtype)
                y_reg = T.alloc_fragment((block_size,), dtype)
                T.copy(x[bx * block_size : (bx + 1) * block_size], x_reg)
                T.copy(y_in[bx * block_size : (bx + 1) * block_size], y_reg)
                for i, j in T.Parallel(threads_arg, npt_arg):
                    k = i * npt_arg + j
                    idx = bx * block_size + k
                    if idx < N:
                        x_reg[k] = T.if_then_else(
                            cond[idx] != T.cast(0, "uint8"),
                            x_reg[k],
                            y_reg[k],
                        )
                T.copy(x_reg, out[bx * block_size : (bx + 1) * block_size])

        return main

    return kernel


class WhereFwdKernel(MultiInputElementwiseKernel):
    """Where: out = cond ? x : y."""

    DEFAULT_THREADS = 512
    INPUTS = (("cond", "mask"), ("x", "tile"), ("y", "tile"))

    @staticmethod
    def _builder_fn():
        return _make_where_kernel

    def forward(self, cond, x, y):
        return self._run(cond=cond, x=x, y=y)
