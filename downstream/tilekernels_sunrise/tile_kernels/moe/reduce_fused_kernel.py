import os
import torch
import tilelang
from tilelang import language as T
from typing import Optional, Union
from tile_kernels.quant.common import *


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        # The TANG `-fstpu-warp-alu` optimization silently miscompiles the
        # ``for k in range(num_topk)`` gather/reduce loop below once
        # ``num_topk >= 4`` (wrong/duplicate reads, NaNs). Disable it here.
        tilelang.PassConfigKey.TL_TANG_DISABLE_WARP_ALU: True,
    },
)
def get_reduce_fused_kernel(
    hidden: int,
    num_topk: int,
    in_dtype: T.dtype,
    out_dtype: T.dtype,
    with_sf: bool,
    with_weights: bool,
    with_x_sf: bool,
):
    num_threads = 128

    num_tokens = T.dynamic('num_tokens')
    num_expanded_tokens = T.dynamic('num_expanded_tokens')

    @T.prim_func
    def reduce_fused_kernel(
        x: T.Tensor[(num_expanded_tokens, hidden), in_dtype],
        topk_weights: T.Tensor[(num_tokens, num_topk), T.float32],
        token_topk_to_pos: T.Tensor[(num_tokens, num_topk), T.int32],
        out: T.Tensor[(num_tokens, hidden), out_dtype],
        sf: T.Tensor[(1,), T.float32],
        x_sf: T.Tensor[(num_expanded_tokens,), T.float32],
    ):
        with T.Kernel(num_tokens, threads=num_threads) as (pid_token,):
            reduced_fragment = T.alloc_fragment((hidden,), T.float32)
            # NOTE: the per-token `pos`/`weight` values are consumed by every thread
            # and warp of the block (they index the same `x[pos, i]` gather across the
            # whole `hidden` dimension). Holding them in per-thread fragments let
            # non-owning warps read stale/garbage values, which produced
            # non-deterministic wrong results that grow with num_topk. This is an
            # intra-kernel race, so stage the values in
            # shared memory (a single block-wide copy) and add a barrier so all
            # threads observe the same values.
            topk_weights_local = T.alloc_shared((num_topk,), T.float32)
            topk_to_pos_local = T.alloc_shared((num_topk,), T.int32)
            sf_var = T.alloc_var(T.float32)

            T.clear(reduced_fragment)
            if with_sf:
                sf_var = sf[0]
            if with_weights:
                T.copy(topk_weights[pid_token, :], topk_weights_local)
            T.copy(token_topk_to_pos[pid_token, :], topk_to_pos_local)
            # Publish the shared pos/weight writes to every thread before the gather.
            T.sync_threads()

            for k in T.unroll(num_topk):
                pos = topk_to_pos_local[k]
                T.assume(pos < num_expanded_tokens)
                if pos >= 0:
                    s = T.alloc_var(T.float32)
                    s = 1
                    if with_weights:
                        s = topk_weights_local[k]

                    if with_x_sf:
                        s *= x_sf[pos]
                    for i in T.Parallel(hidden):
                        reduced_fragment[i] += x[pos, i] * s

            for i in T.Parallel(hidden):
                out[pid_token, i] = T.Select(with_sf, reduced_fragment[i] * sf_var, reduced_fragment[i])

    return reduce_fused_kernel


def reduce_fused(
    x: Union[torch.Tensor, QuantTensor],
    topk_weights: Optional[torch.Tensor],
    token_topk_to_pos: torch.Tensor,
    fp8_format: str = '',
    sf: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reduce expanded expert outputs back to token-level by weighted summation.

    Args:
        x: Expanded tensor of shape (num_expanded_tokens, hidden), or a
            ``QuantTensor`` ``(data, sf)`` to include per-token sf factors.
        topk_weights: Optional top-k routing weights of shape (num_tokens, num_topk).
        token_topk_to_pos: Mapping from (token, topk) to expanded position,
            shape (num_tokens, num_topk).
        fp8_format: Optional FP8 output format (``'e4m3'`` or empty string).
        sf: Optional FP32 scalar tensor for output scaling.
        out: Optional pre-allocated output tensor of shape (num_tokens, hidden).

    Returns:
        Reduced tensor of shape (num_tokens, hidden).
    """
    if isinstance(x, tuple):
        x, x_sf = x
    else:
        x_sf = None
    num_expanded_tokens, hidden = x.shape
    num_tokens, num_topk = token_topk_to_pos.shape
    assert hidden % 256 == 0

    in_dtype = x.dtype
    out_dtype = in_dtype
    if fp8_format != '':
        if fp8_format == 'e4m3':
            out_dtype = torch.float8_e4m3fn
        else:
            assert False
    else:
        assert sf is None, 'Only FP8 output supports sf.'

    if out is not None:
        num_tokens_, hidden_ = out.shape
        assert num_tokens == num_tokens_ and hidden == hidden_
    else:
        out = torch.empty((num_tokens, hidden), dtype=out_dtype, device=x.device)

    if x_sf is not None:
        num_expanded_tokens_ = x_sf.shape[0]
        assert num_expanded_tokens == num_expanded_tokens_

    kernel = get_reduce_fused_kernel(
        hidden,
        num_topk,
        T.dtype(in_dtype),
        T.dtype(out_dtype),
        sf is not None,
        topk_weights is not None,
        x_sf is not None,
    )
    if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
        print(kernel.get_kernel_source())

    if num_tokens > 0:
        kernel(x,
               topk_weights if topk_weights is not None else torch.zeros((num_tokens, num_topk), dtype=torch.float32, device='cpu').ptpu(),
               token_topk_to_pos, out,
               sf if sf is not None else torch.zeros((1,), dtype=torch.float32, device='cpu').ptpu(),
               x_sf if x_sf is not None else torch.zeros((num_expanded_tokens,), dtype=torch.float32, device='cpu').ptpu())

    return out
