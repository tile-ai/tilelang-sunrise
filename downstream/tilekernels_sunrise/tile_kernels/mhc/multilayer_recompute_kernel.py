import math

import torch

from tile_kernels.config import get_is_tang, get_num_vec_cores, get_pdl, is_ascend
from tile_kernels.mhc.multilayer_recompute_cuda import mhc_multilayer_recompute_cuda
from tile_kernels.utils import align


def _make_ptr_tables_batched(
    tensor_lists: list[list[torch.Tensor]],
    device: torch.device,
) -> list[torch.Tensor]:
    sizes = [len(tl) for tl in tensor_lists]
    total = sum(sizes)
    if total == 0:
        return [torch.empty(0, device=device, dtype=torch.int64) for _ in tensor_lists]

    # An asynchronous pinned upload of the pointer table can race the kernel launch on TANG.
    pinned = torch.empty(total, dtype=torch.int64, device='cpu', pin_memory=not get_is_tang())
    offset = 0
    for tl in tensor_lists:
        for i, t in enumerate(tl):
            pinned[offset + i] = t.data_ptr()
        offset += len(tl)

    gpu_buf = pinned.to(device=device, non_blocking=not get_is_tang())

    tables: list[torch.Tensor] = []
    offset = 0
    for sz in sizes:
        tables.append(gpu_buf[offset : offset + sz])
        offset += sz
    return tables


def mhc_multilayer_recompute(
    initial_residual: torch.Tensor,
    pre_mix_list: list[torch.Tensor],
    layer_output_list: list[torch.Tensor],
    post_mix_list: list[torch.Tensor],
    comb_mix_list: list[torch.Tensor],
    layer_input_list: list[torch.Tensor],
    residual_list: list[torch.Tensor],
) -> None:
    num_layers = len(pre_mix_list)
    num_post = len(layer_output_list)
    assert num_layers == len(layer_input_list)
    assert num_post == len(post_mix_list) == len(comb_mix_list) == len(residual_list)
    assert num_post == num_layers - 1 or num_post == num_layers, (
        f'post count ({num_post}) must be num_layers-1 or num_layers (num_layers={num_layers})'
    )
    assert num_layers > 0

    mhc_mult = initial_residual.shape[-2]
    hidden = initial_residual.shape[-1]
    if initial_residual.numel() == 0:
        return

    (
        pre_mix_ptrs,
        layer_input_ptrs,
        residual_ptrs,
        layer_output_ptrs,
        post_mix_ptrs,
        comb_mix_ptrs,
    ) = _make_ptr_tables_batched(
        [pre_mix_list, layer_input_list, residual_list, layer_output_list, post_mix_list, comb_mix_list],
        device=initial_residual.device,
    )

    if is_ascend():
        from tile_kernels.mhc.multilayer_recompute_asc import mhc_multilayer_recompute_asc

        h_blk = align(hidden, 64)
        h_blk = min(h_blk, 8192)
        mhc_multilayer_recompute_asc(
            initial_residual.view(-1, mhc_mult, hidden),
            pre_mix_ptrs,
            layer_output_ptrs,
            post_mix_ptrs,
            comb_mix_ptrs,
            layer_input_ptrs,
            residual_ptrs,
            h_blk=h_blk,
            num_vec_cores=get_num_vec_cores(),
        )
    else:
        h_blk = math.gcd(512, hidden)
        mhc_multilayer_recompute_cuda(
            initial_residual.view(-1, mhc_mult, hidden),
            pre_mix_ptrs,
            layer_output_ptrs,
            post_mix_ptrs,
            comb_mix_ptrs,
            layer_input_ptrs,
            residual_ptrs,
            n_thr=64,
            h_blk=h_blk,
            use_pdl=get_pdl(),
        )
