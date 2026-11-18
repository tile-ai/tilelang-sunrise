import os

import pytest
import torch

import tile_kernels
from tile_kernels.config import get_device
from tile_kernels.testing.numeric import calc_diff, count_bytes
from tile_kernels.torch import rotary_embedding_ref

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


@pytest.mark.parametrize('head_split', [False, True])
@pytest.mark.parametrize('nheads', [1, 5, 32, 128])
@pytest.mark.parametrize('dim', [64, 128])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('interleaved', [True, False])
@pytest.mark.parametrize('conjugate', [True, False])
def test_rotary_embedding_3d(nheads, dim, interleaved, conjugate, dtype, head_split):
    torch.manual_seed(0)
    device = get_device()
    seqlen = 2001
    seqlen_total = 8 * 1024

    query = torch.randn(seqlen, nheads + int(head_split), dim + 128, dtype=dtype, device=device)[:, :nheads, -dim:]
    key = torch.randn(seqlen, nheads + int(head_split), dim + 128, dtype=dtype, device=device)[:, :nheads, -dim:]
    cos_sin = torch.randn(seqlen_total, dim, dtype=torch.float32, device=device)
    positions = torch.randint(0, seqlen_total, (seqlen,), dtype=torch.int32, device=device)

    cos, sin = cos_sin.chunk(2, dim=-1)
    ref_query = rotary_embedding_ref(query, cos, sin, positions, interleaved=interleaved, conjugate=conjugate)
    ref_key = rotary_embedding_ref(key, cos, sin, positions, interleaved=interleaved, conjugate=conjugate)

    tile_kernels.transform.apply_rotary(
        query,
        cos_sin,
        key,
        positions=positions,
        interleaved=interleaved,
        conjugate=conjugate,
    )

    assert calc_diff(query, ref_query) <= 1e-8  # TODO update to bitwise match when torch upgrades and supports fma
    assert calc_diff(key, ref_key) <= 1e-8  # TODO update to bitwise match when torch upgrades and supports fma


@pytest.mark.parametrize(
    'batch,nheads,dim,positions_mode,interleaved,conjugate,seqlen_offset',
    [
        pytest.param(3, 1, 64, 'none', True, False, 17, id='b3-h1-d64-none-offset-interleaved'),
        pytest.param(1, 32, 128, 'int32', False, False, 0, id='b1-h32-d128-int32-neox'),
        pytest.param(3, 64, 64, 'broadcast', True, False, 0, id='b3-h64-d64-broadcast-interleaved'),
        pytest.param(3, 128, 128, 'int64', False, True, 0, id='b3-h128-d128-int64-neox-conjugate'),
    ],
)
def test_rotary_embedding_4d(batch, nheads, dim, positions_mode, interleaved, conjugate, seqlen_offset):
    torch.manual_seed(0)
    device = get_device()
    seqlen = 2048
    seqlen_total = seqlen + seqlen_offset

    query = torch.randn(batch, seqlen, nheads, dim + 16, dtype=torch.bfloat16, device=device)[..., -dim:]
    cos_sin = torch.randn(seqlen_total, dim, dtype=torch.float32, device=device)

    if positions_mode == 'broadcast':
        positions = torch.randint(0, seqlen_total - seqlen_offset, (seqlen,), dtype=torch.int32, device=device)
        positions = positions.unsqueeze(0).expand(batch, -1)
    elif positions_mode in ('int32', 'int64'):
        positions_dtype = torch.int32 if positions_mode == 'int32' else torch.int64
        positions = torch.randint(0, seqlen_total - seqlen_offset, (batch, seqlen), dtype=positions_dtype, device=device)
    else:
        positions = None

    cos, sin = cos_sin.chunk(2, dim=-1)
    ref = rotary_embedding_ref(
        query,
        cos,
        sin,
        positions,
        seqlen_offsets=seqlen_offset,
        interleaved=interleaved,
        conjugate=conjugate,
    )
    tile_kernels.transform.apply_rotary(
        query,
        cos_sin_cache=cos_sin,
        positions=positions,
        interleaved=interleaved,
        conjugate=conjugate,
        seqlen_offset=seqlen_offset,
    )

    assert calc_diff(query, ref) <= 1e-8  # TODO update to bitwise match when torch upgrades and supports fma


@pytest.mark.benchmark
@pytest.mark.parametrize('head_split', [False, True])
@pytest.mark.parametrize('has_positions', [False, True])
@pytest.mark.parametrize('interleaved', [False, True])
@pytest.mark.parametrize('nheads', [1, 32, 64, 128])
def test_rotary_embedding_benchmark(nheads, interleaved, has_positions, head_split, benchmark_timer, benchmark_record):
    torch.manual_seed(0)
    device = get_device()
    batch = 1
    seqlen = 8192
    dim = 64
    pad_size = 128
    dtype = torch.bfloat16

    query = torch.randn(batch, seqlen, nheads + int(head_split), dim + pad_size, dtype=dtype, device=device)
    query = query[:, :, :nheads, -dim:]
    cos_sin = torch.randn(seqlen, dim, dtype=torch.float32, device=device)
    positions = torch.randint(0, seqlen, (batch, seqlen), dtype=torch.int32, device=device) if has_positions else None

    def bench_fn():
        tile_kernels.transform.apply_rotary(
            query,
            cos_sin_cache=cos_sin,
            positions=positions,
            interleaved=interleaved,
            conjugate=False,
        )

    t_us = benchmark_timer(bench_fn)
    num_bytes = count_bytes(query, cos_sin, positions) + count_bytes(query)
    benchmark_record(
        kernel='rope',
        operation='fwd',
        params={
            'head_split': head_split,
            'has_positions': has_positions,
            'interleaved': interleaved,
            'nheads': nheads,
            'seqlen': seqlen,
        },
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )
