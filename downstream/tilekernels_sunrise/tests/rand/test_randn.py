import os

import pytest
import torch

import tile_kernels
from tile_kernels.config import get_device, get_is_tang
from tile_kernels.testing.bench import make_param_id


os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


_DTYPES = ('bfloat16', 'float32')


def test_set_seed_resets_torch_seed():
    tile_kernels.rand.set_seed(42)
    first = torch.randn(8)

    tile_kernels.rand.set_seed(42)
    repeated = torch.randn(8)

    assert torch.equal(first, repeated)


@pytest.mark.parametrize('dtype', _DTYPES)
def test_randn_seed(dtype):
    dtype = getattr(torch, dtype)
    input = torch.empty((128, 8192), dtype=dtype, device=get_device())
    tile_kernels.rand.set_seed(42)
    first = tile_kernels.rand.randn(input.shape, dtype=input.dtype, device=input.device)

    tile_kernels.rand.set_seed(42)
    repeated = tile_kernels.rand.randn_like(input)

    assert repeated.shape == input.shape
    assert repeated.dtype == input.dtype
    assert repeated.device == input.device
    assert torch.equal(first, repeated)


@pytest.mark.parametrize('dtype', _DTYPES)
@pytest.mark.parametrize(
    'shape',
    [
        (1,),
        (127, 129),
        (7, 11, 13),
        (16383,),
        (16384,),
        (16385,),
        (1024, 1024),
    ],
)
def test_randn_distribution(dtype, shape):
    dtype = getattr(torch, dtype)
    tile_kernels.rand.set_seed(42)
    output = tile_kernels.rand.randn(shape, dtype=dtype, device=get_device())

    assert output.shape == shape
    assert output.dtype == dtype
    assert torch.isfinite(output).all()

    if output.numel() >= 1024 * 1024:
        sample = output.float()
        if get_is_tang():
            torch.ptpu.synchronize()
            sample = sample.cpu()
        assert abs(sample.mean().item()) < 0.02
        assert abs(sample.std().item() - 1.0) < 0.02
        assert abs((sample > 0).float().mean().item() - 0.5) < 0.02


@pytest.mark.benchmark
@pytest.mark.parametrize(
    'params',
    [{'shape': shape, 'dtype': dtype} for shape in ((1024, 1024), (4096, 4096), (4096, 262144)) for dtype in _DTYPES],
    ids=make_param_id,
)
def test_randn_benchmark(benchmark_timer, benchmark_record, params):
    shape = params['shape']
    dtype = getattr(torch, params['dtype'])

    output = tile_kernels.rand.randn(shape, dtype=dtype, device=get_device())
    time_us = benchmark_timer(lambda: tile_kernels.rand.randn(shape, dtype=dtype, device=get_device()))

    num_bytes = output.numel() * output.element_size()
    benchmark_record(
        kernel='randn',
        operation='fwd',
        params=params,
        time_us=time_us,
        bandwidth_gbs=num_bytes / time_us / 1e3,
    )
