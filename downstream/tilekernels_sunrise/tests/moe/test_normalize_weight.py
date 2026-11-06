import os
import torch

import pytest

import tile_kernels
from tile_kernels.testing.generator import generate_topk_idx, generate_moe_params, get_test_level
from tile_kernels.config import get_device
from tile_kernels.testing.numeric import count_bytes
from tile_kernels.torch import normalize_weight as torch_normalize_weight
from tile_kernels.testing.bench import make_param_id

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


def generate_test_data(params):
    num_topk = params['num_topk']

    topk_idx = generate_topk_idx(params)
    num_tokens = topk_idx.shape[0]
    topk_weights = torch.rand((num_tokens, num_topk), dtype=torch.float32, device=get_device())

    return (topk_weights, num_tokens)


@pytest.mark.parametrize('params', list(generate_moe_params(get_test_level())), ids=make_param_id)
def test_normalize_weight(params):
    topk_weights, _ = generate_test_data(params)

    denominator, normalized_weights = tile_kernels.moe.normalize_weight(topk_weights)

    denom_ref, norm_ref = torch_normalize_weight(topk_weights)
    torch.testing.assert_close(denominator, denom_ref, atol=2e-6, rtol=0)
    torch.testing.assert_close(normalized_weights, norm_ref, atol=2e-6, rtol=0)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', list(generate_moe_params(0)), ids=make_param_id)
def test_normalize_weight_benchmark(benchmark_timer, benchmark_record, params):
    topk_weights, num_tokens = generate_test_data(params)

    denominator, normalized_weights = tile_kernels.moe.normalize_weight(topk_weights)

    t_us = benchmark_timer(lambda: tile_kernels.moe.normalize_weight(topk_weights))
    num_bytes = count_bytes(topk_weights, denominator, normalized_weights)
    bandwidth_gbs = num_bytes / t_us / 1e3

    params.pop('num_send_tokens')
    benchmark_record(
        kernel='normalize_weight',
        operation='fwd',
        params={'num_tokens': num_tokens, **params},
        time_us=t_us,
        bandwidth_gbs=bandwidth_gbs,
    )
