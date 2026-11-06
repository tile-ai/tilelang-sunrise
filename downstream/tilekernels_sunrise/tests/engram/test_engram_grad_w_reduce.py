import os

import pytest
import torch

from tile_kernels.config import get_device, get_num_sms, is_ascend, set_num_sms
from tile_kernels.engram import grad_w_reduce
from tile_kernels.rand import randn
from tile_kernels.testing.bench import make_param_id
from tile_kernels.testing.generator import generate_hidden_sizes, generate_num_sms, get_test_level
from tile_kernels.testing.numeric import count_bytes

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


def grad_w_reduce_ref(grad_w_partial, weight_hidden, weight_embed, grad_weight_hidden, grad_weight_embed):
    grad_w_sum = grad_w_partial.sum(0)
    grad_weight_hidden += grad_w_sum * weight_embed.float()
    grad_weight_embed += grad_w_sum * weight_hidden.float()


def generate_test_data(params):
    hidden_size = params['hidden']
    hc_mult = 4
    num_persistent_blocks = get_num_sms() * 2 // hc_mult if is_ascend() else get_num_sms()
    device = get_device()
    grad_w_partial = randn(num_persistent_blocks, hc_mult, hidden_size, dtype=torch.float32, device=device)
    weight_hidden = randn(hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    weight_embed = randn(hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    return (grad_w_partial, weight_hidden, weight_embed)


def generate_test_params(level: int) -> list[dict]:
    partials_per_sm = 2 if is_ascend() else 1
    return [
        {
            'hidden': hidden_size,
            'num_sms': num_sms * partials_per_sm,
        }
        for hidden_size in generate_hidden_sizes(64 if is_ascend() else 256)
        for num_sms in generate_num_sms(level)
    ]


@pytest.mark.parametrize('params', generate_test_params(get_test_level()), ids=make_param_id)
def test_engram_grad_w_reduce(params):
    hidden_size = params['hidden']

    if not is_ascend():
        set_num_sms(params['num_sms'])
    grad_w_partial, weight_hidden, weight_embed = generate_test_data(params)
    hc_mult = grad_w_partial.shape[1]

    # Correctness
    device = get_device()
    grad_wh_ref = randn(hc_mult, hidden_size, dtype=torch.float32, device=device)
    grad_we_ref = randn(hc_mult, hidden_size, dtype=torch.float32, device=device)
    grad_weight_hidden = grad_wh_ref.clone()
    grad_weight_embed = grad_we_ref.clone()
    grad_w_reduce_ref(grad_w_partial, weight_hidden, weight_embed, grad_wh_ref, grad_we_ref)
    grad_w_reduce(grad_w_partial, weight_hidden, weight_embed, grad_weight_hidden, grad_weight_embed)
    torch.testing.assert_close(grad_weight_hidden, grad_wh_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(grad_weight_embed, grad_we_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(0), ids=make_param_id)
def test_engram_grad_w_reduce_benchmark(benchmark_timer, benchmark_record, params):
    hidden_size = params['hidden']

    if not is_ascend():
        set_num_sms(params['num_sms'])
    grad_w_partial, weight_hidden, weight_embed = generate_test_data(params)
    hc_mult = grad_w_partial.shape[1]
    device = get_device()
    grad_weight_hidden = randn(hc_mult, hidden_size, dtype=torch.float32, device=device)
    grad_weight_embed = randn(hc_mult, hidden_size, dtype=torch.float32, device=device)

    t_us = benchmark_timer(lambda: grad_w_reduce(grad_w_partial, weight_hidden, weight_embed, grad_weight_hidden, grad_weight_embed))

    num_bytes = count_bytes(grad_w_partial, weight_hidden, weight_embed, grad_weight_hidden, grad_weight_embed)
    bandwidth_gbs = num_bytes / t_us / 1e3
    benchmark_record(
        kernel='grad_w_reduce',
        operation='fwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=bandwidth_gbs,
    )
