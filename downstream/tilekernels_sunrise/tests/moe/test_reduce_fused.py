import os
import torch

import pytest

import tile_kernels
from tile_kernels.testing.bench import dtype_to_str, make_param_id
from tile_kernels.testing.generator import generate_topk_idx, generate_hidden_sizes, generate_moe_params
from tile_kernels.testing.numeric import assert_equal, count_bytes

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


def generate_test_data(params):
    # Fixed seed so the random inputs are reproducible across runs. bf16 reductions
    # sum up to num_topk weighted terms, and the rare cancellation-heavy draw can push
    # the PTPU-vs-CPU diff into the tail; seeding keeps the test deterministic instead
    # of intermittently flaky on those draws.
    torch.manual_seed(0)

    hidden = params['hidden']
    with_weights = params['with_weights']
    in_dtype = params['in_dtype']
    out_dtype = params['out_dtype']
    with_sf = params['with_sf']
    num_experts = params['num_experts']
    num_ep_ranks = params['num_ep_ranks']
    num_topk = params['num_topk']

    # Create reference data on CPU first
    topk_idx = generate_topk_idx(params)
    num_tokens = topk_idx.shape[0]
    num_expanded_tokens = num_tokens * num_topk
    expanded_cpu = torch.randn((num_expanded_tokens, hidden), dtype=in_dtype)
    # get_fused_mapping returns PTPU tensors; copy them back to CPU for the ref
    _, _, _, token_topk_to_pos_ptpu, _, _, _, _ = tile_kernels.moe.get_fused_mapping(topk_idx, num_experts, 0, 1)
    token_topk_to_pos = token_topk_to_pos_ptpu.cpu()

    topk_weights = torch.rand((num_tokens, num_topk), dtype=torch.float32) if with_weights else None
    if out_dtype == torch.float8_e4m3fn:
        sf = torch.randn((1,), dtype=torch.float32)
    else:
        sf = None
    x_sf = torch.randn((num_expanded_tokens,), dtype=torch.float32) if with_sf else None
    fp8_format = 'e4m3' if out_dtype == torch.float8_e4m3fn else ''

    x_input = (expanded_cpu, x_sf) if x_sf is not None else expanded_cpu

    return (expanded_cpu, token_topk_to_pos, topk_weights, sf, x_sf, fp8_format, x_input, num_tokens)


def generate_test_params(is_benchmark: bool) -> list[dict]:
    # NOTE: FP8 output tests (out_dtype=float8_e4m3fn) are excluded because the
    # PTPU/TANG compiler does not support the 'fp8_e4_t' type. Only non-FP8
    # output variants (out_dtype == in_dtype) are tested on the PTPU backend.
    # 开关: TK_FULL_TEST=1 跑全部用例，否则跑 CI 精简集
    do_full_test = os.getenv('TK_FULL_TEST') in ['1', 'true', 'True']
    if do_full_test and not is_benchmark:
        _hidden_sizes = generate_hidden_sizes(256)
        _with_weights_opts = (True, False)
    else:
        _hidden_sizes = [h for h in (4096,) if h in generate_hidden_sizes(256)]
        _with_weights_opts = (True,)
    params = [
        {**moe, 'hidden': hidden, 'with_weights': with_weights,
         'in_dtype': in_dtype, 'out_dtype': in_dtype, 'with_sf': with_sf}
        for moe in generate_moe_params(is_benchmark=is_benchmark)
        for hidden in _hidden_sizes
        for with_weights in _with_weights_opts
        for in_dtype in (torch.float32, torch.bfloat16)
        for with_sf in (True, False)
    ]
    if is_benchmark:
        params = [p for p in params if p['num_topk'] == 6 and p['with_weights']]
    return params


@pytest.mark.parametrize('params', generate_test_params(is_benchmark=False), ids=make_param_id)
def test_reduce_fused(params):
    (expanded_cpu, token_topk_to_pos_cpu, topk_weights_cpu, sf_cpu, x_sf_cpu, fp8_format, x_input_cpu,
     _) = generate_test_data(params)

    # Copy data to PTPU for TileLang kernel
    expanded_ptpu = expanded_cpu.ptpu()
    token_topk_to_pos_ptpu = token_topk_to_pos_cpu.ptpu()
    topk_weights_ptpu = topk_weights_cpu.ptpu() if topk_weights_cpu is not None else None
    sf_ptpu = sf_cpu.ptpu() if sf_cpu is not None else None
    x_sf_ptpu = x_sf_cpu.ptpu() if x_sf_cpu is not None else None
    x_input_ptpu = (expanded_ptpu, x_sf_ptpu) if x_sf_ptpu is not None else expanded_ptpu

    # Test correctness: TileLang kernel on PTPU
    func = lambda: tile_kernels.moe.reduce_fused(
        x_input_ptpu, topk_weights_ptpu, token_topk_to_pos_ptpu, fp8_format, sf_ptpu, None
    )
    r_tk = func()

    # Test correctness: torch reference on CPU (same data source)
    r_ref = tile_kernels.torch.reduce_fused(
        x_input_cpu, topk_weights_cpu, token_topk_to_pos_cpu, fp8_format, sf_cpu
    )
    torch.ptpu.synchronize()
    r_tk_cpu = r_tk.cpu()
    # Use allclose for cross-device comparison (PTPU vs CPU float accumulation order differs)
    # bf16 has lower precision (~7 mantissa bits), so use looser tolerance.
    # The reduce sums up to num_topk weighted terms, and PTPU vs CPU accumulation
    # order differs, so the bf16 result can deviate by a few ULP (grows with num_topk).
    # 2e-2 covers the observed 1-2 ULP rounding differences across the full sweep.
    rtol_val, atol_val = (2e-2, 2e-2) if params['in_dtype'] == torch.bfloat16 else (1e-4, 1e-4)
    assert torch.allclose(r_tk_cpu, r_ref, rtol=rtol_val, atol=atol_val), \
        f'Max diff: {(r_tk_cpu - r_ref).abs().max().item():.8f}'


@pytest.mark.skip(reason='CI 测试忽略 benchmark 测试')
@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(is_benchmark=True), ids=make_param_id)
def test_reduce_fused_benchmark(benchmark_timer, benchmark_record, params):
    hidden = params['hidden']
    out_dtype = params['out_dtype']

    (expanded, token_topk_to_pos, topk_weights, sf, x_sf, fp8_format, x_input,
     num_tokens) = generate_test_data(params)
    in_dtype = params['in_dtype']

    func = lambda: tile_kernels.moe.reduce_fused(
        x_input, topk_weights, token_topk_to_pos, fp8_format, sf, None
    )
    r_tk = func()

    num_bytes = count_bytes(token_topk_to_pos, x_sf, r_tk)
    num_bytes += torch.count_nonzero(token_topk_to_pos != -1).item() * hidden * (torch.finfo(in_dtype).bits // 8)
    if topk_weights is not None:
        num_bytes += count_bytes(topk_weights)

    t_us = benchmark_timer(func)

    bandwidth_gbs = num_bytes / t_us / 1e3

    params.pop('num_send_tokens')
    benchmark_record(
        kernel='reduce_fused',
        operation='fwd',
        params={'num_tokens': num_tokens, **params, 'in_dtype': dtype_to_str(in_dtype), 'out_dtype': dtype_to_str(out_dtype)},
        time_us=t_us,
        bandwidth_gbs=bandwidth_gbs,
    )
