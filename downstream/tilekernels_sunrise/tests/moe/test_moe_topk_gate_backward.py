import os

import pytest
import torch

import tile_kernels
from tile_kernels.config import get_device
from tile_kernels.rand import randn
from tile_kernels.testing.bench import make_param_id
from tile_kernels.testing.generator import generate_num_tokens, get_test_level
from tile_kernels.testing.moe import construct_logits
from tile_kernels.testing.numeric import assert_equal, count_bytes
from tile_kernels.torch import moe_topk_gate_backward as torch_moe_topk_gate_backward

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


_CORE_CONFIGS = [
    (256, 2, 8),
]

_FULL_CONFIGS = [
    (72, 1, 6),
    (32, 2, 6),
    (64, 2, 6),
    (96, 2, 6),
    (16, 2, 6),
    (36, 2, 6),
    (108, 2, 6),
    (128, 2, 6),
    (144, 2, 6),
    (256, 2, 8),
]


def generate_test_params(level: int) -> list[dict]:
    configs = _CORE_CONFIGS if level == 0 else _FULL_CONFIGS
    params = [
        {
            'num_tokens': num_tokens,
            'num_padded_tokens': num_padded_tokens,
            'num_seqs': num_seqs,
            'num_topk': num_topk,
            'num_routed_experts': num_routed_experts,
            'num_shared_experts': num_shared_experts,
            'bias_exists': bias_exists,
            'image_bias_exists': image_bias_exists,
            'aux_exists': aux_exists,
        }
        for num_tokens in generate_num_tokens(level, backward_only=True)
        for num_padded_tokens in (0, 10)
        # Padded tokens sit at the tail of every sequence
        for num_seqs in ([1] if level == 0 else [1, 2])
        for num_routed_experts, num_shared_experts, num_topk in configs
        for bias_exists in ([True] if level == 0 else [True, False])
        for image_bias_exists in ([True] if level == 0 else [True, False])
        for aux_exists in (False, True)
    ]
    return params


def get_kwargs(params, fix_routing):
    num_tokens, num_padded_tokens = params['num_tokens'], params['num_padded_tokens']
    num_seqs = params['num_seqs']
    num_routed_experts = params['num_routed_experts']
    num_topk = params['num_topk']
    bias_exists, image_bias_exists = params['bias_exists'], params['image_bias_exists']
    mask = None
    if num_padded_tokens > 0:
        mask = torch.ones(num_seqs, num_tokens + num_padded_tokens, dtype=torch.bool, device=get_device())
        mask[:, -num_padded_tokens:] = False
        mask = mask.flatten()
    num_tokens = num_seqs * (num_tokens + num_padded_tokens)

    fix_routing_mask = None
    unmapped_topk_idx = torch.zeros((num_tokens, num_topk), dtype=torch.int64, device=get_device())

    bias = randn(num_routed_experts, dtype=torch.float, device=get_device()) if bias_exists else None
    image_bias = randn(num_routed_experts, dtype=torch.float, device=get_device()) if image_bias_exists else None
    image_token_mask = torch.rand(num_tokens, dtype=torch.float, device=get_device()) < 0.5 if image_bias_exists else None

    if fix_routing:
        fix_routing_mask = torch.ones((num_tokens,), dtype=torch.bool, device=get_device())
        unmapped_topk_idx = (
            torch.rand((num_tokens, num_routed_experts), dtype=torch.float32, device=get_device())
            .topk(num_topk, dim=1, largest=True, sorted=False)
            .indices
        )

    return dict(
        mask=mask,
        bias=bias,
        image_bias=image_bias,
        image_token_mask=image_token_mask,
        fix_routing_mask=fix_routing_mask,
        unmapped_topk_idx=unmapped_topk_idx,
    )


def get_backward_kwargs(params, kwargs):
    num_tokens = params['num_seqs'] * (params['num_tokens'] + params['num_padded_tokens'])
    num_seqs = params['num_seqs']
    num_routed_experts = params['num_routed_experts']
    aux_exists = params['aux_exists']
    mask = kwargs['mask'] if aux_exists else None
    grad_scores_sum = None
    if aux_exists:
        mask = mask if mask is not None else torch.ones(num_tokens, dtype=torch.bool, device=get_device())
        grad_scores_sum = randn((num_seqs, num_routed_experts), dtype=torch.float, device=get_device())

    return dict(
        mask=mask,
        grad_scores_sum=grad_scores_sum,
    )


@pytest.mark.parametrize('params', generate_test_params(get_test_level()), ids=make_param_id)
def test_moe_topk_gate_backward(params):
    num_tokens = params['num_seqs'] * (params['num_tokens'] + params['num_padded_tokens'])
    num_routed_experts = params['num_routed_experts']
    num_shared_experts = params['num_shared_experts']
    num_topk = params['num_topk']
    scoring_func = 'sqrtsoftplus'
    use_shared_as_routed = False

    routed_scaling_factor = 1.5
    ep_rank = 0

    # Correctness test
    for fix_routing in (False, True):
        kwargs = get_kwargs(params, fix_routing)
        backward_kwargs = get_backward_kwargs(params, kwargs)
        logits = construct_logits(params, kwargs)
        scores = torch.empty_like(logits)
        topk_idx, topk_weights = tile_kernels.moe.moe_topk_gate_forward(
            logits,
            num_topk,
            use_shared_as_routed,
            num_shared_experts,
            routed_scaling_factor,
            ep_rank,
            scoring_func,
            **kwargs,
            scores=scores,
        )
        grad_topk_weights = randn((num_tokens, num_topk), dtype=torch.float, device=get_device())
        args = (
            scores,
            topk_idx,
            topk_weights,
            grad_topk_weights,
            routed_scaling_factor,
            scoring_func,
        )

        grad_logits = tile_kernels.moe.moe_topk_gate_backward(*args, **backward_kwargs)
        grad_logits_out = torch.empty((num_tokens, num_routed_experts), dtype=torch.float32, device=get_device())
        grad_logits_out = tile_kernels.moe.moe_topk_gate_backward(*args, **backward_kwargs, out=grad_logits_out)

        assert_equal(grad_logits, grad_logits_out)

        # Compare against torch reference
        grad_logits_ref = torch_moe_topk_gate_backward(*args, **backward_kwargs)
        atol = 1e-5 * grad_logits_ref.abs().max().item() if num_tokens > 0 else 0.0
        assert torch.allclose(grad_logits, grad_logits_ref, rtol=1e-5, atol=atol), (
            f'{num_routed_experts=}, {num_topk=}, {fix_routing=}, aux_exists={params["aux_exists"]}\n'
            f'Max diff: {(grad_logits - grad_logits_ref).abs().max().item()}'
        )

        # Padded tokens must get exactly zero gradient
        mask = kwargs['mask']
        if mask is not None:
            assert_equal(grad_logits[~mask], torch.zeros_like(grad_logits[~mask]))


def generate_benchmark_func(params):
    num_tokens = params['num_seqs'] * (params['num_tokens'] + params['num_padded_tokens'])
    num_topk = params['num_topk']
    routed_scaling_factor = 1.5
    ep_rank = 0
    fix_routing = False

    kwargs = get_kwargs(params, fix_routing)
    backward_kwargs = get_backward_kwargs(params, kwargs)
    logits = randn((num_tokens, params['num_routed_experts']), dtype=torch.float, device=get_device())
    scores = torch.empty_like(logits)
    topk_idx, topk_weights = tile_kernels.moe.moe_topk_gate_forward(
        logits,
        num_topk,
        False,
        params['num_shared_experts'],
        routed_scaling_factor,
        ep_rank,
        'sqrtsoftplus',
        **kwargs,
        scores=scores,
    )
    grad_topk_weights = randn((num_tokens, num_topk), dtype=torch.float, device=get_device())

    args = (
        scores,
        topk_idx,
        topk_weights,
        grad_topk_weights,
        routed_scaling_factor,
        'sqrtsoftplus',
    )
    grad_logits = tile_kernels.moe.moe_topk_gate_backward(*args, **backward_kwargs)
    func = lambda: tile_kernels.moe.moe_topk_gate_backward(*args, **backward_kwargs)
    num_bytes = count_bytes(scores, topk_idx, topk_weights, grad_topk_weights, backward_kwargs['grad_scores_sum'], grad_logits)
    return func, num_bytes


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(0), ids=make_param_id)
def test_moe_topk_gate_backward_benchmark(benchmark_timer, benchmark_record, params):
    func, num_bytes = generate_benchmark_func(params)
    t_us = benchmark_timer(func)
    bandwidth_gbs = num_bytes / t_us / 1e3

    benchmark_record(
        kernel='moe_topk_gate_backward',
        operation='bwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=bandwidth_gbs,
    )
