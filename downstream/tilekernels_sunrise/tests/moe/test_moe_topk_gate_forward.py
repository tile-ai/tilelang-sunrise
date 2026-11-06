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
from tile_kernels.torch import moe_topk_gate_forward as torch_moe_topk_gate_forward
from tile_kernels.torch.topk import sqrt_softplus_ref

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
            'num_topk': num_topk,
            'num_routed_experts': num_routed_experts,
            'num_shared_experts': num_shared_experts,
            'bias_exists': bias_exists,
            'image_bias_exists': image_bias_exists,
            'force_random_exists': force_random_exists,
            'use_shared_as_routed': use_shared_as_routed,
            'scores_exists': scores_exists,
        }
        for num_tokens in generate_num_tokens(level)
        for num_padded_tokens in (0, 10)
        for num_routed_experts, num_shared_experts, num_topk in configs
        for bias_exists in ([True] if level == 0 else [True, False])
        for image_bias_exists in ([True] if level == 0 else [True, False])
        for force_random_exists in ([True] if level == 0 else [True, False])
        for use_shared_as_routed in (False, True)
        for scores_exists in ([True] if level < 2 else [True, False])
    ]
    return params


def get_kwargs(params, fix_routing):
    num_tokens, num_padded_tokens = params['num_tokens'], params['num_padded_tokens']
    num_routed_experts = params['num_routed_experts']
    num_shared_experts = params['num_shared_experts']
    use_shared_as_routed = params['use_shared_as_routed']
    num_topk = params['num_topk']
    bias_exists, image_bias_exists = params['bias_exists'], params['image_bias_exists']
    force_random_exists = params['force_random_exists']
    mask = None
    if num_padded_tokens > 0:
        mask = torch.ones(num_tokens + num_padded_tokens, dtype=torch.bool, device=get_device())
        mask[-num_padded_tokens:] = False

    to_physical_map = None
    logical_count = None
    fix_routing_mask = None
    force_random = None
    unmapped_topk_idx = torch.zeros((num_tokens + num_padded_tokens, num_topk), dtype=torch.int64, device=get_device())

    bias = randn(num_routed_experts, dtype=torch.float, device=get_device()) if bias_exists else None
    image_bias = randn(num_routed_experts, dtype=torch.float, device=get_device()) if image_bias_exists else None
    image_token_mask = torch.rand(num_tokens + num_padded_tokens, dtype=torch.float, device=get_device()) < 0.5 if image_bias_exists else None

    if use_shared_as_routed:
        # Contiguous mapping
        # Keep this static JIT parameter stable across token counts so the
        # precompile and full-shape tests share the same kernel cache entry.
        num_max_duplicate_experts = 3
        num_logical_experts = num_routed_experts + num_shared_experts
        logical_count = torch.randint(1, num_max_duplicate_experts + 1, (num_logical_experts,), dtype=torch.int32, device=get_device())
        row_offsets = (logical_count.cumsum(dim=0) - logical_count).to(torch.int32)
        col_offsets = torch.arange(num_max_duplicate_experts, dtype=torch.int32, device=get_device())
        to_physical_map = torch.where(
            col_offsets < logical_count.view(-1, 1),
            row_offsets.view(-1, 1) + col_offsets,
            -1,
        )

    if fix_routing:
        fix_routing_mask = torch.ones((num_tokens + num_padded_tokens,), dtype=torch.bool, device=get_device())
        unmapped_topk_idx = torch.randint(
            0,
            num_routed_experts,
            (num_tokens + num_padded_tokens, num_topk),
            dtype=torch.int64,
            device=get_device(),
        )

    if force_random_exists:
        force_random = torch.rand(num_tokens + num_padded_tokens, device=get_device()) < 0.5
        # Force random do not appear with fix_routing, which leads to conflict
        if fix_routing:
            force_random = torch.logical_and(torch.logical_not(fix_routing_mask), force_random)

    return dict(
        mask=mask,
        bias=bias,
        image_bias=image_bias,
        image_token_mask=image_token_mask,
        fix_routing_mask=fix_routing_mask,
        to_physical_map=to_physical_map,
        logical_count=logical_count,
        unmapped_topk_idx=unmapped_topk_idx,
        force_random=force_random,
    )


def assert_uniform_distribution(values: torch.Tensor, num_bins: int, description: str):
    """Assert that values are approximately uniformly distributed across [0, num_bins) using chi-squared test."""
    if values.numel() < num_bins * 10:
        return
    if values.device.type == 'ptpu':
        torch.ptpu.synchronize()
        values = values.cpu()
    counts = torch.bincount(values.to(torch.int64), minlength=num_bins).float()
    expected = values.numel() / num_bins
    chi_squared = ((counts - expected) ** 2 / expected).sum().item()
    degrees_of_freedom = num_bins - 1
    # Threshold: chi-squared < 4 * degrees_of_freedom
    assert chi_squared < 4 * degrees_of_freedom, (
        f'Force random expert distribution is not uniform ({description}): '
        f'chi_squared={chi_squared:.2f}, threshold={4 * degrees_of_freedom}, '
        f'counts={counts.tolist()}, expected={expected:.1f}'
    )


@pytest.mark.parametrize('params', generate_test_params(get_test_level()), ids=make_param_id)
def test_moe_topk_gate_forward(params):
    num_tokens, num_padded_tokens = params['num_tokens'], params['num_padded_tokens']
    num_routed_experts = params['num_routed_experts']
    num_shared_experts = params['num_shared_experts']
    num_topk = params['num_topk']
    scoring_func = 'sqrtsoftplus'
    use_shared_as_routed = params['use_shared_as_routed']

    routed_scaling_factor = 1.5
    ep_rank = 0

    # Correctness test
    for fix_routing in (False, True):
        kwargs = get_kwargs(params, fix_routing)
        logits = construct_logits(params, kwargs)
        scores = torch.empty_like(logits) if params['scores_exists'] else None
        args = (
            logits,
            num_topk,
            use_shared_as_routed,
            num_shared_experts,
            routed_scaling_factor,
            ep_rank,
            scoring_func,
        )

        def clone(tensor_dict: dict):
            return {k: v.clone() if k == 'unmapped_topk_idx' else v for (k, v) in tensor_dict.items()}

        kwargs_ref = clone({k: v for k, v in kwargs.items() if k != 'force_random'})
        kwargs_out = clone(kwargs)

        topk_idx, topk_weights = tile_kernels.moe.moe_topk_gate_forward(*args, **kwargs, scores=scores)
        num_physical_topk = num_topk + num_shared_experts if use_shared_as_routed else num_topk
        topk_idx_out = torch.empty((num_tokens + num_padded_tokens, num_physical_topk), dtype=torch.int64, device=get_device())
        topk_weights_out = torch.empty(num_tokens + num_padded_tokens, num_physical_topk, dtype=torch.float32, device=get_device())
        scores_out = torch.empty_like(logits) if params['scores_exists'] else None
        tile_kernels.moe.moe_topk_gate_forward(*args, **kwargs_out, out=(topk_idx_out, topk_weights_out), scores=scores_out)

        assert_equal(topk_idx, topk_idx_out)
        assert_equal(topk_weights, topk_weights_out)
        assert_equal(kwargs['unmapped_topk_idx'], kwargs_out['unmapped_topk_idx'])

        # Compare against torch reference for tokens where force_random is False
        topk_idx_ref, topk_weights_ref = torch_moe_topk_gate_forward(*args, **kwargs_ref)

        # Mask to select tokens that are NOT force_random (i.e. should match ref)
        force_random_mask = kwargs['force_random']
        mask = kwargs['mask']
        logical_count = kwargs['logical_count']
        if force_random_mask is not None:
            force_random_mask = torch.logical_and(
                force_random_mask, mask if mask is not None else torch.ones(num_tokens + num_padded_tokens, dtype=torch.bool, device=get_device())
            )
            compare_mask = ~force_random_mask
        else:
            compare_mask = torch.ones(topk_idx.size(0), dtype=torch.bool, device=get_device())

        if params['scores_exists']:
            assert_equal(scores, scores_out)
            routed = compare_mask if mask is None else torch.logical_and(compare_mask, mask)
            assert torch.allclose(scores, torch.where(routed.unsqueeze(1), sqrt_softplus_ref(logits), 0.0))

        unmapped_topk_idx = kwargs['unmapped_topk_idx']
        unmapped_topk_idx_ref = kwargs_ref['unmapped_topk_idx']

        sorted_topk_idx, _ = topk_idx[compare_mask].sort(dim=1)
        sorted_topk_idx_ref, _ = topk_idx_ref[compare_mask].sort(dim=1)

        sorted_unmapped_topk_idx = unmapped_topk_idx[compare_mask].sort(dim=1)[0]
        sorted_unmapped_topk_idx_ref = unmapped_topk_idx_ref[compare_mask].sort(dim=1)[0]

        sorted_topk_weights, _ = topk_weights[compare_mask].sort(dim=1)
        sorted_topk_weights_ref, _ = topk_weights_ref[compare_mask].sort(dim=1)

        assert_equal(sorted_topk_idx, sorted_topk_idx_ref)
        assert_equal(sorted_unmapped_topk_idx, sorted_unmapped_topk_idx_ref)
        assert torch.allclose(sorted_topk_weights, sorted_topk_weights_ref), (
            f'{sorted_topk_weights=}\n'
            f'{sorted_topk_weights_ref=}\n'
            f'{scoring_func=}, {num_routed_experts=}, {num_shared_experts=}, {num_topk=}, {fix_routing=}\n'
            f'Different topk weights: \n'
            f'{[(sorted_topk_weights[i], sorted_topk_weights_ref[i]) for i in range(topk_weights.size(0)) if not torch.equal(sorted_topk_weights[i], sorted_topk_weights_ref[i])]}'
        )

        # Check distribution for force random tokens
        if force_random_mask is not None and force_random_mask.any():
            fr_topk_idx = topk_idx[force_random_mask]
            fr_topk_weights = topk_weights[force_random_mask]
            num_logical_experts = num_routed_experts + (num_shared_experts if use_shared_as_routed else 0)
            num_physical_experts = torch.sum(logical_count).item() if logical_count is not None else num_logical_experts

            # Weights should be positive
            if fr_topk_weights.numel() > 0:
                assert (fr_topk_weights > 0).all(), (
                    f'Force random tokens should have positive weights for valid experts, got: {fr_topk_weights[fr_topk_weights <= 0]}'
                )

            # Validity: random experts should in range
            assert torch.all(torch.logical_and(fr_topk_idx < num_physical_experts, fr_topk_idx >= 0)).item()

            # Uniformity: random experts should be uniformly distributed
            assert_uniform_distribution(fr_topk_idx.view(-1), num_physical_experts, 'rank')

            # all -1 for unmapped
            assert torch.all(unmapped_topk_idx[force_random_mask] == -1).item()

    # Check sort stability
    if not params['force_random_exists']:
        logits = torch.zeros((num_tokens + num_padded_tokens, num_routed_experts), dtype=torch.float, device=get_device())
        args = (
            logits,
            num_topk,
            use_shared_as_routed,
            num_shared_experts,
            routed_scaling_factor,
            ep_rank,
            scoring_func,
        )
        fix_routing = False
        kwargs = get_kwargs(params, fix_routing)
        kwargs['bias'], kwargs['image_bias'], kwargs['image_token_mask'] = None, None, None
        unmapped_topk_idx = kwargs['unmapped_topk_idx']
        scores = torch.empty_like(logits) if params['scores_exists'] else None
        topk_idx, topk_weights = tile_kernels.moe.moe_topk_gate_forward(*args, **kwargs, scores=scores)
        assert torch.all(
            unmapped_topk_idx[:num_tokens, :num_topk] == torch.arange(0, num_topk, dtype=unmapped_topk_idx.dtype, device=unmapped_topk_idx.device)
        )


def generate_benchmark_func(params):
    num_tokens, num_padded_tokens = params['num_tokens'], params['num_padded_tokens']
    num_routed_experts = params['num_routed_experts']
    routed_scaling_factor = 1.5
    ep_rank = 0
    fix_routing = False

    logits = randn((num_tokens + num_padded_tokens, num_routed_experts), dtype=torch.float, device=get_device())
    scores = torch.empty_like(logits) if params['scores_exists'] else None
    kwargs = get_kwargs(params, fix_routing)

    args = (
        logits,
        params['num_topk'],
        params['use_shared_as_routed'],
        params['num_shared_experts'],
        routed_scaling_factor,
        ep_rank,
        'sqrtsoftplus',
    )
    func = lambda: tile_kernels.moe.moe_topk_gate_forward(*args, **kwargs, scores=scores)
    num_bytes = count_bytes(logits)
    return func, num_bytes


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(0), ids=make_param_id)
def test_moe_topk_gate_forward_benchmark(benchmark_timer, benchmark_record, params):
    func, num_bytes = generate_benchmark_func(params)
    t_us = benchmark_timer(func)
    bandwidth_gbs = num_bytes / t_us / 1e3

    benchmark_record(
        kernel='moe_topk_gate_forward',
        operation='fwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=bandwidth_gbs,
    )
