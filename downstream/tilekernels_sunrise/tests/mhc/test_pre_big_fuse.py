import pytest
import torch
from tile_kernels.modeling.mhc.ops import (
    mhc_pre_big_fuse,
    mhc_pre_big_fuse_post,
)

from tile_kernels.testing.numeric import count_bytes
from tile_kernels.torch.mhc import (
    mhc_post_ref,
    mhc_pre_apply_mix_ref,
    mhc_pre_norm_fn_ref,
    mhc_pre_split_mixes_ref,
    sinkhorn_normalize_ref,
)


def generate_big_fuse_test_data(
    n1: int,
    mhc_mult: int,
    hidden_size: int,
    rms_eps: float = 1e-6,
    mhc_pre_eps: float = 1e-6,
    mhc_sinkhorn_eps: float = 1e-6,
    mhc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 10,
    n_splits: int = 16,
) -> dict[str, torch.Tensor | float]:
    """Generate test data for pre_big_fuse correctness and benchmark tests."""
    n0 = 1
    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2
    device = 'ptpu'

    arange = torch.arange(mhc_mult, device=device)
    scale_factor = 1 + arange.mul(0.01).view(1, 1, -1, 1)

    residual = (
        torch.randn((n0, n1, mhc_mult, hidden_size), dtype=torch.float)
        .to(device=device)
        .mul(scale_factor)
        .bfloat16()
    )

    fn = (
        torch.randn((mhc_mult3, mhc_mult, hidden_size), dtype=torch.float)
        .to(device=device)
        * 1e-4
        * (1 + arange.mul(0.01).view(1, -1, 1))
    ).flatten(1, 2)

    mhc_scale = torch.randn((3,), dtype=torch.float).to(device=device) * 0.1
    mhc_base = torch.randn((mhc_mult3,), dtype=torch.float).to(device=device) * 0.1

    return {
        'residual': residual,
        'fn': fn,
        'mhc_scale': mhc_scale,
        'mhc_base': mhc_base,
        'rms_eps': rms_eps,
        'mhc_pre_eps': mhc_pre_eps,
        'mhc_sinkhorn_eps': mhc_sinkhorn_eps,
        'mhc_post_mult_value': mhc_post_mult_value,
        'sinkhorn_repeat': sinkhorn_repeat,
        'n_splits': n_splits,
    }


def big_fuse_reference(
    residual: torch.Tensor,
    fn: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference implementation of pre_big_fuse using pure PyTorch ops on CPU.

    Uses PyTorch reference implementations (not TileLang kernels) to provide
    a clean ground truth that is free of accelerator-specific numerical issues.
    """
    _ = n_splits  # unused by reference
    mhc_mult = residual.shape[-2]

    mixes = mhc_pre_norm_fn_ref(residual, fn, None, rms_eps)
    pre_mix, post_mix, comb_mix = mhc_pre_split_mixes_ref(
        mixes, mhc_scale, mhc_base, mhc_mult, mhc_post_mult_value, mhc_pre_eps,
    )
    comb_mix = sinkhorn_normalize_ref(comb_mix, repeat=sinkhorn_repeat, eps=mhc_sinkhorn_eps)
    layer_input = mhc_pre_apply_mix_ref(residual, pre_mix)

    return post_mix, comb_mix, layer_input


@pytest.mark.parametrize('n1', [512, 1024, 2048, 8192])
@pytest.mark.parametrize('hidden_size', [1280, 2560, 4096])
@pytest.mark.parametrize('mhc_mult', [4])
def test_correctness(
    n1: int,
    hidden_size: int,
    mhc_mult: int,
) -> None:
    """Verify that mhc_pre_big_fuse matches the reference decomposed implementation."""
    test_data = generate_big_fuse_test_data(
        n1=n1, mhc_mult=mhc_mult, hidden_size=hidden_size,
    )

    post_mix_fused, comb_mix_fused, layer_input_fused = mhc_pre_big_fuse(
        test_data['residual'],
        test_data['fn'],
        test_data['mhc_scale'],
        test_data['mhc_base'],
        rms_eps=test_data['rms_eps'],
        mhc_pre_eps=test_data['mhc_pre_eps'],
        mhc_sinkhorn_eps=test_data['mhc_sinkhorn_eps'],
        mhc_post_mult_value=test_data['mhc_post_mult_value'],
        sinkhorn_repeat=test_data['sinkhorn_repeat'],
        n_splits=test_data['n_splits'],
    )
    torch.ptpu.synchronize()

    post_mix_ref, comb_mix_ref, layer_input_ref = big_fuse_reference(**test_data)

    # Move to CPU before converting to float (must be in this order on ptpu).
    torch.ptpu.synchronize()
    layer_input_fused = layer_input_fused.cpu()
    layer_input_ref = layer_input_ref.cpu()
    post_mix_fused = post_mix_fused.cpu()
    post_mix_ref = post_mix_ref.cpu()
    comb_mix_fused = comb_mix_fused.cpu()
    comb_mix_ref = comb_mix_ref.cpu()

    # post_mix/comb_mix: sinkhorn normalization in fp32 accumulates small
    # differences from computation reordering. Measured max over 5 seeds x
    # all shapes is 1.58e-6, so atol=1e-4 leaves ample headroom. The original
    # assertions passed no atol at all, falling back to allclose's
    # rtol=1e-5/atol=1e-8 — far too strict for values near zero.
    pm_diff = (post_mix_fused.float() - post_mix_ref.float()).abs().max().item()
    cm_diff = (comb_mix_fused.float() - comb_mix_ref.float()).abs().max().item()
    assert torch.allclose(post_mix_fused.float(), post_mix_ref.float(), atol=1e-4), \
        f'post_mix max diff {pm_diff:.6e} at n1={n1}, hidden_size={hidden_size}'
    assert torch.allclose(comb_mix_fused.float(), comb_mix_ref.float(), atol=1e-4), \
        f'comb_mix max diff {cm_diff:.6e} at n1={n1}, hidden_size={hidden_size}'
    # layer_input is bf16; accumulation-order differences can flip the last bit,
    # which is value/128 for bfloat16 (~0.0156 for values around 2.0).
    assert torch.allclose(layer_input_fused.float(), layer_input_ref.float(), atol=2e-2)


@pytest.mark.parametrize('n1', [512, 1024, 2048, 8192])
@pytest.mark.parametrize('hidden_size', [1280, 2560, 4096])
@pytest.mark.parametrize('mhc_mult', [4])
@pytest.mark.benchmark
def test_correctness_benchmark(
    benchmark_timer,
    benchmark_record,
    n1: int,
    hidden_size: int,
    mhc_mult: int,
) -> None:
    """Benchmark mhc_pre_big_fuse kernel performance (fwd pass)."""
    test_data = generate_big_fuse_test_data(
        n1=n1, mhc_mult=mhc_mult, hidden_size=hidden_size,
    )

    # Clone tensors so the benchmark operates on fresh copies each time.
    residual_tl = test_data['residual'].clone()
    fn_tl = test_data['fn'].clone()
    mhc_scale_tl = test_data['mhc_scale'].clone()
    mhc_base_tl = test_data['mhc_base'].clone()

    # Warmup: prime the PTU before timing.
    torch.ptpu.synchronize()
    post_mix_fused, comb_mix_fused, layer_input_fused = mhc_pre_big_fuse(
        residual_tl, fn_tl, mhc_scale_tl, mhc_base_tl,
        rms_eps=test_data['rms_eps'],
        mhc_pre_eps=test_data['mhc_pre_eps'],
        mhc_sinkhorn_eps=test_data['mhc_sinkhorn_eps'],
        mhc_post_mult_value=test_data['mhc_post_mult_value'],
        sinkhorn_repeat=test_data['sinkhorn_repeat'],
        n_splits=test_data['n_splits'],
    )
    torch.ptpu.synchronize()

    t_save_us = benchmark_timer(lambda: mhc_pre_big_fuse(
        residual_tl, fn_tl, mhc_scale_tl, mhc_base_tl,
        rms_eps=test_data['rms_eps'],
        mhc_pre_eps=test_data['mhc_pre_eps'],
        mhc_sinkhorn_eps=test_data['mhc_sinkhorn_eps'],
        mhc_post_mult_value=test_data['mhc_post_mult_value'],
        sinkhorn_repeat=test_data['sinkhorn_repeat'],
        n_splits=test_data['n_splits'],
    ))
    num_bytes_save = count_bytes(
        residual_tl, fn_tl, mhc_scale_tl, mhc_base_tl,
        post_mix_fused, comb_mix_fused, layer_input_fused,
    )

    benchmark_record(
        kernel='mhc_pre_big_fuse',
        operation='fwd',
        params={
            'residual': f'shape={residual_tl.shape}, dtype={residual_tl.dtype}',
            'fn': f'shape={fn_tl.shape}, dtype={fn_tl.dtype}',
            'mhc_scale': f'shape={mhc_scale_tl.shape}, dtype={mhc_scale_tl.dtype}',
            'mhc_base': f'shape={mhc_base_tl.shape}, dtype={mhc_base_tl.dtype}',
            'rms_eps': test_data['rms_eps'],
            'mhc_pre_eps': test_data['mhc_pre_eps'],
            'mhc_sinkhorn_eps': test_data['mhc_sinkhorn_eps'],
            'mhc_post_mult_value': test_data['mhc_post_mult_value'],
            'sinkhorn_repeat': test_data['sinkhorn_repeat'],
            'n_splits': test_data['n_splits'],
        },
        time_us=t_save_us,
        bandwidth_gbs=num_bytes_save / t_save_us / 1e3,
    )


# ============================================================
# Tests for the fused pre_big_fuse + post_fwd kernel
# ============================================================

def big_fuse_post_reference(
    residual: torch.Tensor,
    fn: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference: pre_big_fuse → post_fwd using pure PyTorch (CPU)."""
    post_mix, comb_mix, layer_input = big_fuse_reference(
        residual, fn, mhc_scale, mhc_base,
        rms_eps, mhc_pre_eps, mhc_sinkhorn_eps,
        mhc_post_mult_value, sinkhorn_repeat, n_splits,
    )
    out = mhc_post_ref(layer_input, residual, post_mix, comb_mix)
    return post_mix, comb_mix, out


@pytest.mark.parametrize('n1', [512, 1024, 2048, 8192])
@pytest.mark.parametrize('hidden_size', [1280, 2560, 4096, 7168, 8192])
@pytest.mark.parametrize('mhc_mult', [4])
def test_fused_correctness(
    n1: int,
    hidden_size: int,
    mhc_mult: int,
) -> None:
    """Verify mhc_pre_big_fuse_post matches reference (pre_big_fuse + post_fwd)."""
    test_data = generate_big_fuse_test_data(
        n1=n1, mhc_mult=mhc_mult, hidden_size=hidden_size,
    )

    post_mix_fused, comb_mix_fused, out_fused = mhc_pre_big_fuse_post(
        test_data['residual'],
        test_data['fn'],
        test_data['mhc_scale'],
        test_data['mhc_base'],
        rms_eps=test_data['rms_eps'],
        mhc_pre_eps=test_data['mhc_pre_eps'],
        mhc_sinkhorn_eps=test_data['mhc_sinkhorn_eps'],
        mhc_post_mult_value=test_data['mhc_post_mult_value'],
        sinkhorn_repeat=test_data['sinkhorn_repeat'],
        n_splits=test_data['n_splits'],
    )
    torch.ptpu.synchronize()

    post_mix_ref, comb_mix_ref, out_ref = big_fuse_post_reference(**test_data)

    torch.ptpu.synchronize()
    out_fused = out_fused.cpu()
    out_ref = out_ref.cpu()
    post_mix_fused = post_mix_fused.cpu()
    post_mix_ref = post_mix_ref.cpu()
    comb_mix_fused = comb_mix_fused.cpu()
    comb_mix_ref = comb_mix_ref.cpu()

    # post_mix and comb_mix should be identical (same Phase 1 in both paths);
    # measured max over 5 seeds x all shapes is 1.58e-6, so atol=1e-4 leaves
    # ample headroom. Kernel output is bit-identical across repeated calls, so
    # these diffs are pure fp32 accumulation order, not non-determinism.
    pm_diff = (post_mix_fused.float() - post_mix_ref.float()).abs().max().item()
    cm_diff = (comb_mix_fused.float() - comb_mix_ref.float()).abs().max().item()
    assert torch.allclose(post_mix_fused.float(), post_mix_ref.float(), atol=1e-4), \
        f'post_mix max diff {pm_diff:.6e} at n1={n1}, hidden_size={hidden_size}'
    assert torch.allclose(comb_mix_fused.float(), comb_mix_ref.float(), atol=1e-4), \
        f'comb_mix max diff {cm_diff:.6e} at n1={n1}, hidden_size={hidden_size}'

    # out: fused kernel keeps intermediates in fp32 (more accurate);
    # reference quantizes layer_input→bf16 before post_fwd (lossy).
    # Allow 4 bf16 ULP tolerance (~0.25 at value 8.0).
    out_diff = (out_fused.float() - out_ref.float()).abs()
    max_diff = out_diff.max().item()
    assert max_diff < 0.25, (
        f'out max diff {max_diff:.2e} exceeds 4 bf16 ULP tolerance at '
        f'n1={n1}, hidden_size={hidden_size}'
    )
