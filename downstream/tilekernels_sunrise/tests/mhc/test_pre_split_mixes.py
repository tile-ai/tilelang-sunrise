import pytest
import torch

from tile_kernels.config import get_device
from tile_kernels.mhc.pre_split_mixes_kernel import (
    mhc_pre_split_mixes_bwd,
    mhc_pre_split_mixes_fwd,
)
from tile_kernels.rand import randn
from tile_kernels.testing.numeric import count_bytes
from tile_kernels.torch.mhc import mhc_pre_split_mixes_ref


def generate_pre_split_mixes_test_data(n0: int, n1: int, mhc_mult: int, device: str | None = None) -> dict[str, torch.Tensor]:
    device = get_device() if device is None else device
    mhc_mult3 = mhc_mult * 2 + mhc_mult * mhc_mult

    input_mixes = randn((n0, n1, mhc_mult3), dtype=torch.float, device=device)
    mhc_scale = randn((3,), dtype=torch.float, device=device)
    mhc_base = randn((mhc_mult3,), dtype=torch.float, device=device)

    pre_layer_mix_grad = randn((n0, n1, mhc_mult, 1), dtype=torch.float, device=device)
    post_layer_mix_grad = randn((n0, n1, mhc_mult, 1), dtype=torch.float, device=device)
    comb_res_mix_grad = randn((n0, n1, mhc_mult, mhc_mult), dtype=torch.float, device=device)

    return {
        'input_mixes': input_mixes,
        'mhc_scale': mhc_scale,
        'mhc_base': mhc_base,
        'pre_layer_mix_grad': pre_layer_mix_grad,
        'post_layer_mix_grad': post_layer_mix_grad,
        'comb_res_mix_grad': comb_res_mix_grad,
        'mhc_post_mult_value': 2.0,
        'mhc_pre_eps': 1e-2,
    }


def _run_pre_split_mixes_fwd(
    input_mixes: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_mult: int,
    mhc_post_mult_value: float,
    mhc_pre_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2
    outer_shape = input_mixes.shape[:-1]
    num_tokens = input_mixes.numel() // mhc_mult3
    input_mixes_2d = input_mixes.reshape(num_tokens, mhc_mult3)
    pre_layer_mix = input_mixes_2d.new_empty(num_tokens, mhc_mult)
    post_layer_mix = input_mixes_2d.new_empty(num_tokens, mhc_mult)
    comb_res_mix = input_mixes_2d.new_empty(num_tokens, mhc_mult2)

    mhc_pre_split_mixes_fwd(
        input_mixes_2d,
        mhc_scale,
        mhc_base,
        pre_layer_mix,
        post_layer_mix,
        comb_res_mix,
        mhc_post_mult_value,
        mhc_pre_eps,
    )
    return (
        pre_layer_mix.view(*outer_shape, mhc_mult, 1),
        post_layer_mix.view(*outer_shape, mhc_mult, 1),
        comb_res_mix.view(*outer_shape, mhc_mult, mhc_mult),
    )


def _run_pre_split_mixes_bwd(
    pre_layer_mix_grad: torch.Tensor,
    post_layer_mix_grad: torch.Tensor,
    comb_res_mix_grad: torch.Tensor,
    input_mixes: torch.Tensor,
    post_layer_mix: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_mult: int,
    mhc_post_mult_value: float,
    input_mixes_grad: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2
    num_tokens = input_mixes.numel() // mhc_mult3

    input_mixes_2d = input_mixes.reshape(num_tokens, mhc_mult3)
    post_layer_mix_2d = post_layer_mix.reshape(num_tokens, mhc_mult)
    if input_mixes_grad is None:
        input_mixes_grad = torch.empty_like(input_mixes_2d)
    else:
        input_mixes_grad = input_mixes_grad.reshape(num_tokens, mhc_mult3)

    mhc_scale_grad = torch.empty_like(mhc_scale)
    mhc_base_grad = torch.empty_like(mhc_base)

    mhc_pre_split_mixes_bwd(
        pre_layer_mix_grad.reshape(num_tokens, mhc_mult),
        post_layer_mix_grad.reshape(num_tokens, mhc_mult),
        comb_res_mix_grad.reshape(num_tokens, mhc_mult2),
        input_mixes_2d,
        post_layer_mix_2d,
        mhc_scale,
        mhc_base,
        input_mixes_grad,
        mhc_scale_grad,
        mhc_base_grad,
        mhc_post_mult_value=mhc_post_mult_value,
    )

    return input_mixes_grad.view_as(input_mixes), mhc_scale_grad, mhc_base_grad


@pytest.mark.parametrize('n0', [1, 2])
@pytest.mark.parametrize('n1', [1001, 1024, 4096])
@pytest.mark.parametrize('mhc_mult', [4])
def test_pre_split_mixes_comprehensive(n0: int, n1: int, mhc_mult: int) -> None:
    test_data = generate_pre_split_mixes_test_data(n0=n0, n1=n1, mhc_mult=mhc_mult)

    pre_layer_mix_tl, post_layer_mix_tl, comb_res_mix_tl = _run_pre_split_mixes_fwd(
        test_data['input_mixes'],
        test_data['mhc_scale'],
        test_data['mhc_base'],
        mhc_mult,
        test_data['mhc_post_mult_value'],
        test_data['mhc_pre_eps'],
    )

    input_mixes_grad_tl, mhc_scale_grad_tl, mhc_base_grad_tl = _run_pre_split_mixes_bwd(
        test_data['pre_layer_mix_grad'],
        test_data['post_layer_mix_grad'],
        test_data['comb_res_mix_grad'],
        test_data['input_mixes'],
        post_layer_mix_tl,
        test_data['mhc_scale'],
        test_data['mhc_base'],
        mhc_mult,
        test_data['mhc_post_mult_value'],
    )

    # PTPU has no sigmoid_backward kernel; evaluate the autograd reference on CPU.
    test_data = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in test_data.items()}
    pre_layer_mix_tl, post_layer_mix_tl, comb_res_mix_tl, input_mixes_grad_tl, mhc_scale_grad_tl, mhc_base_grad_tl = (
        t.cpu() for t in (pre_layer_mix_tl, post_layer_mix_tl, comb_res_mix_tl, input_mixes_grad_tl, mhc_scale_grad_tl, mhc_base_grad_tl)
    )
    input_mixes_ref = test_data['input_mixes'].clone().requires_grad_()
    mhc_scale_ref = test_data['mhc_scale'].clone().requires_grad_()
    mhc_base_ref = test_data['mhc_base'].clone().requires_grad_()
    pre_layer_mix_ref, post_layer_mix_ref, comb_res_mix_ref = mhc_pre_split_mixes_ref(
        input_mixes_ref,
        mhc_scale_ref,
        mhc_base_ref,
        mhc_mult,
        test_data['mhc_post_mult_value'],
        test_data['mhc_pre_eps'],
    )
    torch.autograd.backward(
        [pre_layer_mix_ref, post_layer_mix_ref, comb_res_mix_ref],
        [
            test_data['pre_layer_mix_grad'],
            test_data['post_layer_mix_grad'],
            test_data['comb_res_mix_grad'],
        ],
    )
    assert input_mixes_ref.grad is not None
    assert mhc_scale_ref.grad is not None
    assert mhc_base_ref.grad is not None

    torch.testing.assert_close(pre_layer_mix_tl, pre_layer_mix_ref, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(post_layer_mix_tl, post_layer_mix_ref, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(comb_res_mix_tl, comb_res_mix_ref, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(input_mixes_grad_tl, input_mixes_ref.grad, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(mhc_scale_grad_tl, mhc_scale_ref.grad, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(mhc_base_grad_tl, mhc_base_ref.grad, rtol=1e-5, atol=2e-5)


_PRE_SPLIT_BENCH_CASES = [
    (1, 512, 4),
    (1, 8192, 4),
]


@pytest.mark.benchmark
@pytest.mark.parametrize('n0,n1,mhc_mult', _PRE_SPLIT_BENCH_CASES)
def test_pre_split_mixes_fwd_benchmark(
    n0: int,
    n1: int,
    mhc_mult: int,
    benchmark_timer,
    benchmark_record,
) -> None:
    test_data = generate_pre_split_mixes_test_data(n0=n0, n1=n1, mhc_mult=mhc_mult)
    input_mixes = test_data['input_mixes']
    mhc_scale = test_data['mhc_scale']
    mhc_base = test_data['mhc_base']
    post_mult_value = test_data['mhc_post_mult_value']
    pre_eps = test_data['mhc_pre_eps']
    num_tokens = n0 * n1

    pre_out, post_out, comb_out = _run_pre_split_mixes_fwd(
        input_mixes,
        mhc_scale,
        mhc_base,
        mhc_mult,
        post_mult_value,
        pre_eps,
    )
    io_gb = count_bytes(input_mixes, mhc_scale, mhc_base, pre_out, post_out, comb_out) / 1e9

    t_us = benchmark_timer(
        lambda: _run_pre_split_mixes_fwd(input_mixes, mhc_scale, mhc_base, mhc_mult, post_mult_value, pre_eps),
    )

    benchmark_record(
        kernel='mhc_pre_split_mixes',
        operation='pre_split_mixes_forward',
        params={'n0': n0, 'n1': n1, 'mhc_mult': mhc_mult},
        time_us=t_us,
        bandwidth_gbs=io_gb / t_us * 1e6,
        extras={'io_gb': io_gb, 'num_tokens': num_tokens},
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('n0,n1,mhc_mult', _PRE_SPLIT_BENCH_CASES)
def test_pre_split_mixes_bwd_benchmark(
    n0: int,
    n1: int,
    mhc_mult: int,
    benchmark_timer,
    benchmark_record,
) -> None:
    test_data = generate_pre_split_mixes_test_data(n0=n0, n1=n1, mhc_mult=mhc_mult)
    input_mixes = test_data['input_mixes']
    mhc_scale = test_data['mhc_scale']
    mhc_base = test_data['mhc_base']
    mhc_post_mult_value = test_data['mhc_post_mult_value']
    mhc_pre_eps = test_data['mhc_pre_eps']
    num_tokens = n0 * n1

    _pre_out, post_out, _comb_out = _run_pre_split_mixes_fwd(
        input_mixes,
        mhc_scale,
        mhc_base,
        mhc_mult,
        mhc_post_mult_value,
        mhc_pre_eps,
    )
    input_mixes_grad = torch.empty_like(input_mixes)
    mhc_scale_grad = torch.empty_like(mhc_scale)
    mhc_base_grad = torch.empty_like(mhc_base)
    grads = (test_data['pre_layer_mix_grad'], test_data['post_layer_mix_grad'], test_data['comb_res_mix_grad'])
    io_gb = (
        count_bytes(
            *grads,
            input_mixes,
            mhc_scale,
            mhc_base,
            input_mixes_grad,
            mhc_scale_grad,
            mhc_base_grad,
        )
        / 1e9
    )

    def _run():
        _run_pre_split_mixes_bwd(
            test_data['pre_layer_mix_grad'],
            test_data['post_layer_mix_grad'],
            test_data['comb_res_mix_grad'],
            input_mixes,
            post_out,
            mhc_scale,
            mhc_base,
            mhc_mult,
            mhc_post_mult_value,
            input_mixes_grad=input_mixes_grad,
        )

    _run()
    t_us = benchmark_timer(_run)

    benchmark_record(
        kernel='mhc_pre_split_mixes',
        operation='pre_split_mixes_backward',
        params={'n0': n0, 'n1': n1, 'mhc_mult': mhc_mult},
        time_us=t_us,
        bandwidth_gbs=io_gb / t_us * 1e6,
        extras={'io_gb': io_gb, 'num_tokens': num_tokens},
    )
