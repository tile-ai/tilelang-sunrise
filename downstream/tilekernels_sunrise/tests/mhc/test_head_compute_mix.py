from typing import Callable

import pytest
import torch
from tile_kernels.modeling.mhc.ops import mhc_head_compute_mix
from tile_kernels.torch.mhc import mhc_head_compute_mix_ref
from tile_kernels.testing.numeric import count_bytes


def generate_head_compute_mix_test_data(
    n0: int, n1: int, mhc_mult: int, device: str = 'ptpu'
) -> dict[str, torch.Tensor]:
    input_mix = torch.randn((n0, n1, mhc_mult), dtype=torch.float).to(device=device)
    mhc_scale = torch.randn(1, dtype=torch.float).to(device=device)
    mhc_base = torch.randn(mhc_mult, dtype=torch.float).to(device=device)
    output_mix_grad = torch.randn((n0, n1, mhc_mult), dtype=torch.float).to(device=device)

    return {
        'input_mix': input_mix,
        'mhc_scale': mhc_scale,
        'mhc_base': mhc_base,
        'output_mix_grad': output_mix_grad,
        'mhc_pre_eps': 1e-2,
    }


def _tester(
    impl: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor],
    test_data: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_mix_ = test_data['input_mix'].clone().requires_grad_()
    mhc_scale_ = test_data['mhc_scale'].clone().requires_grad_()
    mhc_base_ = test_data['mhc_base'].clone().requires_grad_()
    output_mix_ = impl(input_mix_, mhc_scale_, mhc_base_, test_data['mhc_pre_eps'])
    # torch.autograd.backward([output_mix_], [test_data['output_mix_grad']])
    # return output_mix_, input_mix_.grad, mhc_scale_.grad, mhc_base_.grad
    return output_mix_


@pytest.mark.parametrize('n0', [1, 2])
@pytest.mark.parametrize('n1', [1024, 4096])
@pytest.mark.parametrize('mhc_mult', [4])
def test_head_compute_mix_comprehensive(n0: int, n1: int, mhc_mult: int) -> None:
    test_data = generate_head_compute_mix_test_data(n0=n0, n1=n1, mhc_mult=mhc_mult)

    # output_mix_tl, grad_input_mix_tl, grad_mhc_scale_tl, grad_mhc_base_tl = _tester(
    #     mhc_head_compute_mix, test_data
    # )
    # output_mix_ref, grad_input_mix_ref, grad_mhc_scale_ref, grad_mhc_base_ref = _tester(
    #     mhc_head_compute_mix_ref, test_data
    # )

    # torch.testing.assert_close(output_mix_tl, output_mix_ref, rtol=1e-4, atol=1e-5)
    # torch.testing.assert_close(grad_input_mix_tl, grad_input_mix_ref, rtol=1e-4, atol=1e-5)
    # torch.testing.assert_close(grad_mhc_scale_tl, grad_mhc_scale_ref, rtol=1e-4, atol=1e-5)
    # torch.testing.assert_close(grad_mhc_base_tl, grad_mhc_base_ref, rtol=1e-4, atol=1e-5)

    output_mix_tl = _tester(
        mhc_head_compute_mix, test_data
    )
    output_mix_ref = _tester(
        mhc_head_compute_mix_ref, test_data
    )
    output_mix_tl = output_mix_tl.cpu()

    torch.testing.assert_close(output_mix_tl, output_mix_ref, rtol=1e-4, atol=1e-5)

@pytest.mark.parametrize('n0', [1, 2])
@pytest.mark.parametrize('n1', [1024, 4096])
@pytest.mark.parametrize('mhc_mult', [4])
@pytest.mark.benchmark
def test_head_compute_mix_comprehensive_benchmark(benchmark_timer, benchmark_record, n0: int, n1: int, mhc_mult: int) -> None:
    test_data = generate_head_compute_mix_test_data(n0=n0, n1=n1, mhc_mult=mhc_mult)

    input_mix_tl = test_data['input_mix'].clone()
    mhc_scale_tl = test_data['mhc_scale'].clone()
    mhc_base_tl = test_data['mhc_base'].clone()

    output_mix_tl = mhc_head_compute_mix(input_mix_tl, mhc_scale_tl, mhc_base_tl, test_data['mhc_pre_eps'])

    t_save_us = benchmark_timer(lambda: mhc_head_compute_mix(input_mix_tl, mhc_scale_tl, mhc_base_tl, test_data['mhc_pre_eps']))
    num_bytes_save = count_bytes(input_mix_tl, mhc_scale_tl, mhc_base_tl, output_mix_tl)

    benchmark_record(
        kernel='mhc_head_compute_mix',
        operation='fwd',
        params={'input_mix': input_mix_tl.cpu() if input_mix_tl is not None else input_mix_tl,
                'mhc_scale': mhc_scale_tl.cpu() if mhc_scale_tl is not None else mhc_scale_tl,
                'mhc_base': mhc_base_tl.cpu() if mhc_base_tl is not None else mhc_base_tl,
                'mhc_pre_eps': test_data['mhc_pre_eps']},
        time_us=t_save_us,
        bandwidth_gbs=num_bytes_save / t_save_us / 1e3,
    )
