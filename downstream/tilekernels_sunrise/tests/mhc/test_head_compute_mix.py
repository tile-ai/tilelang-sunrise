import pytest
import torch

from tile_kernels.config import get_device, get_num_sms, get_num_vec_cores, is_ascend
from tile_kernels.mhc.head_compute_mix_kernel import mhc_head_compute_mix_bwd, mhc_head_compute_mix_fwd
from tile_kernels.rand import randn
from tile_kernels.testing.numeric import count_bytes
from tile_kernels.torch.mhc import mhc_head_compute_mix_ref


def generate_head_compute_mix_test_data(n0: int, n1: int, mhc_mult: int, device: str | None = None) -> dict[str, torch.Tensor]:
    device = get_device() if device is None else device
    input_mix = randn((n0, n1, mhc_mult), dtype=torch.float, device=device)
    mhc_scale = randn(1, dtype=torch.float, device=device)
    mhc_base = randn(mhc_mult, dtype=torch.float, device=device)
    output_mix_grad = randn((n0, n1, mhc_mult), dtype=torch.float, device=device)

    return {
        'input_mix': input_mix,
        'mhc_scale': mhc_scale,
        'mhc_base': mhc_base,
        'output_mix_grad': output_mix_grad,
        'mhc_pre_eps': 1e-2,
    }


def _run_head_compute_mix_fwd(
    input_mix: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_pre_eps: float,
) -> torch.Tensor:
    assert input_mix.ndim == 3
    mhc_mult = input_mix.shape[-1]
    output_mix = torch.empty_like(input_mix)
    mhc_head_compute_mix_fwd(
        input_mix.view(-1, mhc_mult),
        mhc_scale,
        mhc_base,
        output_mix.view(-1, mhc_mult),
        mhc_pre_eps,
    )
    return output_mix.view_as(input_mix)


def _run_head_compute_mix_bwd(
    output_mix_grad: torch.Tensor,
    input_mix: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mhc_mult = input_mix.shape[-1]
    input_mix_grad = torch.empty_like(input_mix)
    num_sms = get_num_vec_cores() if is_ascend() else get_num_sms()
    mhc_scale_grad_partial = torch.empty(
        num_sms,
        *mhc_scale.shape,
        dtype=mhc_scale.dtype,
        device=mhc_scale.device,
    )
    mhc_base_grad_partial = torch.empty(
        num_sms,
        *mhc_base.shape,
        dtype=mhc_base.dtype,
        device=mhc_base.device,
    )
    mhc_head_compute_mix_bwd(
        output_mix_grad.view(-1, mhc_mult),
        input_mix.view(-1, mhc_mult),
        mhc_scale,
        mhc_base,
        input_mix_grad.view(-1, mhc_mult),
        mhc_scale_grad_partial,
        mhc_base_grad_partial,
        num_sms,
    )
    return input_mix_grad, mhc_scale_grad_partial.sum(0), mhc_base_grad_partial.sum(0)


@pytest.mark.parametrize('n0', [1, 2])
@pytest.mark.parametrize('n1', [1001, 1024, 4096, 24576])
@pytest.mark.parametrize('mhc_mult', [4])
def test_head_compute_mix_comprehensive(n0: int, n1: int, mhc_mult: int) -> None:
    test_data = generate_head_compute_mix_test_data(n0=n0, n1=n1, mhc_mult=mhc_mult)

    output_mix_tl = _run_head_compute_mix_fwd(
        test_data['input_mix'],
        test_data['mhc_scale'],
        test_data['mhc_base'],
        test_data['mhc_pre_eps'],
    )
    grad_input_mix_tl, grad_mhc_scale_tl, grad_mhc_base_tl = _run_head_compute_mix_bwd(
        test_data['output_mix_grad'],
        test_data['input_mix'],
        test_data['mhc_scale'],
        test_data['mhc_base'],
    )

    input_mix_ref = test_data['input_mix'].cpu().clone().requires_grad_()
    mhc_scale_ref = test_data['mhc_scale'].cpu().clone().requires_grad_()
    mhc_base_ref = test_data['mhc_base'].cpu().clone().requires_grad_()
    output_mix_ref = mhc_head_compute_mix_ref(input_mix_ref, mhc_scale_ref, mhc_base_ref, test_data['mhc_pre_eps'])
    torch.autograd.backward([output_mix_ref], [test_data['output_mix_grad'].cpu()])
    assert input_mix_ref.grad is not None
    assert mhc_scale_ref.grad is not None
    assert mhc_base_ref.grad is not None

    torch.testing.assert_close(output_mix_tl.cpu(), output_mix_ref, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(grad_input_mix_tl.cpu(), input_mix_ref.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(grad_mhc_scale_tl.cpu(), mhc_scale_ref.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(grad_mhc_base_tl.cpu(), mhc_base_ref.grad, rtol=1e-4, atol=1e-5)


_HEAD_COMPUTE_BENCH_CASES = [
    (1, 512, 4),
    (1, 8192, 4),
]


@pytest.mark.benchmark
@pytest.mark.parametrize('n0,n1,mhc_mult', _HEAD_COMPUTE_BENCH_CASES)
def test_head_compute_mix_fwd_benchmark(
    n0: int,
    n1: int,
    mhc_mult: int,
    benchmark_timer,
    benchmark_record,
) -> None:
    test_data = generate_head_compute_mix_test_data(n0=n0, n1=n1, mhc_mult=mhc_mult)
    input_mix = test_data['input_mix']
    mhc_scale = test_data['mhc_scale']
    mhc_base = test_data['mhc_base']
    eps = test_data['mhc_pre_eps']
    num_tokens = n0 * n1

    output_mix = torch.empty_like(input_mix)
    io_gb = count_bytes(input_mix) * 2 / 1e9  # read + write, same shape

    input_mix_2d = input_mix.view(num_tokens, mhc_mult)
    output_mix_2d = output_mix.view(num_tokens, mhc_mult)
    t_us = benchmark_timer(lambda: mhc_head_compute_mix_fwd(input_mix_2d, mhc_scale, mhc_base, output_mix_2d, eps))

    benchmark_record(
        kernel='mhc_head_compute_mix',
        operation='head_compute_mix_forward',
        params={'n0': n0, 'n1': n1, 'mhc_mult': mhc_mult},
        time_us=t_us,
        bandwidth_gbs=io_gb / t_us * 1e6,
        extras={'io_gb': io_gb, 'num_tokens': num_tokens},
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('n0,n1,mhc_mult', _HEAD_COMPUTE_BENCH_CASES)
def test_head_compute_mix_bwd_benchmark(
    n0: int,
    n1: int,
    mhc_mult: int,
    benchmark_timer,
    benchmark_record,
) -> None:
    test_data = generate_head_compute_mix_test_data(n0=n0, n1=n1, mhc_mult=mhc_mult)
    input_mix = test_data['input_mix']
    mhc_scale = test_data['mhc_scale']
    mhc_base = test_data['mhc_base']
    out_grad = test_data['output_mix_grad']
    num_tokens = n0 * n1
    io_gb = count_bytes(input_mix) * 2 / 1e9  # read + write, same shape

    m = mhc_mult
    input_mix_grad = torch.empty_like(input_mix)
    num_sms = get_num_vec_cores() if is_ascend() else get_num_sms()
    mhc_scale_grad_partial = torch.empty(
        num_sms,
        *mhc_scale.shape,
        dtype=mhc_scale.dtype,
        device=mhc_scale.device,
    )
    mhc_base_grad_partial = torch.empty(
        num_sms,
        *mhc_base.shape,
        dtype=mhc_base.dtype,
        device=mhc_base.device,
    )
    out_grad_2d = out_grad.view(-1, m)
    input_mix_2d = input_mix.view(-1, m)
    input_mix_grad_2d = input_mix_grad.view(-1, m)

    t_us = benchmark_timer(
        lambda: mhc_head_compute_mix_bwd(
            out_grad_2d,
            input_mix_2d,
            mhc_scale,
            mhc_base,
            input_mix_grad_2d,
            mhc_scale_grad_partial,
            mhc_base_grad_partial,
            num_sms,
        )
    )

    benchmark_record(
        kernel='mhc_head_compute_mix',
        operation='head_compute_mix_backward',
        params={'n0': n0, 'n1': n1, 'mhc_mult': mhc_mult},
        time_us=t_us,
        bandwidth_gbs=io_gb / t_us * 1e6,
        extras={'io_gb': io_gb, 'num_tokens': num_tokens},
    )
