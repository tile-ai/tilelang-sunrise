from typing import Callable

import pytest
import torch
from tile_kernels.modeling.mhc.ops import sinkhorn_normalize
from tile_kernels.torch.mhc import sinkhorn_normalize_ref
from tile_kernels.testing.numeric import count_bytes


def generate_sinkhorn_test_data(
    n0: int, n1: int, mhc: int, device: str = 'ptpu'
) -> dict[str, torch.Tensor]:
    comb_res_mix = torch.randn((n0, n1, mhc, mhc), dtype=torch.float32).to(device=device)
    out_grad = torch.randn((n0, n1, mhc, mhc), dtype=torch.float32).to(device=device)

    return {
        'comb_res_mix': comb_res_mix,
        'out_grad': out_grad,
        'repeat': 10,
        'eps': 1e-6,
    }


def _tester(
    impl: Callable[[torch.Tensor, int, float], torch.Tensor],
    test_data: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    comb_res_mix_ = test_data['comb_res_mix'].clone().requires_grad_()
    out_ = impl(comb_res_mix_, test_data['repeat'], test_data['eps'])
    return out_
    # torch.autograd.backward([out_], [test_data['out_grad']])
    # return out_, comb_res_mix_.grad


@pytest.mark.parametrize('n0', [1, 2])
@pytest.mark.parametrize('n1', [1, 1024, 4096])
@pytest.mark.parametrize('mhc', [4])
def test_sinkhorn_comprehensive(n0: int, n1: int, mhc: int) -> None:
    test_data = generate_sinkhorn_test_data(n0=n0, n1=n1, mhc=mhc)

    # out_tl, grad_tl = _tester(sinkhorn_normalize, test_data)
    # out_ref, grad_ref = _tester(sinkhorn_normalize_ref, test_data)

    out_tl = _tester(sinkhorn_normalize, test_data)
    out_ref = _tester(sinkhorn_normalize_ref, test_data)

    out_tl = out_tl.cpu()
    out_ref = out_ref.cpu()

    torch.testing.assert_close(out_tl, out_ref)
    # torch.testing.assert_close(grad_tl, grad_ref)

@pytest.mark.parametrize('n0', [1, 2])
@pytest.mark.parametrize('n1', [1, 1024, 4096])
@pytest.mark.parametrize('mhc', [4])
@pytest.mark.benchmark
def test_sinkhorn_comprehensive_benchmark(benchmark_timer, benchmark_record,
                                          n0: int, n1: int, mhc: int) -> None:
    test_data = generate_sinkhorn_test_data(n0=n0, n1=n1, mhc=mhc)

    comb_res_mix_tl = test_data['comb_res_mix'].clone()
    out_tl = sinkhorn_normalize(comb_res_mix_tl, test_data['repeat'], test_data['eps'])

    t_save_us = benchmark_timer(lambda: sinkhorn_normalize(comb_res_mix_tl, test_data['repeat'], test_data['eps']))
    num_bytes_save = count_bytes(comb_res_mix_tl)

    benchmark_record(
        kernel='sinkhorn_normalize',
        operation='fwd',
        params={'comb_res_mix': comb_res_mix_tl.cpu() if comb_res_mix_tl is not None else comb_res_mix_tl,
                'repeat': test_data['repeat'],
                'eps': test_data['eps']},
        time_us=t_save_us,
        bandwidth_gbs=num_bytes_save / t_save_us / 1e3,
    )
