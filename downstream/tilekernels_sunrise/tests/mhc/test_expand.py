from typing import Callable

import pytest
import torch
from tile_kernels.modeling.mhc.ops import expand_to_mhc
from tile_kernels.torch.mhc import expand_to_mhc_ref
from tile_kernels.testing.numeric import count_bytes


def generate_expand_test_data(
    n0: int, n1: int, mhc_mult: int, h: int, device: str = 'ptpu'
) -> dict[str, torch.Tensor]:
    torch.random.manual_seed(42)

    x = torch.randn(n0, n1, h, dtype=torch.bfloat16).to(device)
    o_grad = torch.randn(n0, n1, mhc_mult, h, dtype=torch.bfloat16).to(device)

    return {'x': x, 'o_grad': o_grad, 'mhc_mult': mhc_mult}


@pytest.mark.parametrize('n0', [1, 2])
@pytest.mark.parametrize('n1', [1024, 4096])
@pytest.mark.parametrize('mhc_mult', [2, 4, 8])
@pytest.mark.parametrize('h', [1280, 2560, 7168])
def test_expand_comprehensive(n0: int, n1: int, mhc_mult: int, h: int) -> None:
    test_data = generate_expand_test_data(n0=n0, n1=n1, mhc_mult=mhc_mult, h=h)

    with torch.no_grad():
        out_tl = expand_to_mhc(test_data['x'], test_data['mhc_mult'])
        out_ref = expand_to_mhc_ref(test_data['x'], test_data['mhc_mult'])
        torch.ptpu.synchronize()
        out_tl = out_tl.cpu()

    torch.testing.assert_close(out_tl, out_ref)

    # NOTE backward skip temporarily
    # def _tester(
    #     impl: Callable[[torch.Tensor, int], torch.Tensor],
    # ) -> tuple[torch.Tensor, torch.Tensor | None]:
    #     x_ = test_data['x'].clone().requires_grad_()
    #     o_ = impl(x_, test_data['mhc_mult'])
    #     torch.autograd.backward([o_], [test_data['o_grad']])
    #     return o_, x_.grad

    # o_tl, x_grad_tl = _tester(expand_to_mhc)
    # o_ref, x_grad_ref = _tester(expand_to_mhc_ref)

    # torch.testing.assert_close(o_tl, o_ref)
    # torch.testing.assert_close(x_grad_tl, x_grad_ref)

@pytest.mark.parametrize('n0', [1, 2])
@pytest.mark.parametrize('n1', [1024, 4096])
@pytest.mark.parametrize('mhc_mult', [2, 4, 8])
@pytest.mark.parametrize('h', [1280, 2560, 7168])
@pytest.mark.benchmark
def test_expand_comprehensive_benchmark(benchmark_timer, benchmark_record, n0: int, n1: int, mhc_mult: int, h: int) -> None:
    test_data = generate_expand_test_data(n0=n0, n1=n1, mhc_mult=mhc_mult, h=h)

    with torch.no_grad():
        out_tl = expand_to_mhc(test_data['x'], test_data['mhc_mult'])
        out_tl = out_tl.cpu()

    x_tl = test_data['x'].clone()
    mhc_mult_tl = test_data['mhc_mult']
    output_tl = expand_to_mhc(x_tl, mhc_mult_tl)

    t_save_us = benchmark_timer(lambda: expand_to_mhc(x_tl, mhc_mult_tl))
    num_bytes_save = count_bytes(x_tl, output_tl)

    benchmark_record(
        kernel='expand_to_mhc',
        operation='fwd',
        params={'x': x_tl.cpu() if x_tl is not None else x_tl,
                'mhc_mult': mhc_mult_tl},
        time_us=t_save_us,
        bandwidth_gbs=num_bytes_save / t_save_us / 1e3,
    )
