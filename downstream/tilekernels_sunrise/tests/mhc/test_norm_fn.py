import pytest
import torch
from tile_kernels.modeling.mhc.ops import mhc_pre_norm_fn
from tile_kernels.torch.mhc import mhc_pre_norm_fn_ref
from tile_kernels.testing.numeric import count_bytes


def generate_norm_fn_test_data(
    n1: int,
    mhc_mult: int,
    hidden_size: int,
    generate_normw: bool,
) -> dict[str, torch.Tensor]:
    n0 = 1
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    mhc_hidden_size = mhc_mult * hidden_size
    device = 'ptpu'

    residual = (
        torch.randn((n0, n1, mhc_mult, hidden_size), dtype=torch.float).to(device=device)
        .mul(1 + torch.arange(mhc_mult, device=device).mul(0.01).view(1, 1, -1, 1))
        .bfloat16()
    )

    fn = (
        torch.randn((mhc_mult3, mhc_mult, hidden_size), dtype=torch.float).to(device=device)
        * 1e-4
        * (1 + torch.arange(mhc_mult, device=device).mul(0.01).view(1, -1, 1))
    ).flatten(1, 2)

    if generate_normw:
        normw = torch.randn((mhc_hidden_size,), dtype=torch.float).to(device=device) * 0.1 + 1.0
    else:
        normw = None

    out_grad = torch.randn((n0, n1, mhc_mult3), dtype=torch.float).to(device=device)

    return {
        'residual': residual,
        'fn': fn,
        'normw': normw,
        'out_grad': out_grad,
        'mhc_norm_eps': 1e-6,
    }


@pytest.mark.parametrize('n1', [4096, 8192])
@pytest.mark.parametrize('hidden_size', [1280, 2560, 7168])
@pytest.mark.parametrize('generate_normw', [False, True])
def test_correctness(
    n1: int,
    hidden_size: int,
    generate_normw: bool,
) -> None:
    mhc_mult = 4

    test_data = generate_norm_fn_test_data(
        n1=n1,
        mhc_mult=mhc_mult,
        hidden_size=hidden_size,
        generate_normw=generate_normw,
    )

    residual_tl = test_data['residual'].clone().requires_grad_()
    fn_tl = test_data['fn'].clone().requires_grad_()
    normw_tl = test_data['normw'].clone().requires_grad_() if test_data['normw'] is not None else None

    residual_ref = test_data['residual'].clone().requires_grad_()
    fn_ref = test_data['fn'].clone().requires_grad_()
    normw_ref = test_data['normw'].clone().requires_grad_() if test_data['normw'] is not None else None

    torch.ptpu.synchronize()
    out_tl = mhc_pre_norm_fn(
        residual_tl,
        fn_tl,
        normw_tl,
        test_data['mhc_norm_eps'],
    )
    torch.ptpu.synchronize()

    # residual_tl_grad = residual_tl.untyped_storage().grad_from_mhc_post = torch.zeros_like(residual_tl)
    # torch.autograd.backward([out_tl], [test_data['out_grad']])

    torch.backends.cuda.matmul.allow_tf32 = True
    out_ref = mhc_pre_norm_fn_ref(
        residual_ref,
        fn_ref,
        normw_ref,
        test_data['mhc_norm_eps'],
    )

    # torch.autograd.backward([out_ref], [test_data['out_grad']])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.ptpu.synchronize()
    out_tl = out_tl.cpu()

    torch.testing.assert_close(out_tl, out_ref, atol=1e-3, rtol=1e-3)
    # torch.testing.assert_close(residual_tl_grad, residual_ref.grad)
    # torch.testing.assert_close(fn_tl.grad, fn_ref.grad, atol=3e-2, rtol=1e-3)


@pytest.mark.parametrize('n1', [13, 48, 512])
@pytest.mark.parametrize('hidden_size', [1280, 2560, 4096, 7168])
def test_split_k_correctness(n1: int, hidden_size: int) -> None:
    mhc_mult = 4

    test_data = generate_norm_fn_test_data(
        n1=n1,
        mhc_mult=mhc_mult,
        hidden_size=hidden_size,
        generate_normw=False,
    )

    residual_tl = test_data['residual'].clone().requires_grad_()
    fn_tl = test_data['fn'].clone().requires_grad_()

    residual_ref = test_data['residual'].clone().requires_grad_()
    fn_ref = test_data['fn'].clone().requires_grad_()

    torch.ptpu.synchronize()
    out_tl = mhc_pre_norm_fn(
        residual_tl,
        fn_tl,
        None,
        test_data['mhc_norm_eps'],
        n_splits=16,
    )
    torch.ptpu.synchronize()

    torch.backends.cuda.matmul.allow_tf32 = True
    out_ref = mhc_pre_norm_fn_ref(
        residual_ref,
        fn_ref,
        None,
        test_data['mhc_norm_eps'],
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    out_tl = out_tl.cpu()

    torch.testing.assert_close(out_tl, out_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize('n1', [4096, 8192])
@pytest.mark.parametrize('hidden_size', [1280, 2560, 7168])
@pytest.mark.parametrize('generate_normw', [False, True])
@pytest.mark.benchmark
def test_correctness_benchmark(benchmark_timer, benchmark_record, n1:int, hidden_size:int, generate_normw:int)->None:
    mhc_mult = 4
    test_data = generate_norm_fn_test_data(
        n1=n1,
        mhc_mult=mhc_mult,
        hidden_size=hidden_size,
        generate_normw=generate_normw,
    )

    residual_tl = test_data['residual'].clone()
    fn_tl = test_data['fn'].clone()
    normw_tl = test_data['normw'].clone() if test_data['normw'] is not None else None
    out_tl = mhc_pre_norm_fn(
        residual_tl,
        fn_tl,
        normw_tl,
        test_data['mhc_norm_eps'],
    )
    t_save_us = benchmark_timer(lambda: mhc_pre_norm_fn(residual_tl, fn_tl, normw_tl, test_data['mhc_norm_eps']))
    num_bytes_save = count_bytes(residual_tl, fn_tl, normw_tl, out_tl)

    benchmark_record(
        kernel='mhc_pre_norm_fn',
        operation='fwd',
        params={'residual_tl': residual_tl.cpu() if residual_tl is not None else residual_tl,
                'fn_tl': fn_tl.cpu() if fn_tl is not None else fn_tl,
                'normw_tl': normw_tl.cpu() if normw_tl is not None else normw_tl,
                'mhc_norm_eps': test_data['mhc_norm_eps']},
        time_us=t_save_us,
        bandwidth_gbs=num_bytes_save / t_save_us / 1e3,
    )


@pytest.mark.parametrize('n1', [13, 48, 512])
@pytest.mark.parametrize('hidden_size', [1280, 2560, 4096, 7168])
@pytest.mark.benchmark
def test_split_k_correctness_benchmark(benchmark_timer, benchmark_record, n1:int, hidden_size:int):
    mhc_mult = 4
    test_data = generate_norm_fn_test_data(
        n1=n1,
        mhc_mult=mhc_mult,
        hidden_size=hidden_size,
        generate_normw=False,
    )

    residual_tl = test_data['residual'].clone()
    fn_tl = test_data['fn'].clone()

    torch.ptpu.synchronize()
    out_tl = mhc_pre_norm_fn(
        residual_tl,
        fn_tl,
        None,
        test_data['mhc_norm_eps'],
        n_splits=16,
    )
    torch.ptpu.synchronize()

    t_save_us = benchmark_timer(lambda: mhc_pre_norm_fn(residual_tl, fn_tl, None, test_data['mhc_norm_eps'], n_splits=16))
    num_bytes_save = count_bytes(residual_tl, fn_tl, out_tl)

    benchmark_record(
        kernel='mhc_pre_norm_fn_split_k',
        operation='fwd',
        params={'residual_tl': residual_tl.cpu() if residual_tl is not None else residual_tl,
                'fn_tl': fn_tl.cpu() if fn_tl is not None else fn_tl,
                'normw_tl': None,
                'mhc_norm_eps': test_data['mhc_norm_eps'],
                'n_splits': 16},
        time_us=t_save_us,
        bandwidth_gbs=num_bytes_save / t_save_us / 1e3,
    )
