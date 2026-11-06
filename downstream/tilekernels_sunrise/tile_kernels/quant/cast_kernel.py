import torch
from tilelang import language as T

from tile_kernels.quant.cast_cuda import get_cast_kernel_cuda
from tile_kernels.config import get_num_vec_cores, is_ascend


def cast(
    x: torch.Tensor,
    dtype: torch.dtype,
    stochastic_rounding: bool = False,
) -> torch.Tensor:
    """Cast a contiguous tensor to ``dtype``, using or not using stochastic rounding.

    Args:
        x: Input tensor.
        dtype: Target type to cast.
        stochastic_rounding: Use stochastic rounding if value is ``True``. Should be ``False`` if casting from small types to larger ones.

    Returns:
        Casted tensor with the same shape and device as the input.
    """
    assert x.is_contiguous()

    x_dtype = T.dtype(x.dtype)
    y_dtype = T.dtype(dtype)
    n = x.numel()

    if stochastic_rounding:
        assert (x_dtype, y_dtype) in [(T.float32, T.bfloat16), (T.float32, T.float8_e4m3fn), (T.bfloat16, T.float8_e4m3fn)], (
            f'stochastic rounding has not support casting from {x_dtype} to {y_dtype} yet'
        )
        assert n % 4 == 0, f'stochastic rounding requires x.numel to be multiples of 4, but given {n}'

    y = torch.empty_like(x, dtype=dtype)

    if n == 0:
        return y

    if is_ascend():
        from tile_kernels.quant.cast_asc import get_cast_kernel_asc

        assert x_dtype == T.float32, 'Ascend only supports fp32 as input yet'
        assert y_dtype == T.bfloat16, 'Ascend only supports bf16 as output yet'
        assert stochastic_rounding, 'Ascend only supports stochastic rounding yet'
        # other dtypes or not-stochastic-rounding not implemented yet

        for n_mod in [256, 128, 4]:
            if n % n_mod == 0:
                kernel = get_cast_kernel_asc(x_dtype, y_dtype, stochastic_rounding, n_mod, get_num_vec_cores())
                break
    else:
        for n_mod in [8 * 256, 256, 128, 32, 4, 1]:
            if n % n_mod == 0:
                kernel = get_cast_kernel_cuda(x_dtype, y_dtype, stochastic_rounding, n_mod)
                break

    kernel(x.view(-1), y.view(-1))
    return y
