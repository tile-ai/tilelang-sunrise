import math
import torch


def assert_equal(x: torch.Tensor, y: torch.Tensor,
                    check_dtype: bool = True,
                    check_shape: bool = True,
                    check_stride: bool = True) -> None:
    # Move to CPU for device-agnostic comparison (avoid unsupported ops on PTPU, etc.)
    x = x.cpu()
    y = y.cpu()
    assert not check_dtype or x.dtype == y.dtype, \
        f'Tensor dtypes are not equal: {x.dtype} vs {y.dtype}'
    assert not check_shape or x.shape == y.shape, \
        f'Tensor shapes are not equal: {x.shape} vs {y.shape}'
    assert not check_stride or x.numel() == 0 or x.stride() == y.stride(), \
        f'Tensor strides are not equal: {x.stride()} vs {y.stride()}'
    assert x.device == y.device, \
        f'Tensor devices are not equal: {x.device} vs {y.device}'
    # Hints: The tensor with a size of [32768, 1] and a stride of [1, 32768] is considered contiguous,
    # but using .view will cause an error. Therefore, .flatten is used to ensure the stride of the last dimension is 1.
    mask = x != y
    assert torch.equal(x.contiguous().flatten().view(torch.uint8), y.contiguous().flatten().view(torch.uint8)), \
        f'Tensor values are not equal: {x.shape=} vs {y.shape=}\n' \
        f'mask={torch.nonzero(mask)}\n' \
        f'{x[mask]}\nvs\n{y[mask]}' \


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    # Compute sum((x - y)^2) / sum(x^2 + y^2) directly. The mathematically
    # equivalent ``1 - 2 * sum(x*y) / sum(x^2 + y^2)`` form suffers catastrophic
    # cancellation when x ~= y: ``sim`` is within float32 eps of 1, so ``1 - sim``
    # collapses to 0 (CPU) or eps noise (PTPU) instead of the true ~1e-16 value.
    # Move to CPU first so the result is a plain float usable in
    # ``assert diff < tol`` without a device scalar cast (PTPU lacks it).
    x, y = x.float().cpu(), y.float().cpu()
    denominator = (x * x + y * y).sum()
    if denominator.item() == 0:
        return 0.0
    return float(((x - y) * (x - y)).sum() / denominator)


def check_bias(x: torch.Tensor, ref_x: torch.Tensor) -> None:
    count = x.numel()
    if count == 0:
        return
    less_count = (x < ref_x).sum()
    equal_count = (x == ref_x).sum()
    less_ratio = (less_count + equal_count / 2) / count

    # Suppose whether a number is larger or smaller after casting is an independent variable
    # and has exact possibility of 0.5 to be larger.
    # Then `less_count` follows a binomial distribution B(count, 0.5)
    # The standard deviation is sqrt(count * 0.5 * 0.5) = sqrt(count) / 2
    # Then `less_ratio` has a standard deviation of sqrt(count) / (2 * count) = 1 / (2 * sqrt(count))
    # When `count` is large enough, the central limit theorem applies, and we have:
    # less_ratio` ~ N(0.5, 1 / (4 * count))
    # So 99.99999% confidence interval should be something like this:
    # (-c / sqrt(count), c / sqrt(count)) around 0.5
    allowed_diff_ratio = 10 / math.sqrt(x.numel())
    assert abs(less_ratio - 0.5) < allowed_diff_ratio, \
        f'Less than ratio not close to 0.5 (size = {x.numel()}): {less_ratio=:.4f}\n' \
        f'Expected:\n  {ref_x.view(-1, 4)}\n' \
        f'Actual:\n   {x.view(-1, 4)}\n' \
        f'  Less than: {less_count}\n  Equal to:  {equal_count}\n  Greater than: {count - less_count - equal_count}\n'\


def count_bytes(*tensors) -> int:
    total = 0
    for t in tensors:
        if isinstance(t, (tuple, list)):
            total += count_bytes(*t)
        elif t is not None:
            total += t.numel() * t.element_size()
    return total
