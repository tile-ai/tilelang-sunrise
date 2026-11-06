"""Bitwise and/or/xor/not kernels."""

import tilelang.language as T
import torch

from tileops._kernels.elementwise._base import (
    BinaryKernel,
    Uint8StorageBinaryKernel,
    UnaryKernel,
)
from tileops._kernels.elementwise._dtype import BITWISE_DTYPES
from tileops._kernels.elementwise.call_spec import (
    BinaryElementwiseFwdInterface,
    UnaryElementwiseFwdInterface,
)
from tileops._utils import get_device

__all__ = [
    "BitwiseAndBoolStorageFwdKernel",
    "BitwiseAndFwdKernel",
    "BitwiseNotFwdKernel",
    "BitwiseOrBoolStorageFwdKernel",
    "BitwiseOrFwdKernel",
    "BitwiseXorBoolStorageFwdKernel",
    "BitwiseXorFwdKernel",
]


class BitwiseAndFwdKernel(BinaryKernel, BinaryElementwiseFwdInterface):
    """Element-wise bitwise AND: y = a & b (integer inputs)."""

    SUPPORTED_DTYPES = BITWISE_DTYPES
    INT64_SPLIT = True

    @staticmethod
    def op_func(a, b):
        return a & b


class BitwiseAndBoolStorageFwdKernel(Uint8StorageBinaryKernel, BinaryElementwiseFwdInterface):
    """Element-wise bitwise AND on uint8-backed bool storage."""

    preferred_over = frozenset({"bitwise_and"})

    @staticmethod
    def op_func(a, b):
        return T.bitwise_and(a, b)


class BitwiseOrFwdKernel(BinaryKernel, BinaryElementwiseFwdInterface):
    """Element-wise bitwise OR: y = a | b (integer inputs)."""

    SUPPORTED_DTYPES = BITWISE_DTYPES
    INT64_SPLIT = True

    @staticmethod
    def op_func(a, b):
        return a | b


class BitwiseOrBoolStorageFwdKernel(Uint8StorageBinaryKernel, BinaryElementwiseFwdInterface):
    """Element-wise bitwise OR on uint8-backed bool storage."""

    preferred_over = frozenset({"bitwise_or"})

    @staticmethod
    def op_func(a, b):
        return T.bitwise_or(a, b)


class BitwiseXorFwdKernel(BinaryKernel, BinaryElementwiseFwdInterface):
    """Element-wise bitwise XOR: y = a ^ b (integer inputs)."""

    SUPPORTED_DTYPES = BITWISE_DTYPES
    INT64_SPLIT = True

    @staticmethod
    def op_func(a, b):
        return a ^ b


class BitwiseXorBoolStorageFwdKernel(Uint8StorageBinaryKernel, BinaryElementwiseFwdInterface):
    """Element-wise bitwise XOR on uint8-backed bool storage."""

    preferred_over = frozenset({"bitwise_xor"})

    @staticmethod
    def op_func(a, b):
        return T.bitwise_xor(a, b)


class BitwiseNotFwdKernel(UnaryKernel, UnaryElementwiseFwdInterface):
    """Element-wise bitwise NOT (~x) for bool/integer inputs.

    Uses ``-x - 1`` (two's-complement ``~x``) instead of ``T.bitwise_not``,
    which fails on vectorized ``int4`` CUDA types.

    Takes the base class's looping strategy: one element per thread reads four bytes
    where the dtype allows sixteen, which costs most of the read bandwidth. A bool
    input is still coerced to the scalar path, by the dtype rule in ``UnaryKernel``.
    """

    SUPPORTED_DTYPES = BITWISE_DTYPES
    INT64_SPLIT = True

    def __init__(self, N_total, dtype, config=None):
        if dtype == torch.int16 and get_device() == "ptpu":
            self.DEFAULT_STRATEGY = "direct"
        super().__init__(N_total, dtype, config)

    @staticmethod
    def op_func(x):
        if x.dtype == "bool":
            return x == T.cast(0, "bool")
        if x.dtype == "uint8":
            return T.cast(255, "uint8") - x
        return -x - T.cast(1, x.dtype)
