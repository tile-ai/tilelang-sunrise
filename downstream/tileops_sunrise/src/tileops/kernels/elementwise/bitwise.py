"""Bitwise and/or/xor/not kernels."""

import tilelang.language as T
import torch

from tileops.utils import get_device

from ._base import (
    _BITWISE_DTYPES,
    BinaryKernel,
    UnaryKernel,
    _Uint8StorageBinaryKernel,
)

__all__ = [
    "BitwiseAndBoolStorageFwdKernel",
    "BitwiseAndFwdKernel",
    "BitwiseNotFwdKernel",
    "BitwiseOrBoolStorageFwdKernel",
    "BitwiseOrFwdKernel",
    "BitwiseXorBoolStorageFwdKernel",
    "BitwiseXorFwdKernel",
]


class BitwiseAndFwdKernel(BinaryKernel):
    """Element-wise bitwise AND: y = a & b (integer inputs)."""

    SUPPORTED_DTYPES = _BITWISE_DTYPES
    INT64_SPLIT = True

    @staticmethod
    def op_func(a, b):
        return a & b


class BitwiseAndBoolStorageFwdKernel(_Uint8StorageBinaryKernel):
    """Element-wise bitwise AND on uint8-backed bool storage."""

    @staticmethod
    def op_func(a, b):
        return T.bitwise_and(a, b)


class BitwiseOrFwdKernel(BinaryKernel):
    """Element-wise bitwise OR: y = a | b (integer inputs)."""

    SUPPORTED_DTYPES = _BITWISE_DTYPES
    INT64_SPLIT = True

    @staticmethod
    def op_func(a, b):
        return a | b


class BitwiseOrBoolStorageFwdKernel(_Uint8StorageBinaryKernel):
    """Element-wise bitwise OR on uint8-backed bool storage."""

    @staticmethod
    def op_func(a, b):
        return T.bitwise_or(a, b)


class BitwiseXorFwdKernel(BinaryKernel):
    """Element-wise bitwise XOR: y = a ^ b (integer inputs)."""

    SUPPORTED_DTYPES = _BITWISE_DTYPES
    INT64_SPLIT = True

    @staticmethod
    def op_func(a, b):
        return a ^ b


class BitwiseXorBoolStorageFwdKernel(_Uint8StorageBinaryKernel):
    """Element-wise bitwise XOR on uint8-backed bool storage."""

    @staticmethod
    def op_func(a, b):
        return T.bitwise_xor(a, b)


class BitwiseNotFwdKernel(UnaryKernel):
    """Element-wise bitwise NOT (~x) for bool/integer inputs.

    Uses ``-x - 1`` (two's-complement ``~x``) instead of ``T.bitwise_not``,
    which fails on vectorized ``int4`` CUDA types.

    Takes the base class's looping strategy: one element per thread reads four bytes
    where the dtype allows sixteen, which costs most of the read bandwidth. A bool
    input is still coerced to the scalar path, by the dtype rule in ``UnaryKernel``.
    """

    SUPPORTED_DTYPES = _BITWISE_DTYPES
    INT64_SPLIT = True

    def __init__(self, N_total, dtype, config=None, tune=False):
        # PTCC cannot construct the packed int16 XOR mask; retain scalar loads for it.
        if dtype == torch.int16 and get_device() == "ptpu" and "strategy" not in (config or {}):
            config = {**(config or {}), "strategy": "direct"}
        super().__init__(N_total, dtype, config, tune)

    @staticmethod
    def op_func(x):
        if x.dtype == "bool":
            return x == T.cast(0, "bool")
        if x.dtype == "uint8":
            return T.cast(255, "uint8") - x
        return -x - T.cast(1, x.dtype)
