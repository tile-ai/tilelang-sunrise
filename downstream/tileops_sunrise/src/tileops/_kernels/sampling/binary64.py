"""Binary64 CDF arithmetic on integer words where native double loses precision.

Only the sampling hierarchy uses this adapter. Its float32 leaves and CUDA
expressions remain unchanged. Integer arithmetic rounds each operation to nearest,
ties to even; no native float64 instruction or 128-bit integer is required.
"""

import tilelang.language as T


class Binary64:
    """Emit the original CUDA expressions or integer-only PTPU equivalents."""

    def __init__(self, software: bool):
        self.software = software
        self.dtype = "int64" if software else "float64"

    def cast(self, value):
        if self.software:
            return T.call_pure_extern(
                "int64", "tileops::sampling_soft64::from_f32", T.cast(value, "float32")
            )
        return T.cast(value, "float64")

    def add(self, a, b):
        if self.software:
            return T.call_pure_extern("int64", "tileops::sampling_soft64::add", a, b)
        return a + b

    def sub(self, a, b):
        if self.software:
            return T.call_pure_extern("int64", "tileops::sampling_soft64::sub", a, b)
        return a - b

    def mul(self, a, b):
        if self.software:
            return T.call_pure_extern("int64", "tileops::sampling_soft64::mul", a, b)
        return a * b

    def gt(self, a, b):
        if self.software:
            return T.call_pure_extern("bool", "tileops::sampling_soft64::gt", a, b)
        return a > b

    def max_zero(self, value):
        if self.software:
            return T.call_pure_extern("int64", "tileops::sampling_soft64::max_zero", value)
        return T.max(value, T.cast(0, "float64"))
