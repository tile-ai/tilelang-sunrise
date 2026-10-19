"""TANG descriptor geometry derived from the common GEMM operand regions."""

from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang.utils.language import retrieve_offset, retrieve_stride


class TangGemmBase(GemmBase):
    """Keep row strides and final-axis origins local to TANG lowering.

    Leading region indices, including a pipeline stage, stay in access_ptr;
    the TANG templates consume only the final-axis origin separately.
    """

    @property
    def stride_A(self):
        return retrieve_stride(self.ARegion)[-2]

    @property
    def stride_B(self):
        return retrieve_stride(self.BRegion)[-2]

    @property
    def offset_A(self):
        return retrieve_offset(self.ARegion)[-1]

    @property
    def offset_B(self):
        return retrieve_offset(self.BRegion)[-1]
