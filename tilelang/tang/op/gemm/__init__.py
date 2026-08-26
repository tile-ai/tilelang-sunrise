"""TANG GEMM registrations."""

from tilelang.tang.target import target_is_stcu, target_is_stcuv2
from tilelang.tileop.gemm.registry import register_gemm_impl

from .gemm_tmma import GEMM_INST_TMMA, GemmTMMA
from .gemm_tcgen5 import GEMM_INST_TCGEN5, GemmTangTCGEN5, GemmTangWGMMA


register_gemm_impl("tang.tmma", GEMM_INST_TMMA, target_is_stcu, GemmTMMA)
register_gemm_impl("tang.tcgen5", GEMM_INST_TCGEN5, target_is_stcuv2, GemmTangTCGEN5)
register_gemm_impl("tang.wgmma", "tang.wgmma", target_is_stcuv2, GemmTangWGMMA)
