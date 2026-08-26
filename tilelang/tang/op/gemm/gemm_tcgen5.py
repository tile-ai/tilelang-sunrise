"""STCUV2 TCGEN5 lowering for the TANG backend."""

from __future__ import annotations

from math import prod

from tilelang import language as T
from tilelang.layout import Layout
from tilelang.tang.op.gemm.gemm_tmma import _as_const_int, _make_tang_ab_layout
from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang.transform.simplify import _Simplify
from tvm import tirx
from tvm.ir import Range
from tvm.target import Target


GEMM_INST_TCGEN5 = "tang.tcgen5"


_ELE_TYPE_NAMES = {
    -1: "void",
    0: "int8_t",
    1: "uint8_t",
    2: "__fp16",
    3: "__bf16",
    4: "float",
    5: "tl::fp4_e2m1_t",
    6: "tl::fp4_e2m1_nv_t",
    8: "fp8_e4_t",
    9: "fp8_e5_t",
    10: "tl::fp6_e2m3_t",
    11: "tl::fp6_e3m2_t",
    12: "tl::fp4_e2m1_mix_t",
}


def _annotation_int(annotations, name: str, default: int) -> int:
    value = annotations.get(name, default)
    return _as_const_int(value, name)


def _dtype_ele_code(dtype, other_dtype, scale_format: int) -> int:
    name = str(dtype)
    other = str(other_dtype)
    if "float8_e4m3" in name:
        return 8
    if "float8_e5m2" in name:
        return 9
    if "float6_e2m3" in name:
        return 10
    if "float6_e3m2" in name:
        return 11
    if "float4_e2m1" in name:
        return 12 if "float4_e2m1" not in other else (6 if scale_format == 0 else 5)
    raise ValueError(f"Unsupported TANG scaled-GEMM operand dtype: {dtype}")


def _access_ptr(region, access: str):
    extents = [_as_const_int(r.extent, "region extent") for r in region.region[-2:]]
    return T.access_ptr(region, access, extent=prod(extents), ignore_last_ndim=2)


class GemmTangTCGEN5(GemmBase):
    """Lower STCUV2 TCGEN5 template calls."""

    @property
    def allow_f8f6f4_mixed_dtypes(self) -> bool:
        return True

    def _warp_partition(self, target: Target, thread_nums: int) -> tuple[int, int]:
        return self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, GEMM_INST_TCGEN5)

    def infer_layout(self, target: Target, thread_nums: int):
        layouts = {
            self.C: Layout([self.M, self.N], lambda i, j: [i, j]),
        }
        if self.A.scope() in ("shared", "shared.dyn"):
            layouts[self.A] = _make_tang_ab_layout(
                _as_const_int(self.A.shape[-2], "A row extent"),
                _as_const_int(self.A.shape[-1], "A column extent"),
                self.A.dtype.bits,
                _as_const_int(self.offset_A, "A offset"),
                True,
                self.trans_A,
            )
        if self.B.scope() in ("shared", "shared.dyn"):
            layouts[self.B] = _make_tang_ab_layout(
                _as_const_int(self.B.shape[-2], "B row extent"),
                _as_const_int(self.B.shape[-1], "B column extent"),
                self.B.dtype.bits,
                _as_const_int(self.offset_B, "B offset"),
                False,
                self.trans_B,
            )
        return layouts

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_var: tirx.Var,
        mbar_phase_expr: tirx.PrimExpr | None = None,
    ):
        del layout_map, thread_var, mbar_phase_expr
        thread_nums = _as_const_int(thread_bounds.extent, "thread extent")
        warp_m, warp_n = self._warp_partition(target, thread_nums)
        annotations = self.annotations
        k_step = _annotation_int(annotations, "tang_k_step", 1)
        a_format = _annotation_int(annotations, "tang_a_format", -1)
        b_format = _annotation_int(annotations, "tang_b_format", -1)
        offset_a = _as_const_int(self.offset_A, "A offset")
        offset_b = _as_const_int(self.offset_B, "B offset")

        a_ptr = _access_ptr(self.ARegion, "r")
        b_ptr = _access_ptr(self.BRegion, "r")
        c_ptr = _access_ptr(self.CRegion, "rw")

        if _annotation_int(annotations, "tang_legacy_blockscaled", 0):
            if self.SFARegion is None or self.SFBRegion is None or self.SFTmemRegion is None:
                raise ValueError("TANG block-scaled GEMM requires scale_a, scale_b and scale_tmem regions")
            scale_vec = _annotation_int(annotations, "tang_scale_vec", 1)
            scale_format = _annotation_int(annotations, "tang_scale_format", 1)
            scale_block = _annotation_int(annotations, "tang_scale_block", 32)
            if a_format < 0:
                a_format = _dtype_ele_code(self.A.dtype, self.B.dtype, scale_format)
            if b_format < 0:
                b_format = _dtype_ele_code(self.B.dtype, self.A.dtype, scale_format)
            a_stride_bytes = self.stride_A * self.A.dtype.bits // 8
            b_stride_bytes = self.stride_B * self.B.dtype.bits // 8
            stride_sfa = _as_const_int(self.SFARegion.buffer.shape[-1], "scale_a stride")
            stride_sfb = _as_const_int(self.SFBRegion.buffer.shape[-1], "scale_b stride")
            template = (
                f"tl::gemm_tang_tcgen5_scale<{self.M}, {self.N}, {self.K}, {warp_m}, {warp_n}, "
                f"{a_stride_bytes}, {b_stride_bytes}, {offset_a}, {offset_b}, "
                f"{int(self.trans_A)}, {int(self.trans_B)}, {self.k_pack}, {k_step}, "
                f"{int(self.trans_A)}, {int(not self.trans_B)}, {scale_vec}, {scale_format}, "
                f"{scale_block}, {stride_sfa}, {stride_sfb}, {_ELE_TYPE_NAMES[a_format]}, "
                f"{_ELE_TYPE_NAMES[b_format]}>"
            )
            sfa_ptr = T.access_ptr(self.SFARegion, "r")
            sfb_ptr = T.access_ptr(self.SFBRegion, "r")
            sft_ptr = _access_ptr(self.SFTmemRegion, "rw")
            call = tirx.call_intrin(
                "handle",
                tirx.op.Op.get("tl.tl_tang_gemm"),
                tirx.StringImm(template),
                a_ptr,
                b_ptr,
                sfa_ptr,
                sfb_ptr,
                c_ptr,
                sft_ptr,
                self.clear_accum,
            )
        else:
            formats = ""
            if a_format >= 0 or b_format >= 0:
                formats = f", {_ELE_TYPE_NAMES[a_format]}, {_ELE_TYPE_NAMES[b_format]}"
            template = (
                f"tl::gemm_tang_tcgen5<{self.M}, {self.N}, {self.K}, {warp_m}, {warp_n}, "
                f"{self.stride_A}, {self.stride_B}, {offset_a}, {offset_b}, "
                f"{int(self.trans_A)}, {int(self.trans_B)}, {self.k_pack}, {k_step}, "
                f"{int(self.trans_A)}, {int(not self.trans_B)}{formats}>"
            )
            call = tirx.call_intrin(
                "handle",
                tirx.op.Op.get("tl.tl_tang_gemm"),
                tirx.StringImm(template),
                a_ptr,
                b_ptr,
                c_ptr,
                self.clear_accum,
            )

        @T.prim_func
        def _gemm_tang_tcgen5() -> None:
            T.evaluate(call)

        return _Simplify(_gemm_tang_tcgen5, inline_let=True)


class GemmTangWGMMA(GemmBase):
    """Reject the unvalidated STCUV2 WGMMA dispatch path."""

    def infer_layout(self, target: Target, thread_nums: int):
        del target, thread_nums
        return {}

    def lower(self, layout_map, target, thread_bounds, thread_var, mbar_phase_expr=None):
        del layout_map, target, thread_bounds, thread_var, mbar_phase_expr
        raise ValueError("TANG stcuv2 WGMMA lowering remains unverified and is disabled.")
