"""STCUV2 TCGEN5 lowering for the TANG backend."""

from __future__ import annotations

from math import prod

from tilelang import language as T
from tilelang.layout import Layout
from tilelang.tang.op.gemm.gemm_tmma import _as_const_int, _make_tang_ab_layout
from tilelang.tang.op.gemm.gemm_tcgen05_descriptors import is_any_fp4, is_unpacked_fp4
from tilelang.tileop.gemm.gemm_base import GemmBase
from tvm import tirx
from tvm.ir import Range
from tvm.target import Target


GEMM_INST_TCGEN5 = "tang.tcgen5"


# Scale factors a ScaleVecType makes the tensorcore consume per operand row.
_SCALE_VEC_SF_PER_ROW = {1: 4, 2: 8, 3: 16}  # X1 / X2 / X4

# Bytes of one operand row a single MMA instruction can read.
_MMA_OPERAND_ROW_BYTES = 128


def _check_operand_row_bytes(name: str, K: int, bits: int) -> None:
    """Validate the per-instruction operand row size.

    The limit applies to bytes read, not the number of elements. Split the
    reduction with T.Pipelined when an operand row exceeds the limit.
    """
    row_bytes = (K * bits + 7) // 8
    if row_bytes <= _MMA_OPERAND_ROW_BYTES:
        return
    max_k = _MMA_OPERAND_ROW_BYTES * 8 // bits
    raise ValueError(
        f"TANG tcgen5 GEMM: operand {name} needs a {row_bytes}-byte row per MMA "
        f"(K={K} x {bits}-bit), over the {_MMA_OPERAND_ROW_BYTES}-byte cap. Use "
        f"K <= {max_k} for this dtype, tiling the reduction with T.Pipelined if "
        f"it is longer."
    )


def _annotation_int(annotations, name: str, default: int) -> int:
    value = annotations.get(name, default)
    return _as_const_int(value, name)


def _check_scale_vec(scale_vec: int, K: int, scale_block: int) -> None:
    """Validate scale-vector shape against the staged scale count.

    The scale descriptor must agree with the number of staged scale factors
    to avoid reading beyond the staged region or leaving factors unused.
    """
    if scale_vec not in _SCALE_VEC_SF_PER_ROW:
        raise ValueError(f"TANG scaled GEMM: scale_vec must be X1/X2/X4 (ScaleVecType 1/2/3), got {scale_vec}")
    if scale_block <= 0 or K % scale_block != 0:
        raise ValueError(f"TANG scaled GEMM: K ({K}) must be a positive multiple of scale_block ({scale_block})")
    want = _SCALE_VEC_SF_PER_ROW[scale_vec]
    have = K // scale_block
    if have != want:
        raise ValueError(
            f"TANG scaled GEMM: scale_vec X{want} expects {want} scale factors per row, but "
            f"K={K} with scale_block={scale_block} gives {have}. Use a scale_vec whose width "
            f"matches K/scale_block (X1->4, X2->8, X4->16), or retile K."
        )


def _dtype_ele_code(dtype, other_dtype, scale_format: int) -> int:
    """Map a block-scaled operand dtype to its tang::ptx::EleType code.

    For fp4 the code depends on the MMA kind, and the kind is implied by whether
    the *other* operand is fp4: fp4 x fp4 is mxf4 / mxf4nvf4, which reads a
    2-per-byte packed operand (eFP4_E2M1 / eNVFP4_E2M1, chosen by scale_format);
    fp4 mixed with fp8 or fp6 is mxf8f6f4, which reads the unpacked layout of
    one code per byte (eFP4_E2M1_MIX).

    Those are two different physical layouts, so the dtype has to agree with the
    kind. Hardware does not validate it: feeding a packed operand to the mixed
    kind (or vice versa) reads the row at the wrong density and returns a
    plausible-looking wrong result, so disagreement is rejected here.
    """
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
    # Unpacked first: its dtype string contains "float4_e2m1" as a substring.
    if is_unpacked_fp4(name):
        if is_any_fp4(other):
            raise ValueError(
                f"TANG scaled GEMM: {dtype} against {other_dtype} is fp4 x fp4, which is the "
                "mxf4 / mxf4nvf4 kind and reads a 2-per-byte packed operand. float4_e2m1_unpacked "
                "is the mxf8f6f4 layout (one code per byte), only valid when the other operand is "
                "fp8 or fp6. Use float4_e2m1fn for both operands."
            )
        return 12
    if is_any_fp4(name):
        if not is_any_fp4(other):
            raise ValueError(
                f"TANG scaled GEMM: {dtype} against {other_dtype} is the mxf8f6f4 kind, which "
                "reads fp4 unpacked -- one e2m1 code in the low nibble of each byte. "
                f"{dtype} is packed 2-per-byte, so the tensorcore would read the row at twice "
                "the density and return a wrong result. Use T.float4_e2m1_unpacked for the fp4 "
                "operand when it is mixed with fp8/fp6."
            )
        return 6 if scale_format == 0 else 5
    raise ValueError(f"Unsupported TANG scaled-GEMM operand dtype: {dtype}")


def _access_ptr(region, access: str):
    extents = [_as_const_int(r.extent, "region extent") for r in region.region[-2:]]
    return T.access_ptr(region, access, extent=prod(extents), ignore_last_ndim=2)


def _completion_mbarrier(mbar):
    """Normalize the ``mbar=`` argument to a barrier expression or None.

    On Blackwell, ``mbar=`` makes the MMA publish its own completion into the
    barrier so a consumer can wait on the parity. TANG expresses the same thing
    with a fence that arrives: the issuing warp binds its tensor-core fence
    group to the barrier and hardware posts the arrive when the group's MMA
    work retires (``fence_tc_arrive_mbarrier``).
    """
    if mbar is None:
        return None
    # The frontend substitutes a const-0 placeholder when the caller omits mbar.
    if isinstance(mbar, (tirx.IntImm, tirx.FloatImm)) and int(mbar.value) == 0:
        return None
    return mbar


class GemmTangTCGEN5(GemmBase):
    """Lower STCUV2 TCGEN5 template calls."""

    @property
    def allow_f8f6f4_mixed_dtypes(self) -> bool:
        return True

    def _warp_partition(self, target: Target, thread_nums: int) -> tuple[int, int]:
        return self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, GEMM_INST_TCGEN5)

    def _mn_major_partition(self, warp_m: int, warp_n: int) -> tuple[int, int]:
        """Reconcile the warp partition with the 512-B descriptor alignment.

        The MMA requires each warp's shared-memory descriptor base to be 512-B
        aligned, so an MN-major operand's *contiguous* dim is never warp-split.
        The other, K-major operand's free dim can still split (its per-warp step
        is a whole number of shared rows = LBO bytes, a multiple of 512 for the
        supported shapes):

          NN/rr  (b_major only)       : A K-major -> split M (warp_n forced to 1)
          TT/cc  (a_major only)       : B K-major -> split N (warp_m forced to 1)
          NT/cr  (a_major && b_major) : both MN-major -> single warp
          TN/rc  (neither)            : K-major on both -> keep policy partition

        A 512-B guard falls back to single warp when the collapsed per-warp byte
        step is not 512-aligned or the free dim is not divisible by the warps.
        The device template handles the matching per-warp sub-tile base
        (gemm_tcgen05.h).
        """
        a_major = bool(self.trans_A)
        b_major = not bool(self.trans_B)
        total = warp_m * warp_n
        if a_major and b_major:  # NT/cr: neither dim splits with 512-B alignment
            return 1, 1
        if b_major:  # NN/rr: split M through the K-major A operand
            step = (self.M // total) * self.stride_A * (self.A.dtype.bits // 8)
            if total > 1 and self.M % total == 0 and step % 512 == 0:
                return total, 1
            return 1, 1
        if a_major:  # TT/cc: split N through the K-major B operand
            step = (self.N // total) * self.stride_B * (self.B.dtype.bits // 8)
            if total > 1 and self.N % total == 0 and step % 512 == 0:
                return 1, total
            return 1, 1
        # TN/rc (neither MN-major): keep the policy partition unchanged.
        return warp_m, warp_n

    def _cap_subtile(self, warp_m: int, warp_n: int, min_tile: int = 16) -> tuple[int, int]:
        """Clamp the warp partition so every warp's sub-tile is at least
        ``min_tile`` (the tensor-core M/N granularity) in both dims.

        The tensor core requires each MMA's M and N to be a positive multiple of
        16, so a partition that would give ``subM`` or ``subN`` below 16 (e.g.
        ``warp_n=4`` on ``N=32`` -> ``subN=8``) is rejected by ptcc. We reduce the
        offending factor to the largest value that both keeps the sub-tile >= 16
        and evenly divides the dim; surplus warps stay idle (the device template
        guards MMA with ``if (warp < warp_m * warp_n)``). Reducing a factor only
        enlarges the per-warp descriptor step, so the MN-major 512-B alignment is
        preserved.
        """
        max_wm = max(1, self.M // min_tile)
        max_wn = max(1, self.N // min_tile)
        while warp_m > 1 and (warp_m > max_wm or self.M % warp_m != 0):
            warp_m -= 1
        while warp_n > 1 and (warp_n > max_wn or self.N % warp_n != 0):
            warp_n -= 1
        return warp_m, warp_n

    def _check_operand_scopes(self) -> None:
        """Reject operand scopes the stcuv2 tensorcore cannot source an MMA from.

        A and B must be in shared memory. The MMA reads them through
        mma_data_desc, a shared-memory descriptor, so a register operand would
        have its register address reinterpreted as a shared one -- the template
        instantiation is otherwise identical, which is why this fails silently
        rather than crashing.

        Note that staging a fragment into shared does not currently work
        either: the operand has to be filled by the swizzle-aware bulk copy
        from global, and a shared buffer written by an in-kernel copy is read
        back wrong by the MMA. That is a separate gap, tracked in the S3 TODO
        list; the error below deliberately does not suggest it as a fix.

        A in tensor memory (the TS variant) goes through mma_atmem instead and
        is validated by _check_ts_constraints.
        """
        if self.is_gemm_ts():
            self._check_ts_constraints()
            return
        if not self.is_gemm_ss():
            raise ValueError(
                "TANG stcuv2 tcgen5 GEMM needs A and B in shared memory, got "
                f"A in '{self.A.scope()}' and B in '{self.B.scope()}'. There is "
                "no register-operand (RS/SR/RR) MMA on this architecture, and "
                "the operand must additionally be filled by a bulk copy from "
                "global -- a shared buffer written by an in-kernel copy is not "
                "yet read back correctly by the MMA."
            )

    def _check_ts_constraints(self) -> None:
        """Reject TS (A in tensor memory) shapes mma_atmem cannot express.

        A in tensor memory is always K-major and unswizzled, so only the layouts
        with a K-major A are reachable; a transposed A has no encoding and would
        otherwise be silently ignored by the descriptor. B is unconstrained and
        keeps its usual shared-memory descriptor.

        The operand is bit-packed into 32-bit tensor memory cells. Note what
        this check can and cannot see: it only gets the tensor memory buffer,
        never the copy that filled it, so it cannot tell an A operand from an
        accumulator. That matters for a 32-bit A, where the two are the same
        shape and width -- so a 32-bit A must be staged with
        `T.tcgen05_cp(tmem, shared)` (cps2t), which is unambiguous because
        accumulators are restored from a fragment, never from shared. The
        fragment route has no 32-bit entry point today: an unmarked
        `T.copy(fragment, tmem)` of a 32-bit buffer writes the *accumulator*
        layout and the MMA reads garbage, and the `tang_tmem_a_operand` marker
        that would override this is not yet exposed by any frontend primitive
        (tracked as T9 in the doc below).

        See docs/s3_tmem_a_operand_atmem_layout.md for the full constraint list
        and the open items.
        """
        if self.B.scope() not in ("shared", "shared.dyn"):
            raise ValueError(f"TANG stcuv2 tcgen5 GEMM with A in tensor memory needs B in shared memory, got B in '{self.B.scope()}'.")
        if self.trans_A:
            raise ValueError(
                "TANG stcuv2 tcgen5 GEMM with A in tensor memory requires a "
                "K-major A (trans_A=False): tensor-memory operands are always "
                "K-major and unswizzled, so there is no M-major encoding."
            )
        if self.A.dtype.bits not in (8, 16, 32):
            raise ValueError(
                "TANG stcuv2 tcgen5 GEMM with A in tensor memory supports "
                f"8/16/32-bit A, got '{self.A.dtype}'. Sub-byte operands are "
                "not staged on this path yet."
            )

    def infer_layout(self, target: Target, thread_nums: int):
        # Also checked in lower(); infer_layout runs first and would otherwise
        # just skip assigning a layout to a non-shared operand, silently.
        self._check_operand_scopes()
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
        del layout_map
        mbar = _completion_mbarrier(self.mbar)
        if _annotation_int(self.annotations, "use_2cta", 0):
            # A 2CTA MMA is issued by one rank of a cluster pair and writes an
            # accumulator twice as wide as either rank's operands -- the caller's
            # N shape check already assumes that doubling. TANG has no cluster
            # level, so ignoring the flag would leave half the accumulator
            # holding whatever was in tensor memory before.
            raise ValueError(
                "T.tcgen05_gemm(): use_2cta=True has no TANG equivalent. A "
                "two-CTA MMA needs a cluster to pair the ranks and a cluster "
                "barrier to hand the result back, and TANG has neither. Drop "
                "use_2cta and size C to N, not 2*N."
            )
        self._check_operand_scopes()
        k_const = _as_const_int(self.K, "K")
        # The 128-byte cap is on the operand row a single MMA reads out of
        # shared memory; A in tensor memory does not take that path, so it is
        # exempt (the golden atmem cases run past the cap).
        if not self.is_gemm_ts():
            _check_operand_row_bytes("A", k_const, self.A.dtype.bits)
        _check_operand_row_bytes("B", k_const, self.B.dtype.bits)
        thread_nums = _as_const_int(thread_bounds.extent, "thread extent")
        warp_m, warp_n = self._warp_partition(target, thread_nums)
        warp_m, warp_n = self._mn_major_partition(warp_m, warp_n)
        warp_m, warp_n = self._cap_subtile(warp_m, warp_n)
        if mbar is not None:
            # One completion event means one issuing warp. A TANG fence group is
            # private to the warp that issued into it, so an N-warp partition
            # would post N arrives and the barrier would need an arrive count of
            # N -- but the count is fixed at T.alloc_barrier(), out of reach
            # here, and CUDA source says 1. Getting it wrong hangs the consumer
            # or releases it early, so collapse to the single-warp MMA instead
            # and keep the count at 1. Costs MMA throughput; drop mbar= to get
            # the multi-warp partition back.
            warp_m, warp_n = 1, 1
        annotations = self.annotations
        a_format = _annotation_int(annotations, "tang_a_format", -1)
        b_format = _annotation_int(annotations, "tang_b_format", -1)

        a_ptr = _access_ptr(self.ARegion, "r")
        b_ptr = _access_ptr(self.BRegion, "r")
        c_ptr = _access_ptr(self.CRegion, "rw")

        if self.is_gemm_ts():
            # A already sits in tensor memory, so it carries no shared
            # descriptor and no warp sub-tiling: mma_atmem takes a bare column
            # offset, and splitting M across warps would need each warp's A
            # sub-block address re-derived (not worked out yet).
            if warp_m * warp_n != 1:
                raise ValueError(
                    "TANG stcuv2 tcgen5 GEMM with A in tensor memory is "
                    f"single-warp only, but the partition asked for {warp_m}x"
                    f"{warp_n} warps. Launch the kernel with 32 threads, or "
                    "keep A in shared memory for the multi-warp path."
                )
            if a_format >= 0 or b_format >= 0:
                raise ValueError(
                    "TANG stcuv2 tcgen5 GEMM with A in tensor memory does not "
                    "take explicit tang_a_format / tang_b_format element tags "
                    "yet; they exist for the sub-byte operands, which this "
                    "path does not stage."
                )
            if _as_const_int(self.offset_A, "A offset") != 0:
                raise ValueError(
                    "TANG stcuv2 tcgen5 GEMM with A in tensor memory cannot "
                    "slice A along K: mma_atmem reads A from its staged column "
                    "origin and stcuv2 has no descriptor-offset advance for a "
                    "nonzero K origin. Keep the staged A tile whole (K origin "
                    "0)."
                )
            return self._lower_ts_python_emitter(a_ptr, b_ptr, c_ptr, mbar, thread_var, mbar_phase_expr)
        elif _annotation_int(annotations, "tang_legacy_blockscaled", 0):
            if self.SFARegion is None or self.SFBRegion is None or self.SFTmemRegion is None:
                raise ValueError("TANG block-scaled GEMM requires scale_a, scale_b and scale_tmem regions")
            return self._lower_blockscaled_python_emitter(a_ptr, b_ptr, c_ptr, mbar, thread_var, mbar_phase_expr)
        else:
            return self._lower_ss_python_emitter(a_ptr, b_ptr, c_ptr, warp_m, warp_n, mbar, thread_var, mbar_phase_expr, a_format, b_format)

    def _lower_ss_python_emitter(
        self, a_ptr, b_ptr, c_ptr, warp_m, warp_n, mbar, thread_var, mbar_phase_expr=None, a_format=-1, b_format=-1
    ):
        raise NotImplementedError("STCUV2 GEMM is not supported")

    def _lower_ts_python_emitter(self, a_ptr, b_ptr, c_ptr, mbar, thread_var, mbar_phase_expr=None):
        raise NotImplementedError("STCUV2 GEMM is not supported")

    def _lower_blockscaled_python_emitter(self, a_ptr, b_ptr, c_ptr, mbar, thread_var, mbar_phase_expr=None):
        raise NotImplementedError("STCUV2 GEMM is not supported")


class GemmTangWGMMA(GemmBase):
    """Reject the unvalidated STCUV2 WGMMA dispatch path."""

    def infer_layout(self, target: Target, thread_nums: int):
        del target, thread_nums
        return {}

    def lower(self, layout_map, target, thread_bounds, thread_var, mbar_phase_expr=None):
        del layout_map, target, thread_bounds, thread_var, mbar_phase_expr
        raise ValueError("TANG stcuv2 WGMMA lowering remains unverified and is disabled.")
