"""STCU tensor-core GEMM lowering."""

from __future__ import annotations

from tilelang import language as T
from tilelang.layout import Fragment, Layout
from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang.transform.simplify import _Simplify
from tvm import tirx
from tvm.ir import Range
from tvm.target import Target


GEMM_INST_TMMA = "tang.tmma"


def _as_const_int(value, name: str) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, tirx.IntImm):
        return int(value.value)
    raise ValueError(f"TANG TMMA requires constant {name}, but got {value}")


def _make_tang_fragment_c(m: int, n: int, warp_m: int, warp_n: int, element_bits: int) -> Fragment:
    if element_bits == 32:
        atom = Fragment(
            [8, 8],
            forward_fn=lambda i, j: ((i % 4) * 8 + j, i // 4),
        )
    elif element_bits == 16:
        atom = Fragment(
            [8, 8],
            forward_fn=lambda i, j: (i * 4 + j // 2, j % 2),
        )
    else:
        raise ValueError(f"TANG TMMA only supports 16-bit or 32-bit accumulators, got {element_bits}")

    if m % warp_m != 0 or n % warp_n != 0 or warp_m % 8 != 0 or warp_n % 8 != 0:
        raise ValueError(f"Invalid TANG TMMA tile: M={m}, N={n}, warp_m={warp_m}, warp_n={warp_n}")
    warp_layout = atom.repeat([warp_m // 8, warp_n // 8], False, True)
    return warp_layout.repeat([m // warp_m, n // warp_n], True, False)


def _make_tensor_core_swizzle_layout_tang(
    stride: int,
    continuous: int,
    element_bits: int,
    offset: int,
    is_a: bool,
    transposed: bool,
    gemm_m: int | None,
    gemm_n: int | None,
    gemm_k: int | None,
    k_pack: int = 1,
    allow_padding: bool = False,
) -> Layout:
    num_banks = 32
    bank_width_bits = 32
    simd_width = 16
    vector_size = 32 // element_bits
    inner_continuous = continuous - offset
    value_width = 128 // element_bits
    c_extent = value_width if is_a else 8
    r_extent = 8 if is_a else value_width
    rows = c_extent if transposed else r_extent
    cols = r_extent if transposed else c_extent
    tile_size = rows * cols
    # Validation only: the XOR swizzle these describe was replaced by the
    # padding schemes below, but a non-positive max_phase still marks a
    # degenerate (too-narrow) shared tile that neither scheme can address.
    elems_per_bank_row = num_banks * bank_width_bits // element_bits
    per_phase = max(1, elems_per_bank_row // inner_continuous)
    max_phase = min(simd_width // per_phase, inner_continuous // vector_size)
    if max_phase <= 0:
        raise ValueError(
            f"Invalid TANG shared layout: stride={stride}, continuous={continuous}, element_bits={element_bits}, offset={offset}"
        )

    # ---- Bank-conflict padding ----
    # Padding is required to avoid the bank conflicts seen without a row gap.
    # ``pad`` follows the established 128-bit rule -- insert one 128-bit gap
    # when the contiguous dimension is a whole multiple of 256 bits.
    #
    # Gated to offset == 0: for a strided/sliced shared region the read side
    # (LoadFromShared's has_stride_offset path in gemm_tmma.h) applies its own
    # row-stride correction and does not pad, so padding here would desync the
    # two sides. Both sides keep pad=0 there.
    #
    # The width tested is ``inner_continuous`` (the GEMM region), not
    # ``continuous`` (which includes the sliced-off prefix), to match the read
    # side: there PadWords derives from StrideWords == stride_real / vec_size,
    # and stride_real == stride - offset == inner_continuous. Today the
    # offset == 0 guard makes the two spellings identical (offset == 0 implies
    # continuous == inner_continuous), so this is a no-op; it diverges the moment
    # the offset == 0 guard is relaxed for sliced shared regions, and the
    # resulting write/read split would be silent. Keep it in the region form.
    # Restricted to 16-bit operands to match the read side's vec_size == 2 gate
    # (gemm_tmma.h). Applying the same scheme to fp32 increases the shared
    # extent and can exceed the available resource limit, so padding remains
    # restricted to supported 16-bit operands. Other widths stay unpadded.
    pad = (
        128 // element_bits if allow_padding and element_bits == 16 and offset == 0 and (element_bits * inner_continuous) % 256 == 0 else 0
    )

    # ---- Per-tile padding (gated) ----
    # Insert the padding gap after every tile (``tile_size`` elements) instead
    # of after every logical row. This makes the read-side S2R address affine in
    # the unroll index, so the compiler folds ``i * phys_tile`` into the LDS
    # [imm] field and drops the per-tile srl/and/add correction that the per-row
    # scheme forces onto the half-row A walk.
    #
    # The gate must stay PER-OPERAND and be the SAME PREDICATE the read side
    # uses, or that operand is written in one scheme and read in the other and
    # silently corrupts the operand. The read side (S2RLoad in gemm_tmma.h)
    # gates on the GEMM TILE dims: ``isA ? (M_Tile == K_Tile) : (K_Tile ==
    # N_Tile)``, so this must compare gemm_m/gemm_k/gemm_n too.
    #
    # It must NOT be spelled ``stride == inner_continuous``: those are the shared
    # buffer's own extents, and they only coincide with the tile dims when the
    # buffer is exactly as wide as the GEMM K tile. A buffer with slack columns
    # (e.g. ``T.alloc_shared((block_M, block_K * 2))``, then
    # ``T.gemm(A_s[:, :block_K], ...)``) splits the two sides: at
    # block_M=block_K=64 with a 128-wide buffer the read side sees
    # M_Tile == K_Tile (per-tile) while the buffer form sees 64 != 128 (per-row).
    # Regression-tested by the 64x64x64 wide-buffer case; the shipping configs
    # allocate width == block_K, which is why they masked it.
    #   - offset == 0: mirrors the read side's !has_stride_offset. The offset != 0
    #     arm there does its own row-stride correction and never pads.
    #   - element_bits == 16: mirrors vec_size == 2. The read side defines a tile
    #     as 32 words, which equals tile_size / vector_size only for fp16/bf16.
    per_tile_pad = allow_padding and offset == 0 and element_bits == 16 and pad != 0 and (gemm_m == gemm_k if is_a else gemm_k == gemm_n)

    if per_tile_pad:

        def forward_per_tile(row, col):
            idx_in_tile = (row % rows) * cols + (col % cols)
            tile_idx = (row // rows) * (inner_continuous // cols) + col // cols
            return tile_idx * (tile_size + pad) + idx_in_tile

        return Layout([stride, continuous], forward_per_tile)

    def forward(row, col):
        if offset == 0:
            region_col = col
            region_flag = 1
        else:
            # 0 for the unused prefix and 1 for the GEMM region.  Express the
            # piecewise mapping arithmetically: vectorized layout indices may
            # become Ramp nodes, while Select over a Ramp is not accepted by
            # the integer-only Z3 bounds prover used during LowerTileOp.
            region_flag = (col + continuous - offset) // continuous
            region_col = col - region_flag * offset
        idx_in_tile = (row % rows) * cols + (region_col % cols)
        tile_idx = (row // rows) * (inner_continuous // cols) + region_col // cols
        linear = tile_idx * tile_size + idx_in_tile
        new_row = linear // inner_continuous
        linear = linear + (new_row + 1) * offset
        mapped_row = linear // continuous
        mapped_col = linear % continuous - (1 - region_flag) * offset
        if pad != 0:
            # Emit a single linear offset (same convention as the per-tile branch
            # and makeGemmABLayoutPadded): addr = row * (continuous + pad) + col,
            # so consecutive logical rows land in different banks.
            return mapped_row * (continuous + pad) + mapped_col
        return [mapped_row, mapped_col]

    return Layout([stride, continuous], forward)


def _make_tang_ab_layout(
    stride: int,
    continuous: int,
    element_bits: int,
    offset: int,
    is_a: bool,
    transposed: bool,
) -> Layout:
    # Plain unpadded 2-D mapping, for consumers that reach shared memory through
    # a hardware descriptor rather than the TMMA template's S2R load. TCGEN5's
    # mma_data_desc is one: its LBO is the logical row pitch, so it cannot
    # express the padding gaps and data would land where the descriptor does not
    # expect it. TMMA operands go through _make_tang_ab_layout_padded instead.
    return _make_tensor_core_swizzle_layout_tang(
        stride,
        continuous,
        element_bits,
        offset,
        is_a,
        transposed,
        None,
        None,
        None,
    )


def _make_tang_ab_layout_padded(
    stride: int,
    continuous: int,
    element_bits: int,
    offset: int,
    is_a: bool,
    transposed: bool,
    gemm_m: int,
    gemm_n: int,
    gemm_k: int,
    k_pack: int = 1,
) -> Layout:
    # Bank-conflict padded mapping for the TMMA shared A/B operands. Padding is a
    # software scheme: it is only correct because the TMMA template's S2R load
    # mirrors the same gap arithmetic, which is why this entry point is TMMA-only.
    # The gemm_m/n/k tile dims are required because the per-tile gate must be the
    # same predicate the read side uses.
    return _make_tensor_core_swizzle_layout_tang(
        stride,
        continuous,
        element_bits,
        offset,
        is_a,
        transposed,
        gemm_m,
        gemm_n,
        gemm_k,
        k_pack,
        allow_padding=True,
    )


class GemmTMMA(GemmBase):
    """Lower STCU TMMA template calls."""

    def _warp_partition(self, target: Target, thread_nums: int) -> tuple[int, int]:
        return self.policy.compute_warp_partition(
            self.M,
            self.N,
            thread_nums,
            target,
            GEMM_INST_TMMA,
        )

    def infer_layout(self, target: Target, thread_nums: int):
        m_warps, n_warps = self._warp_partition(target, thread_nums)
        warp_m = self.M // m_warps
        warp_n = self.N // n_warps
        k_pack = _as_const_int(self.k_pack if self.k_pack is not None else 1, "k_pack")
        # GEMM tile dims, not the shared buffers' extents. The per-tile padding
        # gate below must compare these because the read side does (M_Tile /
        # N_Tile / K_Tile in gemm_tmma.h); a buffer with slack columns otherwise
        # splits the write and read schemes for that operand.
        gemm_m = _as_const_int(self.M, "GEMM M")
        gemm_n = _as_const_int(self.N, "GEMM N")
        gemm_k = _as_const_int(self.K, "GEMM K")
        layouts = {
            self.C: _make_tang_fragment_c(
                self.M,
                self.N,
                warp_m,
                warp_n,
                self.C.dtype.bits,
            )
        }
        if self.A.scope() in ("shared", "shared.dyn"):
            layouts[self.A] = _make_tang_ab_layout_padded(
                _as_const_int(self.A.shape[-2], "A row extent"),
                _as_const_int(self.A.shape[-1], "A column extent"),
                self.A.dtype.bits,
                _as_const_int(self.offset_A, "A offset"),
                True,
                self.trans_A,
                gemm_m,
                gemm_n,
                gemm_k,
                k_pack,
            )
        if self.B.scope() in ("shared", "shared.dyn"):
            layouts[self.B] = _make_tang_ab_layout_padded(
                _as_const_int(self.B.shape[-2], "B row extent"),
                _as_const_int(self.B.shape[-1], "B column extent"),
                self.B.dtype.bits,
                _as_const_int(self.offset_B, "B offset"),
                False,
                self.trans_B,
                gemm_m,
                gemm_n,
                gemm_k,
                k_pack,
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
        thread_nums = _as_const_int(thread_bounds.extent, "thread extent")
        m_warps, n_warps = self._warp_partition(target, thread_nums)
        # ``clear_accum=True`` is supported, and it makes GemmTensorOp emit a real
        # zero-init over C_local -- the hardware does NOT overwrite the
        # accumulator for us. An earlier revision claimed it did, on the grounds
        # that every MMA chain starts with a `tensor.mul` that ignores C; that
        # instruction feeds a tensor latch, and the chain's closing
        # `tensor.macc.out` is followed by one `add.f32` per element that folds
        # the incoming C back in. See the clear_accum note in
        # src/tl_templates/tang/gemm_tmma.h for the assembly counts and for the
        # poisoned-accumulator test that settles it (pre-filling C_local with
        # 1000.0 instead of 0.0 shifts the result by exactly 1000.0).
        #
        # Callers may therefore pass clear_accum=True on the first T.gemm of a
        # k-loop and skip T.clear; the flag reaches GemmTensorOp as its last
        # template argument, where the zeroing happens.
        clear_accum = _as_const_int(self.clear_accum, "clear_accum") != 0
        offset_a = _as_const_int(self.offset_A, "A offset")
        offset_b = _as_const_int(self.offset_B, "B offset")
        k_pack = _as_const_int(self.k_pack if self.k_pack is not None else 1, "k_pack")
        if k_pack != 1:
            raise ValueError(f"TANG STCU TMMA does not support k_pack={k_pack}; only k_pack=1")
        # The TMMA template always interleaves the shared->register loads with the
        # MMA chain (equivalent to load_overlap_mma for both operands), so the
        # a/b_local_load_type knobs have no effect here. Reject a request for the
        # load_before_mma variant instead of silently ignoring it.
        for operand in ("a", "b"):
            annotation = f"tang_{operand}_local_load_overlap_mma"
            if _as_const_int(self.annotations.get(annotation, 1), f"{operand} local-load policy") == 0:
                raise ValueError(
                    f"TANG STCU TMMA does not support {operand}_local_load_type='load_before_mma'; "
                    "the template always overlaps the local load with the MMA chain"
                )
        template = (
            f"tl::gemm_tang<{self.M}, {self.N}, {self.K}, {m_warps}, {n_warps}, "
            f"{self.stride_A}, {self.stride_B}, {offset_a}, {offset_b}, "
            f"{int(self.trans_A)}, {int(self.trans_B)}, "
            f"{int(clear_accum)}>"
        )
        a_ptr = T.access_ptr(
            self.ARegion,
            "r",
            extent=self.ARegion.region[-2].extent * self.ARegion.region[-1].extent,
            ignore_last_ndim=2,
        )
        b_ptr = T.access_ptr(
            self.BRegion,
            "r",
            extent=self.BRegion.region[-2].extent * self.BRegion.region[-1].extent,
            ignore_last_ndim=2,
        )
        c_ptr = T.access_ptr(
            self.CRegion,
            "rw",
            extent=self.CRegion.region[-2].extent * self.CRegion.region[-1].extent,
            ignore_last_ndim=2,
        )
        call = tirx.call_intrin(
            "handle",
            tirx.op.Op.get("tl.tl_tang_gemm"),
            tirx.StringImm(template),
            a_ptr,
            b_ptr,
            c_ptr,
        )

        @T.prim_func
        def _gemm_tang() -> None:
            T.evaluate(call)

        return _Simplify(_gemm_tang, inline_let=True)
