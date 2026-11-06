import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

from tile_kernels.config import get_max_ub_per_vector_core
from tile_kernels.quant.common import CastOutputConfig
from tile_kernels.utils import align

# BF16 bit-field constants for power-of-two round_sf.
_ABS_MASK_BF16 = 0x7FFF
_BF16_EXP_BIAS = 0x7F00
_EXP_MASK_F32 = 0x7F800000
_F32_EXP_BIAS = 0x7F000000


def _get_block_shape(hidden: int, dtype: T.dtype, num_stages: int, num_per_channels: int) -> tuple[int, int]:
    """Choose the widest valid block_k and largest UB-fitting block_m."""
    stage_budget = get_max_ub_per_vector_core(use_simt=False) // num_stages

    alignment = 256 if hidden % 256 == 0 else 128
    for block_k in range(hidden, alignment - 1, -alignment):
        if hidden % block_k != 0:
            continue
        for block_m in range(32, 0, -1):
            num_sf = align(block_m * align(block_k // num_per_channels, 8), 64)
            ub_usage = block_m * block_k * dtype.bits // 4 + num_sf * 12
            if ub_usage <= stage_budget:
                return block_m, block_k


@tilelang.jit
def get_per_token_cast_and_cast_back_kernel_asc(
    hidden: int,
    token_stride: int,
    dtype: T.dtype,
    out_config: CastOutputConfig,
    num_vec_cores: int,
):
    num_per_channels = out_config.sf_block[1]
    is_bf16_in = dtype == T.bfloat16
    assert dtype in (T.bfloat16, T.float32), 'Ascend per-token cast and cast-back only supports bf16 or fp32 input'
    assert (num_per_channels, out_config.use_e4m3_sf) in ((16, True), (32, False)), 'Ascend supports per-16 with E4M3 SF or per-32 without E4M3 SF'
    assert hidden % 128 == 0, 'Ascend per-token cast and cast-back hidden must be a multiple of 128'

    is_fp4_out = out_config.dtype == T.float4_e2m1fn
    round_sf = out_config.round_sf
    quant_max = 6.0 if is_fp4_out else 448.0
    round_carry = 0x003FFFFF if is_fp4_out else 0x001FFFFF
    quant_max_exp = 0x01000000 if is_fp4_out else 0x04000000

    num_stages = 3

    block_m, block_k = _get_block_shape(hidden, dtype, num_stages, num_per_channels)
    # Per-32 rows with a 128-element tail need four padding scales.
    num_groups = align(block_k // num_per_channels, 8)

    num_sf = align(block_m * num_groups, 64)
    use_bf16_sf = is_bf16_in and round_sf
    sf_inv_dtype = T.bfloat16 if use_bf16_sf else T.float32
    use_bf16_dequant = is_bf16_in and out_config.use_e4m3_sf
    sf_dtype = T.bfloat16 if use_bf16_dequant else T.float32

    num_tokens = T.dynamic('num_tokens')

    @T.macro
    def get_sf_from_inv(sf_inv, sf_inv_bias):
        """Recover power-of-two scales from inverse scales using exponent bits."""
        sf_bits = S.vsub(sf_inv_bias, T.reinterpret(sf_inv, 'uint16x128' if use_bf16_sf else 'uint32x64'))
        return T.reinterpret(sf_bits, 'bfloat16x128' if use_bf16_sf else 'float32x64')

    @T.macro
    def load_sf_dintlv(sf_ub, sf_offset):
        """Load eight FP32 scales and broadcast each to 16 lanes."""
        expanded = S.vld(sf_ub[sf_offset], dist='E2B_B32')
        return S.vintlv(expanded, expanded)

    @T.macro
    def load_sf(sf_ub, sf_offset):
        """Load eight FP32 scales and broadcast each to 32 lanes."""
        low, high = load_sf_dintlv(sf_ub, sf_offset)
        sf0, sf1 = S.vintlv(low, low)
        sf2, sf3 = S.vintlv(high, high)
        return sf0, sf1, sf2, sf3

    @T.macro
    def collect_amax_tile(x_ub, amax_ub, row, col, tile_k, mask_group_low, mask_group_high):
        """Compute and store the maximum absolute value of each scale group in one tile."""
        sf_offset = row * num_groups + col // num_per_channels
        if is_bf16_in:
            abs_mask_u16 = S.vdup(_ABS_MASK_BF16, T.uint16)
            zero_bf16 = S.vdup(0.0, T.bfloat16)
            amax_store_mask = S.pset(32, 'PAT_VL8' if tile_k // num_per_channels == 8 else 'PAT_VL4')
            if num_per_channels == 16:
                values = S.vld(x_ub[row, col])
                abs_max_u16 = S.vand(T.reinterpret(values, 'uint16x128'), abs_mask_u16)
            elif tile_k == 256:
                values0, values1 = S.vld2(x_ub[row, col], dist='DINTLV_B16')
                abs0_u16 = S.vand(T.reinterpret(values0, 'uint16x128'), abs_mask_u16)
                abs1_u16 = S.vand(T.reinterpret(values1, 'uint16x128'), abs_mask_u16)
                abs_max_u16 = S.vmax(abs0_u16, abs1_u16)
            else:
                values = S.vld(x_ub[row, col])
                abs_u16 = S.vand(T.reinterpret(values, 'uint16x128'), abs_mask_u16)
                even, odd = S.vdintlv(abs_u16, abs_u16)
                abs_max_u16 = S.vmax(even, odd)
            amax_u16 = S.vcgmax(abs_max_u16)
            amax_low, _amax_high = S.vintlv(zero_bf16, T.reinterpret(amax_u16, 'bfloat16x128'))
            S.vsts(amax_ub[sf_offset], T.reinterpret(amax_low, 'float32x64'), amax_store_mask, dist='NORM_B32')
        elif num_per_channels == 16:
            amax_store_mask = S.pset(32, 'PAT_VL8')
            values0, values1 = S.vld2(x_ub[row, col], dist='DINTLV_B32')
            abs_values = S.vmax(S.vabs(values0), S.vabs(values1))
            amax = S.vcgmax(abs_values)
            S.vsts(amax_ub[sf_offset], amax, amax_store_mask, dist='NORM_B32')
        else:
            for vector in T.unroll(tile_k // 64, explicit=True):
                values = S.vld(x_ub[row, col + vector * 64])
                abs_values = S.vabs(values)
                amax_lower = S.vcmax(abs_values, mask_group_low)
                amax_upper = S.vcmax(abs_values, mask_group_high)
                S.vsts(amax_ub[sf_offset + vector * 2], amax_lower, dist='ONEPT_B32')
                S.vsts(amax_ub[sf_offset + vector * 2 + 1], amax_upper, dist='ONEPT_B32')

    @T.macro
    def collect_amax(x_ub, amax_ub):
        """Collect per-group maxima across all rows and tiles in the UB block."""
        with T.SimdVF():
            mask_group_low = S.pset(32, 'PAT_VL32')
            mask_group_high = S.vcmps(S.vci(0, T.int32), 31, op='gt')
            tile_k = num_per_channels * 8
            for row in T.serial(block_m):
                for tile in T.serial(block_k // tile_k):
                    collect_amax_tile(x_ub, amax_ub, row, tile * tile_k, tile_k, mask_group_low, mask_group_high)
                if block_k % tile_k:
                    collect_amax_tile(x_ub, amax_ub, row, block_k - 128, 128, mask_group_low, mask_group_high)

    @T.macro
    def compute_sf(amax_ub, sf_ub, sf_inv_ub):
        """Compute scales and inverse scales from group maxima in the requested scale format."""
        with T.SimdVF():
            if round_sf:
                exp_mask_u32 = S.vdup(_EXP_MASK_F32, T.uint32)
                sf_inv_bias_u32 = S.vdup(_F32_EXP_BIAS + quant_max_exp, T.uint32)
            else:
                quant_max_f32 = S.vdup(quant_max, T.float32)

            for sf_batch in T.serial(num_sf // 64):
                sf_offset = sf_batch * 64
                amax = S.vld(amax_ub[sf_offset])
                clamped_amax = S.vmaxs(amax, out_config.clamp_min_value)
                if round_sf:
                    rounded_amax_exp_u32 = S.alloc_var(T.uint32)
                    rounded_amax_exp_u32 = S.vadds(T.reinterpret(clamped_amax, 'uint32x64'), round_carry)
                    rounded_amax_exp_u32 = S.vand(rounded_amax_exp_u32, exp_mask_u32)
                    sf_inv_bits_u32 = S.vsub(sf_inv_bias_u32, rounded_amax_exp_u32)
                    sf_inv = T.reinterpret(sf_inv_bits_u32, 'float32x64')
                else:
                    # BF16 amax with the default clamp rounds to the same E4M3 SF after hardware division.
                    sf = S.vdiv(
                        clamped_amax,
                        quant_max_f32,
                        precision='ftz_true' if use_bf16_dequant and out_config.clamp_min_value == quant_max * 2**-9 else None,
                    )
                    if out_config.use_e4m3_sf:
                        # Match CUDA's finite saturation when the scale itself overflows E4M3.
                        sf_f32 = S.vcvt(S.vcvt(S.vmins(sf, 448.0), T.float8_e4m3fn), T.float32, part=0)
                        if is_bf16_in:
                            S.vsts(sf_ub[sf_offset], S.vcvt(sf_f32, T.bfloat16), dist='PK_B32')
                        else:
                            S.vsts(sf_ub[sf_offset], sf_f32)
                        # Hardware division matches the corrected result for all finite nonnegative E4M3 scales on Ascend950.
                        sf_inv = S.vdiv(S.vdup(1.0, T.float32), sf_f32, precision='ftz_true' if is_bf16_in else None)
                    else:
                        S.vsts(sf_ub[sf_offset], sf)
                        sf_inv = S.vdiv(quant_max_f32, clamped_amax)
                if use_bf16_sf:
                    S.vsts(sf_inv_ub[sf_offset], S.vcvt(sf_inv, T.bfloat16), dist='PK_B32')
                else:
                    S.vsts(sf_inv_ub[sf_offset], sf_inv)

    @T.macro
    def clamp_e4m3(values):
        """Clamp normalized values to the finite E4M3 range."""
        # Rounding SF down can make normalized values exceed E4M3's finite range.
        return S.vmins(S.vmaxs(values, -448.0), 448.0)

    @T.macro
    def cast_and_cast_back_f32_pair(scaled0, scaled1, sf0, sf1):
        """Round-trip two FP32 vectors; return one BF16 or two FP32 dequantized vectors."""
        if is_fp4_out:
            low, high = S.vdintlv(T.reinterpret(scaled0, 'uint16x128'), T.reinterpret(scaled1, 'uint16x128'))
            scaled_bf16 = T.reinterpret(S.vor(high, S.vmins(low, 1)), 'bfloat16x128')
            cast_back_bf16 = S.vcvt(S.vcvt(scaled_bf16, T.float4_e2m1fn), T.bfloat16, part=0)
            if not use_bf16_dequant:
                cast_back0, cast_back1 = S.vintlv(
                    S.vcvt(cast_back_bf16, T.float32, part=0),
                    S.vcvt(cast_back_bf16, T.float32, part=1),
                )
        else:
            cast_back0 = S.vcvt(S.vcvt(clamp_e4m3(scaled0) if out_config.use_e4m3_sf else scaled0, T.float8_e4m3fn), T.float32, part=0)
            cast_back1 = S.vcvt(S.vcvt(clamp_e4m3(scaled1) if out_config.use_e4m3_sf else scaled1, T.float8_e4m3fn), T.float32, part=0)
            if use_bf16_dequant:
                _, cast_back_bits = S.vdintlv(T.reinterpret(cast_back0, 'uint16x128'), T.reinterpret(cast_back1, 'uint16x128'))
                cast_back_bf16 = T.reinterpret(cast_back_bits, 'bfloat16x128')
        if use_bf16_dequant:
            # Finite E2M1/E4M3 * E4M3 products are exactly representable in BF16.
            return S.vmul(cast_back_bf16, sf0)
        else:
            return S.vmul(cast_back0, sf0), S.vmul(cast_back1, sf1)

    @T.macro
    def cast_and_cast_back_bf16(scaled, sf):
        """Quantize a scaled BF16 vector and dequantize with a power-of-two scale."""
        if is_fp4_out:
            quantized = S.vcvt(scaled, T.float4_e2m1fn)
            cast_back = S.vcvt(quantized, T.bfloat16, part=0)
            return S.vmul(cast_back, sf)
        else:
            scaled0_bits, scaled1_bits = S.vintlv(S.vdup(0.0, T.bfloat16), scaled)
            scaled0 = T.reinterpret(scaled0_bits, 'float32x64')
            scaled1 = T.reinterpret(scaled1_bits, 'float32x64')
            quantized0 = S.vcvt(scaled0, T.float8_e4m3fn)
            quantized1 = S.vcvt(scaled1, T.float8_e4m3fn)
            cast_back0 = S.vcvt(quantized0, T.float32, part=0)
            cast_back1 = S.vcvt(quantized1, T.float32, part=0)
            _rounding_bits, result = S.vdintlv(
                T.reinterpret(cast_back0, 'uint16x128'),
                T.reinterpret(cast_back1, 'uint16x128'),
            )
            return S.vmul(T.reinterpret(result, 'bfloat16x128'), sf)

    @T.macro
    def cast_and_cast_back(x_ub, out_ub, row, col, sf0, sf1, sf_inv0, sf_inv1):
        """Load, scale, quantize, dequantize, and store 128 contiguous input values."""
        if is_bf16_in:
            values0 = S.vcvt(S.vld(x_ub[row, col], dist='UNPK_B16'), T.float32, part=0)
            values1 = S.vcvt(S.vld(x_ub[row, col + 64], dist='UNPK_B16'), T.float32, part=0)
            scaled0 = S.vmul(values0, sf_inv0)
            scaled1 = S.vmul(values1, sf_inv1)
        else:
            values0 = S.vld(x_ub[row, col])
            values1 = S.vld(x_ub[row, col + 64])
            scaled0 = S.vmul(values0, sf_inv0)
            scaled1 = S.vmul(values1, sf_inv1)
        output = cast_and_cast_back_f32_pair(scaled0, scaled1, sf0, sf1)
        if use_bf16_dequant:
            S.vsts(out_ub[row, col], output)
        else:
            output0, output1 = output
            if is_bf16_in:
                S.vsts(out_ub[row, col], S.vcvt(output0, T.bfloat16), dist='PK_B32')
                S.vsts(out_ub[row, col + 64], S.vcvt(output1, T.bfloat16), dist='PK_B32')
            else:
                S.vsts(out_ub[row, col], output0)
                S.vsts(out_ub[row, col + 64], output1)

    @T.macro
    def cast_and_cast_back_bf16_no_round(x_ub, out_ub, row, col, sf_lower, sf_upper, sf_inv_lower, sf_inv_upper, zero_bf16):
        """Process 256 BF16 inputs with per-32 FP32 scales and deinterleaved loads."""
        values0, values1 = S.vld2(x_ub[row, col], dist='DINTLV_B16')
        values0_lower_bits, values0_upper_bits = S.vintlv(zero_bf16, values0)
        values1_lower_bits, values1_upper_bits = S.vintlv(zero_bf16, values1)
        values0_lower = T.reinterpret(values0_lower_bits, 'float32x64')
        values0_upper = T.reinterpret(values0_upper_bits, 'float32x64')
        values1_lower = T.reinterpret(values1_lower_bits, 'float32x64')
        values1_upper = T.reinterpret(values1_upper_bits, 'float32x64')

        output0_lower, output1_lower = cast_and_cast_back_f32_pair(
            S.vmul(values0_lower, sf_inv_lower), S.vmul(values1_lower, sf_inv_lower), sf_lower, sf_lower
        )
        output_lower0, output_lower1 = S.vintlv(output0_lower, output1_lower)
        S.vsts(out_ub[row, col], S.vcvt(output_lower0, T.bfloat16), dist='PK_B32')
        S.vsts(out_ub[row, col + 64], S.vcvt(output_lower1, T.bfloat16), dist='PK_B32')

        output0_upper, output1_upper = cast_and_cast_back_f32_pair(
            S.vmul(values0_upper, sf_inv_upper), S.vmul(values1_upper, sf_inv_upper), sf_upper, sf_upper
        )
        output_upper0, output_upper1 = S.vintlv(output0_upper, output1_upper)
        S.vsts(out_ub[row, col + 128], S.vcvt(output_upper0, T.bfloat16), dist='PK_B32')
        S.vsts(out_ub[row, col + 192], S.vcvt(output_upper1, T.bfloat16), dist='PK_B32')

    @T.macro
    def cast_and_cast_back_per16(x_ub, sf_ub, sf_inv_ub, out_ub):
        """Process each row with per-16 scaling in tiles of 128 elements."""
        with T.SimdVF():
            for row in T.serial(block_m):
                for tile in T.serial(block_k // 128):
                    sf_offset = row * num_groups + tile * 8
                    if use_bf16_dequant:
                        sf0 = S.vld(sf_ub[sf_offset], dist='E2B_B16')
                        sf1 = sf0
                    else:
                        sf0, sf1 = load_sf_dintlv(sf_ub, sf_offset)
                    sf_inv0, sf_inv1 = load_sf_dintlv(sf_inv_ub, sf_offset)
                    cast_and_cast_back(x_ub, out_ub, row, tile * 128, sf0, sf1, sf_inv0, sf_inv1)

    @T.macro
    def cast_and_cast_back_per32_tile(x_ub, sf_ub, sf_inv_ub, out_ub, row, col, tile_k, sf_inv_bias, zero_bf16):
        """Quantize and dequantize one per-32 tile of 256 or 128 elements."""
        sf_offset = row * num_groups + col // 32
        if use_bf16_sf and tile_k == 256:
            values0, values1 = S.vld2(x_ub[row, col], dist='DINTLV_B16')
            sf_inv = S.vld(sf_inv_ub[sf_offset], dist='E2B_B16')
            scaled0 = S.vmul(values0, sf_inv)
            scaled1 = S.vmul(values1, sf_inv)
            sf = get_sf_from_inv(sf_inv, sf_inv_bias)
            output0 = cast_and_cast_back_bf16(scaled0, sf)
            output1 = cast_and_cast_back_bf16(scaled1, sf)
            output_lower, output_upper = S.vintlv(output0, output1)
            S.vsts(out_ub[row, col], output_lower, dist='NORM_B16')
            S.vsts(out_ub[row, col + 128], output_upper, dist='NORM_B16')
        elif use_bf16_sf:
            expanded = S.vld(sf_inv_ub[sf_offset], dist='E2B_B16')
            # Widen each scale's broadcast from 16 to 32 lanes for the contiguous load.
            sf_inv, _sf_inv_high = S.vintlv(expanded, expanded)
            values = S.vld(x_ub[row, col])
            scaled = S.vmul(values, sf_inv)
            sf = get_sf_from_inv(sf_inv, sf_inv_bias)
            output = cast_and_cast_back_bf16(scaled, sf)
            S.vsts(out_ub[row, col], output, dist='NORM_B16')
        elif is_bf16_in and tile_k == 256:
            sf_lower, sf_upper = load_sf_dintlv(sf_ub, sf_offset)
            sf_inv_lower, sf_inv_upper = load_sf_dintlv(sf_inv_ub, sf_offset)
            cast_and_cast_back_bf16_no_round(x_ub, out_ub, row, col, sf_lower, sf_upper, sf_inv_lower, sf_inv_upper, zero_bf16)
        else:
            if round_sf:
                sf_invs = load_sf(sf_inv_ub, sf_offset)
                sfs = (
                    get_sf_from_inv(sf_invs[0], sf_inv_bias),
                    get_sf_from_inv(sf_invs[1], sf_inv_bias),
                    get_sf_from_inv(sf_invs[2], sf_inv_bias),
                    get_sf_from_inv(sf_invs[3], sf_inv_bias),
                )
            else:
                sfs = load_sf(sf_ub, sf_offset)
                sf_invs = load_sf(sf_inv_ub, sf_offset)
            cast_and_cast_back(x_ub, out_ub, row, col, sfs[0], sfs[1], sf_invs[0], sf_invs[1])
            if tile_k == 256:
                cast_and_cast_back(x_ub, out_ub, row, col + 128, sfs[2], sfs[3], sf_invs[2], sf_invs[3])

    @T.macro
    def cast_and_cast_back_per32(x_ub, sf_ub, sf_inv_ub, out_ub):
        """Process each row with per-32 scaling, including any 128-element tail."""
        with T.SimdVF():
            bias_u16 = S.vdup(_BF16_EXP_BIAS, T.uint16)
            sf_inv_bias_u32 = S.vdup(_F32_EXP_BIAS, T.uint32)
            sf_inv_bias = bias_u16 if use_bf16_sf else sf_inv_bias_u32
            zero_bf16 = S.vdup(0.0, T.bfloat16)
            for row in T.serial(block_m):
                for tile in T.serial(block_k // 256):
                    cast_and_cast_back_per32_tile(x_ub, sf_ub, sf_inv_ub, out_ub, row, tile * 256, 256, sf_inv_bias, zero_bf16)
                if block_k % 256:
                    cast_and_cast_back_per32_tile(x_ub, sf_ub, sf_inv_ub, out_ub, row, block_k - 128, 128, sf_inv_bias, zero_bf16)

    @T.prim_func
    def per_token_cast_and_cast_back_kernel(x: T.StridedTensor[(num_tokens, hidden), (token_stride, 1), dtype]):
        with T.Kernel(num_vec_cores) as core_id:
            x_ub = T.alloc_shared((block_m, block_k), dtype)
            amax_ub = T.alloc_shared((num_sf,), T.float32)
            sf_ub = T.alloc_shared((num_sf,), sf_dtype)
            sf_inv_ub = T.alloc_shared((num_sf,), sf_inv_dtype)
            out_ub = T.alloc_shared((block_m, block_k), dtype)
            versions = {x_ub: num_stages, amax_ub: num_stages, sf_inv_ub: num_stages, out_ub: num_stages}
            if not round_sf:
                versions[sf_ub] = num_stages
            T.annotate_buffer_versions(versions)

            for pid_token, pid_hidden in T.Persistent(
                [T.ceildiv(num_tokens, block_m), hidden // block_k],
                num_vec_cores,
                core_id,
                group_size=1,
                num_stages=num_stages,
                annotations={'enable_offset': True},
            ):
                T.copy(x[pid_token * block_m, pid_hidden * block_k], x_ub)
                collect_amax(x_ub, amax_ub)
                compute_sf(amax_ub, sf_ub, sf_inv_ub)
                if num_per_channels == 16:
                    cast_and_cast_back_per16(x_ub, sf_ub, sf_inv_ub, out_ub)
                else:
                    cast_and_cast_back_per32(x_ub, sf_ub, sf_inv_ub, out_ub)
                T.copy(out_ub, x[pid_token * block_m, pid_hidden * block_k])

    return per_token_cast_and_cast_back_kernel
