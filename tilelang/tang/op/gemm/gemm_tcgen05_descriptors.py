"""TANG stcuv2 tcgen5 MMA descriptor parameters (Python-side).

Mirrors the constexpr helpers in ``src/tl_templates/tang/gemm_tcgen05.h`` and the
enum values in ``cccl/tang/__ptx/instructions/tc_mma.h``, so the Python emitter
can compute the descriptor template arguments at lowering time and hand them to
the ``tang_tcgen05_mma_ss`` / ``tang_tcgen05_mma_ts`` builtins.

Note: these are *not* bit-packed integers. The STCU backend reads the
``mma_data_desc<>`` / ``mma_desc<>`` marker intrinsics structurally, so the
codegen emits those calls with the constants below as template arguments.
"""

from __future__ import annotations

# tang::ptx::SwizzleMode (cccl/tang/__ptx/instructions/tc_mma.h)
# swizzle_mode_value = swizzle_base(sw_bytes) + atom_offset(atom_bytes)
#   sw_bytes: 32 -> 4, 64 -> 8, 128 -> 12
#   atom_bytes: 8 -> 0, 16 -> 1, 32 -> 2, 64 -> 3
SWIZZLE_NONE = 0
SWIZZLE_SW32_A8 = 4
SWIZZLE_SW32_A16 = 5
SWIZZLE_SW64_A8 = 8
SWIZZLE_SW64_A16 = 9
SWIZZLE_SW64_A32 = 10
SWIZZLE_SW128_A8 = 12
SWIZZLE_SW128_A16 = 13
SWIZZLE_SW128_A32 = 14
SWIZZLE_SW128_A64 = 15

# tang::ptx::TmemType (accumulator)
TMEM_S32 = 0
TMEM_FP32 = 1
TMEM_FP16 = 2
TMEM_BF16 = 3

# tang::ptx::EleType (A/B operands)
ELE_S8 = 0
ELE_U8 = 1
ELE_FP16 = 2
ELE_BF16 = 3
ELE_TF32 = 4
ELE_FP4_E2M1 = 5
ELE_NVFP4_E2M1 = 6
ELE_FP8_E4M3 = 8
ELE_FP8_E5M2 = 9
ELE_FP6_E2M3 = 10
ELE_FP6_E3M2 = 11
ELE_FP4_E2M1_MIX = 12


def tcgen5_operand_swizzle(elem_bytes: int, mn_major: bool) -> int:
    raise NotImplementedError("STCUV2 GEMM descriptor construction is not supported")


def tcgen5_sbo(lbo: int, sw: int) -> int:
    raise NotImplementedError("STCUV2 GEMM descriptor construction is not supported")


def is_unpacked_fp4(dtype) -> bool:
    """True for ``float4_e2m1_unpacked`` (``custom[float4_e2m1_unpacked]8``).

    This is the mxf8f6f4 storage layout for an fp4 operand: one e2m1 code in the
    low nibble of its own byte, as opposed to the 2-per-byte packing mxf4/nvfp4
    read. Matched on the dtype string because operand dtypes reach the emitter
    as both DataType and str.
    """
    return "float4_e2m1_unpacked" in str(dtype)


def is_any_fp4(dtype) -> bool:
    """True for either fp4 layout, packed (``float4_e2m1fn``) or unpacked."""
    return "float4_e2m1" in str(dtype)


def ele_code(dtype: str) -> int:
    """Map a TileLang operand dtype to tang::ptx::EleType."""
    name = str(dtype)
    # Unpacked fp4 is only legal mixed with a non-fp4 operand (the mxf8f6f4
    # kind), which on this unscaled path is whatever the other operand is; the
    # kind-vs-layout agreement is checked by the block-scaled emitter, which is
    # the only one that knows both operands. Tested before the fp8/fp6 names
    # because "custom[float4_e2m1_unpacked]8" is a substring-matched string too.
    if is_unpacked_fp4(name):
        return ELE_FP4_E2M1_MIX
    if "float8_e4m3" in name:
        return ELE_FP8_E4M3
    if "float8_e5m2" in name:
        return ELE_FP8_E5M2
    if "float6_e2m3" in name:
        return ELE_FP6_E2M3
    if "float6_e3m2" in name:
        return ELE_FP6_E3M2
    # bfloat16 must be checked before float16: "bfloat16" contains "float16".
    if "bfloat16" in name:
        return ELE_BF16
    if "float16" in name:
        return ELE_FP16
    if "float32" in name:
        return ELE_TF32
    if "int8" in name:
        return ELE_S8
    if "uint8" in name:
        return ELE_U8
    raise ValueError(f"unsupported TANG tcgen5 operand dtype: {dtype}")


def tmem_code(dtype: str) -> int:
    """Map a TileLang accumulator dtype to tang::ptx::TmemType."""
    name = str(dtype)
    if "float32" in name:
        return TMEM_FP32
    # bfloat16 before float16 for the same substring reason.
    if "bfloat16" in name:
        return TMEM_BF16
    if "float16" in name:
        return TMEM_FP16
    if "int32" in name:
        return TMEM_S32
    raise ValueError(f"unsupported TANG tcgen5 accumulator dtype: {dtype}")
