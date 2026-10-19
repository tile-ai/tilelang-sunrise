"""TANG GEMM dialect built on the common dense and block-scaled protocols."""

from __future__ import annotations

from tvm import tirx

from tilelang._typing import BarrierType, BufferLikeType
from tilelang.language.gemm_op import GemmWarpPolicy, _gemm_blockscaled_impl, _gemm_impl, _legalize_buffer_arg
from tilelang.language.utils import _normalize_annotations, buffer_region_to_tile_region
from tilelang.utils.language import to_buffer_region

__all__ = ["gemm", "gemm_blockscaled"]


def gemm(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    mbar: BarrierType | None = None,
    annotations: dict | None = None,
    *,
    k_step: int = 1,
    a_local_load_type: str = "load_overlap_mma",
    b_local_load_type: str = "load_overlap_mma",
    wc_interleave: int = 0,
    scale_a: BufferLikeType | None = None,
    scale_b: BufferLikeType | None = None,
    scale_tmem: BufferLikeType | None = None,
    scale_vec: int = 1,
    scale_format: int = 1,
    scale_block: int = 32,
    a_format: int = -1,
    b_format: int = -1,
) -> tirx.PrimExpr:
    """Synchronous TANG GEMM with STCU scheduling and optional scale staging.

    Scale operands use the dedicated block-scaled operator. New callers may
    invoke ``gemm_blockscaled`` directly with the common SFA/SFB contract.
    ``wc_interleave`` and the local-load type arguments remain accepted for
    compatibility; the STCU template owns its fixed load/MMA interleaving.
    """
    if not isinstance(k_step, int) or isinstance(k_step, bool) or k_step <= 0:
        raise ValueError(f"k_step must be a positive integer, but got {k_step!r}")
    valid_load_types = {"load_overlap_mma", "load_before_mma"}
    for name, value in (("a_local_load_type", a_local_load_type), ("b_local_load_type", b_local_load_type)):
        if value not in valid_load_types:
            raise ValueError(f"Invalid {name}: {value!r}")
    # These legacy arguments never entered the emitted GEMM protocol.
    del wc_interleave
    ann = _normalize_annotations(annotations)
    ann.update(
        {
            "tang_k_step": k_step,
            "tang_a_format": int(a_format),
            "tang_b_format": int(b_format),
        }
    )
    if scale_a is not None or scale_b is not None or scale_tmem is not None:
        if scale_a is None or scale_b is None or scale_tmem is None:
            raise ValueError("scale_a, scale_b and scale_tmem must all be provided for a TANG block-scaled GEMM")
        return gemm_blockscaled(
            A,
            B,
            C,
            scale_a,
            scale_b,
            transpose_A,
            transpose_B,
            policy,
            clear_accum,
            scale_tmem=scale_tmem,
            scale_vec=scale_vec,
            scale_format=scale_format,
            scale_block=scale_block,
            a_format=a_format,
            b_format=b_format,
            mbar=mbar,
            annotations=ann,
        )
    return _gemm_impl("tl.tileop.gemm", A, B, C, transpose_A, transpose_B, policy, clear_accum, mbar, annotations=ann)


def gemm_blockscaled(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    SFA: BufferLikeType,
    SFB: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    *,
    scale_tmem: BufferLikeType,
    k_start: int | tirx.PrimExpr = 0,
    sf_a_granularity_k: int | None = None,
    sf_b_granularity_k: int | None = None,
    scale_vec: int = 1,
    scale_format: int = 1,
    scale_block: int = 32,
    a_format: int = -1,
    b_format: int = -1,
    mbar: BarrierType | None = None,
    annotations: dict | None = None,
) -> tirx.PrimExpr:
    """Block-scaled TANG GEMM with a caller-owned scale staging workspace.

    SFA/SFB are shared-memory scale tiles already sliced to this K tile.
    ``scale_tmem`` is the TMEM workspace written by scale staging and read by
    the MMA. TANG currently requires ``k_start=0`` and equal A/B granularity.
    """
    workspace = to_buffer_region(_legalize_buffer_arg(scale_tmem))
    if workspace.buffer.scope() != "shared.tmem":
        raise ValueError("TANG block-scaled GEMM requires scale_tmem in shared.tmem")
    ann = _normalize_annotations(annotations)
    ann.update(
        {
            "tang_scale_vec": int({1: 1, 2: 2, 4: 3}.get(scale_vec, scale_vec)),
            "tang_scale_format": int(scale_format),
            "tang_scale_block": int(scale_block),
            "tang_a_format": int(a_format),
            "tang_b_format": int(b_format),
        }
    )
    call = _gemm_blockscaled_impl(
        "tl.tileop.gemm_blockscaled",
        A,
        B,
        C,
        SFA,
        SFB,
        transpose_A,
        transpose_B,
        policy,
        clear_accum,
        mbar,
        k_start=k_start,
        sf_a_granularity_k=scale_block if sf_a_granularity_k is None else sf_a_granularity_k,
        sf_b_granularity_k=scale_block if sf_b_granularity_k is None else sf_b_granularity_k,
        annotations=ann,
    )
    return tirx.call_intrin(
        "handle",
        call.op,
        *call.args,
        buffer_region_to_tile_region(workspace, "rw", [r.extent for r in workspace.region]),
        annotations=call.annotations,
    )
