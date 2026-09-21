"""Explicit TCGEN05 (5th-gen tensor core) primitives — stcuv2 only.

These wrap the TANG stcuv2 backend intrinsics that already exist in the code
generator (see ``src/target/codegen_tang.cc``):

* ``tcgen05_ld`` / ``tcgen05_st`` — TMEM <-> register-fragment movement. They
  delegate to the layout-aware :func:`tilelang.language.copy` lowering, which on
  stcuv2 emits the warp-collective ``ldt_16x256b`` load (``tang_tmem_ld_16x256b``)
  and the ``stt_16x256b`` store (``tang_tmem_st_16x256b``). Both run on the first
  warp of the enclosing ``thread_bounds``; wrap the drain in
  ``T.ws(warp, warp_group_size=32)`` (plus a ``tcgen05_sync_arrive`` /
  ``tcgen05_sync_wait`` handshake) to run them on a non-zero warp.
* ``tcgen05_cp`` / ``tang_cp_tmem_to_shared`` — shared <-> TMEM movement through
  the cps2t / cpt2s copy engines, bypassing the register file. These are the
  *only* spellings: a bare ``T.copy`` between shared and TMEM is rejected,
  matching the CUDA backend, which has no ``T.copy`` path for these scopes
  either. The asymmetric naming is deliberate — ``tcgen05.cp`` is the same
  direction on SM100, while tmem->shared has no CUDA counterpart at all.
* ``tcgen05_sync_arrive`` / ``tcgen05_sync_wait`` — producer/consumer named-barrier
  handshake for cross-warp TMEM handoff. Map to ``sync_arrive`` / ``sync_wait``.
* ``tcgen05_before_thread_sync`` / ``tcgen05_after_thread_sync`` — tensor-core
  ordering fences around a thread-level sync. Both map to
  ``tang::ptx::fence_tc<fence_group>()`` (``tl.tang_fence_tc``); TANG exposes a
  single TC fence that serves both the before- and after-sync roles.

All of these are only meaningful on ``tang -arch=stcuv2``; using them on another
target fails during code generation (no handler for the TANG intrinsic).
"""

from __future__ import annotations

from tilelang._typing import BufferLikeType
from tilelang.utils.language import to_buffer_region
from tilelang.language.copy_op import copy as _copy
from tvm import tirx as tir

_FENCE_TC_OP = "tl.tang_fence_tc"
_SYNC_ARRIVE_OP = "tl.tang_sync_arrive"
_SYNC_WAIT_OP = "tl.tang_sync_wait"


def _scope_of(buf: BufferLikeType) -> str:
    return to_buffer_region(buf).buffer.scope()


def tcgen05_ld(dst: BufferLikeType, src: BufferLikeType) -> tir.PrimExpr | tir.Stmt:
    """Load a tile from tensor memory (TMEM) into a register fragment (stcuv2).

    Args:
        dst: destination register fragment (``T.alloc_fragment``).
        src: source TMEM buffer (``T.alloc_tmem``, scope ``shared.tmem``).

    Delegates to the layout-aware copy lowering, which emits the warp-collective
    ``ldt_16x256b`` (``tang_tmem_ld_16x256b``) TMEM->register load on stcuv2.

    The ldt runs on the first warp of the enclosing ``thread_bounds``. To run it
    on a non-zero warp (warp specialization), wrap the whole drain -- this call
    plus its fragment consumer and a ``tcgen05_sync_wait`` -- in
    ``T.ws(warp, warp_group_size=32)``; that narrows ``thread_bounds`` to the
    chosen warp for both the ldt and its consumer.
    """
    s = _scope_of(src)
    assert s == "shared.tmem", (
        f"tcgen05_ld source must be a TMEM buffer (scope 'shared.tmem', allocate via T.alloc_tmem), but got scope '{s}'."
    )
    return _copy(src, dst)


def tcgen05_st(dst: BufferLikeType, src: BufferLikeType) -> tir.PrimExpr | tir.Stmt:
    """Store a register fragment into tensor memory (TMEM) (stcuv2).

    Args:
        dst: destination TMEM buffer (``T.alloc_tmem``, scope ``shared.tmem``).
        src: source register fragment (``T.alloc_fragment``).

    Delegates to the copy lowering, which emits ``stt_16x256b``
    (``tang_tmem_st_16x256b``) register->TMEM store on stcuv2. Runs on the first
    warp of the enclosing ``thread_bounds``; wrap in
    ``T.ws(warp, warp_group_size=32)`` for warp specialization (see
    :func:`tcgen05_ld`).
    """
    s = _scope_of(dst)
    assert s == "shared.tmem", (
        f"tcgen05_st destination must be a TMEM buffer (scope 'shared.tmem', allocate via T.alloc_tmem), but got scope '{s}'."
    )
    return _copy(src, dst)


_TMEM_SCOPE = "shared.tmem"
_SHARED_SCOPES = ("shared", "shared.dyn")

# Marker read by src/tang/op/copy.cc. Both shared <-> TMEM directions are
# named-primitive only: a bare T.copy() between these scopes is rejected. The
# contracts these copy engines impose (source swizzle mode, row/column
# granularity, single-warp execution, a full barrier) are invisible at a
# T.copy call site, and the CUDA backend has no T.copy path for these scopes
# either -- there `T.tcgen05_cp_warpx4` is likewise the only entry point.
_TCGEN05_CP_MARKER = {"tang_tcgen05_cp": 1}


def _reject_scopes(name: str, dst: BufferLikeType, src: BufferLikeType, want_dst: str, want_src: str) -> None:
    """Raise unless (dst, src) sit in the scopes this copy engine needs."""
    d, s = _scope_of(dst), _scope_of(src)
    ok_dst = d == want_dst if isinstance(want_dst, str) else d in want_dst
    ok_src = s == want_src if isinstance(want_src, str) else s in want_src
    if ok_dst and ok_src:
        return
    hint = ""
    if _TMEM_SCOPE not in (d, s):
        hint = " Neither side is tensor memory; plain T.copy handles this."
    elif d == s == _TMEM_SCOPE:
        hint = " Both sides are tensor memory; there is no TMEM->TMEM copy."
    elif "local.fragment" in (d, s):
        hint = " A fragment side belongs to T.tcgen05_ld (TMEM->fragment) or T.tcgen05_st (fragment->TMEM)."
    elif "global" in (d, s):
        hint = " A global side belongs to T.copy: the fused TMEM->global drain stays on T.copy."
    raise AssertionError(f"{name}(dst, src) needs dst scope '{want_dst}' and src scope '{want_src}', but got dst='{d}', src='{s}'.{hint}")


def tcgen05_cp(dst: BufferLikeType, src: BufferLikeType) -> tir.PrimExpr | tir.Stmt:
    """Copy a tile from shared memory into tensor memory (stcuv2 cps2t).

    Args:
        dst: destination TMEM buffer (``T.alloc_tmem``, scope ``shared.tmem``).
        src: source shared-memory buffer.

    This is the shared->TMEM copy engine, used to stage an MMA **A operand**
    without a register-file round trip. It is the only supported spelling; a
    bare ``T.copy(shared, tmem)`` is rejected by the lowering.

    Named after PTX ``tcgen05.cp``, which is the same direction on SM100. The
    CUDA backend spells its (scale-factor-specific) variant
    ``T.tcgen05_cp_warpx4``.

    The lowering enforces the copy engine's contract and reports violations at
    compile time: the source must have been filled by a sw128a32 bulk copy, its
    rows must be a multiple of 32, and its row pitch a whole number of 128-byte
    swizzle units. See ``docs/s3_tmem_shared_cpt2s_cps2t_layout.md`` §6.2.
    """
    _reject_scopes("tcgen05_cp", dst, src, _TMEM_SCOPE, _SHARED_SCOPES)
    return _copy(src, dst, annotations=dict(_TCGEN05_CP_MARKER))


def tang_cp_tmem_to_shared(dst: BufferLikeType, src: BufferLikeType) -> tir.PrimExpr | tir.Stmt:
    """Copy a tile from tensor memory into shared memory (stcuv2 cpt2s).

    Args:
        dst: destination shared-memory buffer.
        src: source TMEM buffer (``T.alloc_tmem``, scope ``shared.tmem``).

    Staging an accumulator through shared memory keeps it out of the register
    file, which is what makes large drain tiles fit. A bare
    ``T.copy(tmem, shared)`` is rejected; the fused ``T.copy(tmem, global)``
    drain is unaffected and remains the right spelling for that.

    Carries the ``tang_`` prefix because this direction has **no CUDA
    counterpart**: tcgen05 has no TMEM->shared instruction, and on Blackwell
    data leaves TMEM only through registers (``tcgen05.ld``). See
    ``docs/s3_tmem_shared_cpt2s_cps2t_layout.md`` §6.3.

    The lowering requires a 32-bit shared tile of exactly 128 bytes per row and
    a row count that is a multiple of 64. Note the follow-up shared->global copy
    must be tagged ``annotations={"tang_swizzle_atom_bytes": 8}`` to match the
    swizzle cpt2s writes -- see §6.1, this is a known footgun.
    """
    _reject_scopes("tang_cp_tmem_to_shared", dst, src, _SHARED_SCOPES, _TMEM_SCOPE)
    return _copy(src, dst, annotations=dict(_TCGEN05_CP_MARKER))


def tcgen05_before_thread_sync(fence_group: int = 0) -> tir.PrimExpr:
    """Tensor-core fence issued *before* a thread-level sync (stcuv2).

    Orders prior tcgen05 (tensor-core / MMA) async effects ahead of a following
    barrier, so their results are visible after the sync. Maps to
    ``tang::ptx::fence_tc<fence_group>()``.

    Args:
        fence_group: TC fence group id (0 or 1); defaults to 0.
    """
    assert fence_group in (0, 1), "fence_group must be 0 or 1 (fence_tc)"
    return tir.call_intrin("handle", tir.op.Op.get(_FENCE_TC_OP), tir.const(fence_group, "int32"))


def tcgen05_after_thread_sync(fence_group: int = 0) -> tir.PrimExpr:
    """Tensor-core fence issued *after* a thread-level sync (stcuv2).

    Orders subsequent tcgen05 async effects behind a preceding barrier. TANG
    exposes a single TC fence for both roles, so this also maps to
    ``tang::ptx::fence_tc<fence_group>()``.

    Args:
        fence_group: TC fence group id (0 or 1); defaults to 0.
    """
    assert fence_group in (0, 1), "fence_group must be 0 or 1 (fence_tc)"
    return tir.call_intrin("handle", tir.op.Op.get(_FENCE_TC_OP), tir.const(fence_group, "int32"))


def tcgen05_sync_arrive(barrier_id: int = 0) -> tir.PrimExpr:
    """Signal arrival on a TANG named sync barrier (stcuv2).

    Producer→consumer TMEM handoff primitive: the warp that produced TMEM data
    (e.g. after an MMA + ``tcgen05_before_thread_sync`` fence, or after a
    ``tcgen05_st`` + its fence) calls this to signal a consumer/drain warp.
    Maps to ``sync_arrive(barrier_id)``. Typically issued inside a warp guard
    (e.g. ``if T.get_thread_binding() // 32 == producer_warp``).

    Args:
        barrier_id: named barrier id.
    """
    return tir.call_intrin("handle", tir.op.Op.get(_SYNC_ARRIVE_OP), tir.const(barrier_id, "int32"))


def tcgen05_sync_wait(barrier_id: int = 0, producer_count: int = 1, consumer_count: int = 1) -> tir.PrimExpr:
    """Wait on a TANG named sync barrier before reading TMEM (stcuv2).

    Consumer/drain side of the TMEM handoff: the drain warp calls this before
    its first ``tcgen05_ld`` so it observes the producer's committed TMEM.
    Maps to ``sync_wait(barrier_id, producer_count, consumer_count)``. Typically
    issued inside a warp guard for the drain warp.

    Args:
        barrier_id: named barrier id (must match the producer's arrive).
        producer_count: number of producer arrivals to wait for.
        consumer_count: number of consumer warps waiting on this barrier.
    """
    return tir.call_intrin(
        "handle",
        tir.op.Op.get(_SYNC_WAIT_OP),
        tir.const(barrier_id, "int32"),
        tir.const(producer_count, "int32"),
        tir.const(consumer_count, "int32"),
    )
