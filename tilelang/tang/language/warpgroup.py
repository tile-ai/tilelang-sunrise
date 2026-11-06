"""TANG dialect ``WarpSpecialize``: CUDA version + stcuv2 single-warp (32) support.

The CUDA dialect's ``T.ws`` always narrows ``thread_bounds`` by a fixed
128-thread warpgroup.  On TANG stcuv2 the warp-specialized TMEM ``ldt``/``stt``
drain needs the bounds narrowed to a single 32-lane warp instead, so this module
shadows the CUDA symbol inside the TANG dialect with a version that takes
``warp_group_size`` (default 128 for backward compatibility). It is only
meaningful on ``tang -arch=stcuv2``.
"""

from tilelang import _ffi_api
from tilelang.cuda.language.warpgroup import WarpSpecializeFrame  # noqa: F401
from tilelang.language.kernel import get_thread_bindings, get_thread_extents

__all__ = ["WarpSpecialize", "ws"]


def _check_groups_fit_the_block(warp_group_ids: list[int], warp_group_size: int, num_threads: int) -> None:
    """Refuse a warp group whose threads the kernel does not launch."""
    if warp_group_size <= 0:
        raise ValueError(f"T.ws needs a positive warp_group_size, got {warp_group_size}")
    for warp_group_id in warp_group_ids:
        if warp_group_id == -1:  # documented escape hatch: no threadIdx.x binding
            continue
        if warp_group_id < 0:
            raise ValueError(f"T.ws takes a non-negative warp group index (or -1 to skip), got {warp_group_id}")
        start = warp_group_id * warp_group_size
        if start + warp_group_size > num_threads:
            raise ValueError(
                f"T.ws({warp_group_id}, warp_group_size={warp_group_size}) covers threads "
                f"[{start}, {start + warp_group_size}), but the kernel launches only {num_threads}; "
                f"this block holds {num_threads // warp_group_size} such group(s), so the index must be "
                f"< {num_threads // warp_group_size}"
            )


def WarpSpecialize(*warp_group_idx, warp_group_size: int = 128) -> WarpSpecializeFrame:
    """Tools to construct a warp group frame (TANG dialect variant).

    Parameters
    ----------
    warp_group_idx : int
        A integer representing warp group index
        Or a list of integers representing blockDim.(x|y|z)
        if the value is -1, we skip the threadIdx.x binding.
    warp_group_size : int
        Number of threads per specialized group (keyword-only). Defaults to 128
        (warpgroup). Pass ``32`` for single-warp specialization — this narrows
        ``thread_bounds`` to one 32-lane warp, which is required for correct
        warp-specialized TMEM ldt/stt drains on TANG stcuv2.

    Returns
    -------
    res : Tuple[frame.LaunchThreadFrame]
        The result LaunchThreadFrame.
    Examples:
        >>> T.ws(0) -> if tx < 128
        >>> T.ws(1) -> if tx >= 128 and tx < 256
        >>> T.ws(0, 1) -> if tx < 128 or (tx >= 128 and tx < 256)
        >>> T.ws(2, warp_group_size=32) -> if tx >= 64 and tx < 96
    """
    id_x, id_y, id_z = get_thread_bindings()
    ex_x, ex_y, ex_z = get_thread_extents()
    tid = id_x
    if ex_y > 1:
        tid = id_y * ex_x + tid
    if ex_z > 1:
        tid = id_z * (ex_y * ex_x) + tid

    warp_group_ids: list[int] = []
    for warp_group_id in warp_group_idx:
        warp_group_ids.append(warp_group_id)

    assert len(warp_group_ids) > 0, "warp_group_idx must be non-empty"
    _check_groups_fit_the_block(warp_group_ids, warp_group_size, ex_x * ex_y * ex_z)

    return _ffi_api.WarpSpecialize(warp_group_ids, tid, warp_group_size)


# Alias for WarpSpecialize for more concise usage
ws = WarpSpecialize
