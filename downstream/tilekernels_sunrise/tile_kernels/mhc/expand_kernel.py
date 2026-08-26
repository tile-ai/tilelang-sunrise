import tilelang
import torch
from tilelang import language as T

# Module-level hardware constants (overridable, no runtime device query).
_S2_SMEM_BUDGET_BYTES = 64 * 1024  # single-buffer staging within the resource budget
_BLK_H_CANDIDATES = (256, 128, 64, 32, 16, 8)  # 256=512B rows saturate the 128B transaction
_BLK_N_CANDIDATES = (128, 64, 32, 16, 8, 4, 2, 1)  # includes 1 so the fast path always hits
_MIN_BLK_N = 8  # narrower token tiles only lose vectorization, never help this copy
# When hidden spans only a few blk_h tiles (grid_j <= _WIDE_HIDDEN_GRID_J) blk_n
# stays minimal: a few wide rows already keep the SMs busy and the token grid is
# short. For hidden that spans many blk_h tiles (grid_j large), the grid can grow
# huge along j, so blk_n grows with n to cap the token-dimension block count at
# _MAX_TOKEN_BLOCKS, avoiding the scheduling overhead of an enormous grid.
_WIDE_HIDDEN_GRID_J = 16
_MAX_TOKEN_BLOCKS = 64


def choose_expand_blocks(hidden: int, mhc_mult: int,
                         n_tokens: int) -> tuple[int, int]:
    """Analytically pick (blk_n, blk_h) for expand_to_mhc, compile-time only.

    ``blk_h`` is the widest divisor of ``hidden`` (for vectorized T.copy). ``blk_n``
    is kept at ``_MIN_BLK_N`` unless ``hidden`` spans many blk_h tiles (grid_j
    large), in which case it grows with ``n_tokens`` to hold the token-block count
    near ``_MAX_TOKEN_BLOCKS``. This keeps the distinct JIT key count tiny (blk_n
    only depends on a coarse grid_j split, not on every n) while keeping SMs
    saturated. The result always divides ``n_tokens`` so the aligned fast path is
    hit, and the staging buffer fits: ``blk_n * blk_h * 2 <= _S2_SMEM_BUDGET_BYTES``.

    For this memory-bound copy, keep the smallest sensible ``blk_n`` except when
    large hidden dimensions and many tokens require a wider tile to control the
    grid. ``mhc_mult`` does not steer the choice and is accepted only for symmetry
    with the kernel factory.
    """
    # blk_h: widest candidate dividing hidden (engram _choose_blk_d style);
    # a non-dividing hidden (odd/prime) falls back via the caller's alignment gate.
    blk_h = next((b for b in _BLK_H_CANDIDATES if hidden % b == 0),
                 min(hidden, _BLK_H_CANDIDATES[0]))
    grid_j = -(-hidden // blk_h)  # ceildiv

    if n_tokens <= 0:
        return 1, blk_h

    if grid_j <= _WIDE_HIDDEN_GRID_J:
        # Few blk_h tiles -> keep blk_n minimal; token grid is already short.
        target = _MIN_BLK_N
    else:
        # Many blk_h tiles -> grow blk_n so the token-block count stays bounded.
        target = max(_MIN_BLK_N, n_tokens // _MAX_TOKEN_BLOCKS)

    blk_n_cap = min(_S2_SMEM_BUDGET_BYTES // (blk_h * 2), _BLK_N_CANDIDATES[0])
    target = min(target, blk_n_cap)

    # Largest candidate that is <= target and divides n_tokens (fast path);
    # 1 always divides, so this never fails.
    blk_n = next((b for b in _BLK_N_CANDIDATES if b <= target and n_tokens % b == 0),
                 1)
    return blk_n, blk_h


@tilelang.jit
def _expand_to_mhc_fwd_fallback(hidden: int, mhc_mult: int) -> tilelang.JITKernel:
    """Original element-wise fwd, used when n/h are not blk-aligned."""
    n = T.dynamic('num_tokens')
    h = hidden
    mhc = mhc_mult

    blk_n = 32
    blk_h = 128

    @T.prim_func
    def _kernel(
        x: T.Tensor[(n, h), T.bfloat16],
        o: T.Tensor[(n, mhc, h), T.bfloat16],
    ) -> None:
        with T.Kernel(T.ceildiv(n, blk_n), T.ceildiv(h, blk_h)) as (pid_i, pid_j):
            if n > 0:
                xl = T.alloc_fragment((blk_n, blk_h), T.bfloat16)
                T.copy(x[pid_i * blk_n, pid_j * blk_h], xl)
                for m in T.serial(mhc):
                    for ti, tj in T.Parallel(blk_n, blk_h):
                        i = pid_i * blk_n + ti
                        j = pid_j * blk_h + tj
                        if i < n and j < h:
                            o[i, m, j] = xl[ti, tj]

    return _kernel


@tilelang.jit
def expand_to_mhc_fwd_tl(hidden: int, mhc_mult: int,
                         blk_n: int, blk_h: int) -> tilelang.JITKernel:
    """Vectorized fwd kernel using shared-memory T.copy.

    Requires num_tokens and hidden to be multiples of blk_n/blk_h; the caller
    dispatches to ``_expand_to_mhc_fwd_fallback`` otherwise. ``blk_n``/``blk_h``
    come from ``choose_expand_blocks``.
    """
    n = T.dynamic('num_tokens')
    h = hidden
    mhc = mhc_mult

    @T.prim_func
    def expand_to_mhc_fwd_kernel(
        x: T.Tensor[(n, h), T.bfloat16],
        o: T.Tensor[(n, mhc * h), T.bfloat16],
    ) -> None:
        with T.Kernel(T.ceildiv(n, blk_n), T.ceildiv(h, blk_h)) as (pid_i, pid_j):
            if n > 0:
                xs = T.alloc_shared((blk_n, blk_h), T.bfloat16)
                T.copy(x[pid_i * blk_n, pid_j * blk_h], xs)
                T.assume(n % blk_n == 0)
                T.assume(h % blk_h == 0)
                for m_idx in T.unroll(mhc):
                    T.copy(xs, o[pid_i * blk_n, m_idx * h + pid_j * blk_h],
                           disable_tma=True)

    return expand_to_mhc_fwd_kernel


@tilelang.jit
def expand_to_mhc_bwd_tl(hidden: int, mhc_mult: int) -> tilelang.JITKernel:
    n = T.dynamic('num_tokens')
    h = hidden
    mhc = mhc_mult

    blk_n = 32
    blk_h = 128

    @T.prim_func
    def expand_to_mhc_bwd_kernel(
        o_grad: T.Tensor[(n, mhc, h), T.bfloat16],
        x_grad: T.Tensor[(n, h), T.bfloat16],
    ) -> None:
        with T.Kernel(T.ceildiv(n, blk_n), T.ceildiv(h, blk_h)) as (pid_i, pid_j):
            if n > 0:
                xgl = T.alloc_fragment((blk_n, blk_h), T.float32)
                T.fill(xgl, 0)
                for m in T.serial(mhc):
                    for ti, tj in T.Parallel(blk_n, blk_h):
                        i = pid_i * blk_n + ti
                        j = pid_j * blk_h + tj
                        if i < n and j < h:
                            xgl[ti, tj] += o_grad[i, m, j]
                T.copy(xgl, x_grad[pid_i * blk_n, pid_j * blk_h])

    return expand_to_mhc_bwd_kernel
