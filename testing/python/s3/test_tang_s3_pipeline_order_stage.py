"""Manual ``order``/``stage`` schedules use the same handshake as ``num_stages``.

Manual schedules publish the same typed pipeline plan as automatic schedules;
``InjectS3AsyncCopy`` does not reconstruct synchronization from addresses.
These tests verify:

* equivalent manual and automatic schedules lower to the same source;
* the pipeline loop carries no block rendezvous, and both edges are on
  mbarriers (``arrive_on_g2s_fence`` publishes a version, ``arrive_on_tc_fence``
  retires it);
* the depth follows ``stage``: ``max(stage) + 1`` versions per buffer, which
  shows up in the shared-memory footprint and in the number of barrier slots;
* a schedule that rotates nothing at all (``stage=[0, 0, 0]``) keeps the
  blocking copy. The reconstruction must not turn "no rotation" into one.
* every one of the six emission orders, crossed with depth 2, 3 and 4, lowers to
  the shape it earns. ``order`` is not a no-op:
  a permutation moves the GEMM within the emitted body and with it the amount of
  overlap, so the whole cross product is swept rather than one representative.

Every compilation asks for the stcuv2 target explicitly.
"""

import itertools
import re

import pytest

import tilelang
import tilelang.tang.language as T
from tilelang import tvm

STCUV2_TARGET = {"kind": "tang", "arch": "stcuv2"}

M = N = 256
K = 512
BM = BN = 128
BK = 64  # <= the 128 B MMA operand row, so the M1b segmentation does not kick in
THREADS = 128
WARP = 32
MMA_WARP = 2

# One schedule per way of stating it. ``depth`` is ``max(stage) + 1``, the same
# number ``num_stages`` would state.
SCHEDULES = {
    "auto-2": dict(num_stages=2, depth=2),
    "manual-2": dict(order=[0, 1, 2], stage=[0, 0, 1], depth=2),
    "manual-3": dict(order=[0, 1, 2], stage=[0, 0, 2], depth=3),
    "manual-reordered": dict(order=[1, 2, 0], stage=[0, 0, 1], depth=2),
    "manual-flat": dict(order=[0, 1, 2], stage=[0, 0, 0], depth=1),
}

# ---------------------------------------------------------------------------
# The whole order x stage matrix
# ---------------------------------------------------------------------------
#
# ``stage`` is indexed by *statement*, not by position: every order below states
# the same thing -- ``T.copy`` into A is statement 0, ``T.copy`` into B is 1, the
# GEMM is 2, and the stage list puts both copies in stage 0 and the MMA in stage
# ``max``. What a permutation changes is the sequence the peeled loop body emits
# them in, and that sequence decides how much of the rotation survives:
#
# * a copy emitted *before* the GEMM has to be double buffered -- it fills the
#   next version while the MMA reads the current one -- so it gets an mbarrier
#   edge and ``max(stage) + 1`` versions;
# * a copy emitted *after* the GEMM may reuse the version the MMA just retired,
#   under the block rendezvous that is still in the body, so it keeps its
#   blocking drain and one version fewer.
#
# Three shapes come out (see ``SHAPE_EXPECTATIONS``): every operand overlapped,
# one operand overlapped, or neither. All three compute the right answer; only
# the first one is the schedule the async form exists for. That is worth pinning
# per order rather than per class, because it is not obvious from the schedule
# which of the six permutations lands in which shape.
ORDER_STAGE_MATRIX = [(order, stage) for order in itertools.permutations(range(3)) for stage in ([0, 0, 1], [0, 0, 2], [0, 0, 3])]

MATRIX_IDS = [f"order_{''.join(map(str, order))}_stage_{''.join(map(str, stage))}" for order, stage in ORDER_STAGE_MATRIX]

# What each order earns today. Measured, not derived from the schedule: the two
# orders that put the GEMM last overlap both operands, and the shapes of the
# remaining four are not symmetric in a way this test can predict -- which is
# exactly why the mapping is asserted rather than computed.
SHAPE_OF_ORDER = {
    (0, 1, 2): "overlapped",  # A, B, GEMM: both copies run ahead of the MMA
    (1, 0, 2): "overlapped",  # B, A, GEMM: the same body, the two guards swap
    (0, 2, 1): "one-edge",  # A, GEMM, B: only A is filled ahead
    (2, 0, 1): "one-edge",  # GEMM, A, B: the same survivor as above
    (1, 2, 0): "barrier-only",  # B, GEMM, A: neither copy is filled ahead
    (2, 1, 0): "barrier-only",  # GEMM, B, A: same, with the operands swapped
}

# (dynamic shared KiB, mbarrier slots, in-loop __syncthreads, ready arrivals,
# retired arrivals) as a function of ``depth = max(stage) + 1``. The versions
# per buffer are the whole story: ``depth`` for an overlapped copy, ``depth - 1``
# for a blocking one, and two fp16 ``BM x BK`` tiles are 16 KiB each.
SHAPE_EXPECTATIONS = {
    "overlapped": lambda depth: (32 * depth, 4 * depth, 0, 2, 2),
    "one-edge": lambda depth: (16 * (2 * depth - 1), 2 * depth, 2, 1, 1),
    "barrier-only": lambda depth: (32 * (depth - 1), 0, 2, 0, 0),
}


def ws_gemm(order=None, stage=None, num_stages=None):
    """Role-split GEMM: two producer warps, one consumer warp, one idle warp.

    The shape the two producers make is what forces the handshake to be
    per-buffer: A and B are filled by different warps and both are read by a
    third, so each version needs its own ready slot and its own free slot.
    """

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((N, K), T.float16),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=THREADS) as (bx, by):
            A_s = T.alloc_shared((BM, BK), T.float16)
            B_s = T.alloc_shared((BN, BK), T.float16)
            C_t = T.alloc_tmem((BM, BN), T.float32)
            for k in T.Pipelined(K // BK, num_stages=num_stages, order=order, stage=stage):
                with T.ws(0, warp_group_size=WARP):
                    T.copy(A[by * BM, k * BK], A_s)
                with T.ws(1, warp_group_size=WARP):
                    T.copy(B[bx * BN, k * BK], B_s)
                with T.ws(MMA_WARP, warp_group_size=WARP):
                    T.gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=(k == 0))
            T.copy(C_t, C[by * BM, bx * BN])

    return kernel


def source(name: str) -> str:
    """Lower one schedule in this process and return its device source."""
    schedule = dict(SCHEDULES[name])
    del schedule["depth"]
    return tilelang.lower(ws_gemm(**schedule), target=STCUV2_TARGET).kernel_source


def _after_isp_snapshot(name: str):
    original = tilelang.transform.InjectSoftwarePipeline
    snapshots = []

    def spying_factory(*args, **kwargs):
        inner = original(*args, **kwargs)

        @tvm.transform.module_pass(opt_level=0)
        def wrapper(mod, ctx):
            out = inner(mod)
            snapshots.extend(func for _gvar, func in out.functions.items())
            return out

        return wrapper

    schedule = dict(SCHEDULES[name])
    del schedule["depth"]
    tilelang.transform.InjectSoftwarePipeline = spying_factory
    try:
        tilelang.lower(ws_gemm(**schedule), target=STCUV2_TARGET)
    finally:
        tilelang.transform.InjectSoftwarePipeline = original
    assert snapshots
    return next(func for func in snapshots if func.attrs is not None and str(func.attrs.get("global_symbol")) == "kernel")


def _before_s3_async_snapshot(name: str):
    original = tilelang.tang.transform.InjectS3AsyncCopy
    snapshots = []

    def spying_factory(*args, **kwargs):
        inner = original(*args, **kwargs)

        @tvm.transform.module_pass(opt_level=0)
        def wrapper(mod, ctx):
            snapshots.extend(func for _gvar, func in mod.functions.items())
            return inner(mod)

        return wrapper

    tilelang.tang.transform.InjectS3AsyncCopy = spying_factory
    try:
        source(name)
    finally:
        tilelang.tang.transform.InjectS3AsyncCopy = original
    assert snapshots
    return next(func for func in snapshots if func.attrs is not None and str(func.attrs.get("global_symbol")) == "kernel")


# ===========================================================================
# Source helpers
# ===========================================================================


def _pipeline_loop(src: str) -> str:
    """The text between the steady-state pipeline loop's braces.

    The loop is the one ISP emits after peeling the prologue, so it starts past
    the copies that seed versions 0..depth-2. Everything this file asserts about
    the handshake is a property of *this* body: a block rendezvous in it
    re-serialises producer and consumer for every tile.
    """
    start = src.index("for (int k = 0;")
    depth = 0
    for i in range(src.index("{", start), len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start : i + 1]
    raise AssertionError("unterminated pipeline loop")


def _shared_kib(src: str) -> int:
    """The dynamic shared-memory window the kernel maps, in KiB."""
    match = re.search(r"buf_dyn_shmem\[(\d+)\]", src)
    assert match, "no dynamic shared-memory window in the device source"
    return int(match.group(1)) // 1024


def _mbarrier_slots(src: str) -> int:
    """How many mbarrier slots the pipeline declares, or 0 for none.

    The encoding is ``1 + 2 * versions * buffers``: id 0 is unused, and each
    managed buffer version owns a ready slot with its free slot right after it
    (see ``PlanBarriers``). Asserting the number rather than "some mbarrier
    exists" is what makes a wrong version count visible -- a schedule that
    silently dropped to one version would still carry a barrier.
    """
    match = re.search(r"mbarrier_mem\[(\d+)\]", src)
    return int(match.group(1)) if match else 0


# ===========================================================================
# The schedule is a way of stating the plan, not a different plan
# ===========================================================================


def test_the_manual_schedule_lowers_to_the_automatic_one():
    """``order``/``stage`` must produce the same kernel as ``num_stages``.

    This is the claim the whole file rests on: stcuv2 has one pipeline
    protocol, and the two ways of stating a schedule are two ways of stating
    the same one. If they diverged, every assertion below would be asserting
    the behaviour of a second protocol nobody designed.
    """
    auto = source("auto-2")
    manual = source("manual-2")
    # The kernels differ only in the symbol their prim_func is named after.
    normalize = lambda src: re.sub(r"\b\w*_kernel\b", "KERNEL", src)  # noqa: E731
    assert normalize(auto) == normalize(manual)


def test_manual_schedule_publishes_the_same_typed_v2_plan():
    plans = {}
    for name in ("auto-2", "manual-2"):
        found = []

        def visit(node, found=found):
            if isinstance(node, tvm.tirx.For) and "tl.pipeline_dependency_facts" in node.annotations:
                found.extend(node.annotations["tl.pipeline_dependency_facts"].values())

        tvm.tirx.stmt_functor.post_order_visit(_after_isp_snapshot(name).body, visit)
        plans[name] = sorted(tuple(int(value) for value in plan) for plan in found)

    assert plans["manual-2"]
    assert all(plan[0] == 2 and len(plan) == 8 and plan[7] == 0 for plan in plans["manual-2"])
    assert plans["manual-2"] == plans["auto-2"]


def test_manual_schedule_copy_records_carry_the_same_v2_plan():
    records = {}
    for name in ("auto-2", "manual-2"):
        found = []

        def visit(node, found=found):
            if isinstance(node, tvm.tirx.Call) and getattr(node.op, "name", "") == "tl.tang_bulk_async_commit":
                found.append(tuple(int(value) for value in node.args[7:13]))

        tvm.tirx.stmt_functor.post_order_visit(_before_s3_async_snapshot(name).body, visit)
        records[name] = sorted(found)

    assert records["manual-2"]
    assert [record[:-1] for record in records["manual-2"]] == [record[:-1] for record in records["auto-2"]]
    assert all(record[-1] in (0, 1) for record in records["manual-2"] + records["auto-2"])


# ===========================================================================
# Where the synchronisation sits
# ===========================================================================


@pytest.mark.parametrize("name", ["auto-2", "manual-2", "manual-3"])
def test_the_pipeline_loop_has_no_block_rendezvous(name):
    """Both edges are on mbarriers, so the loop body holds no ``__syncthreads``.

    A block-wide barrier in the steady-state body is exactly what the async
    form exists to remove: it would make the producers wait for the consumer's
    MMA and the consumer wait for the producers' DMA, which is the serialised
    schedule the rotation was supposed to overlap.
    """
    body = _pipeline_loop(source(name))
    assert "__syncthreads" not in body, body
    # The blocking form's other tell: the copy drains into a fence group
    # instead of publishing into its version's ready slot.
    assert "fence_mem" not in body, body
    # Ready edge: each producer's copy publishes the version it filled.
    assert body.count("arrive_on_g2s_fence") == 2, body
    # Free edge: the consumer retires both versions it read.
    assert body.count("arrive_on_tc_fence") == 2, body


def test_the_loop_waits_for_the_versions_it_reads():
    """The consumer waits at the top of the body, the producer before reuse.

    Stated separately from the counts above because the *placement* is what the
    parities depend on: the ready wait must precede the MMA that reads the
    version, and the free wait must precede the copy that overwrites it.
    """
    body = _pipeline_loop(source("manual-2"))
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    mma = next(i for i, line in enumerate(lines) if "tcgen05_mma" in line)
    copy = next(i for i, line in enumerate(lines) if "fcpg2s" in line)
    waits = [i for i, line in enumerate(lines) if ".wait(" in line]
    assert any(i < copy for i in waits), body  # free wait, before the overwrite
    assert any(i < mma for i in waits), body  # ready wait, before the read


# ===========================================================================
# The depth the schedule states
# ===========================================================================


@pytest.mark.parametrize("name", ["manual-2", "manual-3"])
def test_stage_sets_the_buffer_depth(name):
    """``max(stage) + 1`` versions, visible in shared memory and in the slots.

    The depth is the one thing a manually scheduled kernel cannot get wrong
    quietly: too few versions and the producer overwrites a version the consumer
    is still reading, too many and the pipeline prefetches tiles it will never
    use. Both are readable from the lowered kernel, so both are asserted.
    """
    depth = SCHEDULES[name]["depth"]
    src = source(name)
    # Two operands, each BM x BK fp16 per version.
    per_version_kib = 2 * BM * BK * 2 // 1024
    assert _shared_kib(src) == depth * per_version_kib
    # One ready slot and one free slot per version per operand.
    assert _mbarrier_slots(src) == 2 * depth * 2


def test_a_schedule_without_a_rotation_keeps_the_blocking_copy():
    """``stage=[0, 0, 0]`` stages nothing, so there is no handshake to build.

    With every statement in stage 0, ISP emits no v2 dependency plan. Inventing
    one would wire a wait nobody arrives on, so the copy keeps the blocking
    form it had.
    """
    src = source("manual-flat")
    assert "tl.pipeline_dependency_facts" not in _after_isp_snapshot("manual-flat").script()
    assert _mbarrier_slots(src) == 0
    assert _mbarrier_slots(source("manual-2")) > 0  # the contrast, not a tautology
    body = _pipeline_loop(src)
    assert "__syncthreads" in body


# ===========================================================================
# Every order, every depth
# ===========================================================================


def matrix_source(order, stage):
    """Lower one point of the matrix."""
    return tilelang.lower(ws_gemm(order=list(order), stage=list(stage)), target=STCUV2_TARGET).kernel_source


def test_the_order_decides_how_much_of_the_loop_overlaps():
    """Order is not a no-op, and this is the map of what it is worth.

    ``stage`` is indexed by statement, so all six permutations state the same
    stage assignment; what changes is where the GEMM sits in the emitted body.
    Measured result: only the orders that put the GEMM last overlap both
    operands. The other four keep a block rendezvous per iteration -- the
    version counts still grow with the depth, but producer and consumer are
    re-serialised every tile, which is the schedule the rotation was supposed to
    remove. The translation accepts them silently and the numbers are right, so
    nothing but this test would notice.

    Pinned as a mapping rather than a rule because the four non-overlapping
    orders do not split evenly in any way the schedule itself states: two keep
    one edge, two keep none.
    """
    for order in itertools.permutations(range(3)):
        assert (SHAPE_OF_ORDER[order] == "overlapped") == (order.index(2) == 2), order
    assert sorted(SHAPE_OF_ORDER.values()) == ["barrier-only"] * 2 + ["one-edge"] * 2 + ["overlapped"] * 2
    # Every permutation is covered exactly once, so a new order cannot be added
    # to the table without this failing.
    assert set(SHAPE_OF_ORDER) == set(itertools.permutations(range(3)))


@pytest.mark.parametrize("order, stage", ORDER_STAGE_MATRIX, ids=MATRIX_IDS)
def test_every_order_stage_combination_lowers_to_its_shape(order, stage):
    """What each of the eighteen combinations compiles to.

    Three facts per point: how many versions were allocated (shared memory), how
    many mbarrier slots the pipeline declared, and what the steady-state body
    does about synchronisation. Together they say whether the operand is
    overlapped (edge on an mbarrier, one version more, no block barrier) or
    serialised (blocking drain, fewer versions, block barrier in the body).
    """
    depth = max(stage) + 1
    shape = SHAPE_OF_ORDER[order]
    kib, slots, syncs, ready, retired = SHAPE_EXPECTATIONS[shape](depth)
    src = matrix_source(order, stage)
    assert _shared_kib(src) == kib
    assert _mbarrier_slots(src) == slots
    body = _pipeline_loop(src)
    assert body.count("__syncthreads") == syncs, body
    assert body.count("arrive_on_g2s_fence") == ready, body
    assert body.count("arrive_on_tc_fence") == retired, body
    # A serialised operand keeps the blocking form's drain; an overlapped one
    # never appears with it.
    assert (body.count("fence_mem") > 0) == (ready < 2), body
    # Nothing may wait on a barrier this schedule never arrives on.
    assert body.count(".wait(") == 2 * ready, body
