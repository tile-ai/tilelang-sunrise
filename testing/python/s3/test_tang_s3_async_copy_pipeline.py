"""Acceptance tests for the stcuv2 async software-pipeline copy commit.

"Overlap" is asserted **structurally**, on the generated device source:
pipelined bulk copies issue without ``fence_mem``, publish to the barrier
created by ``InjectSoftwarePipeline``, and the consumer waits that same barrier
without an in-loop ``__syncthreads``. A missing async form silently keeps the
pipeline serialized.

The ready edge is the shared ``pipeline_mbar`` created and waited by generic
``InjectSoftwarePipeline``. ``FinalizeS3PipelineInjection`` assigns its S3
arrival count, and copy lowering fulfils that declaration. The free edge
remains one TANG mbarrier per operand:
``arrive_on_tc_fence`` + a parity wait protects MMA -> overwrite.
"""

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


def _gemm_kernel(stages):
    """The fp16 TN pipeline shape used by these cases."""

    @T.prim_func
    def kernel(A: T.Tensor((M, K), T.float16), B: T.Tensor((N, K), T.float16), C: T.Tensor((M, N), T.float32)):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
            A_s = T.alloc_shared((BM, BK), T.float16)
            B_s = T.alloc_shared((BN, BK), T.float16)
            C_t = T.alloc_tmem((BM, BN), T.float32)
            for k in T.Pipelined(K // BK, num_stages=stages):
                T.copy(A[by * BM, k * BK], A_s)
                T.copy(B[bx * BN, k * BK], B_s)
                T.gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=(k == 0))
            T.copy(C_t, C[by * BM, bx * BN])

    return kernel


def _plain_gemm_kernel():
    """One non-pipelined fp16 TN tile for the blocking-copy control."""

    @T.prim_func
    def kernel(A: T.Tensor((128, BK), T.float16), B: T.Tensor((128, BK), T.float16), C: T.Tensor((128, 128), T.float32)):
        with T.Kernel(1, threads=128):
            A_s = T.alloc_shared((128, BK), T.float16)
            B_s = T.alloc_shared((128, BK), T.float16)
            C_t = T.alloc_tmem((128, 128), T.float32)
            T.copy(A, A_s)
            T.copy(B, B_s)
            T.gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=True)
            T.copy(C_t, C)

    return kernel


def _source(stages: int) -> str:
    kernel = _gemm_kernel(stages)
    return tilelang.lower(kernel, target=STCUV2_TARGET).kernel_source


def _after_isp_snapshot(stages: int):
    """IR immediately after ISP, before any S3 late pass can repair it."""
    ns = tilelang.transform
    original = ns.InjectSoftwarePipeline
    snapshots = []

    def spying_factory(*args, **kwargs):
        inner = original(*args, **kwargs)

        @tvm.transform.module_pass(opt_level=0)
        def wrapper(mod, ctx):
            out = inner(mod)
            snapshots.extend(func for _gvar, func in out.functions.items())
            return out

        return wrapper

    ns.InjectSoftwarePipeline = spying_factory
    try:
        _source(stages)
    finally:
        ns.InjectSoftwarePipeline = original
    assert snapshots
    return next(func for func in snapshots if "tl.pipeline_dependency_facts" in func.script())


def _after_isp_script(stages: int) -> str:
    return _after_isp_snapshot(stages).script()


def _after_s3_pipeline_finalize_snapshot(stages: int):
    """IR after the S3 post-pass has finalized generic ISP barriers."""
    ns = tilelang.tang.transform
    original = ns.FinalizeS3PipelineInjection
    snapshots = []

    def spying_factory(*args, **kwargs):
        inner = original(*args, **kwargs)

        @tvm.transform.module_pass(opt_level=0)
        def wrapper(mod, ctx):
            out = inner(mod)
            snapshots.extend(func for _gvar, func in out.functions.items())
            return out

        return wrapper

    ns.FinalizeS3PipelineInjection = spying_factory
    try:
        _source(stages)
    finally:
        ns.FinalizeS3PipelineInjection = original
    assert snapshots
    return next(func for func in snapshots if "tl.pipeline_dependency_facts" in func.script())


def _after_s3_pipeline_finalize_script(stages: int) -> str:
    return _after_s3_pipeline_finalize_snapshot(stages).script()


def _snapshot_after(owner, pass_name: str, stages: int):
    """Capture the first PrimFunc carrying S3 async-edge facts after a pass."""
    original = getattr(owner, pass_name)
    snapshots = []

    def spying_factory(*args, **kwargs):
        inner = original(*args, **kwargs)

        @tvm.transform.module_pass(opt_level=0)
        def wrapper(mod, ctx):
            out = inner(mod)
            for _gvar, func in out.functions.items():
                if isinstance(func, tvm.tirx.PrimFunc) and func.attrs is not None and "tl.tang_async_edge_facts" in func.attrs:
                    snapshots.append(func)
            return out

        return wrapper

    setattr(owner, pass_name, spying_factory)
    try:
        _source(stages)
    finally:
        setattr(owner, pass_name, original)
    assert snapshots, f"no async-edge facts survived {pass_name}"
    return snapshots[-1]


def _snapshot_before_s3_sync_finalize(stages: int):
    """Capture the device PrimFunc at the S3 certificate consumer's entry."""
    original = tilelang.tang.transform.FinalizeS3ThreadSyncs
    snapshots = []

    def spying_factory(*args, **kwargs):
        inner = original(*args, **kwargs)

        @tvm.transform.module_pass(opt_level=0)
        def wrapper(mod, ctx):
            for _gvar, func in mod.functions.items():
                if isinstance(func, tvm.tirx.PrimFunc) and func.attrs is not None and "tl.tang_async_edge_facts" in func.attrs:
                    snapshots.append(func)
            return inner(mod)

        return wrapper

    tilelang.tang.transform.FinalizeS3ThreadSyncs = spying_factory
    try:
        _source(stages)
    finally:
        tilelang.tang.transform.FinalizeS3ThreadSyncs = original
    assert snapshots
    return snapshots[-1]


def _snapshot_before_s3_async_copy(stages: int):
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
        _source(stages)
    finally:
        tilelang.tang.transform.InjectS3AsyncCopy = original
    assert snapshots
    return next(func for func in snapshots if func.attrs is not None and str(func.attrs.get("global_symbol")) == "kernel")


def _run_s3_sync_finalize(func, pass_configs=None):
    with tvm.transform.PassContext(config=dict(pass_configs or {})):
        out = tilelang.tang.transform.FinalizeS3ThreadSyncs()(tvm.IRModule.from_expr(func))
    return next(iter(out.functions.values()))


def _storage_sync_count(func) -> int:
    count = 0

    def visit(node):
        nonlocal count
        if (
            isinstance(node, tvm.tirx.Evaluate)
            and isinstance(node.value, tvm.tirx.Call)
            and getattr(node.value.op, "name", "") == "tirx.tvm_storage_sync"
        ):
            count += 1

    tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    return count


def _fact_and_anchor_inventory(func):
    facts = func.attrs["tl.tang_async_edge_facts"]
    roles = []

    def visit(node):
        if isinstance(node, tvm.tirx.AttrStmt) and str(node.attr_key) == "tl.tang_async_edge_anchor":
            roles.append(int(node.value) & 0xFF)

    tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    return len(facts), sorted(roles)


def _edge_form(src: str) -> str:
    """Assert that the pipeline uses the single supported mbarrier mechanism."""
    form = "sync" if "sync_wait(" in src else "mbarrier"
    assert form == "mbarrier", f"the pipeline edges should be mbarriers, got {form}\n{src}"
    return form


# ===========================================================================
# §1  Structural: the pipeline is asynchronous
# ===========================================================================


@pytest.mark.parametrize("stages", [2, 3])
def test_s3_finalizer_completes_the_ready_edge(stages):
    """The S3 post-pass completes ISP's generic barrier contract."""
    script = _after_s3_pipeline_finalize_script(stages)
    expected = ", ".join(["8"] * stages)
    assert f'"barrier_init": {{pipeline_mbar.data: [{expected}]}}' in script, script
    declarations = re.findall(r"tang_pipeline_arrive_count=(\d+)", script)
    assert declarations and set(declarations) == {"4"}, declarations
    assert "T.mbarrier_wait_parity(pipeline_mbar[" in script, script
    assert '"tl.pipeline_dependency_facts"' in script, script


def test_common_isp_keeps_s3_barrier_metadata_backend_neutral():
    script = _after_isp_script(2)
    assert '"barrier_init": {pipeline_mbar.data: [1, 1]}' in script, script
    assert "tang_pipeline_arrive_count" not in script, script
    assert "tl.pipeline_ready_owner=0" in script, script


def test_automatic_pipeline_publishes_typed_v2_plan():
    func = _after_s3_pipeline_finalize_snapshot(3)
    plans = []
    copies = []

    def visit(node):
        if isinstance(node, tvm.tirx.For) and "tl.pipeline_dependency_facts" in node.annotations:
            plans.extend(node.annotations["tl.pipeline_dependency_facts"].values())
        if (
            isinstance(node, tvm.tirx.Call)
            and getattr(node.op, "name", "") in {"tl.tileop.copy", "tl.tileop.tma_copy"}
            and "tl.pipeline_buffer_uid" in node.annotations
        ):
            copies.append(node.annotations)

    tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    assert plans
    assert all(len(plan) == 8 and int(plan[0]) == 2 and int(plan[7]) == 0 for plan in plans)
    assert copies
    assert set(int(copy["tl.pipeline_buffer_uid"]) for copy in copies) == {int(plan[2]) for plan in plans}
    assert all(int(copy["tl.pipeline_ready_owner"]) == 1 for copy in copies)


def test_lowered_copy_record_is_scoped_and_carries_plan_identity():
    func = _snapshot_before_s3_async_copy(3)
    scopes = 0
    records = []

    def visit(node):
        nonlocal scopes
        if isinstance(node, tvm.tirx.AttrStmt) and str(node.attr_key) == "tl.tang_async_copy_record_scope":
            scopes += 1
        if isinstance(node, tvm.tirx.Call) and getattr(node.op, "name", "") == "tl.tang_bulk_async_commit":
            records.append(node)

    tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    assert scopes == len(records) and records
    assert all(len(call.args) in (13, 14) and int(call.args[0]) == 2 for call in records)
    assert all(int(call.args[7]) > 0 and int(call.args[8]) > 0 for call in records)
    assert all(int(call.args[12]) == 1 for call in records)


def test_async_edge_certificate_has_all_six_anchor_roles():
    func = _snapshot_after(tilelang.tang.transform, "InjectS3AsyncCopy", 3)
    fact_count, roles = _fact_and_anchor_inventory(func)
    assert fact_count == 2, fact_count
    assert set(roles) == {1, 2, 3, 4, 5, 6}, roles


@pytest.mark.parametrize("pass_name", ["FlattenBuffer", "StorageRewrite", "SplitHostDevice"])
def test_async_edge_certificate_survives_address_rewriting(pass_name):
    func = _snapshot_after(tilelang.transform, pass_name, 3)
    fact_count, roles = _fact_and_anchor_inventory(func)
    assert fact_count == 2, fact_count
    assert set(roles) == {1, 2, 3, 4, 5, 6}, roles


def test_incomplete_async_edge_certificate_keeps_threadsync_barriers():
    """Anchors without valid function facts are metadata, not a proof."""
    func = _snapshot_before_s3_sync_finalize(3)
    facts = func.attrs["tl.tang_async_edge_facts"]
    invalid = {str(key): [999, *(int(value) for value in values[1:])] for key, values in facts.items()}
    invalid_func = func.with_attr("tl.tang_async_edge_facts", invalid)

    valid_func = _run_s3_sync_finalize(func)
    invalid_func = _run_s3_sync_finalize(invalid_func)

    assert _storage_sync_count(invalid_func) > _storage_sync_count(valid_func)


def test_facts_without_copy_anchors_keep_threadsync_barriers():
    """Function facts cannot certify an unanchored or extra shared writer."""
    func = _snapshot_before_s3_sync_finalize(3)

    def strip_copy_anchor(node):
        if isinstance(node, tvm.tirx.AttrStmt) and str(node.attr_key) == "tl.tang_async_edge_anchor" and int(node.value) & 0xFF == 1:
            return node.body
        return None

    stripped_body = tvm.tirx.stmt_functor.ir_transform(func.body, None, strip_copy_anchor, ["tirx.AttrStmt"])
    stripped_func = func.with_body(stripped_body)

    valid_func = _run_s3_sync_finalize(func)
    stripped_func = _run_s3_sync_finalize(stripped_func)

    assert _storage_sync_count(stripped_func) > _storage_sync_count(valid_func)


def test_s3_sync_finalization_is_independent_of_fence_injection():
    """The fence switch must not bypass certificate-aware ThreadSync cleanup."""
    func = _snapshot_before_s3_sync_finalize(3)
    before = _storage_sync_count(func)
    fence_off = _run_s3_sync_finalize(
        func,
        {tilelang.PassConfigKey.TL_ENABLE_TANG_INJECT_TCGEN05_FENCE: False},
    )
    cleanup_off = _run_s3_sync_finalize(
        func,
        {
            tilelang.PassConfigKey.TL_ENABLE_TANG_INJECT_TCGEN05_FENCE: False,
            tilelang.PassConfigKey.TL_DISABLE_REMOVE_REDUNDANT_SYNCS: True,
        },
    )

    assert _storage_sync_count(fence_off) < before
    assert _storage_sync_count(cleanup_off) == before


def test_mbarrier_initialization_keeps_publication_sync():
    """fence_barrier_init orders init stores but does not rendezvous warps."""
    src = _source(2)
    fences = src.count("tl::fence_barrier_init();")
    published = src.count("tl::fence_barrier_init();\n  __syncthreads();")
    assert fences > 0
    assert published == fences


@pytest.mark.parametrize("stages", [2, 3])
def test_pipelined_copies_commit_into_their_ready_edge(stages):
    """The copy is the leaf, its publication is the arrival, and the consumer waits.

    ISP declares the barrier and per-copy arrival count; lowering mechanically
    emits leaf + record + mbarrier arrival, with no stalling drain.
    """
    src = _source(stages)
    _edge_form(src)

    assert "tl::tang_bulk_g2s_async<" not in src, src
    # The blocking helper is gone: it would stall the warp in fence_mem and then
    # rendezvous the whole block, which is exactly what kills the overlap.
    assert "tl::tang_bulk_g2s<" not in src, src
    assert "tang::ptx::fcpg2s<" in src, src
    # Each logical copy publishes once to ISP's versioned barrier and the
    # consumer waits that slot.
    assert "tang::ptx::fence_g2s_arrive(" not in src, src
    assert src.count("arrive_on_g2s_fence()") >= 4, src
    assert src.count(".wait(") >= 2, src
    # ... and it must be the non-blocking arrive form, not a plain fence_mem.
    assert "fence_mem(tang::ptx::fg_g2s_default)" not in src, src


def test_each_operand_owns_its_barrier_slots():
    """The ready edge is the pipeline's barrier; the free edge stays per operand.

    InjectSoftwarePipeline creates one ``pipeline_mbar`` and both copies of a
    version arrive on that slot, so its init count is the sum of their warps
    (4 + 4). Sharing it is the point: the consumer waits once, in front of the
    MMA, for every producer of that version. The free edge does not share --
    overwriting A's version must not be released by B's MMA -- so it stays two
    TANG slots, one per operand, naming the version the copy is about to reuse.
    """
    src = _source(3)
    body = _loop_body(src)
    pipeline_inits = [int(v) for v in re.findall(r"pipeline_mbar\[\d+\]\.init\((\d+)\)", src)]
    assert pipeline_inits and set(pipeline_inits) == {8}, f"{pipeline_inits}\n{src}"
    assert "pipeline_mbar[(k % 3)].wait((k / 3));" in body, body
    rotated = re.findall(r"mbarrier\[\(\(\(k % 3\) \* 2\) \+ (\d+)\)\]\.wait\(", body)
    assert rotated == [], f"the ready edge is pipeline_mbar, not a per-operand slot:\n{body}"
    free = {int(i) for i in re.findall(r"mbarrier\[\(\(\(\(k \+ 2\) % 3\) \* 2\) \+ (\d+)\)\]\.wait\(", body)}
    assert len(free) == 2, f"one free slot per operand: {free}\n{body}"


def test_every_slot_is_sized_by_its_arrivals():
    """Each slot is initialised with the warps that arrive on it, not the waiters.

    rows=128 with sw128a32 means 8 stripes, so all 4 warps of a 128-thread block
    issue a copy and, on the free edge, an MMA lane slice: every slot a warp
    arrives on is initialised with 4. The count is independent of how many
    warps execute the wait.
    """
    src = _source(3)
    inits = {int(i): int(v) for i, v in re.findall(r"(?<!_)mbarrier\[(\d+)\]\.init\((\d+)\)", src)}
    assert inits, src
    for slot, count in inits.items():
        assert count == 4, f"slot {slot} must be sized by its arrivals: {inits}\n{src}"
    # The ready edge is sized by both operands' issuing warps, once per stage.
    pipeline = [int(v) for v in re.findall(r"pipeline_mbar\[\d+\]\.init\((\d+)\)", src)]
    assert pipeline and set(pipeline) == {8}, pipeline


def test_every_pipelined_operand_has_a_free_edge():
    """The reverse edge: a version may only be overwritten once the MMA is done.

    Both sides are async, so both need an edge: the DMA (fcpg2s) publishes the
    ready edge, the MMA publishes the free edge with its tensor-core fence
    (non-blocking: hardware posts the arrival when the TC group retires). On the
    mbarrier form each is an arrival on a slot, and the producer's wait carries
    the phase parity of the generation it is reusing instead of a waiter count.
    """
    body = _loop_body(_source(3))
    arrives = re.findall(r"(?<!_)mbarrier\[[^\]]+\]\.arrive_on_tc_fence", body)
    waits = re.findall(r"(?<!_)mbarrier\[[^\]]+\]\.wait\(", body)
    assert len(arrives) == 2, f"one TC arrive per operand: {arrives}\n{body}"
    # The ready wait is the shared pipeline barrier, outside this count. What
    # remains is one free wait per operand.
    assert len(waits) == 2, f"{waits}\n{body}"
    # The free edge is guarded: the first versions a rotation produces have never
    # been consumed, so waiting there could only deadlock.
    assert body.count("if (1 <= k)") == 2, body
    assert "arrive_on_g2s_fence()" in body, body


def test_non_pipelined_kernel_keeps_blocking_copies():
    """Failure policy: nothing to pipeline means the pass must not touch it.

    A single-buffered kernel has no rotated version, so it keeps the blocking
    copy and gets no barrier at all. This is the guard against the pass
    "optimizing" a kernel it cannot reason about.
    """
    kernel = _plain_gemm_kernel()
    src = tilelang.lower(kernel, target=STCUV2_TARGET).kernel_source
    assert "_async" not in src, src
    assert "sync_wait(" not in src, src
    assert "fence_g2s_arrive(" not in src, src
    assert "mbarrier" not in src, src
    # Blocking is the leaf plus its own drain: no barrier is attached, but the
    # drain must remain so the copy completes.
    assert "tang::ptx::fcpg2s<" in src, src
    assert "tang::ptx::fence_mem(" in src, src


# ===========================================================================
# §3  The loop is barrier-driven: no block-wide rendezvous (P4)
# ===========================================================================


def _loop_body(src: str) -> str:
    """The generated pipeline loop, braces included."""
    lines = src.splitlines()
    start = next(i for i, line in enumerate(lines) if re.match(r"\s*for \(int k = 0;", line))
    depth = 0
    for i in range(start, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if depth == 0 and i > start:
            return "\n".join(lines[start : i + 1])
    raise AssertionError("unterminated pipeline loop in:\n" + src)


@pytest.mark.parametrize("stages", [2, 3])
def test_pipeline_loop_has_no_block_wide_sync(stages):
    """The two per-iteration ``__syncthreads`` are gone; the fences stay.

    ThreadSync inserts a RAW sync before the MMA and a WAR sync after it for
    exactly the dependencies the ready/free barriers carry, so the
    certificate-aware S3 sync finalizer drops them. The TC/TMEM
    domain fences are *not* dropped: they are what orders the tensor-core
    writes, and the free edge's arrive depends on them.
    """
    src = _source(stages)
    _edge_form(src)
    body = _loop_body(src)
    assert "__syncthreads" not in body, body
    assert "tvm_storage_sync" not in body, body
    # The rendezvous is replaced by the barriers, not by nothing: one shared
    # ready wait (both operands arrive on the same pipeline slot) and one free
    # wait per operand.
    assert body.count(".wait(") == 3, body
    # One ready arrival per operand. The epilogue consumer waits outside the
    # loop, on the slot this iteration already arrived on.
    assert body.count("arrive_on_g2s_fence()") == 2, body
    assert body.count("arrive_on_tc_fence") == 2, body
    # The MMA keeps its domain fences, before and after, and still runs.
    assert body.count("fence_tc<0>()") == 2, body
    assert "tl::tang_tcgen05_mma_ss(" in body, body


def test_syncs_outside_the_pipeline_loop_survive():
    """Only the loop's own rendezvous goes.

    The drain barrier after the loop orders the LDT reads against the collective
    ``tc_dealloc``; removing it would break the TMEM handoff. The allocation
    itself needs no barrier: ``tc_alloc`` writes the base address from every
    thread with the same value, so it publishes nothing (pinned by
    ``test_the_tmem_allocation_needs_no_barrier``). So rather than count
    barriers (the number depends on the drain), pin the one that must stay.
    """
    src = _source(3)
    body = _loop_body(src)
    assert body.count("__syncthreads") == 0, body
    # The drain (ldt) starts only after every warp's accumulator has landed.
    assert re.search(r"fence_tmem<3>\(\);\s*__syncthreads\(\);\s*uint _out", src), src
    assert src.count("__syncthreads") >= 3, src


def test_the_tmem_allocation_needs_no_barrier():
    """``tc_alloc`` is followed by its first use, not by a block rendezvous.

    The TMEM base address is written into a shared word by *every* thread, all
    with the same value, so a reader already wrote what it reads and stcuv2
    needs no rendezvous to publish it. ``LowerSharedTmem`` therefore emits no
    barrier after the reservation (and ThreadSync asks for none: the allocation
    call names the holder through a plain load, which is not a shared write).
    """
    src = _source(3)
    assert re.search(r"tc_alloc<\d+>\([^;]*\);\s*(?!__syncthreads)", src), src
    assert not re.search(r"tc_alloc<\d+>\([^;]*\);\s*__syncthreads\(\);", src), src


def test_barrier_init_is_elected_not_thread_zero():
    """Every mbarrier publication elects its init thread the same way.

    Two producers publish barriers in this kernel -- ``InjectSoftwarePipeline``
    owns ``pipeline_mbar`` and this pass owns the operand handshake's slots --
    and both name the initialising thread with ``tl_shuffle_elect<0>()`` (lane 0
    of warp 0 on TANG), the form ``LowerSharedBarrier`` also emits for a
    ``T.alloc_barrier`` buffer. A ``threadIdx.x == 0`` test would be equivalent
    arithmetic but a second spelling of the same election, and the three
    producers' generated code would no longer read alike.
    """
    src = _source(3)
    assert src.count("tl::tl_shuffle_elect<0>()") >= 2, src
    assert not re.search(r"threadIdx\.x\)\s*==\s*0", src), src


@pytest.mark.parametrize("stages", [2, 3])
def test_prologue_barrier_in_front_of_the_loop_is_gone(stages):
    """The barrier ThreadSync puts between the prologue copies and the loop goes.

    It guards the prologue's async copies -> the MMA's reads, which is exactly
    the ready edge: every copy arrives on the ``pipeline_mbar`` slot the loop's
    first iteration waits on. What sits in front of the previous barrier
    (the tc_alloc publish) is that barrier's business, so this one is a second
    rendezvous for a dependency already carried by the barriers -- and unlike
    them, it would be illegal once the copies are confined to a warp subset.
    """
    src = _source(stages)
    lines = src.splitlines()
    loop = next(i for i, line in enumerate(lines) if re.match(r"\s*for \(int k = 0;", line))
    # Locate the prologue's last copy by its leaf; the following arrival belongs
    # to that copy.
    last_copy = max(i for i, line in enumerate(lines[:loop]) if "tang_bulk_g2s_async<" in line or "fcpg2s" in line)
    window = lines[last_copy + 1 : loop]
    assert not any("__syncthreads" in line for line in window), "\n".join(window)
    # The wait that replaces it is the pipeline barrier's consumer wait, on the
    # first iteration of the loop (after that iteration's own copies, which fill
    # the next version). It names pipeline_mbar, not a per-operand slot.
    head = lines[loop : loop + 24]
    assert any("pipeline_mbar[" in line and ".wait(" in line for line in head), "\n".join(head)


def test_no_async_pipeline_means_nothing_to_remove():
    """No-op guard: a kernel without the barrier markers loses no barrier."""
    kernel = _plain_gemm_kernel()
    src = tilelang.lower(kernel, target=STCUV2_TARGET).kernel_source
    assert "sync_wait(" not in src, src
    assert "mbarrier" not in src, src
    assert "__syncthreads" in src, src


def test_single_stage_pipeline_keeps_blocking_copy():
    """One version cannot carry a pipeline barrier generation."""
    src = _source(1)
    assert "pipeline_mbar" not in src, src
    assert "mbarrier" not in src, src
    assert "fence_mem(" in src, src
    assert "__syncthreads" in src, src
    ir = _snapshot_before_s3_async_copy(1).script()
    assert "tl.tang_bulk_async_commit" not in ir, ir
    assert "tl.tang_async_copy_record_scope" not in ir, ir


@pytest.mark.parametrize("stages", [2, 3])
def test_last_tile_copy_is_ordered_before_the_epilogue_mma(stages):
    """The post-loop MMA must not read a tile whose DMA was never waited for.

    The loop's own ready waits only name the consumers *inside* the loop, so the
    tile the epilogue MMA reads is covered by a barrier of its own: the last
    iteration's copy signals it (conditionally, so the rotated ids keep their
    generations to themselves) and the wait sits right after the loop, once, in
    front of the MMA.
    """
    src = _source(stages)
    _edge_form(src)
    body = _loop_body(src)
    after_loop = src.split(body, 1)[1]

    # The loop's own bound decides which iteration is the last one (it depends on
    # the stage count), so read it from the source instead of deriving it.
    bound = re.search(r"for \(int k = 0; k < (\d+);", body)
    assert bound is not None, body
    loop_last = int(bound.group(1)) - 1
    # ISP peels ``stages - 1`` consumer iterations. Each one waits the pipeline
    # slot of the version it reads; the producer already arrived on that slot
    # inside the loop, so there is no last-iteration ``if (k == last)`` arrive.
    epilogue = re.findall(r"pipeline_mbar\[(\d+)\]\.wait\(([^)]+)\);", after_loop)
    assert len(epilogue) == stages - 1, f"one pipeline wait per peeled consumer:\n{after_loop}"
    before_mma = after_loop.split("tl::tang_tcgen05_mma_ss(")[0]
    assert re.search(r"pipeline_mbar\[\d+\]\.wait\(", before_mma), before_mma
    assert f"if (k == {loop_last})" not in body, body


# ===========================================================================
# §6  ISP states the loop's region (S6)
# ===========================================================================


def test_isp_declares_each_pipelined_loop_region():
    """S6: ISP publishes the role fact beside its rotation contract."""
    script = _after_isp_script(stages=2)
    assert '"tl.pipeline_region_warps"' in script, script
    assert '"tl.pipeline_region_warps": 4' in script, script
