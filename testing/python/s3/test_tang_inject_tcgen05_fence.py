"""stcuv2 fence 自动注入测试 —— 对应 ``src/tang/transform/inject_tcgen05_fence.cc``。

stcuv2 把 ordering 拆成互不覆盖的 **fence 域**：

    fence.tc    —— 只管 TC 指令(TMUL/TMAD/TMACC),组 fg0~fg1
    fence.tmem  —— 只管 TMEM 寄存器搬运(LDT/STT),组 fg0~fg3
    fence.mem   —— 异步拷贝/DMA,组 fg0~fg7

所以一条 ``fence_tc`` **不会** retired 掉 LDT/STT,反之亦然。
``tl.tang.transform.InjectTcgen05Fence`` 在与 CUDA 版 ``InjectTcgen05Fence`` 等价的
交接点上按**邻居所属域**插入正确的指令:

| 邻居                              | 注入                    |
|-----------------------------------|-------------------------|
| TC 指令(mma)                      | ``fence_tc<fg_tc_default=0>``   |
| TMEM 读(LDT / shared.tmem 读)     | ``fence_tmem<fg_ldt_default=3>`` |
| TMEM 写(STT / shared.tmem 写)     | ``fence_tmem<fg_stt_default=2>`` |
| 拷贝引擎(bulk / cpt2s / cps2t)     | 不注入(模板内部用 ``fence_mem``) |

插入位置(与 CUDA 版一致):

1. ``tvm_storage_sync("shared"/"shared.dyn")`` 的相邻侧;
2. ``mbarrier_wait_parity`` / ``tang_sync_wait`` 之后(前向扫描遇到使用);
3. ``ptx_arrive_barrier`` / ``tang_sync_arrive`` 之前(后向扫描遇到使用)。

已存在的 fence 按**域**去重:手写的 ``fence_tc`` 不会抑制 TMEM 侧仍需的
``fence_tmem``。非 stcuv2 目标、或不含 TC/TMEM 操作的函数原样返回。
``tang_tcgen05_mma_arrive`` 刻意不加 fence(mma 的 commit 已带 producer 侧 ordering)。

§1 注入契约(只跑 pass,不跑 ISS)
§2 pipeline 开关(``tl.enable_tang_inject_tcgen05_fence``,**默认开启**)+ ISS 数值验证
"""

import re

import pytest
import torch

import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.jit import JITKernel
from tvm import tirx

STCUV2_TARGET = {"kind": "tang", "arch": "stcuv2"}
STCU_TARGET = {"kind": "tang", "arch": "stcu"}

_FENCE_TC_OP = "tl.tang_fence_tc"
_FENCE_TMEM_OP = "tl.tang_tmem_fence"
_TMEM_READ_OP = "tl.tang_tmem_ld"  # LDT: TMEM -> register, group 3
_TMEM_WRITE_OP = "tl.tang_tmem_st"  # STT: register -> TMEM, group 2
_TC_USE_OP = "tl.tang_tcgen05_mma_ss"  # TC instruction, group 0
_SYNC_WAIT_OP = "tl.tang_sync_wait"
_SYNC_ARRIVE_OP = "tl.tang_sync_arrive"


_RE_FENCE = re.compile(r"fence_(?:tc|tmem)<\d+>\(\)")


def _count_op(stmt, op_name: str) -> int:
    """Count calls of ``op_name`` in ``stmt``."""
    count = 0

    def visitor(node):
        nonlocal count
        if isinstance(node, tirx.Call) and getattr(node.op, "name", None) == op_name:
            count += 1

    tirx.stmt_functor.post_order_visit(stmt, visitor)
    return count


def _tc_fences(stmt) -> int:
    return _count_op(stmt, _FENCE_TC_OP)


def _tmem_fences(stmt) -> int:
    return _count_op(stmt, _FENCE_TMEM_OP)


def _tmem_fence_groups(stmt) -> list:
    """Fence groups used by the injected ``fence_tmem`` calls, sorted."""
    groups = []

    def visitor(node):
        if isinstance(node, tirx.Call) and getattr(node.op, "name", None) == _FENCE_TMEM_OP:
            arg = node.args[0]
            groups.append(int(getattr(arg, "value", arg)))

    tirx.stmt_functor.post_order_visit(stmt, visitor)
    return sorted(groups)


def _fences(stmt) -> int:
    """Total auto-injected fences, across both domains."""
    return _tc_fences(stmt) + _tmem_fences(stmt)


def _use(op_name: str, ref):
    """A minimal TC/TMEM operation; the pass only matches the op identity."""
    return T.call_intrin("handle", tirx.op.Op.get(op_name), ref)


def _tmem_read(ref):
    return _use(_TMEM_READ_OP, ref)


def _tmem_write(ref):
    return _use(_TMEM_WRITE_OP, ref)


def _tc_use(ref):
    return _use(_TC_USE_OP, ref)


def _tmem_fence(group: int):
    """A hand-written TMEM fence, as the kernel would spell it."""
    return _use(_FENCE_TMEM_OP, T.int32(group))


def _tc_fence(group: int = 0):
    """A hand-written TC fence.

    ``tang_tcgen05.tcgen05_before/after_thread_sync`` map to
    ``tl.tang_fence_tc`` (group 0~1). Note that ``T.tcgen05_before_thread_sync``
    (from ``tilelang.language.builtin``) is the CUDA-only spelling: it emits
    ``tl.tcgen05_before_thread_sync``, which the TANG codegen has no handler for.
    """
    return _use(_FENCE_TC_OP, T.int32(group))


def _sync_wait(barrier_id: int = 0):
    return T.call_intrin(
        "handle",
        tirx.op.Op.get(_SYNC_WAIT_OP),
        T.int32(barrier_id),
        T.int32(1),
        T.int32(1),
    )


def _sync_arrive(barrier_id: int = 0):
    return T.call_intrin("handle", tirx.op.Op.get(_SYNC_ARRIVE_OP), T.int32(barrier_id))


def _apply(func, target=STCUV2_TARGET):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tirx.transform.BindTarget(tvm.target.Target(target))(mod)
    return tilelang.tang.transform.InjectTcgen05Fence()(mod)


# ---------------------------------------------------------------------------
# §1 注入契约
# ---------------------------------------------------------------------------


def test_sync_with_tmem_read_after_gets_the_ldt_fence():
    @T.prim_func
    def func():
        with T.Kernel(1):
            C_tmem = T.decl_buffer((1,), T.uint32, scope="shared")
            T.tvm_storage_sync("shared")
            T.evaluate(_tmem_read(C_tmem[0]))

    mod = _apply(func)
    body = mod["main"].body
    assert _fences(body) == 1, "only the TMEM side of the sync is fenced"
    assert _tc_fences(body) == 0, "fence.tc does not retire LDT"
    assert _tmem_fence_groups(body) == [3], "LDT uses fg_ldt_default"


def test_sync_with_tmem_read_before_gets_the_ldt_fence():
    @T.prim_func
    def func():
        with T.Kernel(1):
            C_tmem = T.decl_buffer((1,), T.uint32, scope="shared")
            T.evaluate(_tmem_read(C_tmem[0]))
            T.tvm_storage_sync("shared")
            T.call_extern("handle", "generic_op")

    mod = _apply(func)
    body = mod["main"].body
    assert _fences(body) == 1
    assert _tmem_fence_groups(body) == [3]


def test_sync_with_tmem_write_uses_the_stt_group():
    @T.prim_func
    def func():
        with T.Kernel(1):
            C_tmem = T.decl_buffer((1,), T.uint32, scope="shared")
            T.evaluate(_tmem_write(C_tmem[0]))
            T.tvm_storage_sync("shared")
            T.call_extern("handle", "generic_op")

    mod = _apply(func)
    body = mod["main"].body
    assert _fences(body) == 1
    assert _tc_fences(body) == 0, "fence.tc does not retire STT"
    assert _tmem_fence_groups(body) == [2], "STT uses fg_stt_default"


def test_sync_with_tc_use_gets_the_tc_fence():
    """A TC instruction is not retired by fence.tmem, so it keeps fence.tc."""

    @T.prim_func
    def func():
        with T.Kernel(1):
            C_tmem = T.decl_buffer((1,), T.uint32, scope="shared")
            T.tvm_storage_sync("shared")
            T.evaluate(_tc_use(C_tmem[0]))

    mod = _apply(func)
    body = mod["main"].body
    assert _tc_fences(body) == 1
    assert _tmem_fences(body) == 0


def test_sync_handing_over_both_domains_gets_one_fence_each():
    @T.prim_func
    def func():
        with T.Kernel(1):
            C_tmem = T.decl_buffer((1,), T.uint32, scope="shared")
            T.evaluate(_tc_use(C_tmem[0]))
            T.tvm_storage_sync("shared")
            T.evaluate(_tmem_read(C_tmem[0]))

    mod = _apply(func)
    body = mod["main"].body
    assert _tc_fences(body) == 1, "the TC side needs fence.tc"
    assert _tmem_fence_groups(body) == [3], "the TMEM side needs fence.tmem"


def test_sync_without_tmem_neighbour_is_not_fenced():
    """A barrier that only orders shared-memory traffic needs no fence.

    The TMEM use is separated from the sync by an arrive, so the sync itself has
    no TC/TMEM neighbour; only the arrive is fenced.
    """

    @T.prim_func
    def func():
        with T.Kernel(1):
            mbarrier = T.decl_buffer((1,), T.uint64, scope="shared.barrier")
            C_tmem = T.decl_buffer((1,), T.uint32, scope="shared")
            T.evaluate(_tmem_read(C_tmem[0]))
            T.ptx_arrive_barrier(mbarrier[0])
            T.tvm_storage_sync("shared")
            T.call_extern("handle", "generic_op")

    mod = _apply(func)
    assert _fences(mod["main"].body) == 1, "only the arrive's fence, not the sync's"


def test_tmem_buffer_access_counts_as_a_use():
    """The drain is a plain BufferStore to shared.tmem before lowering."""

    @T.prim_func
    def func():
        with T.Kernel(1):
            C_tmem = T.decl_buffer((4,), T.float32, scope="shared.tmem")
            T.tvm_storage_sync("shared")
            C_tmem[0] = T.float32(1)

    mod = _apply(func)
    body = mod["main"].body
    assert _fences(body) == 1, "a shared.tmem store must trigger the fence"
    assert _tmem_fence_groups(body) == [2], "a store is STT traffic (group 2)"


def test_tmem_buffer_load_uses_the_ldt_group():
    @T.prim_func
    def func():
        with T.Kernel(1):
            C_tmem = T.decl_buffer((4,), T.float32, scope="shared.tmem")
            T.tvm_storage_sync("shared")
            T.call_extern("handle", "generic_op", C_tmem[0])

    mod = _apply(func)
    assert _tmem_fence_groups(mod["main"].body) == [3]


def test_wait_and_arrive_are_rewritten_only_at_tmem_handoffs():
    @T.prim_func
    def func():
        with T.Kernel(1):
            mbarrier = T.decl_buffer((1,), T.uint64, scope="shared.barrier")
            C_tmem = T.decl_buffer((1,), T.uint32, scope="shared")
            T.mbarrier_wait_parity(mbarrier[0], 0)
            T.evaluate(_tmem_read(C_tmem[0]))
            T.ptx_arrive_barrier(mbarrier[0])

    mod = _apply(func)
    # one fence after the wait, one before the arrive
    assert _fences(mod["main"].body) == 2


def test_wait_and_arrive_scan_across_neutral_statements():
    @T.prim_func
    def func():
        with T.Kernel(1):
            mbarrier = T.decl_buffer((1,), T.uint64, scope="shared.barrier")
            C_tmem = T.decl_buffer((1,), T.uint32, scope="shared")
            T.mbarrier_wait_parity(mbarrier[0], 0)
            T.call_extern("handle", "generic_op")
            T.evaluate(_tmem_read(C_tmem[0]))
            T.call_extern("handle", "generic_op")
            T.ptx_arrive_barrier(mbarrier[0])

    mod = _apply(func)
    assert _fences(mod["main"].body) == 2


def test_sync_boundary_stops_wait_lookahead():
    @T.prim_func
    def func():
        with T.Kernel(1):
            mbarrier = T.decl_buffer((1,), T.uint64, scope="shared.barrier")
            C_tmem = T.decl_buffer((1,), T.uint32, scope="shared")
            T.mbarrier_wait_parity(mbarrier[0], 0)
            T.call_extern("handle", "generic_op")
            T.ptx_arrive_barrier(mbarrier[0])
            T.evaluate(_tmem_read(C_tmem[0]))

    mod = _apply(func)
    # the wait's lookahead stops at the arrive, so neither boundary is fenced
    assert _fences(mod["main"].body) == 0


def test_tang_named_barrier_handshake_is_fenced():
    @T.prim_func
    def func():
        with T.Kernel(1):
            C_tmem = T.decl_buffer((1,), T.uint32, scope="shared")
            T.evaluate(_sync_wait())
            T.evaluate(_tmem_read(C_tmem[0]))
            T.evaluate(_sync_arrive())

    mod = _apply(func)
    body = mod["main"].body
    assert _fences(body) == 2, "sync_wait/sync_arrive bracketing a TMEM use need fences"
    assert _tmem_fence_groups(body) == [3, 3]


def test_manual_tc_fence_does_not_suppress_the_tmem_fence():
    """fence.tc cannot retire LDT/STT, so the TMEM side still needs its own fence."""

    @T.prim_func
    def func():
        with T.Kernel(1):
            mbarrier = T.decl_buffer((1,), T.uint64, scope="shared.barrier")
            C_tmem = T.decl_buffer((1,), T.uint32, scope="shared")
            T.mbarrier_wait_parity(mbarrier[0], 0)
            T.evaluate(_tc_fence())
            T.evaluate(_tmem_read(C_tmem[0]))
            T.evaluate(_tc_fence())
            T.ptx_arrive_barrier(mbarrier[0])

    mod = _apply(func)
    body = mod["main"].body
    assert _tc_fences(body) == 2, "the hand-written fences stay as they are"
    assert _tmem_fence_groups(body) == [3, 3], "and the tmem domain is added"


def test_manual_tc_fence_covers_a_tc_neighbour():
    @T.prim_func
    def func():
        with T.Kernel(1):
            C_tmem = T.decl_buffer((1,), T.uint32, scope="shared")
            T.tvm_storage_sync("shared")
            T.evaluate(_tc_fence())
            T.evaluate(_tc_use(C_tmem[0]))

    mod = _apply(func)
    body = mod["main"].body
    assert _fences(body) == 1, "the hand-written fence already retires the TC use"
    assert _tmem_fences(body) == 0


def test_manual_tmem_fence_is_not_duplicated():
    @T.prim_func
    def func():
        with T.Kernel(1):
            mbarrier = T.decl_buffer((1,), T.uint64, scope="shared.barrier")
            C_tmem = T.decl_buffer((1,), T.uint32, scope="shared")
            T.mbarrier_wait_parity(mbarrier[0], 0)
            T.evaluate(_tmem_fence(3))
            T.evaluate(_tmem_read(C_tmem[0]))
            T.evaluate(_tmem_fence(3))
            T.ptx_arrive_barrier(mbarrier[0])

    mod = _apply(func)
    body = mod["main"].body
    assert _fences(body) == 2, "exactly the two hand-written fences"
    assert _tc_fences(body) == 0


def test_no_tmem_ops_leaves_syncs_alone():
    @T.prim_func
    def func():
        with T.Kernel(1):
            T.tvm_storage_sync("shared")
            T.call_extern("handle", "generic_op")

    mod = _apply(func)
    assert _fences(mod["main"].body) == 0


def test_boundary_at_the_tail_of_a_block_reaches_outward():
    """A barrier that *ends* a block must still be fenced for the use after it."""

    @T.prim_func
    def func():
        with T.Kernel(1):
            C_tmem = T.decl_buffer((1,), T.uint32, scope="shared")
            for _ in T.serial(1):
                T.call_extern("handle", "generic_op")
                T.tvm_storage_sync("shared")
            T.evaluate(_tmem_read(C_tmem[0]))

    mod = _apply(func)
    body = mod["main"].body
    assert _tmem_fence_groups(body) == [3], "the loop's trailing barrier has to be fenced for the consumer that follows it"


def test_boundary_at_the_head_of_a_block_reaches_outward():
    @T.prim_func
    def func():
        with T.Kernel(1):
            C_tmem = T.decl_buffer((1,), T.uint32, scope="shared")
            T.evaluate(_tmem_write(C_tmem[0]))
            for _ in T.serial(1):
                T.tvm_storage_sync("shared")
                T.call_extern("handle", "generic_op")

    mod = _apply(func)
    body = mod["main"].body
    assert _tmem_fence_groups(body) == [2], "the loop's leading barrier has to be fenced for the producer before it"


def test_copy_engine_calls_are_not_fenced():
    """The smem<->tmem engines drain themselves with fence.mem inside the template."""

    @T.prim_func
    def func():
        with T.Kernel(1):
            smem = T.decl_buffer((4,), T.float32, scope="shared")
            T.tvm_storage_sync("shared")
            T.evaluate(_use("tl.tang_cp_tmem_to_shared", smem[0]))

    mod = _apply(func)
    assert _fences(mod["main"].body) == 0


def test_non_stcuv2_targets_are_left_untouched():
    @T.prim_func
    def func():
        with T.Kernel(1):
            C_tmem = T.decl_buffer((1,), T.uint32, scope="shared")
            T.tvm_storage_sync("shared")
            T.evaluate(_tmem_read(C_tmem[0]))

    mod = _apply(func, STCU_TARGET)
    assert _fences(mod["main"].body) == 0, "TMEM only exists on stcuv2"


# ---------------------------------------------------------------------------
# §2 pipeline 开关 + ISS 数值
# ---------------------------------------------------------------------------


def _make_tmem_gemm_kernel(M=128, N=128, K=64):
    """fp16 TN GEMM with a TMEM accumulator (the shape used by the TMEM tests)."""

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((N, K), T.float16),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(1, 1, threads=128):
            A_s = T.alloc_shared((M, K), T.float16)
            B_s = T.alloc_shared((N, K), T.float16)
            C_t = T.alloc_tmem((M, N), T.float32)
            T.copy(A, A_s)
            T.copy(B, B_s)
            T.gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=True)
            T.copy(C_t, C)

    return kernel


def _lower_src(pass_configs=None):
    config = dict(pass_configs or {})
    with tvm.target.Target(STCUV2_TARGET), tvm.transform.PassContext(config=config):
        return tilelang.lower(_make_tmem_gemm_kernel(), target=STCUV2_TARGET).kernel_source


def test_pipeline_injection_is_on_by_default():
    """stcuv2 injection defaults to on, mirroring the CUDA pipeline."""
    key = tilelang.PassConfigKey.TL_ENABLE_TANG_INJECT_TCGEN05_FENCE
    default = len(_RE_FENCE.findall(_lower_src()))
    explicit_on = len(_RE_FENCE.findall(_lower_src({key: True})))
    explicit_off = len(_RE_FENCE.findall(_lower_src({key: False})))
    assert default == explicit_on, f"default must match an explicit enable (default={default}, on={explicit_on})"
    assert default > explicit_off, f"default must inject fences, and False must opt out (default={default}, off={explicit_off})"


def test_pipeline_injects_both_domains():
    """The GEMM hands over TC work (mma) and TMEM register traffic (drain/LDT)."""
    src = _lower_src({tilelang.PassConfigKey.TL_ENABLE_TANG_INJECT_TCGEN05_FENCE: True})
    assert re.search(r"fence_tc<0>\(\)", src), "TC neighbours keep fence.tc"
    assert re.search(r"fence_tmem<3>\(\)", src), "LDT neighbours get fence.tmem"


def test_pipeline_flag_can_disable_injection():
    """tl.enable_tang_inject_tcgen05_fence=False restores the uninstrumented pipeline."""
    off = _lower_src({tilelang.PassConfigKey.TL_ENABLE_TANG_INJECT_TCGEN05_FENCE: False})
    on = _lower_src({tilelang.PassConfigKey.TL_ENABLE_TANG_INJECT_TCGEN05_FENCE: True})
    assert len(_RE_FENCE.findall(off)) < len(_RE_FENCE.findall(on))
    assert not re.search(r"fence_tmem<\d+>\(\)", off), "opting out must not leave injected tmem fences behind"


def test_injected_fence_gemm_matches_reference_on_iss():
    """The injected fences must not change the numerical result on the ISS."""
    import os

    if not os.environ.get("TANG_S3_PTCC_PATH") or not os.environ.get("TANG_S3_ISS_PATH"):
        pytest.skip("S3 simulator toolchain not configured (TANG_S3_PTCC_PATH/TANG_S3_ISS_PATH)")

    torch.manual_seed(0)
    M = N = 128
    K = 64
    a = torch.randn(M, K, dtype=torch.float16)
    b = torch.randn(N, K, dtype=torch.float16)

    with tvm.transform.PassContext(config={tilelang.PassConfigKey.TL_ENABLE_TANG_INJECT_TCGEN05_FENCE: True}):
        kernel = JITKernel(
            _make_tmem_gemm_kernel(M, N, K),
            out_idx=[-1],
            target=STCUV2_TARGET,
            execution_backend="simulator",
            verbose=False,
        )

    c = kernel(a.ptpu(), b.ptpu()).cpu()
    ref = (a.float() @ b.float().t()).to(torch.float32)
    torch.testing.assert_close(c, ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    tilelang.testing.main()
