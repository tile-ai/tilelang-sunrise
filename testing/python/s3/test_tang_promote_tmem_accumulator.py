"""stcuv2 PromoteTmemAccumulator — fragment 累加器自动提升到 tensor memory。

stcuv2 没有累加到寄存器的 MMA 形式:SelectInst 把 C 在 shared.tmem 的路由到
tcgen5、C 在 shared 的路由到 wgmma,其余落到只存在于 stcu 的 TMMA。所以
``T.gemm`` 的 C 来自 ``T.alloc_fragment`` 时在 stcuv2 上根本没有可用指令。
``PromoteTmemAccumulator`` 把这种自然写法改写成 tmem 写法(见
src/tang/transform/promote_tmem_accumulator.cc)。

§1 改写契约      test_promote_*        (只跑 pass, 断言 IR)
§2 端到端数值    test_promoted_gemm_numeric_*
§3 拒绝路径      test_promote_rejects_*
"""

import pytest
import torch

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tvm import IRModule, tirx

STCUV2_TARGET = {"kind": "tang", "arch": "stcuv2"}

# tcgen5 reads exactly a 128-byte operand row per MMA, so the legal BK is fixed
# per dtype. Anything else has no legal swizzle -- see CheckOperandRowBytes.
_LEGAL_BK = {"float16": 64, "bfloat16": 64, "float32": 32}


# ===========================================================================
# Shared helpers
# ===========================================================================


def _apply_promotion(func):
    """Run PromoteTmemAccumulator alone, on the IR shape it sees in the pipeline.

    The pass is scheduled right before PipelinePlanning (tilelang/tang/pipeline.py),
    so all it needs is a bound target and a materialized kernel launch.
    """
    mod = IRModule({"main": func})
    mod = tirx.transform.BindTarget(tvm.target.Target(STCUV2_TARGET))(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    return tilelang.tang.transform.PromoteTmemAccumulator()(mod)


def _expect_promotion_error(func, pattern: str):
    """A kernel the pass refuses must fail *here*, with a directed message.

    Refusing is not the same as leaving the kernel alone: a fragment accumulator
    has no lowering on stcuv2 at all, so a pass-through would only resurface as
    the far more opaque "No GEMM implementation registered for instruction
    tang.tmma".
    """
    with pytest.raises(Exception, match=pattern):
        _apply_promotion(func)


def _fragment_matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float32", gemm_annotations=None):
    """The natural fragment-accumulator matmul: T.clear + T.gemm + T.copy out."""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True, annotations=gemm_annotations)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


# ===========================================================================
# §1  Rewrite contract — run the pass alone and assert the resulting IR
# ===========================================================================


def test_promote_rewrites_scope_fill_and_clear_accum():
    """三件事必须同时发生:scope 变 shared.tmem、T.clear 消失、clear_accum 变 k==0。

    三者是一体的:tmem 不可直接存(LowerSharedTmem 拒绝写入数据),所以初始化只能
    靠 MMA 自带的累加器清零,``T.clear`` 必须折进 ``clear_accum``。少做任何一件都
    会算错——留着 T.clear 会在 LowerSharedTmem 处报错,丢掉 clear_accum 则累加器
    从上一个 tile 的残值开始。
    """
    script = _apply_promotion(_fragment_matmul(256, 256, 256, 128, 128, 64))["main"].script()
    assert 'scope="shared.tmem"' in script, script
    assert 'scope="local.fragment"' not in script, script
    assert "T.fill(" not in script, script
    # clear_accum is the 10th gemm argument; k == 0 is what replaced T.bool(False).
    assert "k == 0" in script, script


def test_promote_keeps_gemm_annotations():
    """改写 clear_accum 时不能丢掉 gemm 上的 TANG annotations。

    重建 Call 而不是 CopyOnWrite 会静默丢掉 tang_k_step / tang_scale_*,后者决定
    缩放 GEMM 的描述符,丢了以后是"看起来合理但错"的结果。
    """
    annotations = {"tang_k_step": 4, "tang_scale_block": 64, "preserve_marker": 7}
    func = _fragment_matmul(256, 256, 256, 128, 128, 64, gemm_annotations=annotations)

    def gemm_call(func):
        calls = []

        def collect(node):
            if isinstance(node, tirx.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == "tl.tileop.gemm":
                calls.append(node)

        tirx.stmt_functor.post_order_visit(func.body, collect)
        assert len(calls) == 1
        return calls[0]

    before = gemm_call(func)
    for key, value in annotations.items():
        assert int(before.annotations[key]) == value
    promoted = _apply_promotion(func)["main"]
    after = gemm_call(promoted)
    assert len(after.args) == 13
    tvm.ir.assert_structural_equal(after.annotations, before.annotations)
    assert "k == 0" in promoted.script()


def test_promote_leaves_explicit_tmem_accumulator_alone():
    """已经写成 T.alloc_tmem 的 kernel 必须逐字不变(pass 是幂等的 no-op)。"""

    @T.prim_func
    def explicit_tmem(
        A: T.Tensor((128, 64), "float16"),
        B: T.Tensor((128, 64), "float16"),
        C: T.Tensor((128, 128), "float16"),
    ):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 64), "float16")
            B_shared = T.alloc_shared((128, 64), "float16")
            C_tmem = T.alloc_tmem((128, 128), "float32")
            for k in T.Pipelined(1, num_stages=1):
                T.copy(A[0, k * 64], A_shared)
                T.copy(B[0, k * 64], B_shared)
                T.gemm(A_shared, B_shared, C_tmem, transpose_B=True, clear_accum=k == 0)
            T.copy(C_tmem, C[0, 0])

    # Compare against the same prefix passes the pass itself runs behind, so the
    # only possible difference is the promotion.
    mod = IRModule({"main": explicit_tmem})
    mod = tirx.transform.BindTarget(tvm.target.Target(STCUV2_TARGET))(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    untouched = mod["main"].script()
    promoted = tilelang.tang.transform.PromoteTmemAccumulator()(mod)["main"].script()
    assert promoted == untouched


def test_promote_leaves_non_accumulator_fragment_alone():
    """不是 GEMM 累加器的 fragment 一概不动,哪怕它被逐元素读写。

    这是 pass 的作用域边界:只有 T.gemm 的 C 没有 stcuv2 指令可用,普通 fragment
    的寄存器路径是完好的。
    """

    @T.prim_func
    def elementwise(A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")):
        with T.Kernel(1, threads=128):
            acc = T.alloc_fragment((128, 128), "float32")
            T.clear(acc)
            for i, j in T.Parallel(128, 128):
                acc[i, j] = A[i, j] * 2.0
            T.copy(acc, C)

    script = _apply_promotion(elementwise)["main"].script()
    assert 'scope="local.fragment"' in script, script
    assert "shared.tmem" not in script, script


def test_promoted_kernel_lowers_to_tcgen5_mma():
    """提升之后必须真的走到 tcgen5 MMA,而不是别的指令。"""
    func = _fragment_matmul(256, 256, 256, 128, 128, 64)
    with tvm.transform.PassContext(), tvm.target.Target(STCUV2_TARGET):
        artifact = tilelang.lower(func, target=STCUV2_TARGET)
    assert "tang::ptx::mma<" in artifact.kernel_source, artifact.kernel_source


# ===========================================================================
# §2  End-to-end numerics on the ISS
# ===========================================================================


@pytest.mark.parametrize("dtype, tol", [("float16", 2e-2), ("float32", 2e-2)])
def test_promoted_gemm_numeric(dtype, tol):
    """多 block + K-tiling 的 fragment 写法必须与 torch CPU 结果一致。

    M=N=K=256 配 128x128 的 tile 同时覆盖两件事:4 个 block(检验 drain 的 per-block
    base offset)和 K>BK 的多次迭代(检验折出来的 ``clear_accum=(k==0)`` —— 若它恒为
    真则只剩最后一片 K 的贡献,若恒为假则叠加上一个 tile 的残值)。
    """
    M = N = K = 256
    block_K = _LEGAL_BK[dtype]
    torch_dtype = getattr(torch, dtype)
    kernel = tilelang.compile(
        _fragment_matmul(M, N, K, 128, 128, block_K, dtype),
        out_idx=[2],
        target=STCUV2_TARGET,
        execution_backend="simulator",
    )
    a = torch.randn(M, K, dtype=torch_dtype)
    b = torch.randn(N, K, dtype=torch_dtype)
    out = kernel(a.ptpu(), b.ptpu())
    ref = (a.to(torch.float32) @ b.to(torch.float32).T).to(torch_dtype)
    torch.testing.assert_close(out.cpu(), ref, rtol=tol, atol=tol)


# ===========================================================================
# §3  Rejection paths
# ===========================================================================


@pytest.mark.parametrize("epilogue", ["in_place", "into_other"])
def test_promote_rejects_elementwise_epilogue(epilogue):
    """循环后逐元素访问累加器必须报错,而不是静默算错。

    读回 tmem 需要 LDT 到 fragment,而该路径今天只绑定 warp 0
    (src/tang/op/copy.cc),通用 fragment C 却铺在全部 warp 上,两者对不上。
    原地更新(``C_local[i, j] += 1``)先命中写路径,读到别处则命中读路径,两条
    诊断都要能指出是哪一种。
    """

    @T.prim_func
    def in_place(
        A: T.Tensor((128, 64), "float16"),
        B: T.Tensor((128, 64), "float16"),
        C: T.Tensor((128, 128), "float32"),
    ):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 64), "float16")
            B_shared = T.alloc_shared((128, 64), "float16")
            C_local = T.alloc_fragment((128, 128), "float32")
            T.clear(C_local)
            T.copy(A[0, 0], A_shared)
            T.copy(B[0, 0], B_shared)
            T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            for i, j in T.Parallel(128, 128):
                C_local[i, j] = C_local[i, j] + 1.0
            T.copy(C_local, C[0, 0])

    @T.prim_func
    def into_other(
        A: T.Tensor((128, 64), "float16"),
        B: T.Tensor((128, 64), "float16"),
        C: T.Tensor((128, 128), "float32"),
    ):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 64), "float16")
            B_shared = T.alloc_shared((128, 64), "float16")
            C_local = T.alloc_fragment((128, 128), "float32")
            scaled = T.alloc_fragment((128, 128), "float32")
            T.clear(C_local)
            T.copy(A[0, 0], A_shared)
            T.copy(B[0, 0], B_shared)
            T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            for i, j in T.Parallel(128, 128):
                scaled[i, j] = C_local[i, j] * 2.0
            T.copy(scaled, C[0, 0])

    func, pattern = {
        "in_place": (in_place, r"written element-wise"),
        "into_other": (into_other, r"read element-wise"),
    }[epilogue]
    _expect_promotion_error(func, pattern)


def test_promote_rejects_copy_to_shared():
    """累加器只有到 global 的 drain;拷到 shared 没有对应路径。"""

    @T.prim_func
    def drain_to_shared(
        A: T.Tensor((128, 64), "float16"),
        B: T.Tensor((128, 64), "float16"),
        C: T.Tensor((128, 128), "float32"),
    ):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 64), "float16")
            B_shared = T.alloc_shared((128, 64), "float16")
            C_shared = T.alloc_shared((128, 128), "float32")
            C_local = T.alloc_fragment((128, 128), "float32")
            T.clear(C_local)
            T.copy(A[0, 0], A_shared)
            T.copy(B[0, 0], B_shared)
            T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[0, 0])

    _expect_promotion_error(drain_to_shared, r"only a copy to global memory")


def test_promote_rejects_nonzero_init():
    """非零初值无法折进 clear_accum —— MMA 的清零只能清成 0。"""

    @T.prim_func
    def nonzero_init(
        A: T.Tensor((128, 64), "float16"),
        B: T.Tensor((128, 64), "float16"),
        C: T.Tensor((128, 128), "float32"),
    ):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 64), "float16")
            B_shared = T.alloc_shared((128, 64), "float16")
            C_local = T.alloc_fragment((128, 128), "float32")
            T.fill(C_local, 1.0)
            T.copy(A[0, 0], A_shared)
            T.copy(B[0, 0], B_shared)
            T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C[0, 0])

    _expect_promotion_error(nonzero_init, r"non-zero value")


def test_promote_rejects_uninitialized_accumulator():
    """既没有 T.clear 也没有 clear_accum 时报错,而不是从残值开始累加。

    编译期必须拒绝未初始化的累加器,不能依赖运行时内存初值。
    """

    @T.prim_func
    def no_init(
        A: T.Tensor((128, 64), "float16"),
        B: T.Tensor((128, 64), "float16"),
        C: T.Tensor((128, 128), "float32"),
    ):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 64), "float16")
            B_shared = T.alloc_shared((128, 64), "float16")
            C_local = T.alloc_fragment((128, 128), "float32")
            T.copy(A[0, 0], A_shared)
            T.copy(B[0, 0], B_shared)
            T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C[0, 0])

    _expect_promotion_error(no_init, r"never initialized")


@pytest.mark.parametrize(
    "dtype, block_K, row_bytes, legal_k",
    [
        pytest.param("float16", 32, 64, 64, id="fp16-BK32-half-row"),
        pytest.param("float32", 16, 64, 32, id="fp32-BK16-half-row"),
    ],
)
def test_promote_rejects_narrow_operand_row(dtype, block_K, row_bytes, legal_k):
    """操作数行不足 128 字节时给出可操作报错,而不是 ptcc 那句无指向的诊断。

    窄行没有合法 swizzle,ptcc 只会说 "Unsupported A matrix swizzle layout",既不
    指出是哪个操作数也不给出该用的 K。为 fragment 路径写的 kernel 没有理由把 BK
    调成 tcgen5 的 128 字节行,所以这是提升最常撞上的约束。
    """
    func = _fragment_matmul(256, 256, 256, 128, 128, block_K, dtype)
    _expect_promotion_error(func, rf"spans {row_bytes} bytes per MMA row.*Use K == {legal_k}")


def test_promote_does_not_mask_fragment_operand_error():
    """C 被提升后,fragment 的 A 操作数仍要由 gemm_tcgen5.py 报出原本的错。

    提升不应该把操作数的问题盖过去:A/B 必须在 shared(A 也可在 tmem),这条规则的
    真值在 gemm_tcgen5.py 的 _check_operand_scopes,pass 不重复实现。
    """

    @T.prim_func
    def frag_a(
        A: T.Tensor((128, 64), "float16"),
        B: T.Tensor((128, 64), "float16"),
        C: T.Tensor((128, 128), "float32"),
    ):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            A_frag = T.alloc_fragment((128, 64), "float16")
            B_shared = T.alloc_shared((128, 64), "float16")
            C_local = T.alloc_fragment((128, 128), "float32")
            T.copy(A[0, 0], A_frag)
            T.copy(B[0, 0], B_shared)
            T.clear(C_local)
            T.gemm(A_frag, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C[0, 0])

    with (
        pytest.raises(Exception, match=r"needs A and B in shared memory"),
        tvm.transform.PassContext(),
        tvm.target.Target(STCUV2_TARGET),
    ):
        tilelang.lower(frag_a, target=STCUV2_TARGET)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"] + sys.argv[1:]))
