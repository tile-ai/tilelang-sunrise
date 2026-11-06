"""stcuv2 PromoteTmemAccumulator — fragment 累加器自动提升到 tensor memory。"""

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tvm import IRModule, tirx

STCUV2_TARGET = {"kind": "tang", "arch": "stcuv2"}


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
    """改写 clear_accum 时不能丢掉 gemm 上的 TANG annotations。"""
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
    assert "tl::tang_tcgen05_mma_ss(" in artifact.kernel_source, artifact.kernel_source


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


def test_promote_folds_pure_staging_copy():
    """`C_local -> shared -> global` 纯中转时,折叠成 `C_local -> global`。

    stcuv2 的累加器只有"融合 drain 到 global"这一条出口,所以这种写法原本直接
    报错。当中转 buffer 只被这一对 copy 触碰(单写单读、无逐元素访问)时,省掉
    中转写出的字节完全相同,pass 就把它折掉,让 kernel 原样可用。
    """

    @T.prim_func
    def staged(
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

    script = _apply_promotion(staged)["main"].script()
    assert 'scope="shared.tmem"' in script, script
    # 中转 buffer 已无任何使用者,连分配一起消失
    assert "C_shared" not in script, script


def test_promote_folds_staged_column_slice():
    """中转只为了切片写回时也要能折:这里只写回前 96 列。"""
    M = N = 128
    K = 64

    @T.prim_func
    def sliced(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((N, K), "float16"),
        C: T.Tensor((N, N), "float32"),
    ):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            A_shared = T.alloc_shared((M, K), "float16")
            B_shared = T.alloc_shared((N, K), "float16")
            C_shared = T.alloc_shared((M, N), "float32")
            C_local = T.alloc_fragment((M, N), "float32")
            T.clear(C_local)
            T.copy(A[0, 0], A_shared)
            T.copy(B[0, 0], B_shared)
            T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C_shared)
            T.copy(C_shared[:, 0:96], C[0, 0])

    script = _apply_promotion(sliced)["main"].script()
    assert 'scope="shared.tmem"' in script, script
    assert "C_shared" not in script, script


def test_promote_rejects_staged_tile_with_second_use():
    """中转 buffer 还有第二个使用者时不能折 —— 仍按原样报错。

    省掉中转会连第二个使用者一起丢掉,所以只有"单写单读"才是纯中转。
    """

    @T.prim_func
    def shared_staged_twice(
        A: T.Tensor((128, 64), "float16"),
        B: T.Tensor((128, 64), "float16"),
        C: T.Tensor((128, 128), "float32"),
        D: T.Tensor((128, 128), "float32"),
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
            T.copy(C_shared, D[0, 0])

    _expect_promotion_error(shared_staged_twice, r"only a copy to global memory")


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

    ISS 上 TMEM 上电为零,所以这个 bug 在模拟器里是不可观测的 —— 必须在编译期拦住。
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


# ===========================================================================
# §9  A staging pair that runs under different guards is not folded
# ===========================================================================
#
# The fold moves the *producer*'s copy onto the consumer's global destination
# and deletes the consumer statement, so the two must run exactly where each
# other runs. Requiring the same loop nest is not enough: with the producer at
# the top level and the write-back under ``if cond``, folding makes the global
# write unconditional -- the output is written on iterations where the kernel
# never asked for it. Tracked as the same guard stack now, and reported with
# the reason instead of the generic "it is copied to shared".


def _guarded_staging_pair(where, M=128, N=128, K=64):
    """`C_local -> C_shared -> C`, with the guard on the consumer or producer."""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((N, K), "float16"),
        C: T.Tensor((M, N), "float32"),
        flag: T.int32,
    ):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            A_shared = T.alloc_shared((M, K), "float16")
            B_shared = T.alloc_shared((N, K), "float16")
            C_shared = T.alloc_shared((M, N), "float32")
            C_local = T.alloc_fragment((M, N), "float32")
            T.clear(C_local)
            T.copy(A[0, 0], A_shared)
            T.copy(B[0, 0], B_shared)
            T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            if where == "consumer":
                T.copy(C_local, C_shared)
                if flag > 0:
                    T.copy(C_shared, C[0, 0])
            elif where == "producer":
                if flag > 0:
                    T.copy(C_local, C_shared)
                T.copy(C_shared, C[0, 0])
            else:  # both: same guard, folding is still valid
                if flag > 0:
                    T.copy(C_local, C_shared)
                    T.copy(C_shared, C[0, 0])

    return main


@pytest.mark.parametrize("where", ["consumer", "producer"])
def test_promote_rejects_a_staging_pair_under_different_guards(where):
    """折叠会把 global 写搬过 if 边界,必须拒绝,并说明真正的原因。"""
    _expect_promotion_error(
        _guarded_staging_pair(where),
        r"do not run in the same loop nest and under the same guards",
    )


def test_promote_rejects_an_undrainable_write_back_width():
    """写回列数必须是 8 的倍数,报错要指名 buffer 和修复方向。

    LDT 的列粒度是 8 列(每行一个 32 字节 atom),没有更窄的指令去补余数,所以
    `[:, 0:100]` 这种切片只能拒绝 —— 但拒绝时要说清是哪块 buffer、多少列、怎么
    改,而不是抛一条内部检查。
    """

    @T.prim_func
    def odd_slice(
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
            T.copy(C_shared[:, 0:100], C[0, 0])

    with pytest.raises(Exception) as err:
        tilelang.lower(odd_slice, target=STCUV2_TARGET)

    msg = str(err.value)
    assert "written back to 'C' over 100 columns" in msg, msg
    assert "not a multiple of 8" in msg, msg
    assert "[:, 0:96] is drainable" in msg, msg


def test_promote_folds_a_pair_under_the_same_guard():
    """两边同处一个 if:折叠把写回留在同一个分支里,语义不变,仍要折。"""
    script = _apply_promotion(_guarded_staging_pair("both"))["main"].script()
    assert 'scope="shared.tmem"' in script, script
    assert "C_shared" not in script, script
    assert "if flag > 0:" in script, script


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"] + sys.argv[1:]))
