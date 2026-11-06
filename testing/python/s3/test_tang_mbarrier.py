"""stcuv2 mbarrier 原语测试。

覆盖 T.alloc_barrier / T.mbarrier_arrive / T.mbarrier_wait_parity /
T.mbarrier_expect_tx 四个原语在 TANG 上的 lowering 与拒绝路径。

S3 硬件只实现了四条 mbarrier 原子操作
(expect_tx / complete_tx / arrive / arrive_drop),init 与所有 wait 变体都是
device 库用软件模拟的 —— init 是一次带位布局的存储 + fence.mem,wait 是轮询
64 位 barrier 字高位的 phase 域。于是有两个测试上的后果:

* 没有阻塞 wait 指令,`.wait(parity)` 由 tl_templates/tang/barrier.h 里的自旋
  实现;lowering 断言看不出自旋写错。
* `init` 由 LowerSharedBarrier 注入,TANG 有自己的一份
  (src/tang/transform/lower_shared_barrier.cc)。不复用 CUDA 那份是因为
  src/cuda/transform 在 USE_CUDA=OFF 时整个不编译,而 S3 构建正是这个配置;
  且 CUDA 那份带 cluster barrier 分支,TANG 现在也有（shared.cluster_barrier,
  远端 arrive 走 DSHM）。漏掉 init 不会
  报错,只会让 arrive/wait 在一块未初始化的共享内存上工作 —— §1 的 init 断言
  钉的是这条静默失败,§5 单独钉这个 pass 自身。

expect_tx 只接通指令,不保证与 bulk copy 的 tx 配对语义:递减侧 complete_tx 的
硬件计数被厂商标为不可靠。所以这里只断言它发射得出来,不做数值配对用例。
arrive_expect_tx 同样只断言发射:它调用 __mbarrier_arrive_expect_tx,设备库
把它实现成先 expect_tx 再 arrive,不是一条融合原子指令。
"""

import pathlib
import re

import pytest

import tilelang
import tilelang.testing
import tilelang.tang.language as T
from tilelang import tvm as tvm
from tilelang.tang import transform as tang_transform
from tilelang.testing.tang_ir import mma_descriptors

STCUV2_TARGET = {"kind": "tang", "arch": "stcuv2"}
STCU_TARGET = {"kind": "tang", "arch": "stcu"}


# ===========================================================================
# Shared helpers
# ===========================================================================


def _lower_src(func, target=STCUV2_TARGET):
    with tvm.target.Target(target):
        return tilelang.lower(func, target=target).kernel_source


def _assert_in_ir(txt: str, *patterns: str):
    for pat in patterns:
        assert re.search(pat, txt), f"Expected '{pat}' NOT found in IR:\n{txt}"


def _assert_not_in_ir(txt: str, *patterns: str):
    for pat in patterns:
        assert not re.search(pat, txt), f"Unexpected '{pat}' found in IR:\n{txt}"


# ===========================================================================
# §0  API existence -- the dialect's export chain
# ===========================================================================


def test_mbarrier_apis_exist_in_tang_dialect():
    """四个原语必须能从 tilelang.tang.language 直接拿到。

    TANG 方言本身没有列举 mbarrier 符号,它们是经
    tang -> cuda.language -> language.common 的星号导入链带进来的,所以这条
    断言防的是导入链某一环收窄了 __all__。
    """
    for name in ("alloc_barrier", "mbarrier_arrive", "mbarrier_wait_parity", "mbarrier_expect_tx"):
        assert callable(getattr(T, name)), f"T.{name} is not exported"


# ===========================================================================
# §1  Lowering
# ===========================================================================


def _make_block_sync_kernel(size=128, threads=128, arrive_count=None):
    """用 mbarrier 复刻一次 __syncthreads():写 shared -> arrive -> wait -> 反向读。

    反向读(s[size-1-i])让每个线程取的都是别的线程写的槽,于是结果有一个好写的
    闭式解(reverse(A))。注意它并不证明 barrier 提供了互斥:同样的 shared RAW
    依赖会让 ThreadStorageSync 自动补一条 __syncthreads()(见模块 docstring)。
    """
    count = threads if arrive_count is None else arrive_count

    @T.prim_func
    def kernel(A: T.Tensor((size,), T.float32), B: T.Tensor((size,), T.float32)):
        with T.Kernel(1, threads=threads):
            s = T.alloc_shared((size,), T.float32)
            bar = T.alloc_barrier(count)
            for i in T.Parallel(size):
                s[i] = A[i]
            T.mbarrier_arrive(bar[0])
            T.mbarrier_wait_parity(bar[0], 0)
            for i in T.Parallel(size):
                B[i] = s[size - 1 - i]

    return kernel


def test_barrier_lowering_emits_header_storage_and_calls():
    """barrier.h 必须被条件发射,且存储、init、arrive、wait 四者齐全。

    存储是 `__shared__ uint64_t X_mem[N]` + reinterpret_cast<Barrier*>,所以
    Barrier 必须是 8 字节 POD —— 头文件里有 static_assert 守着。
    """
    src = _lower_src(_make_block_sync_kernel())
    _assert_in_ir(
        src,
        r"#include <tl_templates/tang/barrier\.h>",
        r"__shared__ uint64_t \w+_mem\[1\]",
        r"reinterpret_cast<Barrier\*>",
        r"\.init\(128\)",
        r"\.arrive\(\)",
        r"\.wait\(0\)",
    )


def test_barrier_init_is_injected_under_elected_thread():
    """init 必须真的被注入,并且只由被选中的那一个线程执行。

    这是静默失败的回归:少了 init,编译照过、arrive/wait 照发,只是 barrier 从未
    被写入 arrive count。守卫用 tl_shuffle_elect<0>()(整个 block 选一个线程),
    随后 fence + __syncthreads() 把 init 发布给其余线程。
    """
    src = _lower_src(_make_block_sync_kernel())
    _assert_in_ir(
        src,
        r"tl::tl_shuffle_elect<0>\(\)",
        r"\.init\(",
        r"tl::fence_barrier_init\(\)",
        r"__syncthreads\(\)",
    )


def test_arrive_count_is_taken_verbatim():
    """arrive count 是线程数语义,原样落到 init 上。

    TANG 另有一条 warp 级的 fence-绑定-mbarrier 路径,那条路的 count 是 warp
    数;这里钉住的是逐线程 arrive 这条路,与 CUDA 语义一致。
    """
    src = _lower_src(_make_block_sync_kernel(threads=256, arrive_count=256))
    _assert_in_ir(src, r"\.init\(256\)")


def test_multiple_barriers_get_independent_inits():
    """一次 alloc_barrier([...]) 要分配连续存储,并逐个 init 各自的 count。"""

    @T.prim_func
    def kernel(A: T.Tensor((128,), T.float32), B: T.Tensor((128,), T.float32)):
        with T.Kernel(1, threads=128):
            s = T.alloc_shared((128,), T.float32)
            bars = T.alloc_barrier([128, 128])
            for i in T.Parallel(128):
                s[i] = A[i]
            T.mbarrier_arrive(bars[0])
            T.mbarrier_wait_parity(bars[0], 0)
            for i in T.Parallel(128):
                B[i] = s[127 - i]
            T.mbarrier_arrive(bars[1])
            T.mbarrier_wait_parity(bars[1], 0)

    src = _lower_src(kernel)
    _assert_in_ir(src, r"__shared__ uint64_t \w+_mem\[2\]")
    assert len(re.findall(r"\.init\(128\)", src)) == 2, f"expected two .init(128) calls, got:\n{src}"


def test_expect_tx_lowers_to_expect_transaction():
    """expect_tx 指令存在,必须能发射出来。

    注意这只保证指令发得出、跑得动;它与 bulk copy 的 tx 配对语义**不保证**,
    因为递减侧 complete_tx 的硬件计数不可靠。
    """

    @T.prim_func
    def kernel(A: T.Tensor((128,), T.float32), B: T.Tensor((128,), T.float32)):
        with T.Kernel(1, threads=128):
            s = T.alloc_shared((128,), T.float32)
            bar = T.alloc_barrier(128)
            T.mbarrier_expect_tx(bar[0], 512)
            for i in T.Parallel(128):
                s[i] = A[i]
            T.mbarrier_arrive(bar[0])
            T.mbarrier_wait_parity(bar[0], 0)
            for i in T.Parallel(128):
                B[i] = s[127 - i]

    _assert_in_ir(_lower_src(kernel), r"\.expect_transaction\(512\)")


def test_arrive_expect_tx_lowers_to_device_helper():
    """接到 __mbarrier_arrive_expect_tx,经 Barrier::arrive_expect_tx 发射。

    设备库把这条 shared-memory 接口实现成 expect_tx 之后再 arrive,没有融合
    硬件指令。这里只钉 lowering,不把 tx 计数和 bulk copy 配成数值用例。
    """

    @T.prim_func
    def kernel(A: T.Tensor((128,), T.float32), B: T.Tensor((128,), T.float32)):
        with T.Kernel(1, threads=128):
            s = T.alloc_shared((128,), T.float32)
            bar = T.alloc_barrier(128)
            T.mbarrier_arrive_expect_tx(bar[0], 512)
            T.mbarrier_wait_parity(bar[0], 0)
            for i in T.Parallel(128):
                s[i] = A[i]
            for i in T.Parallel(128):
                B[i] = s[127 - i]

    _assert_in_ir(_lower_src(kernel), r"\.arrive_expect_tx\(512\)")


# ===========================================================================
# §2  Rejection path -- the error must stop in lowering, not in ptcc
# ===========================================================================


def test_alloc_barrier_rejected_on_stcu():
    """S2 没有 mbarrier(整个 __mbarrier_* 家族由 __Tang_ARCH__ >= 200 门控)。

    不提前拒绝的话,ptcc 会在 barrier.h 内部报一个既不提 barrier 也不提 arch
    的错。
    """
    with pytest.raises(ValueError, match=r"stcuv2"):
        _lower_src(_make_block_sync_kernel(), target=STCU_TARGET)


def test_local_cta_arrive_lowers_to_block_arrive():
    """cta_id 只接受常量 0,落到本 CTA 的 .arrive()。

    远端 rank 要走 __set_block_rank 加 DSHM __mbarrier_arrive,这条没有接。
    前端不传 pred,等价于始终 arrive。
    """

    @T.prim_func
    def kernel(A: T.Tensor((128,), T.float32), B: T.Tensor((128,), T.float32)):
        with T.Kernel(1, threads=128):
            s = T.alloc_shared((128,), T.float32)
            bar = T.alloc_cluster_barrier(128)
            for i in T.Parallel(128):
                s[i] = A[i]
            T.mbarrier_arrive(bar, 0)
            T.mbarrier_wait_parity(bar, 0)
            for i in T.Parallel(128):
                B[i] = s[127 - i]

    src = _lower_src(kernel)
    _assert_in_ir(src, r"\.arrive\(\);", r"\.init\(128\)")


def test_remote_cta_arrive_lowers_to_cluster_arrive():
    """非 0 的常量 cta_id 生成远端 arrive：mapa 到 peer CTA 的 mbarrier 后
    DSHM arrive（tl::cluster_arrive_barrier，依赖 LLVM 的 DSHM builtins）。"""

    @T.prim_func
    def kernel(A: T.Tensor((128,), T.float32), B: T.Tensor((128,), T.float32)):
        with T.Kernel(1, threads=128):
            s = T.alloc_shared((128,), T.float32)
            bar = T.alloc_cluster_barrier(128)
            for i in T.Parallel(128):
                s[i] = A[i]
            T.mbarrier_arrive(bar, 1)
            T.mbarrier_wait_parity(bar, 0)
            for i in T.Parallel(128):
                B[i] = s[127 - i]

    src = _lower_src(kernel)
    _assert_in_ir(src, r"tl::cluster_arrive_barrier\([^,]+,\s*\(int\)\(1\)\)")
    _assert_in_ir(src, r"cluster\.h")


def test_nonconstant_cta_arrive_lowers_with_runtime_rank():
    """运行时 rank 不能静默当成 cta 0：它作为寄存器操作数进入远端 arrive。"""

    @T.prim_func
    def kernel(A: T.Tensor((128,), T.int32)):
        with T.Kernel(1, threads=128):
            bar = T.alloc_cluster_barrier(1)
            T.mbarrier_arrive(bar, T.block_rank_in_cluster())
            A[0] = 0

    src = _lower_src(kernel)
    _assert_in_ir(src, r"tl::cluster_arrive_barrier\([^,]+,\s*\(int\)\(")
    _assert_not_in_ir(src, r"\.arrive\(\);")


def test_negative_arrive_expect_tx_rejected():
    """负数 tx 不能交给 unsigned 参数截断。"""

    @T.prim_func
    def kernel(A: T.Tensor((128,), T.float32)):
        with T.Kernel(1, threads=128):
            bar = T.alloc_barrier(1)
            T.mbarrier_arrive_expect_tx(bar[0], -1)
            A[0] = 0

    with pytest.raises(Exception, match=r"transaction count"):
        _lower_src(kernel)


def test_cp_async_barrier_noinc_lowers_to_g2s_fence_arrive():
    """没有 cp.async.mbarrier.arrive。noinc 接到 fence.arrive_mbar.mem。"""

    @T.prim_func
    def kernel(A: T.Tensor((128,), T.float32), B: T.Tensor((128,), T.float32)):
        with T.Kernel(1, threads=128):
            s = T.alloc_shared((128,), T.float32)
            bar = T.alloc_barrier(1)
            T.cp_async_barrier_noinc(bar[0])
            T.mbarrier_wait_parity(bar[0], 0)
            for i in T.Parallel(128):
                s[i] = A[i]
            for i in T.Parallel(128):
                B[i] = s[127 - i]

    _assert_in_ir(_lower_src(kernel), r"\.cp_async_arrive_noinc\(\)")


# ===========================================================================
# §3  GEMM + mbarrier -- the counterpart of CUDA's tcgen05 gemm case
# ===========================================================================
#
# The pattern in testing/python/language/test_tilelang_language_tcgen05_gemm.py
# is `gemm(A_s, B_s, C_tmem, mbar)`: the MMA publishes its own completion into an
# mbarrier and the consumer reads TMEM after `mbarrier_wait_parity`. That file is
# skipped 16/17 on this machine (it needs a real sm_100 card and hard-codes
# target="cuda"), so it cannot be used to exercise TANG.
#
# This section reproduces its **synchronisation** shape: the order between the
# MMA and the TMEM read is established by tcgen05 fences, and the mbarrier only
# carries the cross-thread rendezvous (in place of __syncthreads()). The coverage
# is new: the existing gemm cases never touch an mbarrier, and the mbarrier cases
# of §1-§2 never touch gemm/TMEM.
#
# The variant where the MMA publishes its own completion (the original meaning of
# CUDA's mbar=) is in §4; the two are not the same thing: the gemm here is
# blocking, the one there is not.
#
# The drain goes through T.copy(C_t, D) directly, like every other S3 gemm case.
# The three-hop TMEM->shared->global path is deliberately avoided: it requires an
# explicit tang_swizzle_atom_bytes annotation, a missing one fails silently, and
# mixing that into an mbarrier case would make a failure hard to attribute -- the
# three-hop path has its own coverage in test_tang_tcgen05_cpt2s.py.


def _make_gemm_mbarrier_kernel(num_k_tiles=1, fence_group=0):
    M = N = 128
    BK = 64  # with fp16, K=64 sits exactly on the 128-byte operand row cap
    K = BK * num_k_tiles

    @T.prim_func
    def kernel(A: T.Tensor((M, K), T.float16), B: T.Tensor((N, K), T.float16), D: T.Tensor((M, N), T.float32)):
        with T.Kernel(1, 1, threads=128):
            A_s = T.alloc_shared((M, BK), T.float16)
            B_s = T.alloc_shared((N, BK), T.float16)
            C_t = T.alloc_tmem((M, N), T.float32)
            bar = T.alloc_barrier(128)
            for k in T.serial(num_k_tiles):
                T.copy(A[0, k * BK], A_s)
                T.copy(B[0, k * BK], B_s)
                T.tcgen05_before_thread_sync(fence_group)
                T.gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=(k == 0))
                T.tcgen05_after_thread_sync(fence_group)
            T.mbarrier_arrive(bar[0])
            T.mbarrier_wait_parity(bar[0], 0)
            T.copy(C_t, D)

    return kernel


def test_gemm_mbarrier_lowering_keeps_both_fence_and_barrier():
    """一个 kernel 里 gemm 的 fence 与 mbarrier 必须共存,且顺序是 MMA -> 会合。

    钉这个顺序是因为反过来（先会合再 fence）读到的会是 MMA 未完成的 TMEM。
    """
    src = _lower_src(_make_gemm_mbarrier_kernel())
    _assert_in_ir(
        src,
        r"#include <tl_templates/tang/barrier\.h>",
        r"fence_tc<0>\(\)",
        r"\.init\(128\)",
        r"\.arrive\(\)",
        r"\.wait\(0\)",
    )
    assert re.search(r"fence_tc<0>\(\)[\s\S]*?\.arrive\(\)[\s\S]*?\.wait\(0\)", src), (
        f"expected fence_tc -> arrive -> wait ordering in IR:\n{src}"
    )


# ===========================================================================
# §4  Completion mbarrier -- the MMA publishes its own completion
# ===========================================================================
#
# §3 is "fences establish the order + the mbarrier carries the cross-thread
# rendezvous", with the MMA itself blocking. This section is the real counterpart
# of CUDA's `T.gemm(A, B, C_tmem, mbar=bar)`: the MMA does not block; the issuing
# warp binds its tensor-core fence group to the barrier
# (fence_tc_arrive_mbarrier), the hardware posts the arrive when that group's MMA
# retires, and the consumer waits on the parity.
#
# Three hard constraints determine the assertions below (from
# cccl/tang/__ptx/instructions/fence.h):
#
#   1. A fence group is **warp private** ("synchronize operations in current
#      warp"), so one warp's fence cannot cover a sibling warp's MMA.
#   2. The arrive is **warp level and exactly once** (.noinc), not once per
#      thread.
#   3. So N issuing warps post N arrives, and the barrier's arrive count must
#      equal the warp count -- while the count is fixed in T.alloc_barrier() and
#      is not available to the GEMM lowering.
#
# Lowering therefore collapses the warp partition to a single warp when mbar= is
# used, keeping the arrive count at 1 to match the CUDA source's
# alloc_barrier(1). test_..._collapses_to_one_warp pins exactly that: bringing
# multi-warp back without solving the count at the same time leaves the barrier
# waiting forever.
#
# "Arrive exactly once" is a measured conclusion, not an inference. Three
# controls: wait(0) returns normally; wait(1) hangs (the phase flips only once);
# and without mbar=, wait(0) hangs too (so the wait is not a no-op and really
# needs that arrive to be released). The last two are judged by timeout -- too
# slow and too brittle, so they are not part of CI.


def _make_completion_mbarrier_kernel(api="tcgen05"):
    """CUDA 完成事件模式的 TANG 版。

    三种 api 对应三种提交方式,差别只在 barrier 由谁、什么时候放行:

      "tcgen05"    T.tcgen05_gemm(mbar=) —— 异步接口,wait 由调用方写。
      "sync"       T.gemm(mbar=)        —— 同步接口,wait 由 lowering 自动补,
                                            所以 kernel 里**故意不写** wait。
      "mma_arrive" GEMM 自己同步收尾,再由选出的 warp 手工提交完成事件。
    """
    M = N = 128
    K = 64  # with fp16, K=64 sits exactly on the 128-byte operand row cap

    @T.prim_func
    def kernel(A: T.Tensor((M, K), T.float16), B: T.Tensor((N, K), T.float16), D: T.Tensor((M, N), T.float32)):
        with T.Kernel(1, 1, threads=128):
            A_s = T.alloc_shared((M, K), T.float16)
            B_s = T.alloc_shared((N, K), T.float16)
            C_t = T.alloc_tmem((M, N), T.float32)
            bar = T.alloc_barrier(1)
            T.copy(A, A_s)
            T.copy(B, B_s)
            if api == "mma_arrive":
                T.gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=True)
                if T.get_thread_binding() < 32:
                    T.tcgen05_mma_arrive(bar[0])
                T.mbarrier_wait_parity(bar[0], 0)
            elif api == "sync":
                T.gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=True, mbar=bar[0])
            else:
                T.tcgen05_gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=True, mbar=bar[0])
                T.mbarrier_wait_parity(bar[0], 0)
            T.copy(C_t, D)

    return kernel


def _gemm_partition(kernel):
    """从 IR 指令描述符的 subM/subN 反推 warp_m / warp_n (M=N=128)。"""
    desc = mma_descriptors(kernel, STCUV2_TARGET)
    assert desc.instr is not None, "no tcgen05 MMA instruction descriptor in the lowered IR"
    return desc.partition(128, 128)


@pytest.mark.parametrize("api", ["tcgen05", "sync"])
def test_gemm_completion_mbarrier_lowers_to_a_fence_arrive(api):
    """mbar= 必须把 barrier 传进 MMA 模板,并由消费者等 parity。"""
    src = _lower_src(_make_completion_mbarrier_kernel(api))
    _assert_in_ir(
        src,
        r"#include <tl_templates/tang/barrier\.h>",
        r"\.init\(1\)",
        # With mbar= the completion protocol becomes arrive_on_tc_fence instead
        # of fence_tc.
        r"arrive_on_tc_fence\(\)",
        r"\.wait\(0\)",
    )
    assert re.search(r"arrive_on_tc_fence\(\)[\s\S]*?\.wait\(0\)", src), f"expected MMA issue before the parity wait:\n{src}"


def test_sync_gemm_supplies_the_parity_wait_itself():
    """T.gemm(mbar=) 是同步接口,wait 必须由 lowering 补上。

    CUDA 那边 T.gemm 文档写明「隐式插入 mbarrier_wait_parity」,只有
    T.tcgen05_gemm 才把 wait 留给调用方。TANG 上要是不跟着补,照搬 CUDA 写法的
    kernel 会静默竞争 —— 排水读到的是 MMA 还没写完的 TMEM,不报错只算错。

    钉法是:"sync" 这个 kernel 源码里根本没有 wait,IR 里却必须有。
    """
    src = _lower_src(_make_completion_mbarrier_kernel("sync"))
    _assert_in_ir(src, r"\.wait\(0\)")

    # The other way round: an async interface must not insert a wait on its own,
    # otherwise manual scheduling would lose its point. A tcgen05_gemm kernel
    # that writes no wait is the control here.
    M = N = 128
    K = 64

    @T.prim_func
    def no_wait(A: T.Tensor((M, K), T.float16), B: T.Tensor((N, K), T.float16), D: T.Tensor((M, N), T.float32)):
        with T.Kernel(1, 1, threads=128):
            A_s = T.alloc_shared((M, K), T.float16)
            B_s = T.alloc_shared((N, K), T.float16)
            C_t = T.alloc_tmem((M, N), T.float32)
            bar = T.alloc_barrier(1)
            T.copy(A, A_s)
            T.copy(B, B_s)
            T.tcgen05_gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=True, mbar=bar[0])
            T.copy(C_t, D)

    _assert_not_in_ir(_lower_src(no_wait), r"\.wait\(")


def test_gemm_completion_mbarrier_collapses_to_one_warp():
    """mbar= 下 warp 切分必须收敛成 1x1,否则 arrive 数与 alloc_barrier(1) 对不上。

    同一个 kernel 不传 mbar 时会拿到多 warp 切分,两边对比才能说明这是 mbar=
    造成的收敛,而不是这个 shape 本来就单 warp。
    """
    wm_with, wn_with = _gemm_partition(_make_completion_mbarrier_kernel())
    wm_without, wn_without = _gemm_partition(_make_gemm_mbarrier_kernel())

    assert (wm_with, wn_with) == (1, 1), f"mbar= should collapse the MMA to a single warp, got ({wm_with},{wn_with})"
    assert (wm_without, wn_without) != (1, 1), (
        f"the no-mbar baseline is single-warp too, so this test cannot tell the collapse apart: ({wm_without},{wn_without})"
    )


def test_tcgen05_mma_arrive_lowers_to_a_fence_arrive():
    """裸 T.tcgen05_mma_arrive 发射 arrive_on_tc_fence,而不是落到通用兜底。

    没有这条分支时它会落到 CodeGenC 的 `Unresolved call ir.Op(...)`,既不提 TANG
    也不提替代做法。
    """
    src = _lower_src(_make_completion_mbarrier_kernel("mma_arrive"))
    # The frontend passes the barrier through retrieve_ptr, so codegen sees a
    # pointer and emits `->`.
    _assert_in_ir(src, r"->arrive_on_tc_fence\(\)", r"\.wait\(0\)")


# ===========================================================================
# §5  Pass-level assertions -- TANG's own LowerSharedBarrier
# ===========================================================================
#
# §1 asserts on the generated C++ source and runs the whole pipeline. This section
# calls the pass itself and asserts which calls got injected at the IR level. The
# split of duties: if the pipeline wiring is broken (a wrong FFI export name, a
# wrong arch condition in pass_filter), §1 fails without pointing anywhere; if the
# pass's own rewrite is wrong, §5 names it precisely. The CUDA counterpart of this
# test lives in
# testing/python/transform/test_tilelang_transform_lower_shared_barrier.py.


def _apply_barrier_pass(func):
    """跑到 barrier_init 注解产生为止,然后只跑 TANG 的 LowerSharedBarrier。"""
    target = tvm.target.Target(STCUV2_TARGET)
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    with target:
        mod = tang_transform.LowerSharedBarrier()(mod)
    return mod["main"].body


def _collect_calls(stmt, op_name: str):
    calls = []

    def visit(node):
        if isinstance(node, tvm.tirx.Call) and str(node.op.name) == op_name:
            calls.append(node)

    tvm.tirx.stmt_functor.post_order_visit(stmt, visit)
    return calls


def test_tang_barrier_pass_injects_one_init_per_barrier():
    """单个 barrier:一条 init(count 原样)、一个 elect 守卫、一次发布。"""

    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            _bar = T.alloc_barrier(128)  # noqa: F841

    body = _apply_barrier_pass(func)

    inits = _collect_calls(body, "tirx.ptx_init_barrier_thread_count")
    assert len(inits) == 1
    assert inits[0].args[1].value == 128
    assert len(_collect_calls(body, "tl.tl_shuffle_elect")) == 1
    assert len(_collect_calls(body, "tl.tang_fence_barrier_init")) == 1
    assert len(_collect_calls(body, "tirx.tvm_storage_sync")) >= 1


def test_tang_barrier_pass_keeps_each_arrive_count():
    """一次 alloc_barrier([...]) 的每个 count 都要落到自己那条 init 上。

    守卫只应有一个:所有 init 共用同一个被选中的线程,发布也只需要一次。
    """

    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            _bars = T.alloc_barrier([1, 1, 128, 128])  # noqa: F841

    body = _apply_barrier_pass(func)

    inits = _collect_calls(body, "tirx.ptx_init_barrier_thread_count")
    assert sorted(c.args[1].value for c in inits) == [1, 1, 128, 128]
    assert len(_collect_calls(body, "tl.tl_shuffle_elect")) == 1
    assert len(_collect_calls(body, "tl.tang_fence_barrier_init")) == 1


def test_tang_barrier_pass_is_a_noop_without_barriers():
    """没有 barrier 分配时不得注入任何东西,包括那次发布用的 sync。"""

    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            buf = T.alloc_shared((16,), T.float16)
            buf[0] = T.float16(0)

    body = _apply_barrier_pass(func)

    assert _collect_calls(body, "tirx.ptx_init_barrier_thread_count") == []
    assert _collect_calls(body, "tl.tang_fence_barrier_init") == []


def test_tang_fence_barrier_init_is_a_real_fence_not_a_noop():
    """回归:这个 device helper 曾经是空函数。

    当时依赖"紧接着的 __syncthreads() 会发布 init"这一邻居不变量保序,fence 本身
    什么都不做。空 fence 在 IR 里与真 fence 长得一模一样,所以这条只能用源码级断言
    守住:tang::ptx::mbarrier_init 自带的 fence.mem 是设备库的实现细节(换成
    -fstpu-sreg-mbarrier 就没有了),不能作为 TileLang 的排序保证。
    """
    header = (pathlib.Path(__file__).resolve().parents[3] / "src" / "tl_templates" / "tang" / "barrier.h").read_text()
    assert re.search(
        r"TL_DEVICE void fence_barrier_init\(\)\s*\{\s*asm volatile\(\"fence\.mem;\"\);\s*\}",
        header,
    ), "TANG's fence_barrier_init must emit a real fence.mem, not be a no-op"


def test_tang_fence_barrier_init_is_indexed_under_tang_op():
    """新 op 名字必须落在 tang registry 里(而不是 backend-neutral 的 op/builtin)。"""

    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            _bar = T.alloc_barrier(128)  # noqa: F841

    body = _apply_barrier_pass(func)

    assert len(_collect_calls(body, "tl.tang_fence_barrier_init")) == 1
    # The old backend-neutral name must not appear any more: the producer now uses
    # the TANG spelling.
    assert _collect_calls(body, "tl.ptx_fence_barrier_init") == []


if __name__ == "__main__":
    tilelang.testing.main()
