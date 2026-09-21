"""stcuv2 mbarrier 原语测试。

覆盖 T.alloc_barrier / T.mbarrier_arrive / T.mbarrier_wait_parity /
T.mbarrier_expect_tx 四个原语在 TANG 上的 lowering、拒绝路径与数值语义。

背景(详见 docs/tang_mbarrier.md):S3 硬件只实现了四条 mbarrier 原子操作
(expect_tx / complete_tx / arrive / arrive_drop),init 与所有 wait 变体都是
device 库用软件模拟的 —— init 是一次带位布局的存储 + fence.mem,wait 是轮询
64 位 barrier 字高位的 phase 域。于是有两个测试上的后果:

* 没有阻塞 wait 指令,`.wait(parity)` 由 tl_templates/tang/barrier.h 里的自旋
  实现,所以数值用例才是真正的判据 —— lowering 断言看不出自旋写错。
* `init` 由 LowerSharedBarrier 注入,TANG 有自己的一份
  (src/tang/transform/lower_shared_barrier.cc)。不复用 CUDA 那份是因为
  src/cuda/transform 在 USE_CUDA=OFF 时整个不编译,而 S3 构建正是这个配置;
  且 CUDA 那份带 cluster barrier 分支,TANG 没有 cluster 层级。漏掉 init 不会
  报错,只会让 arrive/wait 在一块未初始化的共享内存上工作 —— §1 的 init 断言
  钉的是这条静默失败,§6 单独钉这个 pass 自身。

§3 数值用例验证的是什么,需要说清楚,否则容易高估:它确认 barrier.h 过得了
ptcc、init 真被注入、arrive 真递减了计数、自旋 wait 能退出、并且连续两轮的
parity 语义正确。它**不**验证互斥 —— TileLang 的 ThreadStorageSync 会因 shared
上的 RAW 依赖自动插入 __syncthreads(),所以把 arrive/wait 全删掉,同一个 kernel
的结果依然精确正确(已实测)。真正证明 wait 是阻塞的、parity 是承载语义的,是
另一个反向对照:传一个永不满足的 parity 会让 kernel 自旋不退出(已实测挂死)。
那个用例没有收进来 —— 判定它得靠超时,在 CI 里既慢又脆。

expect_tx 只接通指令,不保证与 bulk copy 的 tx 配对语义:递减侧 complete_tx 的
硬件计数被厂商标为不可靠。所以这里只断言它发射得出来,不做数值配对用例。
"""

import re

import pytest
import torch

import tilelang
import tilelang.testing
import tilelang.tang.language as T
from tilelang import tvm as tvm
from tilelang.jit import JITKernel
from tilelang.tang import transform as tang_transform

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


def _sim_jit(func):
    return JITKernel(func, out_idx=[-1], target=STCUV2_TARGET, execution_backend="simulator", verbose=False)


# ===========================================================================
# §0  API existence — 方言导出链
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
    因为递减侧 complete_tx 的硬件计数不可靠(docs/tang_mbarrier.md §4)。
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


# ===========================================================================
# §2  拒绝路径 —— 报错必须停在 lowering,而不是 ptcc
# ===========================================================================


def test_alloc_barrier_rejected_on_stcu():
    """S2 没有 mbarrier(整个 __mbarrier_* 家族由 __Tang_ARCH__ >= 200 门控)。

    不提前拒绝的话,ptcc 会在 barrier.h 内部报一个既不提 barrier 也不提 arch
    的错。
    """
    with pytest.raises(ValueError, match=r"stcuv2"):
        _lower_src(_make_block_sync_kernel(), target=STCU_TARGET)


def test_arrive_expect_tx_rejected():
    """ISA 无融合 arrive.expect_tx,静默拆成两条会丢原子性。"""

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

    with pytest.raises(Exception, match=r"fused arrive\.expect_tx|arrive_expect_tx"):
        _lower_src(kernel)


def test_cluster_barrier_rejected():
    """TANG 无 cluster 层级,cluster barrier 在任何 arch 上都没有对应物。"""

    @T.prim_func
    def kernel(A: T.Tensor((128,), T.float32), B: T.Tensor((128,), T.float32)):
        with T.Kernel(1, threads=128):
            s = T.alloc_shared((128,), T.float32)
            bar = T.alloc_cluster_barrier(128)
            for i in T.Parallel(128):
                s[i] = A[i]
            T.mbarrier_arrive(bar[0])
            T.mbarrier_wait_parity(bar[0], 0)
            for i in T.Parallel(128):
                B[i] = s[127 - i]

    with pytest.raises(ValueError, match=r"cluster"):
        _lower_src(kernel)


# ===========================================================================
# §3  数值(ISS)—— 自旋 wait 的唯一判据
# ===========================================================================


@pytest.mark.parametrize(
    "size,threads",
    [
        (128, 128),
        (256, 128),  # 每个线程搬多个元素
        (128, 256),  # 线程数 > 元素数
    ],
)
def test_barrier_block_sync_numeric(size, threads):
    """带 barrier 的 kernel 必须编得过 ptcc、跑得完、结果等于 reverse(A)。

    这条用例的价值在"跑得完":init 若没注入、arrive 若没递减、自旋若不退出,
    kernel 会挂在 wait 上直到超时,而不是给出错的数。数值本身是精确的(整个通路
    只有搬运,没有算术),所以用零容差比较。
    """
    torch.manual_seed(0)
    A = torch.randn(size, dtype=torch.float32)
    jit = _sim_jit(_make_block_sync_kernel(size=size, threads=threads, arrive_count=threads))
    B = jit(A.ptpu()).cpu()
    torch.testing.assert_close(B, torch.flip(A, dims=[0]), rtol=0, atol=0)


def test_barrier_parity_flip_numeric():
    """连续两轮复用同一个 barrier,parity 必须按 0/1 交替。

    单轮用例盖不住相位:barrier 的相位从 0 起,第一轮等 parity 0 是"翻离 0",
    第二轮等 parity 1 是"翻离 1"。若把 parity 钉死成常量,第二轮会永远等不到,
    kernel 挂死 —— 这是本文件里唯一能区分"相位真的在翻"和"第一轮恰好蒙对"的
    用例。

    两个 barrier 各司其职:bars[0] 是写完成,bars[1] 是读完成。少了后者,快线程
    会在慢线程读 s 之前进入下一轮把 s 覆盖掉,那是 kernel 自身的 WAR 竞态,与
    parity 语义无关。
    """
    size = 128
    rounds = 2

    @T.prim_func
    def kernel(A: T.Tensor((rounds, size), T.float32), B: T.Tensor((rounds, size), T.float32)):
        with T.Kernel(1, threads=size):
            s = T.alloc_shared((size,), T.float32)
            bars = T.alloc_barrier([size, size])
            for k in T.serial(rounds):
                for i in T.Parallel(size):
                    s[i] = A[k, i]
                T.mbarrier_arrive(bars[0])
                T.mbarrier_wait_parity(bars[0], k % 2)
                for i in T.Parallel(size):
                    B[k, i] = s[size - 1 - i]
                T.mbarrier_arrive(bars[1])
                T.mbarrier_wait_parity(bars[1], k % 2)

    torch.manual_seed(0)
    A = torch.randn(rounds, size, dtype=torch.float32)
    B = _sim_jit(kernel)(A.ptpu()).cpu()
    torch.testing.assert_close(B, torch.flip(A, dims=[1]), rtol=0, atol=0)


# ===========================================================================
# §4  GEMM + mbarrier —— 对标 CUDA 的 tcgen05 gemm 用例
# ===========================================================================
#
# testing/python/language/test_tilelang_language_tcgen05_gemm.py 那套模式是
# `gemm(A_s, B_s, C_tmem, mbar)` 让 MMA 自己把完成事件发布进 mbarrier,消费者
# `mbarrier_wait_parity` 后读 TMEM。那个文件在这台机器上 16/17 跳过(要 sm_100
# 实卡,且 compile 写死 target="cuda"),无法用来检验 TANG。
#
# 本节复刻的是它的**同步**形状:MMA 与 TMEM 读之间的顺序由 tcgen05 fence 建立,
# mbarrier 只承担跨线程会合(取代 __syncthreads())。这是新增覆盖:既有 gemm 用例
# 不碰 mbarrier,而 §1–§3 的 mbarrier 用例不碰 gemm/TMEM。
#
# 让 MMA 自己发布完成事件的那条(CUDA 的 mbar= 本义)在 §5,两者不是一回事:
# 这里的 gemm 是阻塞的,§5 的不是。
#
# 排水走 T.copy(C_t, D) 直通,与其余 S3 gemm 用例一致。刻意不走
# TMEM->shared->global 那条三跳:那条路要求显式标注 tang_swizzle_atom_bytes,
# 漏标会静默出错,把它和 mbarrier 缠在一个用例里会让失败原因难以归属 —— 三跳
# 本身另有 test_tang_tcgen05_cpt2s.py 覆盖。


def _make_gemm_mbarrier_kernel(num_k_tiles=1, fence_group=0):
    M = N = 128
    BK = 64  # fp16 下 K=64 正好压在每条 MMA 的 128 字节行上限
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


@pytest.mark.parametrize("num_k_tiles", [1, 2])
def test_gemm_mbarrier_rendezvous_numeric(num_k_tiles):
    """gemm 结果经 mbarrier 会合后读出,必须等于 A @ B^T。

    num_k_tiles=2 让 MMA 累加链跨两轮,顺带确认 mbarrier 与 clear_accum 的累加
    语义不互相干扰。
    """
    M = N = 128
    K = 64 * num_k_tiles
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.float16)
    B = torch.randn(N, K, dtype=torch.float16)

    jit = _sim_jit(_make_gemm_mbarrier_kernel(num_k_tiles=num_k_tiles))
    got = jit(A.ptpu(), B.ptpu()).cpu().float()
    want = A.float() @ B.float().T

    rel = (got - want).abs().max().item() / (want.abs().max().item() + 1e-6)
    assert rel < 1e-2, f"gemm+mbarrier rel_err too large: {rel:.3e}"


# ===========================================================================
# §5  完成 mbarrier —— MMA 自己发布完成事件
# ===========================================================================
#
# §4 那套是「fence 建立顺序 + mbarrier 做跨线程会合」,MMA 本身是同步的。这一节
# 是 CUDA `T.gemm(A, B, C_tmem, mbar=bar)` 的真正对应物:MMA 不阻塞,发起 warp 把
# 自己的 tensor-core fence group 绑到 barrier 上(fence_tc_arrive_mbarrier),
# 硬件在该组的 MMA 退休时投递 arrive,消费者等 parity。
#
# 三条硬约束决定了下面的断言(依据 cccl/tang/__ptx/instructions/fence.h):
#
#   1. fence group 是 **warp 私有**的("synchronize operations in current warp"),
#      所以一个 warp 的 fence 覆盖不了兄弟 warp 的 MMA。
#   2. arrive 是 **warp 级且恰好一次**(.noinc),不是每线程一次。
#   3. 于是 N 个发起 warp 就是 N 次 arrive,barrier 的 arrive count 必须等于
#      warp 数 —— 而 count 定在 T.alloc_barrier(),GEMM lowering 时拿不到。
#
# lowering 因此在 mbar= 时把 warp 切分收敛成单 warp,让 arrive 恒为 1,与 CUDA
# 源码的 alloc_barrier(1) 对齐。test_..._collapses_to_one_warp 钉的就是这条:
# 谁要是把多 warp 放回来而不同时解决 count,barrier 会永远等不满而挂死。
#
# 「arrive 恰好一次」是实测结论,不是推断。三个对照(见 docs/tang_mbarrier.md
# T1):wait(0) 正常返回;wait(1) 挂死(说明 phase 只翻了一次);去掉 mbar= 后
# wait(0) 也挂死(说明 wait 不是空转、确实靠这次 arrive 才放行)。后两条靠超时
# 判定,太慢也太脆,没有收进 CI。


def _make_completion_mbarrier_kernel(api="tcgen05"):
    """CUDA 完成事件模式的 TANG 版。

    三种 api 对应三种提交方式,差别只在 barrier 由谁、什么时候放行:

      "tcgen05"    T.tcgen05_gemm(mbar=) —— 异步接口,wait 由调用方写。
      "sync"       T.gemm(mbar=)        —— 同步接口,wait 由 lowering 自动补,
                                            所以 kernel 里**故意不写** wait。
      "mma_arrive" GEMM 自己同步收尾,再由选出的 warp 手工提交完成事件。
    """
    M = N = 128
    K = 64  # fp16 下 K=64 正好压在每条 MMA 的 128 字节行上限

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


def _gemm_partition(src):
    """从 mma_desc 的 subM/subN 反推 warp_m / warp_n (M=N=128)。"""
    m = re.search(r"mma_desc<([^>]*)>", src)
    assert m, f"no mma_desc emitted:\n{src}"
    parts = [p.strip() for p in m.group(1).split(",")]
    # D_FMT, A_FMT, B_FMT, a_major, b_major, subM, subN, K
    return 128 // int(parts[5]), 128 // int(parts[6])


@pytest.mark.parametrize("api", ["tcgen05", "sync"])
def test_gemm_completion_mbarrier_lowers_to_a_fence_arrive(api):
    """mbar= 必须把 barrier 传进 MMA 模板,并由消费者等 parity。"""
    src = _lower_src(_make_completion_mbarrier_kernel(api))
    _assert_in_ir(
        src,
        r"#include <tl_templates/tang/barrier\.h>",
        r"\.init\(1\)",
        # mbar= 下完成协议改为 arrive_on_tc_fence, 而非 fence_tc。
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

    # 反面:异步接口不许自作主张补 wait,否则手工调度就失去意义。这里用一个
    # 不写 wait 的 tcgen05_gemm kernel 对照。
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
    wm_with, wn_with = _gemm_partition(_lower_src(_make_completion_mbarrier_kernel()))
    wm_without, wn_without = _gemm_partition(_lower_src(_make_gemm_mbarrier_kernel()))

    assert (wm_with, wn_with) == (1, 1), f"mbar= should collapse the MMA to a single warp, got ({wm_with},{wn_with})"
    assert (wm_without, wn_without) != (1, 1), (
        f"the no-mbar baseline is single-warp too, so this test cannot tell the collapse apart: ({wm_without},{wn_without})"
    )


@pytest.mark.parametrize("api", ["tcgen05", "sync"])
def test_gemm_completion_mbarrier_numeric(api):
    """MMA 完成事件经 barrier 发布后读出的 TMEM,必须等于 A @ B^T。"""
    M = N = 128
    K = 64
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.float16)
    B = torch.randn(N, K, dtype=torch.float16)

    got = _sim_jit(_make_completion_mbarrier_kernel(api))(A.ptpu(), B.ptpu()).cpu().float()
    want = A.float() @ B.float().T

    rel = (got - want).abs().max().item() / (want.abs().max().item() + 1e-6)
    assert rel < 1e-2, f"completion-mbarrier gemm rel_err too large: {rel:.3e}"


def test_tcgen05_mma_arrive_lowers_to_a_fence_arrive():
    """裸 T.tcgen05_mma_arrive 发射 arrive_on_tc_fence,而不是落到通用兜底。

    没有这条分支时它会落到 CodeGenC 的 `Unresolved call ir.Op(...)`,既不提 TANG
    也不提替代做法。
    """
    src = _lower_src(_make_completion_mbarrier_kernel("mma_arrive"))
    # 前端把 barrier 过了 retrieve_ptr,所以到 codegen 手里是指针,发射的是 `->`。
    _assert_in_ir(src, r"->arrive_on_tc_fence\(\)", r"\.wait\(0\)")


def test_tcgen05_mma_arrive_numeric():
    """手工提交那条路同样要能放行 parity 等待并读出正确结果。"""
    M = N = 128
    K = 64
    torch.manual_seed(7)
    A = torch.randn(M, K, dtype=torch.float16)
    B = torch.randn(N, K, dtype=torch.float16)

    jit = _sim_jit(_make_completion_mbarrier_kernel("mma_arrive"))
    got = jit(A.ptpu(), B.ptpu()).cpu().float()
    want = A.float() @ B.float().T

    rel = (got - want).abs().max().item() / (want.abs().max().item() + 1e-6)
    assert rel < 1e-2, f"tcgen05_mma_arrive gemm rel_err too large: {rel:.3e}"


def test_completion_mbarrier_on_the_tmem_a_path():
    """A 在 TMEM 的 TS 路径同样支持 mbar= (arrive_on_tc_fence + parity wait)。

    早先 TS 模板仍然内联 fence、没接 barrier,所以此处只断言拒绝;现在 Python
    emitter 的 TS 路径也走 arrive_on_tc_fence,数值必须正确。
    """
    M = N = 128
    K = 64

    @T.prim_func
    def kernel(A: T.Tensor((M, K), T.float16), B: T.Tensor((N, K), T.float16), D: T.Tensor((M, N), T.float32)):
        with T.Kernel(1, 1, threads=32):
            A_f = T.alloc_fragment((M, K), T.float16)
            A_t = T.alloc_tmem((M, K), T.float16)
            B_s = T.alloc_shared((N, K), T.float16)
            C_t = T.alloc_tmem((M, N), T.float32)
            bar = T.alloc_barrier(1)
            T.copy(A, A_f)
            T.copy(A_f, A_t)
            T.copy(B, B_s)
            T.gemm(A_t, B_s, C_t, transpose_B=True, clear_accum=True, mbar=bar[0])
            T.mbarrier_wait_parity(bar[0], 0)
            T.copy(C_t, D)

    src = _lower_src(kernel)
    _assert_in_ir(src, r"arrive_on_tc_fence\(\)", r"\.wait\(0\)")

    torch.manual_seed(3)
    A = torch.randn(M, K, dtype=torch.float16)
    B = torch.randn(N, K, dtype=torch.float16)
    got = _sim_jit(kernel)(A.ptpu(), B.ptpu()).cpu().float()
    want = A.float() @ B.float().T
    rel = (got - want).abs().max().item() / (want.abs().max().item() + 1e-6)
    assert rel < 1e-2, f"TS mbar gemm rel_err too large: {rel:.3e}"


# ===========================================================================
# §6  pass 级断言 —— TANG 自有的 LowerSharedBarrier
# ===========================================================================
#
# §1 是在生成的 C++ 源码上断言的,过的是整条 pipeline。这一节直接调 pass 本身,
# 断言的是 IR 层面注入了哪些 call。两者的分工:pipeline 接线断了(比如前端导出
# 名写错、pass_filter 的 arch 条件写错),§1 会挂但指不出原因;pass 自身的重写
# 逻辑写错,§6 会精确指出。CUDA 那份的同类测试在
# testing/python/transform/test_tilelang_transform_lower_shared_barrier.py。


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
    assert len(_collect_calls(body, "tl.ptx_fence_barrier_init")) == 1
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
    assert len(_collect_calls(body, "tl.ptx_fence_barrier_init")) == 1


def test_tang_barrier_pass_is_a_noop_without_barriers():
    """没有 barrier 分配时不得注入任何东西,包括那次发布用的 sync。"""

    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            buf = T.alloc_shared((16,), T.float16)
            buf[0] = T.float16(0)

    body = _apply_barrier_pass(func)

    assert _collect_calls(body, "tirx.ptx_init_barrier_thread_count") == []
    assert _collect_calls(body, "tl.ptx_fence_barrier_init") == []


if __name__ == "__main__":
    tilelang.testing.main()
