"""stcuv2 tensor memory <-> shared 直通拷贝测试(cpt2s / cps2t)。

覆盖 S3 专属的 tensor memory <-> shared 直通拷贝。与 ldt/stt 通路
(test_tang_tcgen05_ldst.py) 的关键区别是 **绕过寄存器堆**:数据由拷贝引擎
直接搬运,不占用 fragment 寄存器,因此可以承载 ldt 路径上会
"ran out of registers" 的大 tile。

两向各有专属原语,裸 `T.copy` 会被 lowering 拒绝(§3 末尾有负向用例):

    T.tcgen05_cp(A_t, A_s)              # shared -> tmem (cps2t)
    T.tang_cp_tmem_to_shared(C_s, C_t)  # tmem -> shared (cpt2s)

命名不对称是有意的:`tcgen05.cp` 在 CUDA 上是同一方向,而 tmem -> shared
没有 CUDA 对应指令,故用 `tang_` 前缀。融合 drain `T.copy(C_t, C_global)`
不受影响,仍是 `T.copy`。

§1 Lowering 断言    test_t2s_lowering_*   (不跑 ISS, 对生成源码做正则)
§2 数值 (via ISS)   test_t2s_*_numeric
§3 反方向 cps2t     test_s2t_*            (shared -> tensor memory)

两条硬约束(见 docs/s3_tmem_shared_cpt2s_cps2t_layout.md):

1. shared tile 必须是 128 字节一行(fp32 即 32 列),行数为 64 的倍数;
   更宽的 N 要在 host 侧切成多个 32 列块循环搬运。
2. cpt2s 按 sw128a8 swizzle 写 shared,所以紧随其后的 shared->global
   bulk copy 必须用同一个 swizzle,即显式标注
   `annotations={"tang_swizzle_atom_bytes": 8}`。漏标会退回默认 atom
   (sw128a32),两侧 XOR 表不一致 -> 数值静默出错。
"""

import re

import pytest
import torch

import tilelang
import tilelang.testing
import tilelang.tang.language as T
from tilelang import tvm as tvm
from tilelang.jit import JITKernel

STCUV2_TARGET = {"kind": "tang", "arch": "stcuv2"}

_TL_DTYPE = {
    "float16": T.float16,
    "bfloat16": T.bfloat16,
    "float32": T.float32,
}
_PT_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

# fp32 下一个 128 字节 shared 行 = 32 列
COLS = 32
# 一次 cpt2s 覆盖 64 行
ROWS = 64

# cpt2s 写 shared 用 sw128a8,回写 global 必须对齐同一 swizzle
_SW128A8 = {"tang_swizzle_atom_bytes": 8}


# ===========================================================================
# Shared kernels / helpers
# ===========================================================================


def _make_colchunk_kernel(M, N, K, dtype, threads=128):
    """gemm -> 按 32 列切块 tmem->shared->global。

    shared tile 为 (M, 32),即单次 T.copy 覆盖全部 M 行,helper 内部按 64
    行循环多次 cpt2s。同时考验 tmem **列**起点随循环变量移动。
    """
    dt = _TL_DTYPE[dtype]
    nchunk = N // COLS

    @T.prim_func
    def kernel(A: T.Tensor((M, K), dt), B: T.Tensor((N, K), dt), C: T.Tensor((M, N), T.float32)):
        with T.Kernel(1, 1, threads=threads):
            A_s = T.alloc_shared((M, K), dt)
            B_s = T.alloc_shared((N, K), dt)
            C_t = T.alloc_tmem((M, N), T.float32)
            C_s = T.alloc_shared((M, COLS), T.float32)
            T.copy(A, A_s)
            T.copy(B, B_s)
            T.gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=True)
            for c in range(nchunk):
                T.tang_cp_tmem_to_shared(C_s, C_t[:, c * COLS : (c + 1) * COLS])
                T.copy(C_s, C[:, c * COLS : (c + 1) * COLS], annotations=_SW128A8)

    return kernel


def _make_tiled_kernel(M, N, K, dtype, threads=128):
    """gemm -> 按 64x32 tile 切分 tmem->shared->global。

    行、列起点都随循环变量移动,覆盖 tmem 地址的 (row, col) 两个分量。
    """
    dt = _TL_DTYPE[dtype]
    nrow, ncol = M // ROWS, N // COLS

    @T.prim_func
    def kernel(A: T.Tensor((M, K), dt), B: T.Tensor((N, K), dt), C: T.Tensor((M, N), T.float32)):
        with T.Kernel(1, 1, threads=threads):
            A_s = T.alloc_shared((M, K), dt)
            B_s = T.alloc_shared((N, K), dt)
            C_t = T.alloc_tmem((M, N), T.float32)
            C_s = T.alloc_shared((ROWS, COLS), T.float32)
            T.copy(A, A_s)
            T.copy(B, B_s)
            T.gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=True)
            for r in range(nrow):
                for c in range(ncol):
                    T.tang_cp_tmem_to_shared(C_s, C_t[r * ROWS : (r + 1) * ROWS, c * COLS : (c + 1) * COLS])
                    T.copy(C_s, C[r * ROWS : (r + 1) * ROWS, c * COLS : (c + 1) * COLS], annotations=_SW128A8)

    return kernel


def _lower_src(func, target=STCUV2_TARGET):
    with tvm.target.Target(target):
        return tilelang.lower(func, target=target).kernel_source


def _sim_run(kernel, M, N, K, dtype):
    pt = _PT_DTYPE[dtype]
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=pt)
    B = torch.randn(N, K, dtype=pt)
    ref = A.float() @ B.float().T
    jit = JITKernel(kernel, out_idx=[-1], target=STCUV2_TARGET, execution_backend="simulator", verbose=False)
    C = jit(A.ptpu(), B.ptpu()).cpu().float()
    return (C - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)


# ===========================================================================
# §1  Lowering assertions
# ===========================================================================


def test_t2s_lowering_emits_cpt2s_and_matching_s2g():
    """tmem->shared 落到 cpt2s,shared->global 用同一 sw128a8 swizzle。"""
    src = _lower_src(_make_colchunk_kernel(128, 64, 64, "float16"))
    assert re.search(r"tang_cp_tmem_to_shared_sw128a8\(", src), "expected the cpt2s helper"
    assert re.search(r"tang_bulk_s2g<tang::ptx::sw128a8>\(", src), "shared->global must use the same swizzle as cpt2s"


def test_t2s_lowering_bypasses_register_file():
    """cpt2s 通路不得退化成 ldt(经寄存器堆)搬运。"""
    src = _lower_src(_make_colchunk_kernel(128, 64, 64, "float16"))
    assert "ldt_16x256b" not in src, "cpt2s path must not stage tensor memory through fragment registers"


def test_t2s_lowering_tmem_address_carries_chunk_offset():
    """回归:tmem 列起点是循环变量时,不能被静默折成 0。

    曾经的 bug:lowering 用 as<IntImmNode>() 取 src_range 起点,遇到
    `c * 32` 这种非常量表达式就回落到 0,导致每个列块都搬第 0 块的数据,
    且不报错。这里要求 tmem 地址实参带上偏移量。
    """
    src = _lower_src(_make_colchunk_kernel(128, 128, 64, "float16"))
    calls = re.findall(r"tang_cp_tmem_to_shared_sw128a8\([^;]*;", src)
    assert calls, "expected at least one cpt2s call"
    assert any(re.search(r"C_t\[0\]\s*\+", c) for c in calls), f"tmem address lost its chunk offset: {calls}"


def test_t2s_lowering_tiled_address_carries_row_and_col_offset():
    """64x32 tile 切分时,tmem 地址要同时带上行、列起点。"""
    src = _lower_src(_make_tiled_kernel(128, 64, 64, "float16"))
    calls = re.findall(r"tang_cp_tmem_to_shared_sw128a8\([^;]*;", src)
    assert calls, "expected at least one cpt2s call"
    # 行起点进地址高 16 位,列起点进低位,两者都必须出现在实参里
    assert any(re.search(r"C_t\[0\]\s*\+", c) for c in calls), f"tmem address lost its tile offset: {calls}"


def test_t2s_rejects_non_128_byte_shared_row():
    """shared 行不是 128 字节(fp32 32 列)时必须在 lowering 期报错。"""

    @T.prim_func
    def kernel(A: T.Tensor((128, 64), T.float16), B: T.Tensor((64, 64), T.float16), C: T.Tensor((128, 64), T.float32)):
        with T.Kernel(1, 1, threads=128):
            A_s = T.alloc_shared((128, 64), T.float16)
            B_s = T.alloc_shared((64, 64), T.float16)
            C_t = T.alloc_tmem((128, 64), T.float32)
            C_s = T.alloc_shared((128, 16), T.float32)  # 64 B/行
            T.copy(A, A_s)
            T.copy(B, B_s)
            T.gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=True)
            T.tang_cp_tmem_to_shared(C_s, C_t[:, 0:16])
            T.copy(C_s, C[:, 0:16], annotations=_SW128A8)

    with pytest.raises(Exception, match="128-byte shared row"):
        _lower_src(kernel)


def test_t2s_rejects_row_count_not_multiple_of_64():
    """一次 cpt2s 搬 64 行,行数不是 64 的倍数必须报错。"""

    @T.prim_func
    def kernel(A: T.Tensor((128, 64), T.float16), B: T.Tensor((64, 64), T.float16), C: T.Tensor((128, 64), T.float32)):
        with T.Kernel(1, 1, threads=128):
            A_s = T.alloc_shared((128, 64), T.float16)
            B_s = T.alloc_shared((64, 64), T.float16)
            C_t = T.alloc_tmem((128, 64), T.float32)
            C_s = T.alloc_shared((32, COLS), T.float32)
            T.copy(A, A_s)
            T.copy(B, B_s)
            T.gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=True)
            T.tang_cp_tmem_to_shared(C_s, C_t[0:32, 0:COLS])
            T.copy(C_s, C[0:32, 0:COLS], annotations=_SW128A8)

    with pytest.raises(Exception, match=r"rows % 64 == 0"):
        _lower_src(kernel)


# ===========================================================================
# §2  Numeric (via ISS)
# ===========================================================================


@pytest.mark.parametrize(
    "M,N,K,dtype",
    [
        (128, 32, 64, "float16"),  # 单列块
        (128, 64, 64, "float16"),  # 2 列块
        (128, 128, 64, "float16"),  # 4 列块
        (64, 64, 64, "float16"),  # 单次 cpt2s 覆盖全部行
        (128, 64, 64, "bfloat16"),
    ],
)
def test_t2s_colchunk_numeric(M, N, K, dtype):
    """tmem->shared->global 必须复现 GEMM 结果(按列块切分)。"""
    rel = _sim_run(_make_colchunk_kernel(M, N, K, dtype), M, N, K, dtype)
    assert rel < 1e-2, f"rel_err too large: {rel:.3e}"


@pytest.mark.parametrize(
    "M,N,K,dtype",
    [
        (128, 32, 64, "float16"),
        (128, 64, 64, "float16"),
        (128, 128, 64, "float16"),
    ],
)
def test_t2s_tiled_numeric(M, N, K, dtype):
    """按 64x32 tile 切分时,行、列起点都要落对位置。"""
    rel = _sim_run(_make_tiled_kernel(M, N, K, dtype), M, N, K, dtype)
    assert rel < 1e-2, f"rel_err too large: {rel:.3e}"


# ===========================================================================
# §3  反方向:shared -> tensor memory (cps2t)
# ===========================================================================
#
# 用途是把 MMA 的 **A 操作数**直接从 shared 送进 tensor memory,不过寄存器堆。
# 相比 `T.copy(fragment, tmem)`,这条路还顺带解决了 32 位 A(tf32)的判别歧义:
# `shared -> shared.tmem` 只可能是操作数暂存,累加器恢复走的是 fragment。
#
# 两条硬约束(ISS 探针实测,见 docs/s3_tmem_shared_cpt2s_cps2t_layout.md):
#
# 1. 一次 cps2t 覆盖 32 行,所以行数必须是 32 的倍数。
# 2. shared 行必须是整数个 128 字节 swizzle 单元,且源 tile 由 sw128a32
#    bulk copy 填入 —— 反 swizzle 的置换表是按 32 字节 atom 测的。


def _make_s2t_gemm_kernel(M, N, K, dtype, threads=32):
    """A -> shared -> TMEM (cps2t) -> mma_atmem(B 在 shared) -> TMEM -> global。

    A 不经 fragment。GEMM 侧走 TS 变体,目前是单 warp。
    """
    dt = _TL_DTYPE[dtype]

    @T.prim_func
    def kernel(A: T.Tensor((M, K), dt), B: T.Tensor((N, K), dt), C: T.Tensor((M, N), T.float32)):
        with T.Kernel(1, 1, threads=threads):
            A_s = T.alloc_shared((M, K), dt)
            A_t = T.alloc_tmem((M, K), dt)
            B_s = T.alloc_shared((N, K), dt)
            C_t = T.alloc_tmem((M, N), T.float32)
            C_f = T.alloc_fragment((M, N), T.float32)
            T.copy(A, A_s)
            T.tcgen05_cp(A_t, A_s)  # cps2t
            T.copy(B, B_s)
            T.tcgen05_gemm(A_t, B_s, C_t, transpose_B=True, clear_accum=True, mbar=None)
            T.tcgen05_ld(C_f, C_t)
            T.copy(C_f, C)

    return kernel


def test_s2t_lowering_emits_cps2t():
    """shared->tmem 落到 cps2t helper,而不是退回普通 copy 循环。

    helper 名字带源布局后缀: A_s 是纯中转 buffer(g2s 走非 swizzle 1D),
    所以每个 lane 的字号是行主序的, 走 ``_linear``; 只有同时是 GEMM shared
    operand 或显式标注了 swizzle 的 buffer 才需要 ``_sw128a32`` 的软件反
    swizzle。见 docs/tang_bulk_copy_gs_layout.md §16.2。
    """
    src = _lower_src(_make_s2t_gemm_kernel(128, 64, 64, "float16"))
    assert re.search(r"tang_cp_shared_to_tmem_linear\(", src), "expected the cps2t helper for an unswizzled staging source"


def test_s2t_lowering_bypasses_register_file():
    """A 的暂存不得经过 fragment(那是 T.copy(fragment, tmem) 的老路)。"""
    src = _lower_src(_make_s2t_gemm_kernel(128, 64, 64, "float16"))
    assert "tang_tmem_st_a_operand" not in src, "cps2t path must not stage A through fragment registers"


def test_s2t_lowering_counts_words_not_elements():
    """cols / row_words 是 32 位字数,不是元素数。

    K=64 的 fp16 行是 128 字节 = 32 个字,所以两个参数都应是 32 而非 64。
    漏掉这次换算会让 helper 多搬一倍的列,越过 tile 边界。
    """
    src = _lower_src(_make_s2t_gemm_kernel(128, 64, 64, "float16"))
    call = re.search(r"tang_cp_shared_to_tmem_linear\(([^;]*)\);", src)
    assert call, "expected a cps2t call"
    args = call.group(1)
    # (smem, taddr, rows=128, cols=32, row_words=32)
    assert re.search(
        r"\(uint32_t\)\(128\),\s*\(uint32_t\)\(32\),\s*"
        r"\(uint32_t\)\(32\)",
        args,
    ), f"cps2t got the wrong row/word counts: {args}"


def test_s2t_rejects_row_count_not_multiple_of_32():
    """一次 cps2t 搬 32 行,行数不是 32 的倍数必须报错。"""

    @T.prim_func
    def kernel(A: T.Tensor((16, COLS), T.float32), C: T.Tensor((16, COLS), T.float32)):
        with T.Kernel(1, 1, threads=32):
            A_s = T.alloc_shared((16, COLS), T.float32)
            A_t = T.alloc_tmem((16, COLS), T.float32)
            C_f = T.alloc_fragment((16, COLS), T.float32)
            T.copy(A, A_s)
            T.tcgen05_cp(A_t, A_s)
            T.copy(A_t, C_f)
            T.copy(C_f, C)

    with pytest.raises(Exception, match=r"rows % 32 == 0"):
        _lower_src(kernel)


def test_s2t_rejects_partial_swizzle_unit_row():
    """shared 行不足一个完整 128 字节 swizzle 单元时必须报错。

    反 swizzle 的置换是以 128 字节单元为周期的,半个单元没有定义。
    """

    @T.prim_func
    def kernel(A: T.Tensor((32, 16), T.float32), C: T.Tensor((32, 16), T.float32)):
        with T.Kernel(1, 1, threads=32):
            A_s = T.alloc_shared((32, 16), T.float32)  # 64 B/行
            A_t = T.alloc_tmem((32, 16), T.float32)
            C_f = T.alloc_fragment((32, 16), T.float32)
            T.copy(A, A_s)
            T.tcgen05_cp(A_t, A_s)
            T.copy(A_t, C_f)
            T.copy(C_f, C)

    with pytest.raises(Exception, match="128-byte swizzle unit"):
        _lower_src(kernel)


def test_s2t_rejects_non_zero_tile_origin():
    """置换以 shared buffer 基址为锚,子区域起点非 0 会读错字。"""

    @T.prim_func
    def kernel(A: T.Tensor((64, COLS), T.float32), C: T.Tensor((32, COLS), T.float32)):
        with T.Kernel(1, 1, threads=32):
            A_s = T.alloc_shared((64, COLS), T.float32)
            A_t = T.alloc_tmem((32, COLS), T.float32)
            C_f = T.alloc_fragment((32, COLS), T.float32)
            T.copy(A, A_s)
            T.tcgen05_cp(A_t, A_s[32:64, :])  # 行起点非 0
            T.copy(A_t, C_f)
            T.copy(C_f, C)

    with pytest.raises(Exception, match="non-zero tile origin"):
        _lower_src(kernel)


@pytest.mark.parametrize(
    "M,N,K,dtype",
    [
        (128, 64, 64, "float16"),
        (128, 64, 64, "bfloat16"),
        (64, 64, 64, "float16"),
        (32, 64, 64, "float16"),
    ],
)
def test_s2t_gemm_numeric(M, N, K, dtype):
    """A 经 shared 直送 tensor memory 后,GEMM 结果必须与参考一致。

    这是整条通路的端到端判据:反 swizzle 的偏移只要错一个字,A 就会取到
    别的行/列,结果立刻发散。

    K 只能取到 64:再大 B 的 shared 行就越过 MMA 的 128 字节上限(见
    gemm_tcgen5.py 的 _check_operand_row_bytes),这和 cps2t 无关。每行多个
    swizzle 单元的情形由下面的往返用例覆盖。
    """
    rel = _sim_run(_make_s2t_gemm_kernel(M, N, K, dtype), M, N, K, dtype)
    assert rel < 1e-2, f"rel_err too large: {rel:.3e}"


def test_s2t_gemm_numeric_tf32_a():
    """32 位 A(tf32)——这条路存在的理由。

    A 走 fragment 时,「fragment -> 32 位 TMEM buffer」和把累加器存回 TMEM 完全
    同形,无从判别;而 shared -> tmem 只可能是操作数暂存,歧义自然消失,所以
    tf32 的 A 必须从 shared 进。

    K 只能取 32:再小 shared 行凑不满一个 128 字节 swizzle 单元,再大 B 就越过
    MMA 的 128 字节行上限 —— fp32 下两个约束正好夹出唯一解。

    tf32 的尾数只有 10 位,所以对着 fp32 参考比的是相对误差而非精确相等。
    """
    rel = _sim_run(_make_s2t_gemm_kernel(128, 64, 32, "float32"), 128, 64, 32, "float32")
    assert rel < 1e-2, f"rel_err too large: {rel:.3e}"


def _make_s2t_roundtrip_kernel(M, N, threads=32):
    """shared -> TMEM (cps2t) -> fragment -> global,不经 GEMM。

    fp32 下元素就是 32 位字,cps2t 写的 (行, 字) 网格和累加器读的 (行, 列)
    网格是同一个,所以整趟应当逐元素还原。这样就把 cps2t 的地址映射和 GEMM
    的操作数约束解耦,能自由测更宽的行。
    """

    @T.prim_func
    def kernel(A: T.Tensor((M, N), T.float32), C: T.Tensor((M, N), T.float32)):
        with T.Kernel(1, 1, threads=threads):
            A_s = T.alloc_shared((M, N), T.float32)
            A_t = T.alloc_tmem((M, N), T.float32)
            A_f = T.alloc_fragment((M, N), T.float32)
            T.copy(A, A_s)
            T.tcgen05_cp(A_t, A_s)  # cps2t
            T.copy(A_t, A_f)
            T.copy(A_f, C)

    return kernel


@pytest.mark.parametrize(
    "M,N",
    [
        (32, 32),  # 一次 cps2t,行宽一个 swizzle 单元
        (128, 32),  # 多个 32 行块
        (32, 64),  # 每行两个 swizzle 单元
        (32, 128),  # 四个
        (128, 64),  # 行块与单元同时多个
    ],
)
def test_s2t_roundtrip_numeric(M, N):
    """shared -> tmem -> global 必须逐元素还原。

    反 swizzle 的 XOR 只要算错,数据不会消失、只会串到别的行 —— 所以这里比
    对的是精确相等,而不是容差。

    N 只能取 8 的 2 的幂倍:读回用的 ldt_16x256b_xN 只有 2 的幂变体,所以像
    96 列(3 个 swizzle 单元)这种宽度无法用本方法验证 —— 那是读回路径的限制,
    与 cps2t 无关。
    """
    torch.manual_seed(0)
    A = torch.randn(M, N, dtype=torch.float32)
    jit = JITKernel(_make_s2t_roundtrip_kernel(M, N), out_idx=[-1], target=STCUV2_TARGET, execution_backend="simulator", verbose=False)
    C = jit(A.ptpu()).cpu()
    assert torch.equal(C, A), f"cps2t round trip mismatched at {(C != A).sum().item()} / {M * N} elements"


# ===========================================================================
# §4  具名原语收口:裸 T.copy 不得跨 shared <-> tensor memory
# ===========================================================================
#
# 两条拷贝引擎通路都背着大量隐含契约(源的 swizzle 模式、行/列粒度、单 warp、
# 全屏障),这些在一个 `T.copy` 调用点上完全看不出来,而最可能的误用(源是用
# 另一种 swizzle atom 填的)是**算错而不报错**。CUDA 后端划的是同一条线:
# src/op/copy.cc 根本没有 shared.tmem 的 lowering 分支,那边唯一入口也是具名
# 原语(T.tcgen05_cp_warpx4)。
#
# 边界要卡准:融合 drain `T.copy(C_t, C_global)` 不在收口范围内。


def _make_bare_copy_s2t_kernel():
    """裸 T.copy(shared -> tmem):应被拒绝,并指向 T.tcgen05_cp。"""

    @T.prim_func
    def kernel(A: T.Tensor((32, 32), T.float32), C: T.Tensor((32, 32), T.float32)):
        with T.Kernel(1, 1, threads=32):
            A_s = T.alloc_shared((32, 32), T.float32)
            A_t = T.alloc_tmem((32, 32), T.float32)
            C_f = T.alloc_fragment((32, 32), T.float32)
            T.copy(A, A_s)
            T.copy(A_s, A_t)
            T.copy(A_t, C_f)
            T.copy(C_f, C)

    return kernel


def _make_bare_copy_t2s_kernel():
    """裸 T.copy(tmem -> shared):应被拒绝,并指向 T.tang_cp_tmem_to_shared。"""

    @T.prim_func
    def kernel(A: T.Tensor((64, 64), T.float16), B: T.Tensor((32, 64), T.float16), C: T.Tensor((64, 32), T.float32)):
        with T.Kernel(1, 1, threads=128):
            A_s = T.alloc_shared((64, 64), T.float16)
            B_s = T.alloc_shared((32, 64), T.float16)
            C_t = T.alloc_tmem((64, 32), T.float32)
            C_s = T.alloc_shared((64, 32), T.float32)
            T.copy(A, A_s)
            T.copy(B, B_s)
            T.gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=True)
            T.copy(C_t, C_s)
            T.copy(C_s, C, annotations=_SW128A8)

    return kernel


def test_bare_copy_shared_to_tmem_is_rejected():
    """裸 T.copy 进 tensor memory 必须报错并指向 T.tcgen05_cp。"""
    with pytest.raises(Exception, match=r"T\.tcgen05_cp"):
        _lower_src(_make_bare_copy_s2t_kernel())


def test_bare_copy_tmem_to_shared_is_rejected():
    """裸 T.copy 出 tensor memory 到 shared 必须指向 T.tang_cp_tmem_to_shared。"""
    with pytest.raises(Exception, match=r"T\.tang_cp_tmem_to_shared"):
        _lower_src(_make_bare_copy_t2s_kernel())


def test_fused_tmem_to_global_drain_still_uses_plain_copy():
    """收口不得误伤融合 drain:T.copy(tmem, global) 仍须正常 lower。

    这是上面两条守卫的边界。drain 走的是 LowerTangTmemDrain,与拷贝引擎无关,
    如果守卫写宽了会把它一并拒掉,而它是最常用的 GEMM 收尾写法。
    """

    @T.prim_func
    def kernel(A: T.Tensor((64, 64), T.float16), B: T.Tensor((32, 64), T.float16), C: T.Tensor((64, 32), T.float32)):
        with T.Kernel(1, 1, threads=128):
            A_s = T.alloc_shared((64, 64), T.float16)
            B_s = T.alloc_shared((32, 64), T.float16)
            C_t = T.alloc_tmem((64, 32), T.float32)
            T.copy(A, A_s)
            T.copy(B, B_s)
            T.gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=True)
            T.copy(C_t, C)

    src = _lower_src(kernel)
    assert "tang_cp_tmem_to_shared" not in src, "融合 drain 不应落到 cpt2s 引擎"


def _build_swapped_args_kernel():
    """实参写反:tcgen05_cp 的 dst 必须是 tensor memory。"""

    @T.prim_func
    def kernel(A: T.Tensor((32, 32), T.float32), C: T.Tensor((32, 32), T.float32)):
        with T.Kernel(1, 1, threads=32):
            A_s = T.alloc_shared((32, 32), T.float32)
            A_t = T.alloc_tmem((32, 32), T.float32)
            T.tcgen05_cp(A_s, A_t)
            T.copy(A_s, C)

    return kernel


def _build_fragment_side_kernel():
    """fragment 侧应走 tcgen05_st,不是 tcgen05_cp。"""

    @T.prim_func
    def kernel(A: T.Tensor((32, 32), T.float32), C: T.Tensor((32, 32), T.float32)):
        with T.Kernel(1, 1, threads=32):
            A_f = T.alloc_fragment((32, 32), T.float32)
            A_t = T.alloc_tmem((32, 32), T.float32)
            T.copy(A, A_f)
            T.tcgen05_cp(A_t, A_f)
            T.copy(A_f, C)

    return kernel


def test_named_primitives_reject_wrong_scopes():
    """方向/scope 写反时,前端要在 tracing 期就报错并指出正确的原语。

    构造放在工厂函数里:`@T.prim_func` 在装饰时即 tracing,断言会在这一刻抛出,
    写成内联的 prim_func 会跑在 pytest.raises 之外。
    """
    with pytest.raises(Exception, match=r"tcgen05_cp\(dst, src\)"):
        _build_swapped_args_kernel()

    with pytest.raises(Exception, match=r"T\.tcgen05_st"):
        _build_fragment_side_kernel()


if __name__ == "__main__":
    tilelang.testing.main()
