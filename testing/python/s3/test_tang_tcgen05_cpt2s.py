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

§1 Lowering 断言    test_t2s_lowering_*   (对生成源码做正则)
§3 反方向 cps2t     test_s2t_*            (shared -> tensor memory)

两条硬约束:

1. shared tile 必须是 128 字节一行(fp32 即 32 列),行数为 64 的倍数;
   更宽的 N 要在 host 侧切成多个 32 列块循环搬运。
2. cpt2s 按 sw128a8 swizzle 写 shared,所以紧随其后的 shared->global
   bulk copy 必须用同一个 swizzle,即显式标注
   `annotations={"tang_swizzle_atom_bytes": 8}`。漏标会退回默认 atom
   (sw128a32),两侧 XOR 表不一致 -> 数值静默出错。
"""

import re

import pytest

import tilelang
import tilelang.testing
import tilelang.tang.language as T
from tilelang import tvm as tvm

STCUV2_TARGET = {"kind": "tang", "arch": "stcuv2"}

_TL_DTYPE = {
    "float16": T.float16,
    "bfloat16": T.bfloat16,
    "float32": T.float32,
}

# One 128-byte shared row holds 32 fp32 columns.
COLS = 32
# One cpt2s call covers 64 rows.
ROWS = 64

# cpt2s writes shared with sw128a8, so the write-back to global must use the
# same swizzle.
_SW128A8 = {"tang_swizzle_atom_bytes": 8}
# sw128a32: the 32-byte-atom swizzle, the one the cps2t un-swizzle map was
# measured against.
_SW128A32 = {"tang_swizzle_atom_bytes": 32}


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


def _indent_at(lines, index):
    """缩进宽度：用来判断某行被哪一层包住。"""
    line = lines[index]
    return len(line) - len(line.lstrip())


def _line_index(lines, needle):
    return next(i for i, line in enumerate(lines) if needle in line)


def _enclosing_header(lines, index):
    """包住第 ``index`` 行的那个开括号行（往上看第一个缩进更小的非空行）。"""
    indent = _indent_at(lines, index)
    for j in range(index - 1, -1, -1):
        if lines[j].strip() and _indent_at(lines, j) < indent:
            return j, lines[j]
    raise AssertionError(f"line {index} is not nested inside anything")


def _assert_wait_opens_the_reader_branch(src):
    """``wait(0)`` 必须是读取那道门里的**第一条**语句。

    这是收紧后的插点（`TmemPublishInjector.PlacePendingWait`）：wait 是轮询
    （`tl::mbarrier_wait` = `while (!mbarrier_try_wait_parity(...))`），所以只该由
    真会读 TMEM 的 warp 执行；落在门外就是全块陪跑。
    """
    lines = src.splitlines()
    wait_i = _line_index(lines, ".wait(0)")
    header_i, header = _enclosing_header(lines, wait_i)
    assert header.strip().startswith("if (") and header.strip().endswith("{"), f"the wait must sit inside a conditional, got: {header!r}"
    assert "threadIdx.x" in header, f"that conditional must be the reader's guard, got: {header!r}"
    assert lines[wait_i - 1].strip() == header.strip(), "the wait must be the first statement inside that guard"
    return wait_i, header_i, header


# ===========================================================================
# §1  Lowering assertions
# ===========================================================================


def test_t2s_lowering_emits_cpt2s_and_matching_s2g():
    """tmem->shared 落到 cpt2s,shared->global 用同一 sw128a8 swizzle。"""
    src = _lower_src(_make_colchunk_kernel(128, 64, 64, "float16"))
    # Check the ISA leaf rather than a helper wrapper.
    assert re.search(r"cpt2s_16x64b_2d<tang::ptx::sw128a8,", src), "expected the cpt2s leaf"
    # The store's swizzle is named by the instruction template argument.
    assert re.search(r"fcps2g<tang::ptx::sw128a8>\(", src), "shared->global must use the same swizzle as cpt2s"
    assert "tang_bulk_s2g" not in src, "the swizzled store must not go back through the helper"


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
    calls = re.findall(r"cpt2s_16x64b_2d<[^;]*;", src)
    assert calls, "expected at least one cpt2s call"
    assert any(re.search(r"C_t\[0\]\s*\+", c) for c in calls), f"tmem address lost its chunk offset: {calls}"


def test_t2s_lowering_splits_rows_across_warps():
    """行块按区域 warp 轮转切分,不再由单 warp 搬整块。

    改前 helper 是 `if (__warpid() == 0) { ... } __syncthreads();`,多 warp block 里
    其余 warp 全程空转;现在 128 行 / 4 warp 的 tile 由 warp 0..1 各搬 64 行。
    切分与消费者对齐后,那道跨 warp 发布
    屏障不再需要,`ThreadSync` 的窄化也才是正当的。
    """
    src = _lower_src(_make_colchunk_kernel(128, 128, 64, "float16"))
    calls = re.findall(r"cpt2s_16x64b_2d<[^;]*;", src)
    assert calls, "expected at least one cpt2s call"
    assert any(">> 5" in c for c in calls), f"each warp must carry its own rows: {calls}"
    assert any("<< 16" in c for c in calls), f"the row offset stays in the address high half: {calls}"
    assert "tang_cp_tmem_to_shared" not in src, "the monolithic helper must be gone"


def test_t2s_lowering_tiled_address_carries_row_and_col_offset():
    """64x32 tile 切分时,tmem 地址要同时带上行、列起点。"""
    src = _lower_src(_make_tiled_kernel(128, 64, 64, "float16"))
    calls = re.findall(r"cpt2s_16x64b_2d<[^;]*;", src)
    assert calls, "expected at least one cpt2s call"
    # The row origin goes into the high 16 address bits (`<< 16`) and the column
    # origin into the low bits of the BufferLoad; both must show up in the args.
    assert any(re.search(r"C_t\[0\]\s*\+", c) for c in calls), f"tmem address lost its tile offset: {calls}"
    assert any("<< 16" in c for c in calls), f"the row offset must reach the address: {calls}"


def test_t2s_rejects_non_128_byte_shared_row():
    """shared 行不是 128 字节(fp32 32 列)时必须在 lowering 期报错。"""

    @T.prim_func
    def kernel(A: T.Tensor((128, 64), T.float16), B: T.Tensor((64, 64), T.float16), C: T.Tensor((128, 64), T.float32)):
        with T.Kernel(1, 1, threads=128):
            A_s = T.alloc_shared((128, 64), T.float16)
            B_s = T.alloc_shared((64, 64), T.float16)
            C_t = T.alloc_tmem((128, 64), T.float32)
            C_s = T.alloc_shared((128, 16), T.float32)  # 64 B per row
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
# §3  The other direction: shared -> tensor memory (cps2t)
# ===========================================================================
#
# Its purpose is to send the MMA's **A operand** straight from shared into
# tensor memory, bypassing the register file. Compared with
# `T.copy(fragment, tmem)`, this path also removes the ambiguity for a 32-bit A
# (tf32): `shared -> shared.tmem` can only be operand staging, whereas
# accumulator recovery goes through a fragment.
#
# Two hard constraints (measured with ISS probes):
#
# 1. One cps2t call covers 32 rows, so the row count must be a multiple of 32.
# 2. The shared row must be a whole number of 128-byte swizzle units, and the
#    source tile must be filled by an sw128a32 bulk copy -- the un-swizzle
#    permutation table was measured against the 32-byte atom.


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
    """shared->tmem 落到 cps2t 指令,而不是退回普通 copy 循环。

    指令本身和它的两层循环都是 IR:
    叶子是 ``__stvm_cps2t_1d``,寻址模式由 ``__stvm_set_mem_dim0_s2t`` 设定,
    每 32 行一组、每 32 位列一条。A_s 是纯中转 buffer(g2s 走非 swizzle 1D),
    所以字号是行主序的(``row_words`` 步长),不需要软件反 swizzle;只有同时是
    GEMM shared operand 或显式标注了 swizzle 的 buffer 才走 sw128a32 的映射。
    """
    src = _lower_src(_make_s2t_gemm_kernel(128, 64, 64, "float16"))
    assert re.search(r"__stvm_cps2t_1d\(", src), "expected the cps2t engine instruction"
    assert re.search(r"__stvm_set_mem_dim0_s2t\(128, 128, 1\)", src), "the engine needs its per-lane addressing mode set up"
    assert re.search(r"threadIdx\.x\)\s*\*\s*32", src), "an unswizzled staging source must use the row-major word map"


def test_s2t_lowering_bypasses_register_file():
    """A 的暂存不得经过 fragment(那是 T.copy(fragment, tmem) 的老路)。"""
    src = _lower_src(_make_s2t_gemm_kernel(128, 64, 64, "float16"))
    assert "tang_tmem_st_a_operand" not in src, "cps2t path must not stage A through fragment registers"


def test_s2t_lowering_counts_words_not_elements():
    """cols / row_words 是 32 位字数,不是元素数。

    K=64 的 fp16 行是 128 字节 = 32 个字,所以列循环上界与行步长都应是 32 而非
    64。漏掉这次换算会让拷贝多搬一倍的列,越过 tile 边界。128 行按每次 32 行切
    成 4 组,所以是两处字面量: ``w < 32`` 与外层 ``blk < 4``。
    """
    src = _lower_src(_make_s2t_gemm_kernel(128, 64, 64, "float16"))
    assert re.search(r"for \(int w = 0; w < 32; \+\+w\)", src), "the 32-bit column count is a per-call word count, not an element count"
    assert re.search(r"for \(int blk = 0; blk < 4; \+\+blk\)", src), "128 rows are staged as 4 groups of 32"
    assert re.search(r"threadIdx\.x\)\s*\*\s*32", src), "the word pitch must be row_words (32), not the element count"


def test_s2t_staging_publishes_before_the_mma():
    """cps2t 之后必须有一条发布边把 TMEM 写交给 MMA 之前。

    这条边 ThreadSync 推不出来(它只推理 buffer,而这里要排序的是拷贝引擎对
    tensor memory 的写与 MMA 的读),所以 lowering 显式发出
    ``tang_publish_tmem_operand`` 标记(带被暂存的 tile 与 arrive 的 warp 数),
    由 ``MaterializeTangMbarriers`` 落实成 mbarrier 握手:staging 的 warp 在区域内
    arrive(它本身就是 fence),第一个读该 tile 的地方之前 wait。
    以前它是设备 helper 尾部的 ``__syncthreads()`` —— 一条任何同步 pass 都看
    不见、且在区域化之后非法的屏障。
    """
    src = _lower_src(_make_s2t_gemm_kernel(128, 64, 64, "float16"))
    staged = src.index("__stvm_cps2t_1d")
    arrived = src.index("arrive_on_s2t_fence", staged)
    waited = src.index(".wait(0)", arrived)
    mma = src.index("tang_tcgen05_mma_ts")
    assert staged < arrived < waited < mma, "the staged A operand must be published and waited for before the MMA reads it"
    assert "tang_publish_tmem_operand" not in src, "the marker is IR-only: the pass must have materialised it"


def test_s2t_lowering_takes_the_deswizzle_map_for_a_swizzled_source():
    """源 tile 是 swizzled 时,字号必须是反 swizzle 映射而不是行主序。

    两种映射只差一个映射函数,判据是**同一份**布局信号(见
    ``LowerTangSharedToTmemCopy``):共享 buffer 的 layout 非恒等(它同时是 ss
    GEMM 的 A 操作数),或拷贝上标了 ``tang_swizzle_atom_bytes``。这里走前一条
    信号。行主序映射里没有 XOR;反 swizzle 的特征就是 atom 序号与行号异或。
    """
    src = _lower_src(_make_s2t_roundtrip_swizzled_kernel(128, 64))
    call = re.search(r"__stvm_cps2t_1d\(([^;]*)\);?", src)
    assert call, "expected the cps2t engine instruction"
    word = call.group(1).split(",", 3)[2]
    assert "^" in word, f"a swizzled source must use the sw128a32 de-swizzle map, got {word}"
    assert ">> 3" in word, f"the de-swizzle must index the 32-byte atom (w >> 3), got {word}"


def test_s2t_roundtrip_swizzled_source_lowering_keeps_the_leaf():
    """反 swizzle 分支不退回设备函数:叶子、寻址模式与两层循环都得在生成码里。"""
    src = _lower_src(_make_s2t_roundtrip_swizzled_kernel(128, 64))
    assert re.search(r"__stvm_set_mem_dim0_s2t\(128, 128, 1\)", src)
    assert re.search(r"for \(int blk = 0; blk < 4; \+\+blk\)", src)
    assert re.search(r"for \(int w = 0; w < 64; \+\+w\)", src)
    assert "copy_tmem_shm.h" not in src, "the cps2t geometry is IR now; the device header must not be pulled in"


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
            A_s = T.alloc_shared((32, 16), T.float32)  # 64 B per row
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
            T.tcgen05_cp(A_t, A_s[32:64, :])  # row origin is not 0
            T.copy(A_t, C_f)
            T.copy(C_f, C)

    with pytest.raises(Exception, match="non-zero tile origin"):
        _lower_src(kernel)


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


def test_s2t_lowering_splits_row_groups_across_warps():
    """32 行组按区域内的 warp 切分,每 warp 一段连续行。

    切法与 cpt2s 那次完全同构(``active = min(region_warps, groups)``、每 warp 一段),
    所以有四种形态要分清楚:组数 >= warp 数时**每 warp 一组、门也发不出来**;
    组数少于 warp 数时只有前 ``groups`` 个 warp 干活(门是真的);组数多于 warp 数时
    每 warp 拿到一个计数循环。
    """
    # 4 warps / 4 groups: one group per warp, no outer loop and no guard
    src = _lower_src(_make_s2t_roundtrip_kernel(128, 32, threads=128))
    assert not re.search(r"for \(int blk", src), "one group per warp needs no run loop at all"
    assert re.search(r"threadIdx\.x\) >> 5\) \* 32\)\) << 16", src), "each warp must stage its own 32-row group"
    assert not re.search(r"if \(\(\(int\)threadIdx\.x\) < 128\)", src), "every warp owns a group: the election guard must fold away"

    # 4 warps / 2 groups: only the first two warps work, the guard must stay
    src = _lower_src(_make_s2t_roundtrip_kernel(64, 32, threads=128))
    assert re.search(r"if \(\(\(int\)threadIdx\.x\) < 64\)", src), "2 groups over 4 warps: only the first two warps may stage"

    # 2 warps / 4 groups: each warp walks a counted loop from its own run start
    src = _lower_src(_make_s2t_roundtrip_kernel(128, 32, threads=64))
    assert re.search(r"for \(int blk = 0; blk < 2; \+\+blk\)", src), "more groups than warps: each warp walks a run of per_warp groups"
    assert re.search(r">> 5\) \* 64\) \+ \(blk \* 32\)", src), "the run must start at warp_id * per_warp groups"


def test_s2t_single_warp_shape_is_unchanged():
    """单 warp 区域的生成的码里不得出现切分留下的痕迹。

    这条是“多 warp 切分没有改到单 warp 路径”的守门用例:单 warp 下只有一组
    循环(列循环)与无门、行偏移不带 warp 项。
    """
    src = _lower_src(_make_s2t_gemm_kernel(128, 64, 64, "float16"))
    assert re.search(r"for \(int blk = 0; blk < 4; \+\+blk\)", src), "a single-warp region keeps the plain group loop"
    assert not re.search(r"for \(int g = 0;", src), "no run loop for one warp"
    assert not re.search(r"threadIdx\.x\) >> 5", src), "a single-warp region must not carry a warp-dependent row offset"


def test_s2t_staging_inside_a_region_publishes_with_an_mbarrier():
    """cps2t 暂存在 T.ws 子区域里时，发布必须走 mbarrier 而不是块级屏障。

    标记是紧跟 staging 发的，区域门会把它一起包住 ⇒ 块级屏障只被进入该分支的
    线程到达（非法且可能挂死）。arrive 不是屏障，所以它留在区域门内；wait 是自旋，
    所以它进一步收窄到**读取分支的第一条** —— 只有真会读 TMEM 的 warp 付这份代价。
    以前这条形状只能响亮拒绝，现在应当正常降级。
    """

    @T.prim_func
    def kernel(A: T.Tensor((128, 32), T.float32), C: T.Tensor((128, 32), T.float32)):
        with T.Kernel(1, 1, threads=128):
            A_s = T.alloc_shared((128, 32), T.float32)
            A_t = T.alloc_tmem((128, 32), T.float32)
            A_f = T.alloc_fragment((128, 32), T.float32)
            T.copy(A, A_s)
            # Two warps do the staging, so the region guard also wraps the
            # publication marker.
            with T.ws(0, warp_group_size=64):
                T.tcgen05_cp(A_t, A_s)
            with T.ws(3, warp_group_size=32):
                T.copy(A_t, A_f)
                T.copy(A_f, C)

    src = _lower_src(kernel)
    lines = src.splitlines()

    # The arrive sits inside the producer's guard: it is not a barrier, so staying
    # in the region is legal.
    arrive_i = _line_index(lines, "arrive_on_s2t_fence")
    _, arrive_header = _enclosing_header(lines, arrive_i)
    assert arrive_header.strip().startswith("if (") and "threadIdx.x" in arrive_header, (
        "the arrive belongs inside the staging region's guard"
    )

    # The wait sits inside the consumer's guard, as that guard's first statement.
    wait_i, _, wait_header = _assert_wait_opens_the_reader_branch(src)
    assert wait_header != arrive_header, "the wait belongs to the reading region, not to the staging one"
    assert _line_index(lines, "__stvm_cps2t_1d") < arrive_i < wait_i, "the order must stay stage -> arrive -> wait"

    # No block barrier may appear inside the region: the span from the staging
    # loop to the arrive must not contain __syncthreads().
    staged = src.index("__stvm_cps2t_1d")
    region_body = src[staged : src.index("arrive_on_s2t_fence")]
    assert "__syncthreads()" not in region_body, "a divergent block barrier must not be emitted inside the region"


def test_s2t_lowering_keeps_the_wait_off_the_non_reading_warps():
    """没有 T.ws 的多 warp 形状：wait 同样收窄进读取门。

    M=64 / K=32 / 128 线程 ⇒ 2 组 / 4 warp：生产者只有前两个 warp，消费者只有第一个。
    以前 wait 落在块作用域，4 个 warp 全在轮询。
    """
    src = _lower_src(_make_s2t_roundtrip_kernel(64, 32, threads=128))
    lines = src.splitlines()
    assert re.search(r"if \(\(\(int\)threadIdx\.x\) < 64\)", src), "2 groups over 4 warps: only the first two warps may stage"
    wait_i, _, wait_header = _assert_wait_opens_the_reader_branch(src)
    assert re.search(r"< 32\)", wait_header), "the reader here is the first warp only"
    arrive_i = _line_index(lines, "arrive_on_s2t_fence")
    _, arrive_header = _enclosing_header(lines, arrive_i)
    assert arrive_header != wait_header, "the arrive and the wait belong to different branches"
    assert arrive_i < wait_i


def test_s2t_staging_over_the_whole_block_is_accepted():
    """区域覆盖整块时没有区域门,发布标记落在块作用域 —— 也应正常降级。"""

    @T.prim_func
    def kernel(A: T.Tensor((128, 32), T.float32), C: T.Tensor((128, 32), T.float32)):
        with T.Kernel(1, 1, threads=128):
            A_s = T.alloc_shared((128, 32), T.float32)
            A_t = T.alloc_tmem((128, 32), T.float32)
            A_f = T.alloc_fragment((128, 32), T.float32)
            T.copy(A, A_s)
            with T.ws(0, warp_group_size=128):
                T.tcgen05_cp(A_t, A_s)
            with T.ws(3, warp_group_size=32):
                T.copy(A_t, A_f)
                T.copy(A_f, C)

    src = _lower_src(kernel)
    assert "__stvm_cps2t_1d" in src
    assert "arrive_on_s2t_fence" in src, "the publish is the mbarrier handshake"
    assert ".wait(0)" in src, "the reader must wait for the arrival"


def _make_s2t_roundtrip_swizzled_kernel(M, K, threads=32):
    """与 ``_make_s2t_roundtrip_kernel`` 同一趟往返,但源 tile 是 **swizzled** 的。

    A_s 的 g2s 填充与 cps2t 拷贝都标上 ``tang_swizzle_atom_bytes: 32``:前者让
    bulk copy 真的按 sw128a32 写,后者告诉 cps2t 必须把那个置换在软件里反回来。
    两侧同一个键,这正是 lowering 里两条布局信号中的“显式标注”那条。

    这会执行 ``tang_sw128a32_deswizzle_word``;XOR 查表错位时数据不会消失、
    只会串到别的 (行, 字) 上,所以判据是精确相等。
    """

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), T.float32),
        C: T.Tensor((M, K), T.float32),
    ):
        with T.Kernel(1, 1, threads=threads):
            A_s = T.alloc_shared((M, K), T.float32)
            A_t = T.alloc_tmem((M, K), T.float32)
            A_f = T.alloc_fragment((M, K), T.float32)
            T.copy(A, A_s, annotations=_SW128A32)
            T.tcgen05_cp(A_t, A_s, annotations=_SW128A32)
            T.copy(A_t, A_f)
            T.copy(A_f, C)

    return kernel


# ===========================================================================
# §4  Named primitives only: a bare T.copy must not cross shared <-> tensor memory
# ===========================================================================
#
# Both copy-engine paths carry a lot of implicit contract (the source's swizzle
# mode, row/column granularity, a single warp, a full barrier) that is invisible
# at a `T.copy` call site, and the most likely misuse -- a source filled with a
# different swizzle atom -- **computes wrong instead of failing**. The CUDA
# backend draws the same line: src/op/copy.cc has no shared.tmem lowering branch
# at all, and the only entry there is a named primitive too
# (T.tcgen05_cp_warpx4).
#
# The boundary has to be exact: the fused drain `T.copy(C_t, C_global)` is not
# part of it.


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
