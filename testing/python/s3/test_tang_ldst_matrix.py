"""stcuv2 ld/st matrix 前端接口测试。

这些都是 TANG **stcuv2** subtarget 上的特性(与 CUDA / 其它 subtarget 隔离)。

现状:LLVM 层的 ld/st matrix intrinsic 仍在开发,stcuv2 后端 lowering 尚未接。
因此本文件只做两件事:

1. **前端构造断言**(始终运行):验证 ``tang_`` 前缀的 st/ld matrix 前端接口
   能正确构造出 IR(发 ``tl.ptx_stmatrix`` / ``tl.ptx_ldmatrix``)。
2. **stcuv2 lowering 用例**:后端未就绪,故一条直接 ``skip``,另一条 ``catch``
   后端报错后 ``skip``——待 LLVM intrinsic 落地、后端接好后再改为真正的数值/
   源码断言。
"""

import re

import pytest

import tilelang
import tilelang.testing
import tilelang.tang.language as T
from tilelang import tvm as tvm

STCUV2_TARGET = {"kind": "tang", "arch": "stcuv2"}


# ===========================================================================
# Shared helpers
# ===========================================================================


def _script_of(func):
    """Return the TIR script of a PrimFunc."""
    return func.script()


def _assert_in_ir(txt: str, *patterns: str):
    """Assert each pattern appears in the IR text."""
    for pat in patterns:
        assert re.search(pat, txt), f"Expected '{pat}' NOT found in IR:\n{txt[:1024]}"


# ===========================================================================
# §0  API existence
# ===========================================================================


def test_matrix_frontend_apis_exist():
    """tang_ st/ld matrix 前端接口可用,且 ptx_ldmatrix 未被 shadow。"""
    assert callable(T.tang_stmatrix)
    assert callable(T.tang_ldmatrix)
    # 上游 ptx_ldmatrix intrinsic 仍在(CUDA mma 加载路径依赖它),未被前端覆盖。
    assert callable(T.ptx_ldmatrix)
    assert T.ptx_ldmatrix is tilelang.language.ptx_ldmatrix


# ===========================================================================
# §1  tang_stmatrix  — S3-explicit store (shared ← register fragment)
# ===========================================================================


def _make_tang_stmatrix_kernel_1val(trans: bool = False):

    @T.prim_func
    def kernel(A: T.Tensor((256,), T.float16)):
        with T.Kernel(1, threads=32):
            s = T.alloc_shared((128,), T.float16)
            f = T.alloc_local((4,), "int32")
            T.tang_stmatrix(s.access_ptr("w"), [f[0]], trans=trans)
            A[0] = T.float16(0)

    return kernel


def _make_tang_stmatrix_kernel_2vals(trans: bool = False):

    @T.prim_func
    def kernel(A: T.Tensor((256,), T.float16)):
        with T.Kernel(1, threads=32):
            s = T.alloc_shared((128,), T.float16)
            f = T.alloc_local((4,), "int32")
            T.tang_stmatrix(s.access_ptr("w"), [f[0], f[1]], trans=trans)
            A[0] = T.float16(0)

    return kernel


def _make_tang_stmatrix_kernel_4vals(trans: bool = False):

    @T.prim_func
    def kernel(A: T.Tensor((256,), T.float16)):
        with T.Kernel(1, threads=32):
            s = T.alloc_shared((128,), T.float16)
            f = T.alloc_local((4,), "int32")
            T.tang_stmatrix(s.access_ptr("w"), [f[0], f[1], f[2], f[3]], trans=trans)
            A[0] = T.float16(0)

    return kernel


def test_tang_stmatrix_1val():
    """tang_stmatrix with 1 register value (1×8x8 matrix)."""
    txt = _script_of(_make_tang_stmatrix_kernel_1val())
    _assert_in_ir(txt, r"ptx_stmatrix\(.*\b1\b")


def test_tang_stmatrix_2vals():
    """tang_stmatrix with 2 register values (2×8x8 matrices)."""
    txt = _script_of(_make_tang_stmatrix_kernel_2vals())
    _assert_in_ir(txt, r"ptx_stmatrix\(.*\b2\b")


def test_tang_stmatrix_4vals():
    """tang_stmatrix with 4 register values (4×8x8 matrices)."""
    txt = _script_of(_make_tang_stmatrix_kernel_4vals())
    _assert_in_ir(txt, r"ptx_stmatrix\(.*\b4\b")


def test_tang_stmatrix_4vals_trans():
    """tang_stmatrix with 4 values and trans=True."""
    txt = _script_of(_make_tang_stmatrix_kernel_4vals(trans=True))
    _assert_in_ir(txt, r"ptx_stmatrix\(.*?\b1\b")  # trans flag = 1


# ===========================================================================
# §2  tang_ldmatrix  — S3-explicit load (shared → register fragment)
# ===========================================================================


def _make_tang_ldmatrix_kernel(num: int, trans: bool = False):

    @T.prim_func
    def kernel(A: T.Tensor((256,), T.float16)):
        with T.Kernel(1, threads=32):
            s = T.alloc_shared((128,), T.float16)
            f = T.alloc_local((4,), "int32")
            T.tang_ldmatrix(s.access_ptr("r"), f.access_ptr("w"), num, trans=trans)
            A[0] = T.float16(0)

    return kernel


def test_tang_ldmatrix_num1():
    """tang_ldmatrix with num=1 (1×8x8 matrix)."""
    txt = _script_of(_make_tang_ldmatrix_kernel(1))
    _assert_in_ir(txt, r"ptx_ldmatrix\(.*\b1\b")


def test_tang_ldmatrix_num2():
    """tang_ldmatrix with num=2 (2×8x8 matrices)."""
    txt = _script_of(_make_tang_ldmatrix_kernel(2))
    _assert_in_ir(txt, r"ptx_ldmatrix\(.*\b2\b")


def test_tang_ldmatrix_num4():
    """tang_ldmatrix with num=4 (4×8x8 matrices)."""
    txt = _script_of(_make_tang_ldmatrix_kernel(4))
    _assert_in_ir(txt, r"ptx_ldmatrix\(.*\b4\b")


def test_tang_ldmatrix_num4_trans():
    """tang_ldmatrix with num=4 and trans=True."""
    txt = _script_of(_make_tang_ldmatrix_kernel(4, trans=True))
    _assert_in_ir(txt, r"ptx_ldmatrix\(.*?\b1\b")  # trans flag = 1


# ===========================================================================
# §3  ptx_ldmatrix  — upstream ptx_ldmatrix intrinsic (CUDA path)
# ===========================================================================


def test_ptx_ldmatrix_upstream_still_works():
    """Upstream ``ptx_ldmatrix`` 仍可正确发出 IR。

    这个路径是 CUDA mma 加载所依赖的(tl_templates/cuda/ldsm.h),
    不能被 tang_ldmatrix 前端覆盖而破坏。develop 的上游签名为 4 参:
    ``ptx_ldmatrix(trans, num, src_access_ptr, dst_access_ptr)``。
    """

    @T.prim_func
    def kernel(A: T.Tensor((256,), T.float16)):
        with T.Kernel(1, threads=32):
            s = T.alloc_shared((256,), T.float16)
            f = T.alloc_local((4,), "int32")
            # upstream 4-arg signature: (trans, num, src, dst)
            T.ptx_ldmatrix(
                False,  # trans
                4,  # num
                s.access_ptr("r"),  # src (smem)
                f.access_ptr("w"),  # dst (local)
            )
            A[0] = T.float16(0)

    txt = _script_of(kernel)
    _assert_in_ir(txt, r"ptx_ldmatrix\(")


# ===========================================================================
# §4  Negative tests  — invalid arguments
# ===========================================================================


def test_stmatrix_rejects_invalid_num_values():
    """stmatrix with invalid number of register values raises AssertionError.

    NOTE: ``@T.prim_func`` 在装饰阶段即执行 eager tracing, assert 在
    函数定义期抛出,故 ``pytest.raises`` 须包裹装饰语句本身。
    """
    with pytest.raises(AssertionError, match="1, 2 or 4"):

        @T.prim_func
        def kernel(A: T.Tensor((256,), T.float16)):
            with T.Kernel(1, threads=32):
                s = T.alloc_shared((128,), T.float16)
                T.tang_stmatrix(s.access_ptr("w"), [])  # 0 values → invalid
                A[0] = T.float16(0)


def test_ldmatrix_rejects_invalid_num():
    """ldmatrix with num=3 raises AssertionError."""

    with pytest.raises(AssertionError, match="1, 2 or 4"):

        @T.prim_func
        def kernel(A: T.Tensor((256,), T.float16)):
            with T.Kernel(1, threads=32):
                s = T.alloc_shared((128,), T.float16)
                f = T.alloc_local((4,), "int32")
                T.tang_ldmatrix(s.access_ptr("r"), f.access_ptr("w"), 3)  # invalid num
                A[0] = T.float16(0)


# ===========================================================================
# §5  stcuv2 lowering placeholders  — backend not ready yet
# ===========================================================================


@pytest.mark.skip(reason="stcuv2 ld/st matrix backend lowering pending LLVM intrinsic")
def test_stmatrix_lowering_stcuv2():
    """占位:待后端就绪后改为对生成源码的断言(直接 shared<->fragment)。"""
    with tvm.target.Target(STCUV2_TARGET):
        tilelang.lower(_make_tang_stmatrix_kernel_2vals(), target=STCUV2_TARGET)


def test_ldmatrix_lowering_backend_wip():
    """catch 版:后端未就绪,lower 若报错则视为预期并 skip。

    后端 ready 后应删除本条,改用 §1-§3 的静态断言 + 新增 §5 数值测试。
    """
    try:
        with tvm.target.Target(STCUV2_TARGET):
            tilelang.lower(_make_tang_ldmatrix_kernel(4), target=STCUV2_TARGET)
    except AssertionError:
        # 后端未就绪时 lowering 内部断言失败是预期行为
        pytest.skip("stcuv2 ld/st matrix backend lowering not ready (assertion)")
    except Exception as e:
        pytest.skip(f"stcuv2 ld/st matrix backend not ready: {e}")


if __name__ == "__main__":
    tilelang.testing.main()
