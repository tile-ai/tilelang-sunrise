"""stcuv2 TMEM 分配/释放测试(隐式 alloc_tmem 路径,对齐官方 tilelang)。

官方 tilelang 只在前端暴露 ``T.alloc_tmem``;TMEM 的分配/释放
(``ptx_init/deallocate_tensor_memory``)是编译器内部 IR 原语,由
``LowerSharedTmem`` pass 自动注入、codegen 消费,用户不直接调用。S3 对齐这一设计:
前端不再暴露 ``T.ptx_*`` / ``T.tang_*`` 的 TMEM 分配包装(仅保留 C++ 侧 IR 原语与
codegen 到 ``tc_alloc`` / ``tc_dealloc`` 的别名)。

因此本文件只覆盖用户实际使用的隐式路径:

Lowering 断言:``T.alloc_tmem`` 在 stcuv2 上经 ``LowerSharedTmem``
注入 ``tang_init/deallocate_tensor_memory``,codegen 生成
``tang::ptx::tc_alloc<N>`` / ``tc_dealloc<N>``。
"""

import re


import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang import tvm as tvm

STCUV2_TARGET = {"kind": "tang", "arch": "stcuv2"}


def _lower_src(func, target=STCUV2_TARGET):
    with tvm.target.Target(target):
        return tilelang.lower(func, target=target).kernel_source


def _make_alloc_tmem_kernel():
    """隐式路径:alloc_tmem 触发 pass 注入的 tc_alloc/tc_dealloc。"""
    # K=64 keeps the fp16 operand row at the 128-byte per-MMA cap.
    M = N = 128
    K = 64

    @T.prim_func
    def kernel(A: T.Tensor((M, K), T.float16), B: T.Tensor((N, K), T.float16), C: T.Tensor((M, N), T.float32)):
        with T.Kernel(1, 1, threads=128):
            A_s = T.alloc_shared((M, K), T.float16)
            B_s = T.alloc_shared((N, K), T.float16)
            C_t = T.alloc_tmem((M, N), T.float32)
            T.copy(A, A_s)
            T.copy(B, B_s)
            T.gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=True)
            T.copy(C_t, C)

    return kernel


def test_alloc_tmem_emits_tc_alloc_dealloc():
    """隐式路径:alloc_tmem 在 stcuv2 上生成 tc_alloc / tc_dealloc。"""
    src = _lower_src(_make_alloc_tmem_kernel())
    assert re.search(r"tc_alloc<\d+>\(", src), "expected tang::ptx::tc_alloc<N>()"
    assert re.search(r"tc_dealloc<\d+>\(", src), "expected tang::ptx::tc_dealloc<N>()"


if __name__ == "__main__":
    tilelang.testing.main()
