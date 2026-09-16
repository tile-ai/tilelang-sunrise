"""stcuv2 TMEM 分配/释放测试(隐式 alloc_tmem 路径,对齐官方 tilelang)。

官方 tilelang 只在前端暴露 ``T.alloc_tmem``;TMEM 的分配/释放
(``ptx_init/deallocate_tensor_memory``)是编译器内部 IR 原语,由
``LowerSharedTmem`` pass 自动注入、codegen 消费,用户不直接调用。S3 对齐这一设计:
前端不再暴露 ``T.ptx_*`` / ``T.tang_*`` 的 TMEM 分配包装(仅保留 C++ 侧 IR 原语与
codegen 到 ``tc_alloc`` / ``tc_dealloc`` 的别名)。

因此本文件只覆盖用户实际使用的隐式路径:

A. Lowering 断言(不跑 ISS):``T.alloc_tmem`` 在 stcuv2 上经 ``LowerSharedTmem``
   注入 ``tang_init/deallocate_tensor_memory``,codegen 生成
   ``tang::ptx::tc_alloc<N>`` / ``tc_dealloc<N>``。

B. ISS 运行断言(真实执行 tc_alloc/tc_dealloc):隐式 ``alloc_tmem`` GEMM 通过
   simulator 后端在 ISS 上跑通(分配 TMEM 累加器 -> gemm 写入 -> drain 回 global),
   并与全精度 torch 参考数值匹配,确保分配/释放在运行期真正生效。
"""

import re

import torch

import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.jit import JITKernel

STCUV2_TARGET = {"kind": "tang", "arch": "stcuv2"}


def _lower_src(func, target=STCUV2_TARGET):
    with tvm.target.Target(target):
        return tilelang.lower(func, target=target).kernel_source


def _sim_jit(func):
    return JITKernel(func, out_idx=[-1], target=STCUV2_TARGET, execution_backend="simulator", verbose=False)


def _rel_max(out, ref) -> float:
    return (out - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)


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


def _make_alloc_tmem_gemm_iss(M, N, K, BM, BN, BK, stages=3):
    """ISS 可运行的隐式 alloc_tmem TN GEMM:C = A @ B^T,累加在 TMEM 中。

    采用与 tcgen5 GEMM 数值用例相同的、经 ISS 验证可跑通的配置(流水化 K-tile +
    TMEM drain),以便真实执行 tc_alloc/tc_dealloc。
    """

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


def test_alloc_tmem_gemm_runs_on_iss():
    """ISS 运行断言:隐式 alloc_tmem GEMM 在 ISS 上真实执行 tc_alloc/tc_dealloc。

    通过 simulator 后端把 kernel 送进 ISS 实际运行:alloc_tmem 分配 TMEM 累加器 ->
    gemm 写入 -> drain 回 global,最后与全精度 torch 参考比较。若 tc_alloc/tc_dealloc
    在运行期不正确(未分配、越界、过早释放等),结果会数值失配或直接崩溃。
    """
    M = N = 128
    K = 64
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.float16)
    B = torch.randn(N, K, dtype=torch.float16)
    ref = A.float() @ B.float().T

    jit = _sim_jit(_make_alloc_tmem_gemm_iss(M, N, K, 128, 128, 64))
    C = jit(A.ptpu(), B.ptpu()).cpu().float()

    assert tuple(C.shape) == (M, N)
    rel = _rel_max(C, ref)
    assert rel < 2e-2, f"rel_diff={rel:.4f} >= tol=2e-2"


if __name__ == "__main__":
    tilelang.testing.main()
