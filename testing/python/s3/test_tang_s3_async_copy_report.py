"""The async-copy pass must say *why* it refused.

A refused rewrite is always correct -- the copies just stay blocking -- and that
made it invisible: the only symptom of a missed rewrite was "the pipeline never
overlapped", with nothing to point at. With ``TL_S3_ASYNC_COPY_REPORT=1`` the
pass names the reason, and these tests pin both halves: it reports a real
refusal, and it stays quiet when the rewrite happens.
"""

import pytest
import tilelang
import tilelang.tang.language as T
from tilelang import tvm

STCUV2_TARGET = {"kind": "tang", "arch": "stcuv2"}

M = N = 256
K = 512
BM = BN = 128
BK = 64


def _plain_pipelined_gemm():
    """The shape the rewrite is for: it must be rewritten, silently."""

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((N, K), T.float16),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(2, 2, threads=128) as (bx, by):
            A_s = T.alloc_shared((BM, BK), T.float16)
            B_s = T.alloc_shared((BN, BK), T.float16)
            C_t = T.alloc_tmem((BM, BN), T.float32)
            for k in T.Pipelined(K // BK, num_stages=2):
                T.copy(A[by * BM, k * BK], A_s)
                T.copy(B[bx * BN, k * BK], B_s)
                T.gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=(k == 0))
            T.copy(C_t, C[by * BM, bx * BN])

    return kernel


def _loop_inside_a_thread_guard():
    """A pipelined loop inside ``T.ws``: the counts cannot be read.

    The whole loop sits in a warp-specialised region, so the number of warps
    that would reach the loop's barrier waits is not knowable from here. The
    pass has to refuse, and that refusal is exactly what the report is for.
    """

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((N, K), T.float16),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(2, 2, threads=128) as (bx, by):
            A_s = T.alloc_shared((BM, BK), T.float16)
            B_s = T.alloc_shared((BN, BK), T.float16)
            C_t = T.alloc_tmem((BM, BN), T.float32)
            with T.ws(0, warp_group_size=32):
                for k in T.Pipelined(K // BK, num_stages=2):
                    T.copy(A[by * BM, k * BK], A_s)
                    T.copy(B[bx * BN, k * BK], B_s)
                    T.gemm(A_s, B_s, C_t, transpose_B=True, clear_accum=(k == 0))
            T.copy(C_t, C[by * BM, bx * BN])

    return kernel


def _lower(kernel):
    with tvm.target.Target(STCUV2_TARGET):
        return tilelang.lower(kernel, target=STCUV2_TARGET).kernel_source


def test_refusal_names_its_reason(monkeypatch, capfd):
    monkeypatch.setenv("TL_S3_ASYNC_COPY_REPORT", "1")
    src = _lower(_loop_inside_a_thread_guard())
    err = capfd.readouterr().err
    assert "keeping the blocking copies" in err, err
    assert "free edge" in err, err
    # The fallback is the blocking form: the copy keeps its own drain and no
    # barrier is attached to it.
    assert src.count("tang_bulk_g2s_async") == 0, src
    assert src.count("fcpg2s") > 0, src
    assert src.count("fence_g2s_arrive(") == 0, src
    assert src.count("sync_wait(") == 0, src


def test_rewritten_loop_stays_quiet(monkeypatch, capfd):
    monkeypatch.setenv("TL_S3_ASYNC_COPY_REPORT", "1")
    src = _lower(_plain_pipelined_gemm())
    err = capfd.readouterr().err
    assert "keeping the blocking copies" not in err, err
    assert src.count("tang_bulk_g2s_async") == 0, src
    # Either edge form proves the rewrite happened: the copy publishes a named
    # barrier arrival, or it arrives on the mbarrier the consumer waits.
    assert "fence_g2s_arrive(" in src or "arrive_on_g2s_fence()" in src, src


@pytest.mark.parametrize("value", ["", "0"])
def test_report_is_off_by_default(monkeypatch, capfd, value):
    monkeypatch.setenv("TL_S3_ASYNC_COPY_REPORT", value)
    _lower(_loop_inside_a_thread_guard())
    assert "keeping the blocking copies" not in capfd.readouterr().err
