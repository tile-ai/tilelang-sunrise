from tilelang import tvm as tvm
from tilelang.backend.target import determine_target
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang.utils.device import get_current_device
import pytest
import torch

auto_target = tvm.target.Target(determine_target(tl.env.get_default_target()))


@pytest.mark.parametrize(
    "block_M, block_N, block_K, threads, vec_load_b, dtype",
    [
        (64, 64, 32, 128, 8, T.float16),
    ],
)
def test_loop_tail_split(block_M, block_N, block_K, threads, vec_load_b, dtype):
    N = tvm.te.var("n")
    K = tvm.te.var("k")

    def before():
        @T.prim_func
        def main(
            B: T.Tensor((K, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), threads=threads) as (bx):
                B_shared = T.alloc_shared((block_K, block_N), dtype)
                thread_bindings = T.thread_binding(0, threads, "threadIdx.x")
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                    t = thread_bindings
                    for i in T.unroll(0, block_N * block_K // (threads * vec_load_b)):
                        for vec in T.Parallel(vec_load_b):
                            B_shared[
                                i * (threads * vec_load_b // block_N) + t // (block_N // vec_load_b),
                                t % (block_N // vec_load_b) * (block_N // vec_load_b) + vec,
                            ] = T.if_then_else(
                                k * block_K + i * (threads * vec_load_b // block_N) + t // (block_N // vec_load_b) < K
                                and bx * block_N + t % (block_N // vec_load_b) * (block_N // vec_load_b) < N,
                                B[
                                    k * block_K + i * (threads * vec_load_b // block_N) + t // (block_N // vec_load_b),
                                    bx * block_N + t % (block_N // vec_load_b) * (block_N // vec_load_b) + vec,
                                ],
                                T.float16(0),
                            )

        return tvm.IRModule({"main": main})

    def after():
        @T.prim_func
        def main(
            B: T.Tensor((K, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), threads=threads) as (bx):
                B_shared = T.alloc_shared((block_K, block_N), dtype)
                thread_bindings = T.thread_binding(0, threads, "threadIdx.x")
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                    t = thread_bindings
                    for i in T.unroll(0, block_N * block_K // (threads * vec_load_b)):
                        if (k * block_K + i * (threads * vec_load_b // block_N) + t // (block_N // vec_load_b)) * N % vec_load_b == 0:
                            for vec in T.vectorized(vec_load_b):
                                B_shared[
                                    i * (threads * vec_load_b // block_N) + t // (block_N // vec_load_b),
                                    t % (block_N // vec_load_b) * (block_N // vec_load_b) + vec,
                                ] = T.if_then_else(
                                    k * block_K + i * (threads * vec_load_b // block_N) + t // (block_N // vec_load_b) < K
                                    and bx * block_N + t % (block_N // vec_load_b) * (block_N // vec_load_b) < N,
                                    B[
                                        k * block_K + i * (threads * vec_load_b // block_N) + t // (block_N // vec_load_b),
                                        bx * block_N + t % (block_N // vec_load_b) * (block_N // vec_load_b) + vec,
                                    ],
                                    T.float16(0),
                                )
                        else:
                            for vec in T.serial(vec_load_b):
                                B_shared[
                                    i * (threads * vec_load_b // block_N) + t // (block_N // vec_load_b),
                                    t % (block_N // vec_load_b) * (block_N // vec_load_b) + vec,
                                ] = T.if_then_else(
                                    k * block_K + i * (threads * vec_load_b // block_N) + t // (block_N // vec_load_b) < K
                                    and bx * block_N + t % (block_N // vec_load_b) * (block_N // vec_load_b) < N,
                                    B[
                                        k * block_K + i * (threads * vec_load_b // block_N) + t // (block_N // vec_load_b),
                                        bx * block_N + t % (block_N // vec_load_b) * (block_N // vec_load_b) + vec,
                                    ],
                                    T.float16(0),
                                )

        return tvm.IRModule({"main": main})

    with tvm.target.Target(auto_target):
        mod = tvm.tirx.transform.BindTarget(auto_target)(before())
        mod = tl.transform.MaterializeKernelLaunch()(mod)
        mod = tl.transform.LayoutInference()(mod)
        mod = tvm.tirx.transform.Simplify()(mod)
        ref_mod = tvm.tirx.transform.BindTarget(auto_target)(after())
        ref_mod = tl.transform.MaterializeKernelLaunch()(ref_mod)
        ref_mod = tvm.tirx.transform.Simplify()(ref_mod)
        # Note(tzj): The structures are equal except one more "for" loop after the LayoutInference pass
        # This loop is "for vec in T.parallel(1)",
        # Since the loop var "vec" is never used in the loop body, it does not affect the correctness
        tvm.ir.structural_equal(mod, ref_mod)
        # tvm.ir.assert_structural_equal(mod, ref_mod)


def test_static_ragged_copy_minimizes_full_thread_padding():
    n = 514
    threads = 128

    @T.prim_func
    def main(
        A: T.Tensor((n,), T.float32),
        B: T.Tensor((n,), T.float32),
    ):
        with T.Kernel(1, threads=threads):
            T.copy(A, B)

    with tvm.target.Target(auto_target):
        artifact = tl.lower(main, target=auto_target, enable_device_compile=False)

    kernel_source = str(artifact.kernel_source)
    assert "__launch_bounds__(128, 1)" in kernel_source
    assert "for (int i = 0; i < 5; ++i)" in kernel_source
    assert "threadIdx.x) >> 1)) < 257" in kernel_source
    assert "float2" not in kernel_source
    assert "threadIdx.x) < 1" not in kernel_source


def test_static_ragged_fp8_copy_minimizes_full_thread_padding():
    n = 3072
    threads = 128

    @T.prim_func
    def main(
        B: T.Tensor((n,), T.float8_e4m3),
    ):
        with T.Kernel(1, threads=threads):
            S = T.alloc_shared((n,), T.float8_e4m3)
            T.copy(S, B, disable_tma=True)

    with tvm.target.Target(auto_target):
        artifact = tl.lower(main, target=auto_target, enable_device_compile=False)

    kernel_source = str(artifact.kernel_source)
    assert "__launch_bounds__(128, 1)" in kernel_source
    if auto_target.kind.name == "tang":
        assert "for (int i = 0; i < 6; ++i)" in kernel_source
        assert "fp8_e4_4_t" in kernel_source
    else:
        assert "for (int i = 0; i < 3; ++i)" in kernel_source
        assert "fp8_e4_8_t" in kernel_source
    assert "fp8_e4_16_t" not in kernel_source


def test_static_ragged_copy_allows_1024_elements_384_threads():
    n = 1024
    threads = 384

    @T.prim_func
    def main(
        A: T.Tensor((n,), T.float32),
        B: T.Tensor((n,), T.float32),
    ):
        with T.Kernel(1, threads=threads):
            T.copy(A, B, coalesced_width=1)

    with tvm.target.Target(auto_target):
        artifact = tl.lower(main, target=auto_target, enable_device_compile=False)

    kernel_source = str(artifact.kernel_source)
    assert "__launch_bounds__(384, 1)" in kernel_source
    assert "for (int i = 0; i < 3; ++i)" in kernel_source
    assert "B[((i * 384) + ((int)threadIdx.x))]" in kernel_source
    assert "(((int)threadIdx.x) >> 7)) < 8" in kernel_source
    assert "threadIdx.x) < 128" not in kernel_source


@pytest.mark.parametrize("block_n", [24, 40, 48, 64, 96])
def test_column_broadcast_fragment_tile_width_lowers(block_n):
    # Regression for issue #2394: LayoutInference used to synthesize a zero-extent
    # leftover iterator for non-power-of-two column broadcasts, then divide by zero.
    m, n = 256, block_n * 4
    block_m = 64

    @T.prim_func
    def main(D_in: T.Tensor((n,), T.bfloat16), Out: T.Tensor((m, n), T.bfloat16)):
        with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=128) as (bx, by):
            d_local = T.alloc_fragment((block_n,), T.float32)
            d_shared = T.alloc_shared((block_n,), T.bfloat16)
            x = T.alloc_fragment((block_m, block_n), T.float32)
            xs = T.alloc_shared((block_m, block_n), T.bfloat16)

            T.copy(D_in[bx * block_n], d_shared)
            T.copy(d_shared, d_local)
            for i, j in T.Parallel(block_m, block_n):
                x[i, j] = d_local[j] * 2.0
            T.copy(x, xs)
            T.copy(xs, Out[by * block_m, bx * block_n])

    with tvm.target.Target(auto_target):
        artifact = tl.lower(main, target=auto_target, enable_device_compile=False)

    assert artifact.kernel_source


@tl.jit(out_idx=[1])
def _column_broadcast_fragment_kernel(block_n):
    m, n = 256, block_n * 4
    block_m = 64

    @T.prim_func
    def main(D_in: T.Tensor((n,), T.float32), Out: T.Tensor((m, n), T.float32)):
        with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=128) as (bx, by):
            d_local = T.alloc_fragment((block_n,), T.float32)
            d_shared = T.alloc_shared((block_n,), T.float32)
            x = T.alloc_fragment((block_m, block_n), T.float32)
            xs = T.alloc_shared((block_m, block_n), T.float32)

            T.copy(D_in[bx * block_n], d_shared)
            T.copy(d_shared, d_local)
            for i, j in T.Parallel(block_m, block_n):
                x[i, j] = d_local[j] * 2.0
            T.copy(x, xs)
            T.copy(xs, Out[by * block_m, bx * block_n])

    return main


@pytest.mark.parametrize("block_n", [24, 32, 40, 48, 96])
def test_column_broadcast_fragment_values(block_n):
    # Numerical regression for issue #2394: the column broadcast must match D*2.
    kernel = _column_broadcast_fragment_kernel(block_n)

    device = get_current_device()
    d = torch.arange(block_n * 4, device=device, dtype=torch.float32)
    out = kernel(d)
    expected = d.unsqueeze(0).expand(256, -1) * 2.0

    if device.type == "ptpu":
        torch.ptpu.synchronize()
    assert torch.equal(out.cpu(), expected.cpu())


def test_layout_inference_shared_scan_no_empty_use_list():
    # Smoke test for RunInferStep's else branch on a shared buffer at kStrict.
    # cumsum's InferScanLayout emits a linear layout for its shared operand at
    # kStrict, and that buffer is absent from use_list_ (addToUseList tracks
    # only fragment buffers). At kStrict update_queue is false, so the else
    # branch sets the layout and returns before the use_list_ enqueue; this
    # exercises the Set path only. The empty-entry bug guarded by a8285369
    # needs a non-fragment buffer returned at kCommon (update_queue true) --
    # see test_layout_inference_shared_staging_pad_no_empty_use_list below,
    # which reaches it through the tang copy staging pad.
    @T.prim_func
    def main(A: T.Tensor((128,), "float32"), Out: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((128,), "float32")
            T.copy(A, S)
            T.cumsum(S, dim=0, reverse=False)
            T.copy(S, Out)

    with tvm.target.Target(auto_target):
        mod = tvm.IRModule({"main": main})
        mod = tvm.tirx.transform.BindTarget(auto_target)(mod)
        mod = tl.transform.MaterializeKernelLaunch()(mod)
        mod = tl.transform.LayoutInference()(mod)
    assert mod is not None
    # Reaching here without crashing (and with a well-formed module) is the check.


def test_layout_inference_shared_staging_pad_no_empty_use_list():
    # The kCommon-path counterpart of the kStrict smoke test above. A
    # fragment→shared 2D copy with the tang copy staging pad enabled makes
    # InferLayout return the shared dst *at kCommon*: the pad skips kStrict
    # (tang copy.cc returns early at `level == kStrict`), so the shared
    # buffer first enters layout_map during FinishInferQueue, where update_queue
    # is true. That buffer is non-fragment and absent from use_list_
    # (addToUseList tracks only fragments), so before a8285369 the else branch's
    # `use_list_[dst]` inserted an empty entry and InferInFreeMode crashed on
    # `uf.Find(infer_indices[0])`. Reaching here without crashing is the check.
    if auto_target.kind.name != "tang":
        pytest.skip("copy staging pad is tang-only")

    @T.prim_func
    def main(A: T.Tensor((64, 64), "float32"), Out: T.Tensor((64, 64), "float32")):
        with T.Kernel(1, threads=128):
            frag = T.alloc_fragment((64, 64), "float32")
            S = T.alloc_shared((64, 64), "float32")
            for i, j in T.Parallel(64, 64):
                frag[i, j] = A[i, j]
            T.copy(frag, S)
            T.copy(S, Out)

    with tvm.target.Target(auto_target):
        mod = tvm.IRModule({"main": main})
        mod = tvm.tirx.transform.BindTarget(auto_target)(mod)
        mod = tl.transform.MaterializeKernelLaunch()(mod)
        with tl.transform.PassContext(config={tl.PassConfigKey.TL_ENABLE_COPY_STAGING_PAD: True}):
            mod = tl.transform.LayoutInference()(mod)
    assert mod is not None
    # Reaching here without crashing (and with a well-formed module) is the check.


if __name__ == "__main__":
    tilelang.testing.main()
