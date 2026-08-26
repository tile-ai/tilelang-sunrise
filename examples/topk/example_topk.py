import tilelang
import tilelang.language as T
import torch
import itertools
import argparse
from tilelang.utils.device import get_current_device


def get_configs():
    iter_params = dict(
        blk_m=[64, 128, 256],
        threads=[128, 256, 512],
    )
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]


@tilelang.autotune(configs=get_configs())
@tilelang.jit
def tl_topk(logits, topk, blk_m, threads=128):
    M, N = T.const("M, N")
    dtype = T.float32

    logits: T.Tensor([M, N], dtype)
    topk_gates = T.empty([M, topk], dtype)
    topk_indices = T.empty([M, topk], T.int32)

    with T.Kernel(T.ceildiv(M, blk_m), threads=threads) as bx:
        logits_frag = T.alloc_fragment([blk_m, N], dtype=dtype)
        max_val = T.alloc_fragment([blk_m], dtype=dtype)
        expand_max_idx = T.alloc_fragment([blk_m, N], T.int32)
        max_idx = T.alloc_fragment([blk_m], T.int32)

        T.copy(logits[bx * blk_m, 0], logits_frag)

        for k in T.serial(topk):
            T.reduce_max(logits_frag, max_val, dim=1, clear=True)

            for i, j in T.Parallel(blk_m, N):
                expand_max_idx[i, j] = T.if_then_else(max_val[i] == logits_frag[i, j], j, N)

            T.reduce_min(expand_max_idx, max_idx, dim=1, clear=True)

            for i, j in T.Parallel(blk_m, N):
                logits_frag[i, j] = T.if_then_else(j == max_idx[i], -10000.0, logits_frag[i, j])

            for i in T.Parallel(blk_m):
                topk_gates[bx * blk_m + i, k] = max_val[i]
                topk_indices[bx * blk_m + i, k] = max_idx[i]

    return topk_gates, topk_indices


def ref_program(logits, top_k):
    top_k_gates, top_k_indices = logits.topk(top_k, dim=1)

    return top_k_gates, top_k_indices.to(torch.int32)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=320, help="num_tokens")
    parser.add_argument("--N", type=int, default=128, help="num_experts")
    parser.add_argument("--topk", type=int, default=6, help="topk")
    parser.add_argument("--blk_m", type=int, default=64, help="blk_m")
    args = parser.parse_args(argv)
    M, N, topk, blk_m = args.M, args.N, args.topk, args.blk_m

    logits = torch.rand((M, N), device=get_current_device(), dtype=torch.float32)

    tl_gates, tl_indices = tl_topk(logits, topk, blk_m=blk_m)

    torch_gates, torch_indices = ref_program(logits, topk)

    # test accuracy
    torch.testing.assert_close(tl_gates.cpu(), torch_gates.cpu())
    torch.testing.assert_close(tl_indices.cpu(), torch_indices.cpu())

    # profile
    from tilelang.profiler import do_bench

    tilelang_latency = do_bench(lambda: tl_topk(logits, topk, blk_m=blk_m))
    print(f"Tilelang latency: {tilelang_latency}")


def run_regression_perf(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=320, help="num_tokens")
    parser.add_argument("--N", type=int, default=128, help="num_experts")
    parser.add_argument("--topk", type=int, default=6, help="topk")
    parser.add_argument("--blk_m", type=int, default=64, help="blk_m")
    # In benchmark mode, ignore process-wide sys.argv unless an explicit argv is provided.
    args = parser.parse_args(argv or [])
    M, N, topk, blk_m = args.M, args.N, args.topk, args.blk_m

    logits = torch.rand((M, N), device=get_current_device(), dtype=torch.float32)

    tl_gates, tl_indices = tl_topk(logits, topk, blk_m=blk_m)

    torch_gates, torch_indices = ref_program(logits, topk)

    torch.testing.assert_close(tl_gates.cpu(), torch_gates.cpu())
    torch.testing.assert_close(tl_indices.cpu(), torch_indices.cpu())

    from tilelang.profiler import do_bench

    return do_bench(lambda: tl_topk(logits, topk, blk_m=blk_m))


if __name__ == "__main__":
    main()
