import tilelang
import tilelang.language as T
import sys  # noqa: F401
from test_utils import DEVICE

import torch


@tilelang.jit(out_idx=[-1], pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def tilelang_chunk_local_cumsum_scalar(
    # task config
    B,
    S,
    H,
    chunk_size=64,
    is_varlen=False,
    head_first=False,
    reverse=False,
    input_dtype=T.float16,
    output_dtype=T.float32,
    # kernel config
    block_S=64,
    threads=256,
    use_fragment=False,
):
    G_shape = (B, H, S) if head_first else (B, S, H)
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), "chunk_size must be a power of 2"
    assert chunk_size == block_S, "chunk_size must be equal to block_S"

    @T.prim_func
    def kernel(
        G: T.Tensor(G_shape, dtype=input_dtype),
        G_new: T.Tensor(G_shape, dtype=output_dtype),
    ):
        with T.Kernel(T.ceildiv(S, block_S), B * H, threads=threads) as (bs, bbh):
            bb, bh = bbh // H, bbh % H
            G_shared = T.alloc_shared((1, block_S), dtype=output_dtype, scope="shared")
            if head_first:
                T.copy(G[bb, bh, bs * block_S : (bs + 1) * block_S], G_shared)
            else:
                T.copy(G[bb, bs * block_S : (bs + 1) * block_S, bh], G_shared)
            if use_fragment:
                G_fragment = T.alloc_fragment((1, block_S), dtype=output_dtype, scope="shared")
                T.copy(G_shared, G_fragment)
                T.cumsum(G_fragment, dim=1, reverse=reverse)
                if head_first:
                    T.copy(G_fragment, G_new[bb, bh, bs * block_S : (bs + 1) * block_S])
                else:
                    T.copy(G_fragment, G_new[bb, bs * block_S : (bs + 1) * block_S, bh])
            else:
                T.cumsum(G_shared, dim=1, reverse=reverse)
                if head_first:
                    T.copy(G_shared, G_new[bb, bh, bs * block_S : (bs + 1) * block_S])
                else:
                    T.copy(G_shared, G_new[bb, bs * block_S : (bs + 1) * block_S, bh])

    return kernel


def prepare_cumsum_input(
    B,
    S,
    H,
    dtype,
):
    G = torch.randn(B, S, H, dtype=dtype).to(DEVICE)
    return G


def prepare_cumsum_output(
    B,
    S,
    H,
    dtype,
):
    G_new = torch.empty(B, S, H, dtype=dtype).to(DEVICE)
    return G_new


def run_test(
    B,
    S,
    H,
    chunk_size,
    reverse,
    head_first,
    input_dtype,
    output_dtype,
    threads,
    use_fragment,
):
    G = prepare_cumsum_input(B, S, H, getattr(torch, input_dtype))
    seq_dim = 2 if head_first else 1
    chunks = G.cpu().float().split(chunk_size, dim=seq_dim)
    if reverse:
        chunks = tuple(chunk.flip(seq_dim).cumsum(seq_dim).flip(seq_dim) for chunk in chunks)
    else:
        chunks = tuple(chunk.cumsum(seq_dim) for chunk in chunks)
    G_new_ref = torch.cat(chunks, dim=seq_dim).to(getattr(torch, output_dtype))

    # tilelang cumsum
    block_S = chunk_size
    kernel = tilelang_chunk_local_cumsum_scalar(
        B=B,
        S=S,
        H=H,
        chunk_size=chunk_size,
        reverse=reverse,
        head_first=head_first,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        block_S=block_S,
        threads=threads,
        use_fragment=use_fragment,
    )
    G_new_tilelang = kernel(G)
    try:
        torch.testing.assert_close(G_new_tilelang.cpu(), G_new_ref, rtol=1e-2, atol=1e-2)
        print("cumsum G_new passed")
    except Exception as e:
        print(f"cumsum G_new failed: {e}")


def main():
    run_test(
        B=1,
        S=64,
        H=1,
        chunk_size=64,
        reverse=True,
        head_first=False,
        input_dtype=T.float32,
        output_dtype=T.float32,
        threads=128,
        use_fragment=False,
    )


if __name__ == "__main__":
    main()
