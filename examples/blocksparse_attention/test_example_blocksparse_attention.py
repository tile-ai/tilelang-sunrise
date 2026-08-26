import tilelang.testing
import example_tilelang_block_sparse_attn
import example_tilelang_sparse_gqa_decode_paged
import example_tilelang_sparse_gqa_decode_varlen_indice
import example_tilelang_sparse_gqa_decode_varlen_mask
import pytest


@pytest.mark.skip(reason="This is test case for triton, not tilelang.")
def test_block_sparse_attn_triton():
    import block_sparse_attn_triton

    block_sparse_attn_triton.main()


def test_example_tilelang_block_sparse_attn():
    example_tilelang_block_sparse_attn.main()


def test_example_tilelang_sparse_gqa_decode_varlen_indice():
    example_tilelang_sparse_gqa_decode_varlen_indice.main(batch=1, max_cache_seqlen=2048)


def test_example_tilelang_sparse_gqa_decode_varlen_mask():
    example_tilelang_sparse_gqa_decode_varlen_mask.main(batch=1, max_cache_seqlen=2048)


def test_example_tilelang_sparse_gqa_decode_paged():
    example_tilelang_sparse_gqa_decode_paged.main(batch=1, max_cache_seqlen=2048, num_pages=128)


if __name__ == "__main__":
    tilelang.testing.main()
