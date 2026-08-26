# ruff: noqa
import pytest
import tilelang
import tilelang.testing

import topk_selector
import fp8_lighting_indexer
import sparse_mla_fwd
import sparse_mla_fwd_pipelined
import sparse_mla_bwd


def test_example_topk_selector():
    topk_selector.test_topk_selector()


@tilelang.testing.pytest.mark.skip("FP8 is not supported on S2.")
def test_example_fp8_lighting_indexer():
    fp8_lighting_indexer.test_fp8_lighting_indexer(S=512, SKV=1024, H=32, HKV=1, D=64, kv_stride=1)


def test_example_sparse_mla_fwd():
    # small shapes for testing
    sparse_mla_fwd.test_sparse_mla_fwd(S=16, SKV=64, H=8, HKV=1, DQK=576, DV=64, topk=64, check_correctness=False, block_I=32, threads=128)


@pytest.mark.skip(reason="On S2, NOT support mbarrier yet.")
def test_example_sparse_mla_fwd_pipelined():
    # small shapes for testing
    sparse_mla_fwd_pipelined.test_sparse_mla_fwd_pipelined(S=256, SKV=512, H=64, HKV=1, DQK=576, DV=512, topk=256, check_correctness=False)


def test_example_sparse_mla_bwd():
    sparse_mla_bwd.test_sparse_mla_bwd(
        S=16, SKV=32, H=8, HKV=1, DQKV=576, DV=512, topk=32, check_correctness=False, block_I=32, block_size=32, threads=128
    )


@pytest.mark.skip(reason="TANGLaunch TANG_ERROR_LAUNCH_OUT_OF_RESOURCES, no need to test this.")
def test_example_sparse_mla_bwd_large_h():
    # test for large H
    sparse_mla_bwd.test_sparse_mla_bwd(S=128, SKV=256, H=128, HKV=1, DQKV=576, DV=512, topk=128, check_correctness=False)


if __name__ == "__main__":
    tilelang.testing.main()
