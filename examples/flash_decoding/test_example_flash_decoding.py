import os
import pytest
import tilelang.testing

import example_gqa_decode
import example_mha_inference
import example_gqa_decode_varlen_logits

_is_cutedsl = os.environ.get("TILELANG_TARGET", "").lower() == "cutedsl"


def test_ptpu_gqa_decode_uses_s2_resource_config(monkeypatch):
    monkeypatch.setattr(example_gqa_decode, "is_ptpu_available", lambda: True)
    example_gqa_decode.get_heuristic_config.cache_clear()
    try:
        config, sm_version = example_gqa_decode.get_heuristic_config()
        assert config == {
            "block_N": 64,
            "block_H": 8,
            "num_split": 16,
            "num_stages": 1,
            "threads": 128,
        }
        assert sm_version == 0
    finally:
        example_gqa_decode.get_heuristic_config.cache_clear()


@pytest.mark.skipif(_is_cutedsl, reason="CuTeDSL backend does not support alloc_global yet")
def test_example_example_gqa_decode():
    example_gqa_decode.main(do_bench=False)


@pytest.mark.skipif(_is_cutedsl, reason="CuTeDSL backend does not support alloc_global yet")
def test_example_example_mha_inference():
    example_mha_inference.main(BATCH=1, H=32, Q_CTX=128, KV_CTX=2048, D_HEAD=128, causal=False)


def test_example_example_gqa_decode_varlen_logits():
    example_gqa_decode_varlen_logits.main()


if __name__ == "__main__":
    tilelang.testing.main()
