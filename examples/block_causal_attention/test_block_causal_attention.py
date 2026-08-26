import pytest
import tilelang.testing
from tilelang.utils.target import determine_target, target_is_tang

from . import block_causal_attention
from . import block_causal_attention_varlen

_TARGET = determine_target(tilelang.env.get_default_target(), return_object=True)
_IS_TANG = target_is_tang(_TARGET)


def test_block_causal_attention_fixed():
    block_causal_attention.test_block_causal_attention_all_block_sizes()


@pytest.mark.parametrize("dllm_block", block_causal_attention_varlen._SUPPORTED_DLLM_BLOCKS)
def test_block_causal_attention_varlen_default_blocks(dllm_block):
    block_causal_attention_varlen._run_varlen_case(
        [128, 256, 384],
        heads=2,
        dim=64,
        dllm_block=dllm_block,
    )


def test_block_causal_attention_varlen_block32():
    block_causal_attention_varlen._run_varlen_case(
        [128, 256],
        heads=2,
        dim=64,
        dllm_block=16,
        block_size=32,
    )


@pytest.mark.skipif(
    _IS_TANG,
    reason="block_size=128 and dim=128 require 110544 bytes of dynamic shared memory, exceeding S2 launch resources",
)
def test_block_causal_attention_varlen_block128():
    block_causal_attention_varlen._run_varlen_case(
        [512, 1024],
        heads=2,
        dim=128,
        dllm_block=64,
        block_size=128,
    )


if __name__ == "__main__":
    tilelang.testing.main()
