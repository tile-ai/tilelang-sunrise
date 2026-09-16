import pytest
import torch
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


@pytest.mark.skipif(not _IS_TANG, reason="TANG vectorized mask loop regression")
@pytest.mark.parametrize("seed", [0])
def test_tang_varlen_vectorized_mask_gradients(seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    inputs = [torch.randn(768, 2, 64, dtype=torch.float16, generator=generator) for _ in range(4)]
    q_ref, k_ref, v_ref = [value.clone().requires_grad_(True) for value in inputs[:3]]
    q, k, v = [value.to("ptpu").requires_grad_(True) for value in inputs[:3]]
    cu_seqlens = torch.tensor([0, 128, 384, 768], dtype=torch.int32)
    expected = block_causal_attention_varlen.block_causal_attention_varlen_ref(q_ref, k_ref, v_ref, cu_seqlens, 8)
    actual = block_causal_attention_varlen.block_causal_attention_varlen(q, k, v, cu_seqlens.to("ptpu"), 8, block_size=64)
    actual.backward(inputs[3].to("ptpu"))
    torch.ptpu.synchronize()
    expected.backward(inputs[3])
    torch.testing.assert_close(actual.cpu(), expected, atol=0.02, rtol=0.02)
    for got, reference in [(q.grad, q_ref.grad), (k.grad, k_ref.grad), (v.grad, v_ref.grad)]:
        torch.testing.assert_close(got.cpu(), reference, atol=0.05, rtol=0.05)


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
