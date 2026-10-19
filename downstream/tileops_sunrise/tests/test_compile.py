# This test validates the compatibility of TileOps operators with torch.compile().
# Check: https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html

import pytest
import torch

from tests.ops.attention.test_gqa_sliding_window import GroupedQueryAttentionSlidingWindowFwdTest
from tests.test_base import FixtureBase
from tileops.ops import GroupedQueryAttentionDenseFwdOp


class MhaCompileFixture(FixtureBase):
    PARAMS = [
        (
            "B, S, H, D, causal, dtype",
            [
                (8, 1024, 32, 128, False, torch.float16),
                (4, 512, 16, 64, True, torch.bfloat16),
            ],
        ),
    ]


@pytest.mark.full
@pytest.mark.usefixtures("isolated_dynamo")
@MhaCompileFixture
def test_mha_kernel_compile(B: int, S: int, H: int, D: int, causal: bool, dtype: torch.dtype):
    test = GroupedQueryAttentionSlidingWindowFwdTest(B, S, H, H, D, causal, -1, -1, dtype)
    op = GroupedQueryAttentionDenseFwdOp(is_causal=causal)
    inputs = test.gen_inputs()
    op(*inputs)  # Materialize the lazy kernel before testing its compiled call.
    compiled_op = torch.compile(op, fullgraph=True, backend="eager")
    test.check(compiled_op, *inputs, atol=1e-2, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
