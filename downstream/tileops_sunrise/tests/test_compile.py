# This test validates the compatibility of TileOps operators with torch.compile().
# Check: https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html

import pytest
import torch

from tests.compile_contract import compile_fullgraph
from tests.workload_test_base import FixtureBase, TestBase
from tileops.attention import GQADenseFwdOp
from workloads.attention.gqa.dense import GQADensePrefillWorkload


class DenseAttentionCompileTest(GQADensePrefillWorkload, TestBase):
    pass


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
    test = DenseAttentionCompileTest(B, S, S, H, H, D, dtype, is_causal=causal)
    op = GQADenseFwdOp(is_causal=causal)
    inputs = test.gen_inputs()
    op(*inputs)  # Materialize the lazy kernel before testing its compiled call.
    compiled_op = compile_fullgraph(op)
    test.check(op, *inputs, runs=compiled_op)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
