"""Op-level tests for MoePermuteNopadFwdOp (tight, non-padded permute).

Verifies:
  - perm_h: correct gather of hidden_states rows into tight expert layout
  - true_offsets / true_sizes: tight per-expert start and count (int32)
  - expert_first_token_offset: exclusive prefix-sum (int64)
  - fwd_idx consistency: perm_h[fwd_idx[flat_idx]] == hidden_states[flat_idx // K]
"""

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops._ops.moe import ContiguousLayoutSpec, MoEPrePermuteFwdOp
from tileops.backend import OpNotAvailableError
from workloads.device import run_device
from workloads.moe import LegacyTightPermuteWorkload
from workloads.numerics import compare_outputs, verification_for
from workloads.workload_base import manifest_call


class MoePermuteNopadTest(LegacyTightPermuteWorkload, TestBase):
    pass


class MoePermuteNopadFixture(FixtureBase):
    PARAMS = [
        (
            "total_tokens, top_k, num_experts, hidden_size, dtype",
            [
                pytest.param(4, 2, 4, 64, torch.bfloat16, marks=pytest.mark.smoke, id="tiny-bf16"),
                pytest.param(4, 2, 4, 64, torch.float16, marks=pytest.mark.smoke, id="tiny-fp16"),
                pytest.param(16, 2, 8, 128, torch.bfloat16, marks=pytest.mark.full, id="small"),
                # numel = total_tokens * top_k = 10 is NOT a multiple of the gather
                # kernel's ROWS_PER_BLOCK (8), so the last block's `if slot < numel`
                # out-of-bounds guard is actually exercised — every other shape here
                # has numel divisible by 8 and never hits the partial tail.
                pytest.param(
                    5, 2, 4, 64, torch.bfloat16, marks=pytest.mark.full, id="partial-tail-numel10"
                ),
            ],
        ),
    ]


@MoePermuteNopadFixture
def test_moe_permute_nopad_op(total_tokens, top_k, num_experts, hidden_size, dtype):
    test = MoePermuteNopadTest(
        manifest_call(
            "MoEPrePermuteFwdOp",
            {"D": str(dtype).removeprefix("torch.")},
            T=total_tokens,
            K=top_k,
            H=hidden_size,
            num_local_experts=num_experts,
            layout={
                "contiguous": {"packing": "tight", "metadata_kind": "physical_psum", "alignment": 1}
            },
        )
    )
    op = MoEPrePermuteFwdOp(ContiguousLayoutSpec.tight_physical_psum(), num_experts)
    hidden_states, topk_ids = test.gen_inputs()

    outputs = op(hidden_states, topk_ids)
    if hidden_states.device.type == "ptpu":
        torch.ptpu.synchronize()
    outputs_ref = test.ref_program(hidden_states.cpu(), topk_ids.cpu())
    evidence = verification_for(test.verification, (hidden_states, topk_ids))
    compare_outputs(outputs, outputs_ref, evidence)
    flops, nbytes = op.eval_roofline()
    assert flops == 0
    assert nbytes > 0


@pytest.mark.smoke
def test_moe_permute_nopad_explicit_shape_mismatch_raises() -> None:
    hidden_states = torch.randn(5, 16, dtype=torch.float16, device=run_device())
    topk_ids = torch.zeros(4, 2, dtype=torch.int32, device=run_device())
    op = MoEPrePermuteFwdOp(ContiguousLayoutSpec.tight_physical_psum(), 4)
    with pytest.raises(ValueError, match="shape"):
        op(hidden_states, topk_ids)


@pytest.mark.smoke
def test_moe_permute_nopad_cpu_input_raises() -> None:
    hidden_states = torch.randn(4, 16, dtype=torch.float16)
    topk_ids = torch.randint(0, 4, (4, 2), dtype=torch.int32)
    op = MoEPrePermuteFwdOp(ContiguousLayoutSpec.tight_physical_psum(), 4)
    with pytest.raises(OpNotAvailableError, match="do not run on cpu"):
        op(hidden_states, topk_ids)


@pytest.mark.smoke
def test_moe_permute_nopad_invalid_dtype_raises() -> None:
    hidden_states = torch.randn(4, 16, dtype=torch.float32, device=run_device())
    topk_ids = torch.randint(0, 4, (4, 2), dtype=torch.int32, device=run_device())
    op = MoEPrePermuteFwdOp(ContiguousLayoutSpec.tight_physical_psum(), 4)
    with pytest.raises(ValueError, match="float32|dtype|float16"):
        op(hidden_states, topk_ids)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
