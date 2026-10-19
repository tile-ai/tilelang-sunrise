"""Tests for the MHC pre/post ops."""

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.ops import MHCPostFwdOp, MHCPreFwdOp
from workloads.mhc import MHCPostWorkload, MHCPreWorkload


class MHCPreTest(MHCPreWorkload, TestBase):
    pass


class MHCPreFixture(FixtureBase):
    PARAMS = [
        (
            "batch, n_expand, c_x, dtype, tune",
            [
                pytest.param(1, 4, 1280, torch.bfloat16, False, marks=pytest.mark.smoke),
                pytest.param(2, 4, 1920, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(4, 4, 2560, torch.bfloat16, False, marks=pytest.mark.full),
            ],
        ),
    ]


def _cosine_compare(output: torch.Tensor, output_ref: torch.Tensor) -> None:
    """Compare using cosine similarity (MHC ops use bf16 and need looser checks)."""
    output = output.cpu().float()
    output_ref = output_ref.cpu().float()
    cos_sim = F.cosine_similarity(output_ref, output, dim=-1, eps=1e-8)

    # Cosine similarity is undefined for a zero reference row. Larger MHC
    # shapes legitimately produce such rows, so compare those by magnitude.
    ref_nonzero = output_ref.norm(dim=-1) > 1e-6
    if ref_nonzero.any():
        assert cos_sim[ref_nonzero].min() > 0.99, (
            f"cosine similarity too low: {cos_sim[ref_nonzero].min().item()}"
        )
    if (~ref_nonzero).any():
        torch.testing.assert_close(
            output[~ref_nonzero],
            output_ref[~ref_nonzero],
            rtol=1e-2,
            atol=1e-2,
        )


@MHCPreFixture
def test_mhc_pre_op(batch: int, n_expand: int, c_x: int, dtype: torch.dtype, tune: bool) -> None:
    test = MHCPreTest(batch, n_expand, c_x, dtype)
    op = MHCPreFwdOp(tune=tune)
    test.check(op, *test.gen_inputs(), compare=_cosine_compare)


class MHCPostTest(MHCPostWorkload, TestBase):
    pass


class MHCPostFixture(FixtureBase):
    PARAMS = [
        (
            "batch, n_expand, c_x, dtype, tune",
            [
                pytest.param(1, 4, 1280, torch.bfloat16, False, marks=pytest.mark.smoke),
                pytest.param(2, 4, 1920, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(4, 4, 2560, torch.bfloat16, False, marks=pytest.mark.full),
            ],
        ),
    ]


@MHCPostFixture
def test_mhc_post_op(batch: int, n_expand: int, c_x: int, dtype: torch.dtype, tune: bool) -> None:
    test = MHCPostTest(batch, n_expand, c_x, dtype)
    op = MHCPostFwdOp(tune=tune)
    test.check(op, *test.gen_inputs(), compare=_cosine_compare)
