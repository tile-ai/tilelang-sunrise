import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.kernel_base import Kernel
from tileops.ops import GroupedQueryAttentionDenseFwdOp, MultiHeadAttentionBwdOp
from workloads.attention.gqa import GroupedQueryAttentionSquareFwdWorkload
from workloads.attention.mha import MhaBwdWorkload


class _FakeLegacyMhaBwdKernel(Kernel):
    def __init__(
        self,
        batch: int,
        heads: int,
        seq_len: int,
        dim: int,
        is_causal: bool,
        dtype: torch.dtype,
        tune: bool = False,
    ) -> None:
        super().__init__()

    def forward(self, *args: object, **kwargs: object) -> object:
        return None


def _autotune_ci_kernel(kernel: Kernel) -> None:
    base_config = kernel.default_config
    configs = [{**base_config, "threads": threads} for threads in (128, 256)]
    kernel.autotune(warmup=1, rep=1, autotune_configs=configs)


class MhaBwdTest(MhaBwdWorkload, TestBase):
    pass


class MhaBwdFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq_len, heads, dim, causal, dtype, tune",
            [
                pytest.param(
                    1,
                    1024,
                    8,
                    64,
                    False,
                    torch.float16,
                    False,
                    marks=pytest.mark.smoke,
                    id="smoke-bwd-fp16",
                ),
                pytest.param(
                    1,
                    1024,
                    8,
                    64,
                    False,
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.smoke,
                    id="smoke-bwd-bf16",
                ),
                pytest.param(
                    16,
                    2048,
                    16,
                    128,
                    False,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-bwd-fp16-large",
                ),
                pytest.param(
                    4,
                    4096,
                    16,
                    128,
                    False,
                    torch.bfloat16,
                    True,
                    marks=pytest.mark.full,
                    id="full-bwd-bf16-tuned",
                ),
            ],
        ),
    ]


@pytest.mark.smoke
def test_mha_bwd_rejects_legacy_kernel_map_keys() -> None:
    with pytest.raises(ValueError, match="legacy MHA backward kernel_map keys"):
        MultiHeadAttentionBwdOp(
            batch=1,
            heads=8,
            seq_len=128,
            dim=64,
            is_causal=False,
            kernel_map={"mha_bwd_kernel": _FakeLegacyMhaBwdKernel},
        )


@MhaBwdFixture
def test_mha_bwd(
    batch: int, seq_len: int, heads: int, dim: int, causal: bool, dtype: torch.dtype, tune: bool
) -> None:
    test = MhaBwdTest(batch, heads, seq_len, dim, causal, dtype)
    op = MultiHeadAttentionBwdOp(batch, heads, seq_len, dim, causal, tune=False)
    inputs = test.gen_inputs()
    if tune:
        op(*inputs)
        kernels = op._gqa_op.built_kernels("gqa_bwd_kernel")
        (kernel,) = kernels.values()
        _autotune_ci_kernel(kernel)
    test.check(op, *inputs, atol=5e-3, rtol=1e-5)


class SquareAttentionFwdTest(GroupedQueryAttentionSquareFwdWorkload, TestBase):
    pass


class MhaFwdFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq_len, heads, dim, causal, dtype, tune",
            [
                pytest.param(
                    1,
                    1024,
                    8,
                    64,
                    False,
                    torch.float16,
                    False,
                    marks=[pytest.mark.smoke, pytest.mark.packaging],
                    id="smoke-fwd-fp16",
                ),
                pytest.param(
                    1,
                    1024,
                    8,
                    64,
                    False,
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.smoke,
                    id="smoke-fwd-bf16",
                ),
                pytest.param(
                    16,
                    2048,
                    16,
                    128,
                    False,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="full-fwd-fp16",
                ),
                pytest.param(
                    4,
                    4096,
                    16,
                    128,
                    False,
                    torch.bfloat16,
                    True,
                    marks=pytest.mark.full,
                    id="full-fwd-bf16-tuned",
                ),
            ],
        ),
    ]


@MhaFwdFixture
def test_mha_fwd(batch, seq_len, heads, dim, causal, dtype, tune):
    test = SquareAttentionFwdTest(batch, heads, heads, seq_len, dim, causal, dtype)
    op = GroupedQueryAttentionDenseFwdOp(is_causal=causal)
    inputs = test.gen_inputs()
    if tune:
        _autotune_ci_kernel(op._get_kernel((*inputs, None, None, None, None, None)))
    test.check(op, *inputs, atol=5e-3, rtol=1e-5)
