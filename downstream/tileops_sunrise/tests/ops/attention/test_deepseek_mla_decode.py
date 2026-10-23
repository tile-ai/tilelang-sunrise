import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.attention import MLADecodeKernel
from tileops.ops import MultiHeadLatentAttentionDecodeWithKVCacheFwdOp
from workloads.attention.deepseek import MlaDecodeWorkload


class MlaDecodeTest(MlaDecodeWorkload, TestBase):
    pass


class MlaDecodeFixture(FixtureBase):
    PARAMS = [
        (
            "batch, heads, heads_kv, seq_len_kv, dim, dim_pe, dtype, tune",
            [
                pytest.param(
                    32, 128, 1, 8192, 512, 64, torch.float16, False, marks=pytest.mark.smoke
                ),
            ],
        ),
    ]


@MlaDecodeFixture
def test_mla_decode(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len_kv: int,
    dim: int,
    dim_pe: int,
    dtype: torch.dtype,
    tune: bool,
):
    test = MlaDecodeTest(batch, heads, heads_kv, seq_len_kv, dim, dim_pe, dtype)
    op = MultiHeadLatentAttentionDecodeWithKVCacheFwdOp(
        batch, heads, heads_kv, seq_len_kv, dim, dim_pe, tune=tune
    )
    test.check(op, *test.gen_inputs(), atol=1e-3, rtol=1e-3)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "heads,seq_len,dim,split",
    [(8, 33, 64, 1), (48, 65, 64, 1), (8, 33, 64, 2), (48, 65, 64, 4), (128, 64, 512, 1)],
)
def test_mla_tail_split(heads, seq_len, dim, split):
    torch.manual_seed(73)
    q = torch.randn(1, heads, dim, dtype=torch.float16)
    qp = torch.randn(1, heads, 64, dtype=torch.float16)
    kv = torch.randn(1, seq_len, 1, dim, dtype=torch.float16)
    kp = torch.randn(1, seq_len, 1, 64, dtype=torch.float16)
    scores = (
        torch.einsum("bhd,bnd->bhn", q.float(), kv[:, :, 0].float())
        + torch.einsum("bhd,bnd->bhn", qp.float(), kp[:, :, 0].float())
    ) / (dim + 64) ** 0.5
    ref = torch.einsum("bhn,bnd->bhd", scores.softmax(-1), kv[:, :, 0].float()).half()
    kernel = MLADecodeKernel(
        1,
        heads,
        1,
        seq_len,
        dim,
        64,
        torch.float16,
        config={"block_H": 32, "block_N": 32, "num_split": split, "num_stages": 1, "threads": 128},
    )
    out = kernel(*(t.ptpu() for t in (q, qp, kv, kp)))
    torch.ptpu.synchronize()
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
