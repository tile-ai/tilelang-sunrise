import pytest
import torch

from tests.ops.gla_test_utils import (
    cosine_sim,
    get_tolerances,
    gla_fwd_chunked_torch,
)
from tests.test_base import FixtureBase
from tileops.ops import GLAFwdOp

try:
    from fla.ops.gla import chunk_gla
except ImportError:
    chunk_gla = None


class GLAFwdFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune",
            [
                pytest.param(2, 64, 2, 64, 64, 64, torch.float32, False, marks=pytest.mark.smoke),
                pytest.param(2, 64, 2, 64, 64, 64, torch.float16, False, marks=pytest.mark.smoke),
                pytest.param(2, 64, 2, 64, 64, 64, torch.bfloat16, False, marks=pytest.mark.smoke),
                pytest.param(1, 128, 4, 64, 64, 64, torch.float32, False, marks=pytest.mark.full),
                pytest.param(1, 128, 4, 64, 64, 64, torch.float16, False, marks=pytest.mark.full),
                pytest.param(1, 128, 4, 64, 64, 64, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(2, 256, 4, 64, 64, 64, torch.float16, False, marks=pytest.mark.full),
            ],
        ),
    ]


@GLAFwdFixture
def test_gla_fwd(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    torch.manual_seed(42)
    B, T, H, K, V, BC = batch, seq_len, heads, dim_k, dim_v, chunk_size
    scale = K**-0.5

    q = torch.randn(B, T, H, K, dtype=dtype).ptpu() * 0.1
    k = torch.randn(B, T, H, K, dtype=dtype).ptpu() * 0.1
    v = torch.randn(B, T, H, V, dtype=dtype).ptpu() * 0.1
    g = -torch.rand(B, T, H, K, dtype=dtype).ptpu()

    # --- Torch reference ---
    q_cpu, k_cpu, v_cpu, g_cpu = q.cpu(), k.cpu(), v.cpu(), g.cpu()
    ref_o = gla_fwd_chunked_torch(q_cpu, k_cpu, v_cpu, g_cpu, BC, scale=scale)

    # --- FLA reference (if available) ---
    if chunk_gla is not None:
        fla_o, _ = chunk_gla(
            q_cpu.float(), k_cpu.float(), v_cpu.float(), g_cpu.float(), scale=scale
        )
        cos = cosine_sim(ref_o, fla_o)
        print(f"  FLA vs ref o: cosine={cos:.6f}")
        assert cos > 0.99, f"FLA vs ref o cosine too low: {cos:.6f}"

    fwd_op = GLAFwdOp(
        chunk_size=BC,
        scale=scale,
        tune=tune,
    )
    op_o, _ = fwd_op.forward(q, k, v, g)
    # PTPU kernel launches are asynchronous; wait for completion before
    # reading op_o, otherwise the comparison races the kernel.
    torch.ptpu.synchronize()

    tols = get_tolerances(dtype)
    op_o_cpu = op_o.cpu()
    cos = cosine_sim(ref_o, op_o_cpu)
    print(f"  TileOPs vs ref o: cosine={cos:.6f}")
    torch.testing.assert_close(
        op_o_cpu.float(),
        ref_o.float(),
        **tols,
        msg=lambda m: f"o: {m}",
    )

    # --- TileOPs vs FLA ---
    if chunk_gla is not None:
        cos = cosine_sim(fla_o, op_o_cpu)
        print(f"  TileOPs vs FLA o: cosine={cos:.6f}")
        assert cos > 0.99, f"TileOPs vs FLA o cosine too low: {cos:.6f}"
