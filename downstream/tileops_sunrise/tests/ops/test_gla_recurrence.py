

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import GLADecodeOp
from workloads.linear_attention import GLADecodeTest as _GLADecodeTestWorkload


def gla_decode_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    state: torch.Tensor,
    scale: float = -1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for single-step GLA recurrence."""
    DK = q.shape[-1]
    if scale <= 0:
        scale = DK ** -0.5

    q, k, v = q.float(), k.float(), v.float()
    gk = gk.float()
    state = state.float()

    alpha = torch.exp(gk)
    new_state = alpha.unsqueeze(-1) * state + k.unsqueeze(-1) * v.unsqueeze(-2)
    o = scale * torch.einsum("bhk,bhkv->bhv", q, new_state)

    return o, new_state


class GLADecodeTest(_GLADecodeTestWorkload, TestBase):
    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gk: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        o, new_state = gla_decode_torch(q, k, v, gk, state, self.scale)
        return o.to(self.dtype), new_state.to(self.dtype)


def _local_fused_recurrent_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
):
    """Pure-PyTorch stand-in for ``fla.ops.gla.fused_recurrent_gla``.

    FLA is a Triton/CUDA-only package and is not installed on the S2 host, so
    this reproduces the GLA *fused recurrent* algorithm it implements: an
    explicit per-timestep scan with an outer-product state update and a
    log-space key gate ``gk`` (used directly as the per-step log-decay, i.e.
    ``alpha_t = exp(gk_t)``). Written as an independent recurrent scan (not the
    one-shot einsum form of ``gla_decode_torch``) so it is a genuine second
    implementation for cross-checking.

    Layout follows FLA: q/k/gk are ``[B, T, H, DK]``, v is ``[B, T, H, DV]``,
    ``initial_state``/returned state are ``[B, H, DK, DV]``. Everything is
    computed in float32; the output is cast back to the input dtype.

    Returns ``(o, final_state)`` where ``final_state`` is ``None`` when
    ``output_final_state`` is False.
    """
    B, Tn, H, DK = q.shape
    DV = v.shape[-1]
    if scale is None or scale <= 0:
        scale = DK ** -0.5

    in_dtype = q.dtype
    q, k, v, gk = q.float(), k.float(), v.float(), gk.float()

    h = initial_state.float().clone() if initial_state is not None else q.new_zeros(B, H, DK, DV)

    outs = []
    for t in range(Tn):
        alpha = torch.exp(gk[:, t])  # [B, H, DK]
        # State update: decay the running state on the key dim, then add the
        # rank-1 (k ⊗ v) contribution for this step.
        h = alpha.unsqueeze(-1) * h + k[:, t].unsqueeze(-1) * v[:, t].unsqueeze(-2)
        o_t = scale * torch.einsum("bhk,bhkv->bhv", q[:, t], h)  # [B, H, DV]
        outs.append(o_t)

    o = torch.stack(outs, dim=1).to(in_dtype)  # [B, T, H, DV]
    final_state = h if output_final_state else None
    return o, final_state


try:
    from fla.ops.gla import fused_recurrent_gla
except ImportError:
    # FLA unavailable on this host: fall back to the local pure-torch reference
    # so the cross-check test runs instead of being skipped.
    fused_recurrent_gla = _local_fused_recurrent_gla

# Torch reference implementation (test-only)


# Correctness tests


def _get_tolerances(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return {"atol": 5e-4, "rtol": 5e-4}
    elif dtype == torch.float16:
        return {"atol": 1e-2, "rtol": 1e-2}
    else:  # bfloat16
        return {"atol": 2e-2, "rtol": 2e-2}


class GLADecodeFixture(FixtureBase):
    PARAMS = [
        ("batch, heads, dim_k, dim_v, dtype, tune", [
            pytest.param(1, 4, 64, 64, torch.float32, False, marks=pytest.mark.smoke),
            pytest.param(1, 4, 64, 64, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(1, 4, 64, 64, torch.bfloat16, False, marks=pytest.mark.smoke),
            pytest.param(2, 8, 64, 64, torch.float32, False, marks=pytest.mark.full),
            pytest.param(2, 4, 128, 128, torch.float32, False, marks=pytest.mark.full),
            pytest.param(2, 8, 64, 64, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 8, 64, 64, torch.bfloat16, False, marks=pytest.mark.full),
        ]),
    ]


@GLADecodeFixture
def test_gla_decode(
    batch: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    torch.manual_seed(42)
    test = GLADecodeTest(batch, heads, dim_k, dim_v, dtype)
    op = GLADecodeOp(tune=tune)
    tols = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), **tols)


@GLADecodeFixture
def test_gla_decode_multi_step(
    batch: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    """Test multiple sequential decode steps to verify state propagation."""
    torch.manual_seed(42)
    num_steps = 8
    B, H, DK, DV = batch, heads, dim_k, dim_v

    op = GLADecodeOp(tune=tune)
    tols = _get_tolerances(dtype)

    state_op = torch.zeros(B, H, DK, DV, device="ptpu", dtype=dtype)
    state_ref = torch.zeros(B, H, DK, DV, device="cpu", dtype=dtype)

    for _ in range(num_steps):
        q = torch.randn(B, H, DK, dtype=dtype) * 0.1
        k = torch.randn(B, H, DK, dtype=dtype) * 0.1
        v = torch.randn(B, H, DV, dtype=dtype) * 0.1
        gk = -torch.rand(B, H, DK, dtype=dtype)

        o_ref, state_ref = gla_decode_torch(q, k, v, gk, state_ref)
        o_ref = o_ref.to(dtype)
        state_ref = state_ref.to(dtype)

        q = q.ptpu()
        k = k.ptpu()
        v = v.ptpu()
        gk = gk.ptpu()

        with torch.no_grad():
            o_op, state_op = op(q, k, v, gk, state_op)
        torch.ptpu.synchronize()

        torch.testing.assert_close(o_op.cpu(), o_ref, **tols)
        torch.testing.assert_close(state_op.cpu(), state_ref, **tols)


@GLADecodeFixture
def test_gla_decode_vs_fla(
    batch: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    """Compare TileOPs GLA decode against fused_recurrent_gla with T=1.

    Uses FLA's ``fused_recurrent_gla`` when installed, otherwise the local
    pure-torch stand-in ``_local_fused_recurrent_gla`` (see module top).
    """
    torch.manual_seed(42)
    B, H, DK, DV = batch, heads, dim_k, dim_v
    scale = DK ** -0.5

    # Device principle: generate on CPU (seed only takes effect on CPU), run the
    # reference on CPU, and move to PTU only for the tilelang kernel.
    q = torch.randn(B, H, DK, dtype=dtype) * 0.1
    k = torch.randn(B, H, DK, dtype=dtype) * 0.1
    v = torch.randn(B, H, DV, dtype=dtype) * 0.1
    gk = -torch.rand(B, H, DK, dtype=dtype)
    state = torch.randn(B, H, DK, DV, dtype=dtype) * 0.1

    # TileOPs kernel runs on PTU.
    op = GLADecodeOp(scale=scale, tune=tune)
    with torch.no_grad():
        o_tile, s_tile = op(q.ptpu(), k.ptpu(), v.ptpu(), gk.ptpu(), state.ptpu())

    # Reference (FLA or local) on CPU, BTHD layout with T=1: [B,H,D] -> [B,1,H,D]
    o_ref, s_ref = fused_recurrent_gla(
        q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), gk=gk.unsqueeze(1),
        scale=scale, initial_state=state.contiguous(),
        output_final_state=True,
    )
    o_ref = o_ref.squeeze(1).to(dtype)

    tols = _get_tolerances(dtype)
    torch.testing.assert_close(o_tile.cpu(), o_ref, **tols)
    torch.testing.assert_close(s_tile.cpu(), s_ref.to(dtype), **tols)


@pytest.mark.smoke
def test_gla_decode_rejects_manifest_shape_mismatch() -> None:
    op = object.__new__(GLADecodeOp)
    op.batch = 2
    op.heads = 3
    op.dim_k = 4
    op.dim_v = 5
    op.scale = -1.0
    op.dtype = torch.float32

    q = torch.empty(2, 3, 4)
    k = torch.empty(2, 3, 4)
    v = torch.empty(2, 3, 5)
    gk = torch.empty(2, 3, 5)
    state = torch.empty(2, 3, 4, 5)

    with pytest.raises(ValueError, match="gk must have shape"):
        op.forward(q, k, v, gk, state)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
