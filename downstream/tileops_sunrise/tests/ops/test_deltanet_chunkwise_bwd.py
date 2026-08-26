
import pytest
import torch

from tests.test_base import FixtureBase
from tileops.ops import DeltaNetBwdOp


def _differentiable_fwd(q, k, v, beta, chunk_size):
    """Fully differentiable chunked forward matching DeltaNet (ungated)."""
    B, H, S, DK = q.shape
    DV = v.shape[-1]
    BC = chunk_size
    NC = S // BC
    h = q.new_zeros(B, H, DK, DV)
    o_chunks = []
    eye = torch.eye(BC, device=q.device, dtype=torch.float32)
    mask = torch.tril(torch.ones(BC, BC, device=q.device, dtype=torch.float32))
    for c in range(NC):
        sl = slice(c * BC, (c + 1) * BC)
        qc = q[:, :, sl, :].float()
        kc = k[:, :, sl, :].float()
        vc = v[:, :, sl, :].float()
        bc = beta[:, :, sl].float()
        Gram = torch.einsum("bhik,bhjk->bhij", kc, kc)
        M = bc.unsqueeze(-1) * Gram
        A = eye + torch.tril(M, diagonal=-1)
        A_inv = torch.linalg.inv(A)
        wc = A_inv @ (kc * bc.unsqueeze(-1))
        uc = A_inv @ (vc * bc.unsqueeze(-1))
        v_new = uc - wc @ h
        o_part = qc @ h
        attn = (qc @ kc.transpose(-2, -1)) * mask
        o_c = o_part + attn @ v_new
        o_chunks.append(o_c)
        h = h + kc.transpose(-2, -1) @ v_new
    return torch.cat(o_chunks, dim=2)


def deltanet_autograd_bwd_torch(do, q, k, v, beta, chunk_size):
    """Compute backward gradients via autograd on the differentiable forward."""
    q_ = q.float().detach().requires_grad_(True)
    k_ = k.float().detach().requires_grad_(True)
    v_ = v.float().detach().requires_grad_(True)
    beta_ = beta.float().detach().requires_grad_(True)

    o = _differentiable_fwd(q_, k_, v_, beta_, chunk_size)
    loss = (o * do.float()).sum()
    dq, dk, dv, dbeta = torch.autograd.grad(loss, [q_, k_, v_, beta_])
    return dq, dk, dv, dbeta

# Autograd-based reference: differentiable forward -> torch.autograd.grad


# Backward correctness tests

def _get_tolerances(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return {"atol": 1e-3, "rtol": 1e-3}
    elif dtype == torch.float16:
        return {"atol": 5e-3, "rtol": 5e-3}
    else:  # bfloat16
        return {"atol": 2e-2, "rtol": 2e-2}


class DeltaNetBwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune", [
            pytest.param(2, 64, 2, 64, 64, 32, torch.float32, False, marks=pytest.mark.smoke),
            pytest.param(2, 64, 2, 64, 64, 32, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(2, 64, 2, 64, 64, 32, torch.bfloat16, False, marks=pytest.mark.smoke),
            pytest.param(1, 128, 4, 64, 64, 32, torch.float32, False, marks=pytest.mark.full),
            pytest.param(1, 128, 4, 64, 64, 32, torch.float16, False, marks=pytest.mark.full),
            pytest.param(1, 128, 4, 64, 64, 32, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(2, 64, 2, 64, 64, 32, torch.bfloat16, True, marks=pytest.mark.full,
                         id="full-bf16-tuned"),
        ]),
    ]


@DeltaNetBwdFixture
def test_deltanet_bwd(
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
    B, H, S, DK, DV, BC = batch, heads, seq_len, dim_k, dim_v, chunk_size
    # Device principle: generate random inputs on CPU (torch.manual_seed only
    # takes effect on CPU — seeding is a no-op on the ptpu device), run the
    # torch autograd reference on CPU, and move to PTU only for the tilelang
    # kernels. Compare on CPU.
    q_cpu = torch.randn(B, H, S, DK, dtype=dtype) * 0.1
    k_cpu = torch.randn(B, H, S, DK, dtype=dtype) * 0.1
    v_cpu = torch.randn(B, H, S, DV, dtype=dtype) * 0.1
    beta_cpu = torch.rand(B, H, S, dtype=dtype) * 0.5
    do_cpu = torch.randn(B, H, S, DV, dtype=dtype) * 0.1

    q, k, v, beta = q_cpu.ptpu(), k_cpu.ptpu(), v_cpu.ptpu(), beta_cpu.ptpu()
    do = do_cpu.ptpu()

    # Forward to get S for backward kernel (tilelang kernel on PTU)
    from tileops.ops import DeltaNetFwdOp
    fwd_op = DeltaNetFwdOp(chunk_size=BC)
    _o, S_fwd, Aw, Au, w_fwd, u_fwd = fwd_op.forward(q, k, v, beta)

    # Reference via autograd — on CPU
    ref_dq, ref_dk, ref_dv, ref_dbeta = deltanet_autograd_bwd_torch(
        do_cpu, q_cpu, k_cpu, v_cpu, beta_cpu, BC)
    ref_outputs = (ref_dq, ref_dk, ref_dv, ref_dbeta)

    # Kernel
    op = DeltaNetBwdOp(chunk_size=BC, tune=tune)
    op_outputs = op.forward(do, q, k, v, beta, S_fwd, Aw, Au, w_fwd, u_fwd)

    tols = _get_tolerances(dtype)
    names = ["dq", "dk", "dv", "dbeta"]
    for name, ref_out, op_out in zip(names, ref_outputs, op_outputs, strict=True):
        torch.testing.assert_close(
            op_out.cpu(), ref_out.to(dtype).cpu(), **tols,
            msg=lambda m, n=name: f"{n}: {m}",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
