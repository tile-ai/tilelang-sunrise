from typing import Optional

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.attention import (
    GQADecodeBs1Kernel,
    GQADecodeKernel,
    GQADecodeLongContextKernel,
    GQADenseFP8Kernel,
    GQADenseFwdKernel,
    GQADenseSlidingWindowKernel,
    GQADenseWsKernel,
)
from tileops.kernels.attention.gqa_decode import (
    _effective_dense_num_split,
    _gqa_decode_no_split_op,
    _gqa_decode_split_op,
)
from tileops.kernels.kernel_base import Kernel
from tileops.ops import (
    GroupedQueryAttentionBwdOp,
    GroupedQueryAttentionDenseFwdOp,
    GroupedQueryAttentionPrefillVarlenFwdOp,
)
from tileops.utils import get_sm_version
from workloads.attention.gqa import (
    GQAPrefillVarlenFwdWorkload,
    GroupedQueryAttentionBwdWorkload,
    GroupedQueryAttentionSquareFwdWorkload,
)


class GQAPrefillVarlenFwdTest(GQAPrefillVarlenFwdWorkload, TestBase):
    def ref_program(self, q, k, v, cu_seqlens_q, cu_seqlens_kv):
        """PyTorch reference: per-request ragged GQA prefill with bottom-right causal mask."""
        groups = self.heads // self.heads_kv
        head_dim = self.dim
        scale = head_dim**-0.5
        dtype = q.dtype
        outs = []
        for b in range(self.batch):
            q_b = q[cu_seqlens_q[b].item() : cu_seqlens_q[b + 1].item()].float()
            k_b = k[cu_seqlens_kv[b].item() : cu_seqlens_kv[b + 1].item()].float()
            v_b = v[cu_seqlens_kv[b].item() : cu_seqlens_kv[b + 1].item()].float()
            k_b = k_b.repeat_interleave(groups, dim=1)
            v_b = v_b.repeat_interleave(groups, dim=1)
            q_len, kv_len = q_b.shape[0], k_b.shape[0]
            scores = torch.einsum("qhd,khd->hqk", q_b, k_b) * scale
            if self.is_causal:
                offset = kv_len - q_len
                q_pos = torch.arange(q_len, device=q.device)[:, None] + offset
                k_pos = torch.arange(kv_len, device=q.device)[None, :]
                scores = scores.masked_fill(~(k_pos <= q_pos).unsqueeze(0), float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            outs.append(torch.einsum("hqk,khd->qhd", probs, v_b).to(dtype))
        return torch.cat(outs, dim=0).contiguous()


class GroupedQueryAttentionBwdTest(GroupedQueryAttentionBwdWorkload, TestBase):
    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        grad_output: torch.Tensor,
        lse: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_bhsd = q.transpose(1, 2)  # [B, H, S, D]
        k_bhsd = k.transpose(1, 2)
        v_bhsd = v.transpose(1, 2)
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            output_bhsd = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, is_causal=self.is_causal, enable_gqa=True
            )
        output = output_bhsd.transpose(1, 2).contiguous()

        output.backward(grad_output)
        return q.grad, k.grad, v.grad


def _gqa_prefill_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    heads: int,
    heads_kv: int,
    is_causal: bool,
    sm_scale: Optional[float] = None,
    softcap: Optional[float] = None,
    window_size_left: int = -1,
    window_size_right: int = -1,
) -> torch.Tensor:
    q, k, v = q.cpu(), k.cpu(), v.cpu()
    batch, seq_len_q, _, dim = q.shape
    seq_len_kv = k.shape[1]
    groups = heads // heads_kv
    device = q.device
    q_bhsd = q.detach().cpu().transpose(1, 2).float()
    k_bhsd = k.detach().cpu().repeat_interleave(groups, dim=2).transpose(1, 2).float()
    v_bhsd = v.detach().cpu().repeat_interleave(groups, dim=2).transpose(1, 2).float()
    scale = dim**-0.5 if sm_scale is None else sm_scale
    scores = torch.matmul(q_bhsd, k_bhsd.transpose(-2, -1)) * scale
    if softcap is not None and softcap > 0:
        scores = softcap * torch.tanh(scores / softcap)
    offset = seq_len_kv - seq_len_q
    q_pos = torch.arange(seq_len_q, device=q.device)[:, None] + offset
    k_pos = torch.arange(seq_len_kv, device=q.device)[None, :]
    mask = torch.ones((seq_len_q, seq_len_kv), device=q.device, dtype=torch.bool)
    if is_causal:
        mask &= k_pos <= q_pos
    if window_size_left >= 0:
        mask &= k_pos >= q_pos - window_size_left
    if window_size_right >= 0:
        mask &= k_pos <= q_pos + window_size_right
    if is_causal or window_size_left >= 0 or window_size_right >= 0:
        scores = scores.masked_fill(~mask.view(1, 1, seq_len_q, seq_len_kv), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    output = torch.matmul(probs, v_bhsd)
    assert output.shape == (batch, heads, seq_len_q, dim)
    return output.transpose(1, 2).to(device=device, dtype=q.dtype).contiguous()


def _apply_dense_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    rotary_dim: int,
    layout: str,
) -> torch.Tensor:
    x, positions, cos, sin = x.cpu(), positions.cpu(), cos.cpu(), sin.cpu()
    half = rotary_dim // 2
    x_rot = x[..., :rotary_dim].float()
    c = cos[positions].view(1, x.shape[1], 1, half).float()
    s = sin[positions].view(1, x.shape[1], 1, half).float()
    if layout == "neox":
        x0, x1 = x_rot[..., :half], x_rot[..., half:]
    else:
        x0, x1 = x_rot[..., 0::2], x_rot[..., 1::2]
    y0, y1 = x0 * c - x1 * s, x1 * c + x0 * s
    rotated = (
        torch.cat((y0, y1), dim=-1)
        if layout == "neox"
        else torch.stack((y0, y1), dim=-1).flatten(-2)
    )
    return torch.cat((rotated.to(x.dtype), x[..., rotary_dim:]), dim=-1).contiguous()


@pytest.mark.parametrize(
    "is_causal, rope_layout, rotary_dim, dtype",
    [
        (True, None, None, torch.float16),
        (True, "neox", 64, torch.float16),
        (True, "interleaved", 64, torch.float16),
        (True, "neox", None, torch.bfloat16),
        (False, None, None, torch.float16),
        (False, "neox", 64, torch.float16),
    ],
)
@pytest.mark.smoke
def test_gqa_dense_sm90_main_kernel_matches_reference(
    is_causal: bool,
    rope_layout: Optional[str],
    rotary_dim: Optional[int],
    dtype: torch.dtype,
) -> None:
    if not torch.ptpu.is_available() and (not torch.cuda.is_available() or get_sm_version() != 90):
        pytest.skip("Dense warp-specialized prefill requires SM90")
    batch, seq_len_q, seq_len_kv, heads, heads_kv, dim = 1, 160, 270, 8, 2, 128
    q = torch.randn(batch, seq_len_q, heads, dim, device="ptpu", dtype=dtype)
    k = torch.randn(batch, seq_len_kv, heads_kv, dim, device="ptpu", dtype=dtype)
    v = torch.randn_like(k)

    if rope_layout is None:
        op = GroupedQueryAttentionDenseFwdOp(is_causal=is_causal)
        output = op(q, k, v)
        q_ref, k_ref = q, k
    else:
        resolved_rotary_dim = dim if rotary_dim is None else rotary_dim
        angles = torch.randn(seq_len_kv, resolved_rotary_dim // 2) * 0.1
        rope_cos, rope_sin = (
            angles.cos().to(device="ptpu", dtype=dtype),
            angles.sin().to(device="ptpu", dtype=dtype),
        )
        op = GroupedQueryAttentionDenseFwdOp(
            is_causal=is_causal,
            pos_encoding_mode="rope",
            rotary_dim=rotary_dim,
            rope_layout=rope_layout,
        )
        output = op(q, k, v, rope_cos=rope_cos, rope_sin=rope_sin)
        q_positions = torch.arange(seq_len_kv - seq_len_q, seq_len_kv, device="ptpu")
        k_positions = torch.arange(seq_len_kv, device="ptpu")
        q_ref = _apply_dense_rope(
            q,
            q_positions,
            rope_cos,
            rope_sin,
            rotary_dim=resolved_rotary_dim,
            layout=rope_layout,
        )
        k_ref = _apply_dense_rope(
            k,
            k_positions,
            rope_cos,
            rope_sin,
            rotary_dim=resolved_rotary_dim,
            layout=rope_layout,
        )

    torch.testing.assert_close(
        (output).cpu(),
        (
            _gqa_prefill_ref(
                q_ref,
                k_ref,
                v,
                heads=heads,
                heads_kv=heads_kv,
                is_causal=is_causal,
            )
        ).cpu(),
        atol=2e-2 if dtype == torch.bfloat16 else 5e-3,
        rtol=1e-5,
    )
    assert isinstance(
        next(iter(op.iter_kernels())),
        GQADenseFwdKernel if torch.ptpu.is_available() else GQADenseWsKernel,
    )


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("seq_len_q", "seq_len_kv", "sm_scale", "softcap"),
    [(256, 1792, 0.125, 0.0), (255, 1793, 0.0625, 50.0)],
)
def test_gqa_dense_fp8_causal_rectangular_matches_reference(
    seq_len_q: int, seq_len_kv: int, sm_scale: float, softcap: float
) -> None:
    fp8 = getattr(torch, "float8_e4m3fn", None)
    if fp8 is None or not torch.cuda.is_available() or get_sm_version() != 90:
        pytest.skip("native FP8 Dense GQA requires SM90 and float8_e4m3fn")
    batch = 1
    heads, heads_kv, dim = 8, 2, 128
    q = (torch.randn(batch, seq_len_q, heads, dim, device="ptpu") * 0.2).to(fp8)
    k = (torch.randn(batch, seq_len_kv, heads_kv, dim, device="ptpu") * 0.2).to(fp8)
    v = (torch.randn(batch, seq_len_kv, heads_kv, dim, device="ptpu") * 0.2).to(fp8)
    scale = torch.ones((batch, heads_kv), device="ptpu", dtype=torch.float32)

    op = GroupedQueryAttentionDenseFwdOp(
        is_causal=True,
        dtype=torch.float16,
        sm_scale=sm_scale,
        softcap=softcap,
    )
    output = op(q, k, v, scale, scale, scale)
    reference = _gqa_prefill_ref(
        q.to(torch.float16),
        k.to(torch.float16),
        v.to(torch.float16),
        heads=heads,
        heads_kv=heads_kv,
        is_causal=True,
        sm_scale=sm_scale,
        softcap=softcap,
    )
    torch.testing.assert_close((output).cpu(), (reference).cpu(), atol=8e-2, rtol=2e-2)
    assert isinstance(next(iter(op.iter_kernels())), GQADenseFP8Kernel)


@pytest.mark.smoke
@pytest.mark.parametrize("batch", [1, 2])
def test_gqa_dense_reuses_cached_sequence_specializations(batch: int) -> None:
    if not torch.ptpu.is_available() and (not torch.cuda.is_available() or get_sm_version() != 90):
        pytest.skip("Dense warp-specialized prefill requires SM90")
    heads, heads_kv, dim = 8, 2, 128
    op = GroupedQueryAttentionDenseFwdOp()

    for seq_len_q, seq_len_kv in (
        (2, 270),
        (2, 271),
        (160, 270),
        (896, 896),
        (2, 270),
    ):
        q = torch.randn(batch, seq_len_q, heads, dim, device="ptpu", dtype=torch.float16)
        k = torch.randn(batch, seq_len_kv, heads_kv, dim, device="ptpu", dtype=torch.float16)
        v = torch.randn_like(k)
        output = op(q, k, v)
        torch.testing.assert_close(
            (output).cpu(),
            (_gqa_prefill_ref(q, k, v, heads=heads, heads_kv=heads_kv, is_causal=True)).cpu(),
            atol=5e-3,
            rtol=1e-5,
        )

    assert len(list(op.iter_kernels())) == (4 if torch.ptpu.is_available() else 1)


@pytest.mark.parametrize(
    "batch, head_shape, dtype, seq_lens_kv, rope_layout, rotary_dim, kernel_type",
    [
        pytest.param(
            1,
            (8, 2),
            torch.float16,
            (257, 1057),
            None,
            None,
            GQADecodeBs1Kernel,
            id="bs1-fp16-other-head-ratio",
        ),
        pytest.param(
            1,
            (32, 4),
            torch.float16,
            (1024, 1057),
            None,
            None,
            GQADecodeLongContextKernel,
            id="measured-bs1-long-generic",
        ),
        pytest.param(
            2,
            (8, 2),
            torch.bfloat16,
            (257, 511),
            None,
            None,
            GQADecodeKernel,
            id="batched-bf16",
        ),
        pytest.param(
            1,
            (8, 2),
            torch.float16,
            (271,),
            "neox",
            64,
            GQADecodeBs1Kernel,
            id="bs1-fp16-neox-partial",
        ),
        pytest.param(
            1,
            (8, 2),
            torch.float16,
            (1057,),
            "neox",
            128,
            GQADecodeBs1Kernel,
            id="bs1-fp16-neox-full-ctx-tail",
        ),
        pytest.param(
            2,
            (8, 2),
            torch.bfloat16,
            (257,),
            "interleaved",
            128,
            GQADecodeKernel,
            id="batched-bf16-interleaved-full",
        ),
    ],
)
@pytest.mark.smoke
def test_gqa_dense_decode_dispatch_and_dynamic_sequence_lengths(
    batch: int,
    head_shape: tuple[int, int],
    dtype: torch.dtype,
    seq_lens_kv: tuple[int, ...],
    rope_layout: Optional[str],
    rotary_dim: Optional[int],
    kernel_type: type[Kernel],
) -> None:
    if not torch.ptpu.is_available() and (not torch.cuda.is_available() or get_sm_version() != 90):
        pytest.skip("Dense decode requires SM90")
    heads, heads_kv = head_shape
    dim = 128
    op = GroupedQueryAttentionDenseFwdOp(
        pos_encoding_mode="rope" if rope_layout is not None else "none",
        rotary_dim=rotary_dim,
        rope_layout="neox" if rope_layout is None else rope_layout,
    )

    for seq_len_kv in seq_lens_kv:
        q = torch.randn(batch, 1, heads, dim, device="ptpu", dtype=dtype)
        k = torch.randn(batch, seq_len_kv, heads_kv, dim, device="ptpu", dtype=dtype)
        v = torch.randn_like(k)
        if rope_layout is None:
            rope_cos = rope_sin = None
            q_ref, k_ref = q, k
        else:
            assert rotary_dim is not None
            angles = torch.randn(seq_len_kv, rotary_dim // 2) * 0.1
            rope_cos, rope_sin = (
                angles.cos().to(device="ptpu", dtype=dtype),
                angles.sin().to(device="ptpu", dtype=dtype),
            )
            q_ref = _apply_dense_rope(
                q,
                torch.tensor([seq_len_kv - 1], device="ptpu"),
                rope_cos,
                rope_sin,
                rotary_dim=rotary_dim,
                layout=rope_layout,
            )
            k_ref = _apply_dense_rope(
                k,
                torch.arange(seq_len_kv, device="ptpu"),
                rope_cos,
                rope_sin,
                rotary_dim=rotary_dim,
                layout=rope_layout,
            )
        output = op(q, k, v, rope_cos=rope_cos, rope_sin=rope_sin)
        torch.testing.assert_close(
            (output).cpu(),
            (
                _gqa_prefill_ref(q_ref, k_ref, v, heads=heads, heads_kv=heads_kv, is_causal=True)
            ).cpu(),
            atol=1.6e-2 if dtype == torch.bfloat16 else 5e-3,
            rtol=1.6e-2 if dtype == torch.bfloat16 else 1e-5,
        )

    kernels = list(op.iter_kernels())
    assert len(kernels) == 1
    expected_type = kernel_type
    if torch.ptpu.is_available():
        expected_type = GQADenseFwdKernel if rope_layout is not None else GQADecodeKernel
    assert isinstance(kernels[0], expected_type)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "num_split, block_N, real_seqlen_kv, expected",
    [
        pytest.param(32, 64, 1024, 16, id="issue-shape-clamps-to-tiles"),
        pytest.param(32, 64, 1000, 8, id="non-candidate-count-rounded-down"),
        pytest.param(7, 64, 100000, 4, id="non-candidate-ceiling-rounded-down"),
        pytest.param(16, 128, 1024, 8, id="block-128-clamps"),
        pytest.param(32, 64, 100, 1, id="short-sequence-no-split"),
        pytest.param(1, 64, 100000, 1, id="tuned-no-split-stays-no-split"),
    ],
)
def test_gqa_dense_decode_effective_num_split(
    num_split: int, block_N: int, real_seqlen_kv: int, expected: int
) -> None:
    """The tuned num_split is a ceiling shrunk to the runtime KV extent."""
    assert _effective_dense_num_split(num_split, block_N, real_seqlen_kv) == expected


@pytest.mark.smoke
@pytest.mark.parametrize("seqlen_kv", [1, 63, 128, 1024])
def test_gqa_decode_autotune_configs_keep_full_tiles_per_split(seqlen_kv: int) -> None:
    """Every swept config leaves each split one full KV tile; num_split=1 stays comparable."""
    if not torch.ptpu.is_available() and (
        not torch.cuda.is_available() or get_sm_version() not in (80, 89, 90)
    ):
        pytest.skip("GQA decode requires SM80/89/90")
    kernel = GQADecodeKernel(2, 8, 2, seqlen_kv, 128, dtype=torch.float16)
    configs = kernel.autotune_configs
    assert configs, "the sweep must stay non-empty for any positive sequence length"
    for config in configs:
        assert config["num_split"] <= max(1, seqlen_kv // config["block_N"])
    assert any(config["num_split"] == 1 for config in configs)


@pytest.mark.smoke
def test_gqa_decode_tuned_split_count_tracks_runtime_sequence(monkeypatch) -> None:
    """A tuned num_split the sequence cannot fill shrinks instead of pushing
    dispatch into the never-tuned no-split kernel (the reported issue)."""
    if not torch.ptpu.is_available() and (
        not torch.cuda.is_available() or get_sm_version() not in (80, 89, 90)
    ):
        pytest.skip("GQA decode requires SM80/89/90")
    batch, heads, heads_kv, dim = 2, 32, 4, 128
    kernel = GQADecodeKernel(
        batch,
        heads,
        heads_kv,
        1024,
        dim,
        dtype=torch.float16,
        config={"block_H": 64, "block_N": 64, "num_split": 32, "num_stages": 2, "threads": 128},
    )

    calls: list[tuple[str, int]] = []

    def split_spy(*args, **kwargs):
        # num_split is the 12th positional argument of _gqa_decode_split_op
        calls.append(("split", args[11]))
        return _gqa_decode_split_op(*args, **kwargs)

    def no_split_spy(*args, **kwargs):
        calls.append(("no_split", 0))
        return _gqa_decode_no_split_op(*args, **kwargs)

    monkeypatch.setattr("tileops.kernels.attention.gqa_decode._gqa_decode_split_op", split_spy)
    monkeypatch.setattr(
        "tileops.kernels.attention.gqa_decode._gqa_decode_no_split_op", no_split_spy
    )

    # 1024 tokens fill 16 of the tuned 32 splits; 100 cannot fill two
    for seq_len_kv, expected in ((1024, ("split", 16)), (100, ("no_split", 0))):
        q = torch.randn(batch, 1, heads, dim, device="ptpu", dtype=torch.float16)
        k = torch.randn(batch, seq_len_kv, heads_kv, dim, device="ptpu", dtype=torch.float16)
        v = torch.randn_like(k)
        output = kernel(q, k, v)
        torch.testing.assert_close(
            (output).cpu(),
            (_gqa_prefill_ref(q, k, v, heads=heads, heads_kv=heads_kv, is_causal=True)).cpu(),
            atol=5e-3,
            rtol=1e-5,
        )
        assert calls[-1] == expected


@pytest.mark.parametrize(
    "is_causal,use_rope",
    [
        pytest.param(True, False, id="causal"),
        pytest.param(True, True, id="causal-rope"),
        pytest.param(False, False, id="noncausal"),
        pytest.param(False, True, id="noncausal-rope"),
    ],
)
@pytest.mark.smoke
def test_gqa_dense_sm90_sliding_window_kernel_matches_reference(
    is_causal: bool, use_rope: bool
) -> None:
    if not torch.ptpu.is_available() and (not torch.cuda.is_available() or get_sm_version() != 90):
        pytest.skip("Dense sliding-window prefill requires SM90")
    batch, seq_len, heads, heads_kv, dim = 1, 270, 8, 2, 128
    window_size_left = 64
    window_size_right = 0 if is_causal else 32
    sm_scale = 0.125
    softcap = 2.0
    q = torch.randn(batch, seq_len, heads, dim, device="ptpu", dtype=torch.float16)
    k = torch.randn(batch, seq_len, heads_kv, dim, device="ptpu", dtype=torch.float16)
    v = torch.randn_like(k)

    rotary_dim = 64
    if use_rope:
        angles = torch.randn(seq_len, rotary_dim // 2) * 0.1
        rope_cos, rope_sin = (
            angles.cos().to(device="ptpu", dtype=q.dtype),
            angles.sin().to(device="ptpu", dtype=q.dtype),
        )
    else:
        rope_cos = rope_sin = None
    op = GroupedQueryAttentionDenseFwdOp(
        is_causal=is_causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        sm_scale=sm_scale,
        softcap=softcap,
        pos_encoding_mode="rope" if use_rope else "none",
        rotary_dim=rotary_dim if use_rope else None,
    )
    output = op(
        q,
        k,
        v,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
    )
    if use_rope:
        assert rope_cos is not None and rope_sin is not None
        positions = torch.arange(seq_len, device="ptpu")
        q = _apply_dense_rope(
            q, positions, rope_cos, rope_sin, rotary_dim=rotary_dim, layout="neox"
        )
        k = _apply_dense_rope(
            k, positions, rope_cos, rope_sin, rotary_dim=rotary_dim, layout="neox"
        )

    torch.testing.assert_close(
        (output).cpu(),
        (
            _gqa_prefill_ref(
                q,
                k,
                v,
                heads=heads,
                heads_kv=heads_kv,
                is_causal=is_causal,
                sm_scale=sm_scale,
                softcap=softcap,
                window_size_left=window_size_left,
                window_size_right=window_size_right,
            )
        ).cpu(),
        atol=5e-3,
        rtol=1e-5,
    )
    assert isinstance(next(iter(op.iter_kernels())), GQADenseSlidingWindowKernel)


class GroupedQueryAttentionBwdFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq_len, heads, heads_kv, dim, causal, dtype, tune",
            [
                pytest.param(
                    1, 1024, 8, 4, 64, False, torch.float16, False, marks=pytest.mark.smoke
                ),
                pytest.param(
                    1, 1024, 8, 4, 64, False, torch.bfloat16, False, marks=pytest.mark.smoke
                ),
                pytest.param(
                    4, 2048, 64, 4, 128, False, torch.float16, False, marks=pytest.mark.full
                ),
                pytest.param(
                    4, 2048, 64, 4, 128, False, torch.bfloat16, False, marks=pytest.mark.full
                ),
            ],
        ),
    ]


@pytest.mark.smoke
def test_gqa_prefill_varlen_rejects_bad_contract_inputs() -> None:
    q_lens, kv_lens = [64, 32], [128, 96]
    heads, heads_kv, dim = 8, 2, 64
    q = torch.randn(sum(q_lens), heads, dim, device="ptpu", dtype=torch.float16).contiguous()
    k = torch.randn(sum(kv_lens), heads_kv, dim, device="ptpu", dtype=torch.float16).contiguous()
    v = torch.randn_like(k)
    cu_q = torch.tensor(
        [0] + torch.tensor(q_lens).cumsum(0).tolist(), device="ptpu", dtype=torch.int32
    )
    cu_kv = torch.tensor(
        [0] + torch.tensor(kv_lens).cumsum(0).tolist(), device="ptpu", dtype=torch.int32
    )

    op = GroupedQueryAttentionPrefillVarlenFwdOp(
        max(q_lens), max(kv_lens), True, validate_inputs=True
    )
    with pytest.raises(ValueError, match="Expected k shape"):
        op(q, k[:, :, :-1].contiguous(), v, cu_q, cu_kv)
    with pytest.raises(ValueError, match="cu_seqlens_q\\[-1\\].*must equal"):
        op(q[:-1], k, v, cu_q, cu_kv)
    with pytest.raises(ValueError, match="max_seqlen_q"):
        bad_op = GroupedQueryAttentionPrefillVarlenFwdOp(
            max(q_lens) - 1, max(kv_lens), True, validate_inputs=True
        )
        bad_op(q, k, v, cu_q, cu_kv)
    bad_cu = torch.tensor([0, 128, 96], device="ptpu", dtype=torch.int32)
    with pytest.raises(ValueError, match="cu_seqlens_q must be non-decreasing"):
        op(q, k, v, bad_cu, cu_kv)


@pytest.mark.smoke
def test_gqa_prefill_varlen_rejects_unsupported_dtype() -> None:
    """The element type now arrives with the tensors, so the rejection does too."""
    op = GroupedQueryAttentionPrefillVarlenFwdOp(max_seqlen_q=64, max_seqlen_kv=128)
    kwargs = {"dtype": torch.float32, "device": "ptpu"}
    q = torch.randn(64, 8, 64, **kwargs)
    k = torch.randn(128, 2, 64, **kwargs)
    v = torch.randn(128, 2, 64, **kwargs)
    cu_q = torch.tensor([0, 64], device="ptpu", dtype=torch.int32)
    cu_kv = torch.tensor([0, 128], device="ptpu", dtype=torch.int32)
    with pytest.raises(ValueError, match="Expected dtype torch.float16 or torch.bfloat16"):
        op(q, k, v, cu_q, cu_kv)


@GroupedQueryAttentionBwdFixture
def test_gqa_bwd(
    batch: int,
    seq_len: int,
    heads: int,
    heads_kv: int,
    dim: int,
    causal: bool,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = GroupedQueryAttentionBwdTest(batch, heads, heads_kv, seq_len, dim, causal, dtype)
    op = GroupedQueryAttentionBwdOp(batch, heads, heads_kv, seq_len, dim, causal, tune=tune)
    test.check(op, *test.gen_inputs(), atol=5e-3, rtol=1e-5)


_PREFILL_TOLERANCE = {torch.float16: (5e-3, 1e-5), torch.bfloat16: (8e-2, 1e-2)}


class SquareAttentionFwdTest(GroupedQueryAttentionSquareFwdWorkload, TestBase):
    def ref_program(self, q, k, v):
        return _gqa_prefill_ref(
            q,
            k,
            v,
            heads=self.heads,
            heads_kv=self.heads_kv,
            is_causal=self.is_causal,
        ).to(q.device)


class GroupedQueryAttentionFwdFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq_len, heads, heads_kv, dim, causal, dtype, tune",
            [
                pytest.param(
                    1, 1024, 8, 4, 64, False, torch.float16, False, marks=pytest.mark.smoke
                ),
                pytest.param(
                    1, 1024, 8, 4, 64, False, torch.bfloat16, False, marks=pytest.mark.smoke
                ),
                pytest.param(
                    4, 512, 64, 4, 128, False, torch.float16, False, marks=pytest.mark.smoke
                ),
                pytest.param(
                    4, 512, 64, 4, 128, True, torch.float16, False, marks=pytest.mark.smoke
                ),
                pytest.param(
                    4, 2048, 64, 4, 128, False, torch.float16, False, marks=pytest.mark.full
                ),
                pytest.param(
                    4, 2048, 64, 4, 128, False, torch.bfloat16, False, marks=pytest.mark.full
                ),
            ],
        ),
    ]


@GroupedQueryAttentionFwdFixture
def test_gqa_fwd(batch, seq_len, heads, heads_kv, dim, causal, dtype, tune):
    test = SquareAttentionFwdTest(batch, heads, heads_kv, seq_len, dim, causal, dtype)
    op = GroupedQueryAttentionDenseFwdOp(is_causal=causal)
    test.check(op, *test.gen_inputs(), atol=5e-3, rtol=1e-5)


def _gqa_prefill_varlen_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    *,
    batch: int,
    heads: int,
    heads_kv: int,
    is_causal: bool,
    sm_scale: Optional[float] = None,
    softcap: Optional[float] = None,
) -> torch.Tensor:
    groups = heads // heads_kv
    dim = q.shape[-1]
    device = q.device
    q = q.detach().cpu()
    k = k.detach().cpu()
    v = v.detach().cpu()
    cu_seqlens_q = cu_seqlens_q.detach().cpu()
    cu_seqlens_kv = cu_seqlens_kv.detach().cpu()
    scale = dim**-0.5 if sm_scale is None else sm_scale
    outputs = []
    for b in range(batch):
        q_start = int(cu_seqlens_q[b].item())
        q_end = int(cu_seqlens_q[b + 1].item())
        kv_start = int(cu_seqlens_kv[b].item())
        kv_end = int(cu_seqlens_kv[b + 1].item())
        q_bhsd = q[q_start:q_end].transpose(0, 1).float()
        k_i = k[kv_start:kv_end].repeat_interleave(groups, dim=1).permute(1, 0, 2).float()
        v_i = v[kv_start:kv_end].repeat_interleave(groups, dim=1).permute(1, 0, 2).float()
        q_len = q_end - q_start
        kv_len = kv_end - kv_start
        scores = torch.matmul(q_bhsd, k_i.transpose(-2, -1)) * scale
        if softcap is not None and softcap > 0:
            scores = softcap * torch.tanh(scores / softcap)
        if is_causal:
            offset = kv_len - q_len
            q_pos = torch.arange(q_len)[:, None] + offset
            kv_pos = torch.arange(kv_len)[None, :]
            mask = kv_pos <= q_pos
            scores = scores.masked_fill(~mask.view(1, q_len, kv_len), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.matmul(probs, v_i).transpose(0, 1).to(q.dtype).contiguous())
    return torch.cat(outputs, dim=0).to(device)


@pytest.mark.parametrize(
    "batch, seq_len_q, seq_len_kv, heads, heads_kv, dim, causal, dtype",
    [
        pytest.param(
            1,
            128,
            128,
            8,
            2,
            64,
            True,
            torch.float16,
            marks=pytest.mark.smoke,
            id="gqa_ratio4_q_eq_kv_fp16",
        ),
        pytest.param(
            1,
            128,
            384,
            8,
            2,
            64,
            True,
            torch.float16,
            marks=pytest.mark.smoke,
            id="gqa_ratio4_q_lt_kv_fp16",
        ),
        pytest.param(
            1,
            96,
            150,
            8,
            2,
            64,
            True,
            torch.float16,
            marks=pytest.mark.smoke,
            id="gqa_tail_q_and_kv_fp16",
        ),
        pytest.param(
            2,
            128,
            256,
            16,
            4,
            128,
            True,
            torch.float16,
            marks=pytest.mark.smoke,
            id="gqa_ratio4_batch2_dim128_fp16",
        ),
        pytest.param(
            1,
            128,
            256,
            8,
            2,
            64,
            False,
            torch.float16,
            marks=pytest.mark.smoke,
            id="gqa_noncausal_fp16",
        ),
        pytest.param(
            1,
            128,
            256,
            8,
            2,
            64,
            True,
            torch.bfloat16,
            marks=pytest.mark.smoke,
            id="gqa_ratio4_bf16",
        ),
    ],
)
def test_gqa_prefill_fwd(
    batch: int,
    seq_len_q: int,
    seq_len_kv: int,
    heads: int,
    heads_kv: int,
    dim: int,
    causal: bool,
    dtype: torch.dtype,
) -> None:
    q = torch.randn(batch, seq_len_q, heads, dim, device="ptpu", dtype=dtype).contiguous()
    k = torch.randn(batch, seq_len_kv, heads_kv, dim, device="ptpu", dtype=dtype).contiguous()
    v = torch.randn(batch, seq_len_kv, heads_kv, dim, device="ptpu", dtype=dtype).contiguous()
    ref = _gqa_prefill_ref(q, k, v, heads=heads, heads_kv=heads_kv, is_causal=causal)
    op = GroupedQueryAttentionDenseFwdOp(is_causal=causal)
    output = op(q, k, v).view(batch, seq_len_q, heads, dim)
    (atol, rtol) = _PREFILL_TOLERANCE[dtype]
    torch.testing.assert_close(output.cpu(), ref.cpu(), atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_gqa_prefill_fwd_dense_backend_matches_reference() -> None:
    (batch, seq_len_q, seq_len_kv, heads, heads_kv, dim) = (1, 128, 256, 8, 2, 64)
    q = torch.randn(batch, seq_len_q, heads, dim, device="ptpu", dtype=torch.float16).contiguous()
    k = torch.randn(
        batch, seq_len_kv, heads_kv, dim, device="ptpu", dtype=torch.float16
    ).contiguous()
    v = torch.randn(
        batch, seq_len_kv, heads_kv, dim, device="ptpu", dtype=torch.float16
    ).contiguous()
    ref = _gqa_prefill_ref(q, k, v, heads=heads, heads_kv=heads_kv, is_causal=True)
    op = GroupedQueryAttentionDenseFwdOp(is_causal=True)
    output = op(q, k, v).view(batch, seq_len_q, heads, dim)
    torch.testing.assert_close(output.cpu(), ref.cpu(), atol=0.005, rtol=1e-05)


@pytest.mark.smoke
def test_gqa_prefill_fwd_uses_bottom_right_causal_mask() -> None:
    (batch, seq_len_q, seq_len_kv, heads, heads_kv, dim) = (1, 128, 256, 4, 2, 64)
    q = torch.zeros(batch, seq_len_q, heads, dim, device="ptpu", dtype=torch.float16)
    k = torch.zeros(batch, seq_len_kv, heads_kv, dim, device="ptpu", dtype=torch.float16)
    v = torch.zeros(batch, seq_len_kv, heads_kv, dim, device="ptpu", dtype=torch.float16)
    q[..., 0] = 1
    k[..., 0] = 1
    v[:, :128, :, 0] = 1
    v[:, 128:, :, 0] = 100
    op = GroupedQueryAttentionDenseFwdOp(is_causal=True)
    output = op(q, k, v).view(batch, seq_len_q, heads, dim)
    assert output[0, 0, 0, 0] < 2
    assert output[0, -1, 0, 0] > 40


@pytest.mark.smoke
def test_gqa_prefill_fwd_respects_sm_scale() -> None:
    (batch, seq_len_q, seq_len_kv, heads, heads_kv, dim) = (1, 128, 256, 8, 2, 64)
    sm_scale = 0.125
    q = torch.randn(batch, seq_len_q, heads, dim, device="ptpu", dtype=torch.float16).contiguous()
    k = torch.randn(
        batch, seq_len_kv, heads_kv, dim, device="ptpu", dtype=torch.float16
    ).contiguous()
    v = torch.randn(
        batch, seq_len_kv, heads_kv, dim, device="ptpu", dtype=torch.float16
    ).contiguous()
    ref = _gqa_prefill_ref(
        q, k, v, heads=heads, heads_kv=heads_kv, is_causal=True, sm_scale=sm_scale
    )
    op = GroupedQueryAttentionDenseFwdOp(is_causal=True, sm_scale=sm_scale)
    output = op(q, k, v).view(batch, seq_len_q, heads, dim)
    torch.testing.assert_close(output.cpu(), ref.cpu(), atol=0.005, rtol=1e-05)


@pytest.mark.smoke
def test_gqa_prefill_fwd_respects_softcap() -> None:
    (batch, seq_len_q, seq_len_kv, heads, heads_kv, dim) = (1, 128, 256, 8, 2, 64)
    softcap = 2.0
    q = torch.randn(batch, seq_len_q, heads, dim, device="ptpu", dtype=torch.float16).contiguous()
    k = torch.randn(
        batch, seq_len_kv, heads_kv, dim, device="ptpu", dtype=torch.float16
    ).contiguous()
    v = torch.randn(
        batch, seq_len_kv, heads_kv, dim, device="ptpu", dtype=torch.float16
    ).contiguous()
    ref = _gqa_prefill_ref(q, k, v, heads=heads, heads_kv=heads_kv, is_causal=True, softcap=softcap)
    op = GroupedQueryAttentionDenseFwdOp(is_causal=True, softcap=softcap)
    output = op(q, k, v).view(batch, seq_len_q, heads, dim)
    torch.testing.assert_close(output.cpu(), ref.cpu(), atol=0.005, rtol=1e-05)


@pytest.mark.parametrize(
    "q_lens, kv_lens, heads, heads_kv, dim, causal, dtype",
    [
        pytest.param(
            [64, 128],
            [64, 128],
            8,
            2,
            64,
            True,
            torch.float16,
            marks=pytest.mark.smoke,
            id="uniform-causal-fp16",
        ),
        pytest.param(
            [33, 96, 129],
            [64, 128, 256],
            8,
            2,
            64,
            True,
            torch.float16,
            marks=pytest.mark.smoke,
            id="mixed-tail-causal-fp16",
        ),
        pytest.param(
            [64, 96],
            [128, 160],
            8,
            2,
            64,
            False,
            torch.float16,
            marks=pytest.mark.smoke,
            id="mixed-noncausal-fp16",
        ),
        pytest.param(
            [33, 65],
            [33, 65],
            8,
            2,
            64,
            False,
            torch.float16,
            marks=pytest.mark.smoke,
            id="equal-tail-noncausal-fp16",
        ),
        pytest.param(
            [64, 96],
            [128, 160],
            8,
            1,
            64,
            True,
            torch.float16,
            marks=pytest.mark.smoke,
            id="mqa-causal-fp16",
        ),
        pytest.param(
            [96],
            [160],
            8,
            2,
            64,
            True,
            torch.float16,
            marks=pytest.mark.smoke,
            id="batch1-causal-fp16",
        ),
        pytest.param(
            [64, 128],
            [128, 256],
            8,
            2,
            64,
            True,
            torch.bfloat16,
            marks=pytest.mark.smoke,
            id="mixed-causal-bf16",
        ),
    ],
)
def test_gqa_prefill_varlen_fwd(
    q_lens: list[int],
    kv_lens: list[int],
    heads: int,
    heads_kv: int,
    dim: int,
    causal: bool,
    dtype: torch.dtype,
) -> None:
    batch = len(q_lens)
    total_q = sum(q_lens)
    total_kv = sum(kv_lens)
    q = torch.randn(total_q, heads, dim, device="ptpu", dtype=dtype).contiguous()
    k = torch.randn(total_kv, heads_kv, dim, device="ptpu", dtype=dtype).contiguous()
    v = torch.randn(total_kv, heads_kv, dim, device="ptpu", dtype=dtype).contiguous()
    cu_q = torch.tensor(
        [0] + torch.tensor(q_lens).cumsum(0).tolist(), device="ptpu", dtype=torch.int32
    )
    cu_kv = torch.tensor(
        [0] + torch.tensor(kv_lens).cumsum(0).tolist(), device="ptpu", dtype=torch.int32
    )
    ref = _gqa_prefill_varlen_ref(
        q, k, v, cu_q, cu_kv, batch=batch, heads=heads, heads_kv=heads_kv, is_causal=causal
    )
    op = GroupedQueryAttentionPrefillVarlenFwdOp(
        max_seqlen_q=max(q_lens), max_seqlen_kv=max(kv_lens), is_causal=causal
    )
    output = op(q, k, v, cu_q, cu_kv)
    (atol, rtol) = _PREFILL_TOLERANCE[dtype]
    torch.testing.assert_close(output.cpu(), ref.cpu(), atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_gqa_prefill_varlen_respects_sm_scale() -> None:
    (q_lens, kv_lens) = ([64, 96], [128, 160])
    (batch, heads, heads_kv, dim) = (len(q_lens), 8, 2, 64)
    sm_scale = 0.125
    q = torch.randn(sum(q_lens), heads, dim, device="ptpu", dtype=torch.float16).contiguous()
    k = torch.randn(sum(kv_lens), heads_kv, dim, device="ptpu", dtype=torch.float16).contiguous()
    v = torch.randn_like(k)
    cu_q = torch.tensor(
        [0] + torch.tensor(q_lens).cumsum(0).tolist(), device="ptpu", dtype=torch.int32
    )
    cu_kv = torch.tensor(
        [0] + torch.tensor(kv_lens).cumsum(0).tolist(), device="ptpu", dtype=torch.int32
    )
    ref = _gqa_prefill_varlen_ref(
        q,
        k,
        v,
        cu_q,
        cu_kv,
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        is_causal=True,
        sm_scale=sm_scale,
    )
    op = GroupedQueryAttentionPrefillVarlenFwdOp(
        max_seqlen_q=max(q_lens), max_seqlen_kv=max(kv_lens), is_causal=True, sm_scale=sm_scale
    )
    output = op(q, k, v, cu_q, cu_kv)
    torch.testing.assert_close(output.cpu(), ref.cpu(), atol=0.005, rtol=1e-05)


@pytest.mark.smoke
def test_gqa_prefill_varlen_respects_softcap() -> None:
    (q_lens, kv_lens) = ([64, 96], [128, 160])
    (batch, heads, heads_kv, dim) = (len(q_lens), 8, 2, 64)
    softcap = 2.0
    q = torch.randn(sum(q_lens), heads, dim, device="ptpu", dtype=torch.float16).contiguous()
    k = torch.randn(sum(kv_lens), heads_kv, dim, device="ptpu", dtype=torch.float16).contiguous()
    v = torch.randn_like(k)
    cu_q = torch.tensor(
        [0] + torch.tensor(q_lens).cumsum(0).tolist(), device="ptpu", dtype=torch.int32
    )
    cu_kv = torch.tensor(
        [0] + torch.tensor(kv_lens).cumsum(0).tolist(), device="ptpu", dtype=torch.int32
    )
    ref = _gqa_prefill_varlen_ref(
        q,
        k,
        v,
        cu_q,
        cu_kv,
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        is_causal=True,
        softcap=softcap,
    )
    op = GroupedQueryAttentionPrefillVarlenFwdOp(
        max_seqlen_q=max(q_lens), max_seqlen_kv=max(kv_lens), is_causal=True, softcap=softcap
    )
    output = op(q, k, v, cu_q, cu_kv)
    torch.testing.assert_close(output.cpu(), ref.cpu(), atol=0.005, rtol=1e-05)
