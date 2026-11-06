"""Tests for the Native Sparse Attention (NSA) ops: the sparse forward, compression and top-k selection."""

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.attention import NSACompressedVarlenFwdOp, NSATopKVarlenFwdOp, NSAVarlenFwdOp
from tileops.manifest import load_adts, load_manifest, load_workloads
from tileops.manifest._plan import entry_plan
from tileops.manifest._workload import instantiate
from workloads.attention.nsa import (
    NSACompressedFwdWorkload,
    NSAFwdCall,
    NSAFwdWorkload,
    NSATopKWorkload,
)
from workloads.device import host_if_ptpu, run_device, set_run_device
from workloads.numerics import Request, reference_for, verification_for, verify


class NSAFwdTest(NSAFwdWorkload, TestBase):
    pass


class NSAFwdFixture(FixtureBase):
    PARAMS = [
        (
            "batch, heads, c_seq_len, dim, is_causal, scale, block_size, "
            "groups, selected_blocks, dtype, tune, seq_lens",
            [
                pytest.param(
                    1,
                    16,
                    1024,
                    64,
                    True,
                    0.1,
                    32,
                    16,
                    1,
                    torch.float16,
                    False,
                    None,
                    marks=pytest.mark.smoke,
                ),
                # Non-causal, with lengths no block size divides.
                pytest.param(
                    2,
                    16,
                    256,
                    64,
                    False,
                    0.1,
                    32,
                    16,
                    4,
                    torch.float16,
                    False,
                    [100, 156],
                    marks=pytest.mark.smoke,
                ),
                # bfloat16 over two KV heads, eight slots through a two-stage K/V ring.
                pytest.param(
                    2,
                    32,
                    256,
                    128,
                    True,
                    128**-0.5,
                    32,
                    16,
                    8,
                    torch.bfloat16,
                    False,
                    [100, 156],
                    marks=pytest.mark.smoke,
                ),
                # A head dim the MMA K step does not tile selects the general kernel.
                pytest.param(
                    2,
                    16,
                    256,
                    72,
                    True,
                    0.1,
                    32,
                    16,
                    4,
                    torch.float16,
                    False,
                    [100, 156],
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    4,
                    16,
                    8192,
                    64,
                    True,
                    0.1,
                    32,
                    16,
                    1,
                    torch.float16,
                    False,
                    None,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    2,
                    16,
                    8192,
                    64,
                    True,
                    0.1,
                    32,
                    16,
                    4,
                    torch.float16,
                    False,
                    None,
                    marks=pytest.mark.full,
                ),
            ],
        ),
    ]


@NSAFwdFixture
def test_nsa_varlen_op(
    batch: int,
    heads: int,
    c_seq_len: int,
    dim: int,
    is_causal: bool,
    scale: float,
    block_size: int,
    groups: int,
    selected_blocks: int,
    dtype: torch.dtype,
    tune: bool,
    seq_lens: "list[int] | None",
) -> None:
    assert groups % 16 == 0, "Group size must be a multiple of 16 in NSA"

    test = NSAFwdTest(
        batch,
        heads,
        c_seq_len,
        dim,
        is_causal,
        scale,
        block_size,
        groups,
        selected_blocks,
        dtype,
        seq_lens,
    )
    op = NSAVarlenFwdOp(
        is_causal=is_causal,
        sm_scale=scale,
        block_size=block_size,
    )
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs())


class NSACompressedFwdTest(NSACompressedFwdWorkload, TestBase):
    pass


class NSACompressedFwdFixture(FixtureBase):
    PARAMS = [
        (
            "seq_num, c_seq_len, heads, dim_k, dim_v, group, scale, bs, dtype, tune",
            [
                pytest.param(
                    9,
                    8192,
                    32,
                    128,
                    128,
                    16,
                    128**-0.5,
                    32,
                    torch.float16,
                    False,
                    marks=pytest.mark.smoke,
                ),
                # BF16 MMA requires FP32 accumulation.
                pytest.param(
                    1, 65, 16, 64, 64, 16, 0.125, 32, torch.bfloat16, False, marks=pytest.mark.smoke
                ),
            ],
        ),
    ]


@NSACompressedFwdFixture
def test_nsa_compressed_fwd_varlen_op(
    seq_num: int,
    c_seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    group: int,
    scale: float,
    bs: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    assert group % 16 == 0, "Group size must be a multiple of 16 in NSA"
    test = NSACompressedFwdTest(seq_num, c_seq_len, heads, dim_k, dim_v, group, scale, bs, dtype)
    inputs = test.gen_inputs()
    op = NSACompressedVarlenFwdOp(sm_scale=scale, block_size=bs)
    if tune:
        op.request_tune()
    test.check(op, *inputs)


class NSATopKTest(NSATopKWorkload, TestBase):
    pass


class NSATopKFixture(FixtureBase):
    PARAMS = [
        (
            "seq_num, c_seq_len, heads, dim, group, scale, selected_block_num, bs, dtype, tune",
            [
                pytest.param(
                    5, 1024, 32, 128, 16, 1.0, 16, 32, dtype, False, marks=pytest.mark.smoke
                )
                for dtype in (torch.float16, torch.bfloat16)
            ]
            + [
                # A 32-head group fills a 64-row tile with two tokens, so half the warps sort.
                pytest.param(
                    4,
                    2048,
                    64,
                    64,
                    32,
                    0.125,
                    8,
                    32,
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    3, 512, 32, 128, 16, 1.0, 16, 32, torch.float16, False, marks=pytest.mark.full
                ),
                # Two sort pairs per lane must not put barriers under a partial-lane mask.
                pytest.param(
                    1, 65, 16, 64, 16, 0.125, 8, 64, torch.float16, False, marks=pytest.mark.full
                ),
                # Regression: a narrow sort band must pad all 64 requested output slots.
                pytest.param(
                    1, 65, 32, 64, 32, 0.125, 64, 64, torch.float16, False, marks=pytest.mark.full
                ),
                # A 48-head group uses the bounded SM89 layout or the per-token kernel.
                pytest.param(
                    2, 817, 48, 64, 48, 0.125, 16, 32, torch.float16, False, marks=pytest.mark.full
                ),
            ],
        ),
    ]


@NSATopKFixture
def test_nsa_topk_varlen_op(
    seq_num: int,
    c_seq_len: int,
    heads: int,
    dim: int,
    group: int,
    scale: float,
    selected_block_num: int,
    bs: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    assert group % 16 == 0, "Group size must be a multiple of 16 in NSA"

    test = NSATopKTest(seq_num, c_seq_len, heads, dim, group, scale, selected_block_num, bs, dtype)
    inputs = test.gen_inputs()
    op = NSATopKVarlenFwdOp(
        sm_scale=scale,
        selected_block_num=selected_block_num,
        block_size=bs,
    )
    if tune:
        op.request_tune()
    test.check(op, *inputs)


@pytest.mark.smoke
def test_nsa_topk_reference_keeps_fp32_dot_products() -> None:
    """Promoting a rounded half matmul is too late to rank close candidates."""
    torch.manual_seed(1235)
    workload = NSATopKWorkload(1, 512, 32, 64, 16, 1.0, 16, 32, torch.float16)
    q, k, *metadata = workload.gen_inputs()
    assert torch.equal(
        reference_for(workload.ref_program, q, k, *metadata),
        reference_for(workload.ref_program, q.float(), k.float(), *metadata),
    )


@pytest.mark.smoke
def test_nsa_topk_ranks_unquantized_scores() -> None:
    """Priority blocks survive; the remaining slot takes the highest raw score."""
    workload = NSATopKTest(1, 256, 16, 16, 16, 1.0, 4, 32, torch.float16)
    q, k, *metadata = workload.gen_inputs()
    q = torch.zeros_like(q)
    q[-1].fill_(1)
    if k.device.type == "ptpu":
        k_host = torch.arange(k.shape[0] - 1, -1, -1, device="cpu")[:, None, None] * 2**-24
        k_host = k_host.expand(k.shape).to(k.dtype).contiguous()
        k = k_host.to(k.device)
    else:
        k = torch.empty_like(k)
        k.copy_(torch.arange(k.shape[0] - 1, -1, -1, device=k.device)[:, None, None] * 2**-24)
    op = NSATopKVarlenFwdOp(sm_scale=1.0, selected_block_num=4, block_size=32)
    workload.check(op, q, k, *metadata)
    result = op(q, k, *metadata)
    # Sub-1e-5 gaps must still select block 1, alongside priority blocks 0, 6, 7.
    assert torch.equal(
        result[-1, 0].sort().values,
        torch.tensor([0, 1, 6, 7], dtype=torch.int32, device=q.device),
    )


@pytest.mark.smoke
def test_nsa_varlen_reference_returns_the_declared_output() -> None:
    """Benchmarks check every implementation against this reference, so it returns ``o`` as declared."""
    name = "NSAVarlenFwdOp"
    plan = entry_plan(name, load_manifest()[name], load_adts(), resolve=False)
    calls = [
        instantiate(plan, row, case)
        for row in load_workloads(name)
        for case in row.get("dtype_cases") or [{}]
    ]
    call = min(calls, key=lambda c: c.indices["T_q"] * c.indices["H"] * c.indices["D"])
    workload = NSAFwdCall(call)
    output = reference_for(workload.ref_program, *workload.gen_inputs())
    spec = call.specs["o"]
    assert tuple(output.shape) == tuple(spec.shape) and output.dtype == getattr(torch, spec.dtype)


@pytest.mark.smoke
@pytest.mark.parametrize("seed", [17, 203])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("seq_lens", [[17, 48], None])
def test_nsa_sparse_cpu_workload_metadata_and_gate(seed, is_causal, seq_lens) -> None:
    """The host oracle retains this input batch's packed metadata and auxiliary gate."""
    device = run_device()
    set_run_device("cpu")
    try:
        workload = NSAFwdWorkload(2, 4, 65, 8, is_causal, 0.25, 8, 2, 3, torch.float32, seq_lens)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            inputs = workload.gen_inputs()
        assert all(x.device.type == "cpu" for x in inputs)
        assert all(x.dtype == torch.int32 for x in inputs[3:])
        offsets, tokens = inputs[-2:]
        assert offsets[0] == 0 and offsets[-1] == 65
        assert (offsets[1:] > offsets[:-1]).all()
        if seq_lens is not None:
            assert offsets.tolist() == [0, 17, 65]
        expected_tokens = [
            [seq, token]
            for seq, (first, last) in enumerate(zip(offsets[:-1], offsets[1:], strict=True))
            for token in range(int(last - first))
        ]
        assert tokens.tolist() == expected_tokens
        pristine = tuple(x.detach().clone() for x in inputs)
        ungated = workload.ref_program(*inputs).detach()
        gate = torch.linspace(0.25, 1.75, 65 * 4).view(65, 4)
        workload.g_slc = gate.unsqueeze(0).expand(2, -1, -1).clone()
        expected = ungated * gate.unsqueeze(-1)
        assert torch.count_nonzero(expected - ungated) > 0
        result = verify(
            workload.ref_program,
            inputs,
            evidence=verification_for(workload.verification, inputs),
            requests={"host-gate-contract": Request(lambda *args: expected.clone(), inputs)},
            preserve_reference_inputs=True,
        )["host-gate-contract"]
        assert result.checked_outputs == 1
        torch.testing.assert_close(workload.ref_program(*inputs), expected, rtol=0, atol=0)
        assert torch.equal(workload.g_slc[0], gate)
        assert all(
            torch.equal(before, after) for before, after in zip(pristine, inputs, strict=True)
        )
    finally:
        set_run_device(device)


@pytest.mark.smoke
def test_nsa_sparse_reference_moves_auxiliary_gate_to_input_device() -> None:
    """Real torch meta dispatch checks placement only; it does not check device numerics."""
    workload = NSAFwdWorkload(1, 2, 5, 8, True, 0.25, 4, 2, 1, torch.float32)
    workload.g_slc = torch.linspace(0.25, 1.75, 10).view(1, 5, 2)
    output = workload.ref_program(
        torch.empty(5, 2, 8, device="meta"),
        torch.empty(5, 1, 8, device="meta"),
        torch.empty(5, 1, 8, device="meta"),
        torch.zeros(5, 1, 1, dtype=torch.int32, device="meta"),
        torch.ones(5, 1, dtype=torch.int32, device="meta"),
        torch.tensor([0, 5], dtype=torch.int32),
        torch.tensor([[0, i] for i in range(5)], dtype=torch.int32),
    )
    assert output.device.type == "meta" and output.shape == (5, 2, 8)
    assert workload.g_slc.device.type == "cpu"


@pytest.mark.cuda_only
@pytest.mark.sm89
@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_nsa_topk_group48_wide_head(dtype: torch.dtype) -> None:
    """Cover both dtypes of the group48 layout, two KV heads and nontrivial ranking."""
    test = NSATopKTest(2, 258, 96, 256, 48, 0.0625, 4, 64, dtype, seq_lens=[1, 257])
    inputs = test.gen_inputs()
    op = NSATopKVarlenFwdOp(sm_scale=0.0625, selected_block_num=4, block_size=64)
    test.check(op, *inputs)


@pytest.mark.cuda_only
@pytest.mark.sm89
@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_nsa_topk_group48_many_requests(dtype: torch.dtype) -> None:
    """Packed short requests must not allocate sorting space for the whole batch."""
    count = 2049
    test = NSATopKTest(count, count, 48, 256, 48, 0.0625, 1, 64, dtype, seq_lens=[1] * count)
    inputs = test.gen_inputs()
    result = NSATopKVarlenFwdOp(sm_scale=0.0625, selected_block_num=1, block_size=64)(*inputs)
    assert torch.equal(result, torch.zeros_like(result))


@pytest.mark.smoke
def test_nsa_compressed_lse_is_head_major_float32_and_minus_inf_without_a_chunk() -> None:
    """A token before its request's first closed chunk sees no key: zero output and lse
    -inf, the convention every attention op and MergeAttentionStatesFwdOp share."""
    test = NSACompressedFwdTest(1, 65, 16, 64, 64, 16, 0.125, 32, torch.bfloat16)
    inputs = test.gen_inputs()
    o, lse = NSACompressedVarlenFwdOp(sm_scale=0.125, block_size=32)(*inputs)
    assert lse.dtype == torch.float32
    assert lse.shape == (16, 65)
    # torch_ptpu lacks isneginf; PTPU inspects the host copy.
    host_lse = lse.to(host_if_ptpu(lse.device))
    assert torch.isneginf(host_lse[:, :31]).all()
    assert torch.isfinite(lse[:, 31:]).all()
    assert (o[:31] == 0).all()
