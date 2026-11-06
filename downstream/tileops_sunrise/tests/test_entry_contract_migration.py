"""Concrete family selection and Entry identity retained across the API migration.

Selection tests use explicit device facts and never build a kernel. Tests named
``test_device_*`` launch the selected builtin implementation on ``run_device``.
"""

import itertools

import pytest
import torch

from tileops._kernels.attention.call_spec import AttentionCall
from tileops._kernels.gemm import GemvKernel
from tileops._kernels.gemm.call_spec import GEMMCall
from tileops._kernels.linear_attention.call_spec import DeltaNetDecodeCall, GDNCall
from tileops._ops.elementwise import BitwiseAndFwdOp, LogicalAndFwdOp, MaskedFillScalarFwdOp
from tileops._ops.linear_attention.deltanet.recurrent import DeltaNetRecurrentFwdOp
from tileops._ops.op_base import OpNotAvailableError
from tileops._ops.reduction import SumFwdOp
from tileops.attention import GQAPagedFwdOp
from tileops.backend import BUILTIN
from tileops.gemm import GEMMFwdOp
from tileops.linear_attention import GDNFwdOp
from tileops.norm import BatchNormFwdOp, LayerNormFwdOp, RMSNormFwdOp
from workloads.device import run_device, run_device_is_cuda

pytestmark = [pytest.mark.smoke, pytest.mark.in_tree_kernels]

# Selection-only models name the CUDA backend without requiring a CUDA runtime.
CUDA_DEVICE_TYPE = torch.cuda.__name__.removeprefix("torch.")


@pytest.fixture(scope="module", autouse=True)
def _register_modeled_ptpu_device_name():
    """Name an unused private backend for CallSpecs without loading its runtime.

    This registers syntax only. Real device tests still require ``run_device``;
    existing non-PTPU private backend registrations are left intact.
    """
    if torch._C._get_privateuse1_backend_name() == "privateuseone":
        torch.utils.rename_privateuse1_backend("ptpu")


def _selection_facts(device=CUDA_DEVICE_TYPE, arch=90):
    """A modeled board for selection only, never a build or hardware claim."""
    return dict(
        device=torch.device(device, 0),
        arch=arch,
        calibration=None,
        sm_count=80,
        smem_budget=227328 if device == CUDA_DEVICE_TYPE else 131072,
    )


@pytest.mark.parametrize(
    "m,n,trans_a,trans_b,expected",
    [
        (1, 8, False, True, "gemv"),
        (8, 1, False, False, "gemv"),
        (1, 8, False, False, "gemm_tma"),
        (8, 1, False, True, "gemm_tma"),
        (8, 8, False, False, "gemm_tma"),
        (1, 1, False, False, "gemv"),
        # The new generic implementation can serve what the old TMA-only set refused.
        (1, 8, True, True, "gemm_cp_async"),
        (8, 1, True, False, "gemm_cp_async"),
    ],
)
def test_gemm_positive_regions(m, n, trans_a, trans_b, expected):
    op = GEMMFwdOp(trans_a=trans_a, trans_b=trans_b, target=BUILTIN)
    call = GEMMCall(
        **_selection_facts(),
        m=m,
        n=n,
        k=64,
        dtype=torch.float16,
        trans_a=trans_a,
        trans_b=trans_b,
    )
    assert op.key_for("gemm", call) == expected
    assert not op._entries_built("gemm")


def test_gemv_preserves_the_old_vector_layout_truth_table():
    for m, n, ta, tb in itertools.product((1, 8), (1, 8), (False, True), (False, True)):
        call = GEMMCall(
            **_selection_facts(),
            m=m,
            n=n,
            k=64,
            dtype=torch.float16,
            trans_a=ta,
            trans_b=tb,
        )
        expected = not ta and ((m == 1 and tb) or (n == 1 and not tb))
        assert (GemvKernel.refusal(call) is None) is expected


@pytest.mark.parametrize("device,arch", [(CUDA_DEVICE_TYPE, 80), ("ptpu", None)])
def test_gemm_generic_region_uses_the_stated_device(device, arch):
    call = GEMMCall(
        **_selection_facts(device, arch),
        m=1,
        n=8,
        k=64,
        dtype=torch.float16,
        trans_b=True,
    )
    assert GEMMFwdOp(target=BUILTIN).key_for("gemm", call) == "gemm_cp_async"


@pytest.mark.parametrize(
    "dtype,dk,dv,arch,expected",
    [
        (torch.float32, 128, 128, 90, "deltanet_decode_fp32"),
        (torch.float32, 64, 64, 80, "deltanet_decode_fp32"),
        (torch.float16, 128, 128, 90, "deltanet_decode_warp"),
        (torch.bfloat16, 128, 128, 90, "deltanet_decode_warp"),
        (torch.float16, 64, 128, 90, "deltanet_decode"),
        (torch.float16, 128, 64, 90, "deltanet_decode"),
        (torch.float16, 128, 128, 80, "deltanet_decode"),
    ],
)
def test_deltanet_decode_positive_regions(dtype, dk, dv, arch, expected):
    call = DeltaNetDecodeCall(
        **_selection_facts(arch=arch),
        batch=1,
        heads=4,
        dim_k=dk,
        dim_v=dv,
        dtype=dtype,
    )
    op = DeltaNetRecurrentFwdOp(target=BUILTIN)
    assert op.key_for("deltanet_decode", call) == expected
    assert not op._entries_built("deltanet_decode")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_deltanet_decode_ptpu_selects_a_portable_owner(dtype):
    call = DeltaNetDecodeCall(
        **_selection_facts("ptpu", None),
        batch=1,
        heads=4,
        dim_k=128,
        dim_v=128,
        dtype=dtype,
    )
    expected = "deltanet_decode_fp32" if dtype == torch.float32 else "deltanet_decode"
    assert DeltaNetRecurrentFwdOp(target=BUILTIN).key_for("deltanet_decode", call) == expected


@pytest.mark.parametrize(
    "device,arch,tokens,expected",
    [
        (CUDA_DEVICE_TYPE, 80, 1, "gdn_dense_decode"),
        (CUDA_DEVICE_TYPE, 90, 1, "gdn_dense_decode"),
        (CUDA_DEVICE_TYPE, 90, 65, "gdn_dense_prefill"),
        (CUDA_DEVICE_TYPE, 89, 65, "gdn_chunk_prefill"),
        ("ptpu", None, 1, "gdn_dense_decode"),
        ("ptpu", None, 65, "gdn_chunk_prefill"),
    ],
)
def test_current_gdn_inference_regions(device, arch, tokens, expected):
    """Current inference ownership; this makes no claim about retired training APIs."""
    call = GDNCall(
        **_selection_facts(device, arch),
        batch=1,
        seq_len=tokens,
        heads=4,
        value_heads=4,
        dim_k=64,
        dim_v=64,
        dtype=torch.bfloat16,
        scale=0.125,
        num_sequences=1,
        l2norm=True,
        beta_sigmoid=True,
        allow_neg_eigval=True,
    )
    assert GDNFwdOp(target=BUILTIN).key_for("gdn", call) == expected


@pytest.mark.parametrize(
    "batch,dtype,page_size,expected",
    [
        (1, torch.float16, 256, "gqa_decode_paged_bs1_kernel"),
        (1, torch.bfloat16, 256, "gqa_paged_varlen_kernel"),
        (2, torch.float16, 256, "gqa_paged_varlen_kernel"),
        (1, torch.float16, 192, "gqa_paged_varlen_kernel"),
    ],
)
@pytest.mark.parametrize("device,arch", [(CUDA_DEVICE_TYPE, 90), ("ptpu", None)])
def test_read_only_paged_decode_regions(batch, dtype, page_size, expected, device, arch):
    call = AttentionCall(
        **_selection_facts(device, arch),
        batch=batch,
        heads=32,
        heads_kv=4,
        dim=128,
        page_size=page_size,
        max_pages_per_req=32,
        max_seqlen_q=1,
        seqlen_kv=8192,
        dtype=dtype,
        cache_dtype=dtype,
    )
    # New packed attention outranks the general decode owner; BS1 still wins on CUDA.
    if device == "ptpu":
        expected = "gqa_paged_varlen_kernel"
    assert GQAPagedFwdOp(target=BUILTIN).key_for("gqa_paged", call) == expected


@pytest.mark.parametrize("call_type", [GEMMCall, DeltaNetDecodeCall, GDNCall, AttentionCall])
def test_concrete_calls_resolve_cpu_facts_and_preserve_explicit_arch(call_type):
    call = call_type(device=torch.device("cpu"))
    assert (call.arch, call.calibration, call.sm_count, call.smem_budget) == (-1, None, 0, 0)
    explicit = call_type(device=torch.device("cpu"), arch=80)
    assert explicit.arch == 80
    assert (explicit.sm_count, explicit.smem_budget) == (0, 0)
    assert explicit.arch == 80


@pytest.mark.parametrize("dtype", [torch.int32, torch.float64])
def test_batchnorm_invalid_dtype_precedes_entry_build(dtype):
    op = BatchNormFwdOp(target=BUILTIN)
    with pytest.raises(ValueError, match="dtype"):
        op(torch.zeros(1, 8, 32, dtype=dtype), torch.zeros(8), torch.ones(8))
    assert not op._entries_built("batch_norm_fwd_infer")


def test_batchnorm_builtin_refuses_cpu_without_building():
    op = BatchNormFwdOp(target=BUILTIN)
    with pytest.raises(OpNotAvailableError, match="cpu"):
        op(torch.zeros(1, 8, 32), torch.zeros(8), torch.ones(8))
    assert not op._entries_built("batch_norm_fwd_infer")


def _device():
    device = torch.device(run_device())
    if not run_device_is_cuda() and device.type != "ptpu":
        pytest.skip("requires a real builtin GPU entry")
    return device


def test_device_concrete_calls_read_current_facts_without_fabricating_an_arch():
    from tileops._utils import device_facts

    device = _device()
    expected = device_facts(device.index)
    for cls in (GEMMCall, DeltaNetDecodeCall, GDNCall, AttentionCall):
        call = cls(device=device)
        assert (call.arch, call.calibration, call.sm_count, call.smem_budget) == expected
        unstated = cls()
        assert (
            unstated.arch,
            unstated.calibration,
            unstated.sm_count,
            unstated.smem_budget,
        ) == expected
        assert cls(device=device, arch=80).arch == 80


def _dtype_case(name):
    if name == "sum":
        return (
            SumFwdOp(dim=-1, target=BUILTIN),
            "reduce",
            lambda dt: (torch.randn(8, 128, dtype=dt),),
        )
    if name in {"rms", "layer"}:
        cls = RMSNormFwdOp if name == "rms" else LayerNormFwdOp
        count = 2 if name == "rms" else 3
        return (
            cls((256,), target=BUILTIN),
            name + "_norm",
            lambda dt: (
                torch.randn(16, 256, dtype=dt),
                *[torch.randn(256, dtype=dt) for _ in range(count - 1)],
            ),
        )
    if name == "moe":
        from tileops._ops.moe import ContiguousLayoutSpec, MoEPostPermuteFwdOp

        op = MoEPostPermuteFwdOp(ContiguousLayoutSpec.tight_physical_psum(), target=BUILTIN)
        return (
            op,
            "post_permute",
            lambda dt: (
                torch.randn(32, 128, dtype=dt),
                torch.rand(16, 2),
                torch.arange(32, dtype=torch.int32),
            ),
        )
    from tileops._ops.mamba.ssd_chunk_coupling import SSDChunkCouplingFwdOp

    return (
        SSDChunkCouplingFwdOp(64, target=BUILTIN),
        "ssd_chunk_coupling",
        lambda dt: (torch.randn(1, 128, 1, 64, dtype=dt), torch.randn(1, 128, 1, 64, dtype=dt)),
    )


@pytest.mark.parametrize("name", ["sum", "rms", "layer", "moe", "ssd"])
def test_device_concrete_ops_reuse_an_entry_and_separate_dtypes(name):
    device = _device()
    op, interface, inputs = _dtype_case(name)
    first = None
    for dtype, count in zip(
        (torch.float16, torch.float16, torch.bfloat16, torch.float16), (1, 1, 2, 2), strict=True
    ):
        host = inputs(dtype)
        args = tuple(t.to(device) for t in host)
        output = op(*args)
        assert output.dtype == dtype
        if name == "sum":
            torch.testing.assert_close(output.cpu(), host[0].sum(-1), atol=2e-2, rtol=2e-2)
        entries = tuple(op._entries_built(interface).values())
        if first is None:
            assert len(entries) == 1
            first = entries[0]
        else:
            assert entries[0] is first
            assert len(entries) == count
    entries = tuple(op._entries_built(interface).values())
    assert len(entries) == 2 and entries[0] is not entries[1]
    assert len(list(op.iter_kernels())) == 2


@pytest.mark.parametrize("kind", ["bitwise", "logical", "masked_fill"])
def test_device_storage_roundtrip_keeps_entries_and_required_classes_separate(kind):
    device = _device()
    op = {
        "bitwise": BitwiseAndFwdOp,
        "logical": LogicalAndFwdOp,
        "masked_fill": lambda **kw: MaskedFillScalarFwdOp(value=1, **kw),
    }[kind](target=BUILTIN)
    first = None
    for step, dtype in enumerate(
        (torch.bool, torch.int32 if kind == "bitwise" else torch.float32, torch.bool)
    ):
        if kind == "masked_fill":
            x = torch.zeros(64, dtype=dtype)
            y = torch.tensor([True, False] * 32)
        elif dtype == torch.bool:
            x = torch.tensor([True, False] * 32)
            y = ~x if step == 0 else x
        elif kind == "bitwise":
            x = torch.arange(64, dtype=torch.int32)
            y = x + 1
        else:
            x = torch.tensor([0.0, 1.0] * 32)
            y = x
        expected = (
            x.masked_fill(y, 1)
            if kind == "masked_fill"
            else (x & y if kind == "bitwise" else torch.logical_and(x, y))
        )
        assert torch.equal(op(x.to(device), y.to(device)).cpu(), expected)
        entries = tuple(op._entries_built("elementwise").values())
        if step == 0:
            first = entries[0]
        assert entries[0] is first
        assert len(entries) == (1 if step == 0 else 2)
    assert entries[0] is not entries[1]
    if kind != "masked_fill":
        assert type(entries[0]) is not type(entries[1])
        assert len({type(kernel) for kernel in op.iter_kernels()}) == 2
