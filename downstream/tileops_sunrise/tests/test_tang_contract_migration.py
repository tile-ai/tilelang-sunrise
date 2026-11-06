"""Public contracts retained across the Entry/CallSpec migration."""

import pytest
import torch

import tileops._kernels.reduction as reduction_kernels
import tileops._ops.elementwise as elementwise
import tileops._ops.reduction as reduction_ops
from tileops._kernels.tiling import ALIGNMENT, align_up
from tileops.backend import BUILTIN
from tileops.manifest import load_workloads

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize(
    "value, alignment, expected",
    [
        (256, 256, 256),
        (100, 256, 256),
        (257, 256, 512),
        (0, 256, 0),
        (10, 8, 16),
        (8, 8, 8),
        (9, 8, 16),
    ],
)
def test_alignment_rounds_up(value, alignment, expected):
    assert align_up(value, alignment) == expected


@pytest.mark.parametrize("alignment", [0, -1])
def test_alignment_refuses_nonpositive_boundaries(alignment):
    with pytest.raises(ValueError, match="positive"):
        align_up(10, alignment)


def test_alignment_preserves_power_of_two_edges_and_reduction_exports():
    assert isinstance(ALIGNMENT, int) and ALIGNMENT == 256
    for power in range(1, 12):
        alignment = 2**power
        assert align_up(alignment - 1, alignment) == alignment
        assert align_up(alignment, alignment) == alignment
        assert align_up(alignment + 1, alignment) == 2 * alignment
    assert reduction_kernels.DEFAULT_ALIGNMENT == ALIGNMENT
    assert "DEFAULT_ALIGNMENT" in reduction_kernels.__all__
    assert all(not name.startswith("_") for name in reduction_kernels.__all__)
    assert isinstance(reduction_ops.__all__, list)


_FP8_REFUSALS = [
    "ReLUFwdOp",
    "BitwiseNotFwdOp",
    "BitwiseAndFwdOp",
    "MulFwdOp",
    "EqFwdOp",
    "LtFwdOp",
    "GeFwdOp",
    "PowFwdOp",
    "DivFwdOp",
    "FloorDivideFwdOp",
    "RemainderFwdOp",
    "LerpScalarFwdOp",
    "MaximumFwdOp",
    "MinimumFwdOp",
    "LogicalAndFwdOp",
    "LogicalOrFwdOp",
    "LogicalNotFwdOp",
    "LeakyReLUFwdOp",
    "ELUFwdOp",
    "HardtanhFwdOp",
    "PReLUFwdOp",
]
_UNARY = {
    "ReLUFwdOp",
    "BitwiseNotFwdOp",
    "LogicalNotFwdOp",
    "LeakyReLUFwdOp",
    "ELUFwdOp",
    "HardtanhFwdOp",
}


@pytest.mark.parametrize("op_name", _FP8_REFUSALS)
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_elementwise_refuses_fp8_before_resolving_a_kernel(op_name, dtype, monkeypatch):
    op = getattr(elementwise, op_name)(target=BUILTIN)

    def unreachable(*args, **kwargs):
        raise AssertionError("an invalid dtype reached kernel resolution")

    monkeypatch.setattr(op, "kernel_for", unreachable)
    x = torch.empty(16, dtype=dtype)
    with pytest.raises(ValueError, match="dtype is outside"):
        op(x) if op_name in _UNARY else op(x, x)


@pytest.mark.parametrize(
    "op_name, dtype",
    [
        (name, dtype)
        for name in ("PowFwdOp", "DivFwdOp", "FloorDivideFwdOp", "RemainderFwdOp")
        for dtype in (torch.bool, torch.int32)
    ]
    + [("LerpScalarFwdOp", torch.int32)],
)
def test_float_arithmetic_refuses_integer_inputs_before_build(op_name, dtype, monkeypatch):
    op = getattr(elementwise, op_name)(target=BUILTIN)

    def unreachable(*args, **kwargs):
        raise AssertionError("an invalid dtype reached kernel resolution")

    monkeypatch.setattr(op, "kernel_for", unreachable)
    x = torch.empty(16, dtype=dtype)
    with pytest.raises(ValueError, match="dtype is outside"):
        op(x, x)


@pytest.mark.parametrize(
    "op_name",
    [
        "EqFwdOp",
        "NeFwdOp",
        "LtFwdOp",
        "LeFwdOp",
        "GtFwdOp",
        "GeFwdOp",
        "LogicalAndFwdOp",
        "LogicalOrFwdOp",
    ],
)
def test_public_predicate_outputs_keep_boolean_dtype(op_name):
    x = torch.empty(16, dtype=torch.float16, device="meta")
    assert getattr(elementwise, op_name)(target=BUILTIN)(x, x).dtype is torch.bool


def test_manifest_workloads_accept_the_public_op_class():
    assert load_workloads(elementwise.ReLUFwdOp) == load_workloads("ReLUFwdOp")


@pytest.mark.parametrize("op_name", ["ReLUFwdOp", "AddFwdOp", "SiLUAndMulFwdOp"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_public_elementwise_outputs_preserve_the_input_dtype(op_name, dtype):
    x = torch.empty(4, 16, dtype=dtype, device="meta")
    op = getattr(elementwise, op_name)(target=BUILTIN)
    output = op(x, x) if op_name == "AddFwdOp" else op(x)
    assert output.dtype is dtype


def test_configuration_is_reported_from_the_resolved_entry_unless_the_op_overrides_it():
    from tileops._kernels.elementwise.call_spec import UnaryElementwiseFwdInterface
    from tileops._kernels.kernel_base import Kernel
    from tileops.backend import _registry as registry

    class ConfiguredCPUEntry(Kernel, UnaryElementwiseFwdInterface):
        devices = frozenset({"cpu"})
        supported_archs = None

        def __init__(self, call):
            self.config = {"threads": 128}

        def forward(self, x):
            return torch.relu(x)

    state = registry.snapshot()
    try:
        registry.register_kernel_type("ReLUFwdOp", "contract_cpu", ConfiguredCPUEntry)
        op = elementwise.ReLUFwdOp(target=BUILTIN)
        assert op.kernel_config() is None
        op(torch.zeros(16))
        assert op.kernel_config() == {"threads": 128}
        op.config = {"explicit": True}
        assert op.kernel_config() == {"explicit": True}
    finally:
        registry.restore(state)
