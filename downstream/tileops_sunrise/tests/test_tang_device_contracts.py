"""Device regressions retained through the Entry/CallSpec migration."""

import pytest
import torch

from tileops._kernels.elementwise.activations import LeakyReLUFwdKernel
from tileops._ops.elementwise import LeakyReLUFwdOp
from tileops.backend import BUILTIN
from tileops.backend import _registry as registry
from tileops.norm import BatchNormBwdOp, BatchNormFwdOp
from workloads.device import run_device, run_device_is_cuda
from workloads.elementwise import LegacyLeakyReLUWorkload
from workloads.norm import (
    batch_norm_backward,
    batch_norm_backward_verification,
    batch_norm_forward_result,
    batch_norm_forward_verification,
    batch_norm_fwd_ref,
)
from workloads.numerics import compare_outputs

pytestmark = [pytest.mark.smoke, pytest.mark.in_tree_kernels]


def _device():
    device = torch.device(run_device())
    if not run_device_is_cuda() and device.type != "ptpu":
        pytest.skip("requires a real GPU kernel entry")
    return device


@pytest.mark.parametrize("training", [False, True])
def test_batch_norm_entry_reuse_and_shape_dtype_specialization(training):
    device = _device()
    op = BatchNormFwdOp(training=training, target=BUILTIN)
    interface = "batch_norm_fwd_train" if training else "batch_norm_fwd_infer"
    first_entry = None
    for c, length, dtype in (
        (8, 32, torch.float16),
        (8, 32, torch.float16),
        (12, 48, torch.bfloat16),
        (8, 32, torch.float32),
    ):
        x = torch.randn(1, c, length, dtype=dtype)
        weight, bias = torch.randn(c), torch.randn(c)
        mean, var = torch.zeros(c), torch.ones(c)
        expected = batch_norm_fwd_ref(x, weight, bias, mean, var, training)
        inputs = tuple(t.to(device) for t in (x, mean, var, weight, bias))
        compare_outputs(
            batch_norm_forward_result(op, *inputs), expected, batch_norm_forward_verification(dtype)
        )
        entries = tuple(op._entries_built(interface).values())
        if first_entry is None:
            first_entry = entries[0]
        assert entries[0] is first_entry
        assert len(entries) == (
            1 if c == 8 and dtype == torch.float16 else 2 if dtype == torch.bfloat16 else 3
        )


def test_batch_norm_backward_entry_reuse_and_specialization():
    device = _device()
    op = BatchNormBwdOp(target=BUILTIN)
    first_entry = None
    for c, length, dtype in (
        (8, 32, torch.float16),
        (8, 32, torch.float16),
        (12, 48, torch.bfloat16),
        (8, 32, torch.float32),
    ):
        x = torch.randn(1, c, length, dtype=dtype)
        grad = torch.randn_like(x)
        weight = torch.randn(c)
        var, mean = torch.var_mean(x.float(), dim=(0, 2), correction=0)
        inputs = (grad, x, mean, torch.rsqrt(var + 1e-5), weight)
        expected = batch_norm_backward(*inputs)
        *tensors, weight_arg = (t.to(device) for t in inputs)
        actual = op(*tensors, weight=weight_arg)
        compare_outputs(actual, expected, batch_norm_backward_verification(dtype))
        entries = tuple(op._entries_built("batch_norm_bwd").values())
        if first_entry is None:
            first_entry = entries[0]
        assert entries[0] is first_entry
        assert len(entries) == (
            1 if c == 8 and dtype == torch.float16 else 2 if dtype == torch.bfloat16 else 3
        )


@pytest.mark.parametrize("threads,npt", [(128, 4), (256, 4), (512, 8), (1024, 8)])
def test_leaky_relu_runs_each_explicit_launch_through_an_entry(threads, npt):
    class FixedLeaky(LeakyReLUFwdKernel):
        preferred_over = frozenset({"leaky_relu"})

        @property
        def default_config(self):
            return {"strategy": self.strategy, "threads": threads, "num_per_thread": npt}

    device = _device()
    state = registry.snapshot()
    try:
        registry.register_kernel_type("LeakyReLUFwdOp", "fixed_leaky", FixedLeaky)
        op = LeakyReLUFwdOp(target=BUILTIN)
        case = LegacyLeakyReLUWorkload(28685, torch.float16)
        x = case.gen_inputs()[0]
        actual = op(x.to(device))
        compare_outputs(actual, case.ref_program(x.cpu()), case.verification(x.cpu()))
        kernel = next(op.iter_kernels())
        assert isinstance(kernel, FixedLeaky)
        assert (kernel.config["threads"], kernel.config["num_per_thread"]) == (threads, npt)
    finally:
        registry.restore(state)


def test_retune_launches_the_program_matching_the_winning_config():
    class TwoConfigLeaky(LeakyReLUFwdKernel):
        preferred_over = frozenset({"leaky_relu"})

        @property
        def default_config(self):
            return {"strategy": self.strategy, "threads": 128, "num_per_thread": 4}

        @property
        def autotune_configs(self):
            return [{"threads": 128, "num_per_thread": 4}, {"threads": 256, "num_per_thread": 4}]

    device = _device()
    state = registry.snapshot()
    try:
        registry.register_kernel_type("LeakyReLUFwdOp", "two_config_leaky", TwoConfigLeaky)
        op = LeakyReLUFwdOp(target=BUILTIN)
        case = LegacyLeakyReLUWorkload(28685, torch.float16)
        x = case.gen_inputs()[0]
        xp = x.to(device)
        op(xp)
        kernel = next(op.iter_kernels())
        program = kernel.program(kernel._compile)
        op(xp)
        assert kernel.program(kernel._compile) is program
        kernel.autotune(warmup=1, rep=2)
        cfg = kernel.config
        assert (cfg["threads"], cfg["num_per_thread"]) in {(128, 4), (256, 4)}
        assert kernel.program(kernel._compile).jit is kernel.kernel(
            cfg["threads"], cfg["num_per_thread"]
        )
        compare_outputs(op(xp), case.ref_program(x.cpu()), case.verification(x.cpu()))
    finally:
        registry.restore(state)
