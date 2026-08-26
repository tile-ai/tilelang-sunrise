import re

import torch
import tilelang
import tilelang.testing
from tilelang import language as T
from tilelang.utils.device import get_current_device


@tilelang.jit()
def _issue_1734_layout_kernel():
    @T.prim_func
    def main(
        A: T.Tensor[(2, 512), T.float32],
        B: T.Tensor[(2, 512), T.float32],
        C: T.Tensor[(2,), T.float32],
    ):
        with T.Kernel(1, threads=256):
            A_local = T.alloc_fragment((2, 512), T.float32)
            B_local = T.alloc_fragment((2, 512), T.float32)
            C_local = T.alloc_fragment((2,), T.float32)

            T.copy(A, A_local)
            T.copy(B, B_local)
            T.copy(C, C_local)

            for i in T.serial(0, 2):
                for j in T.Parallel(512):
                    if C_local[i] >= 0:
                        B_local[i, j] = A_local[i, j]

            T.copy(B_local, B)

    return main


def test_issue_1734():
    """Test that loop-invariant if statements are hoisted out of loops."""

    @tilelang.jit()
    def kernel():
        @T.prim_func
        def main(
            A: T.Tensor[(2, 512), T.float32],
            B: T.Tensor[(2, 512), T.float32],
            C: T.Tensor[(2,), T.float32],
        ):
            with T.Kernel(1, threads=256):
                A_local = T.alloc_fragment((2, 512), T.float32)
                B_local = T.alloc_fragment((2, 512), T.float32)
                C_local = T.alloc_fragment((2,), T.float32)

                T.copy(A, A_local)
                T.copy(B, B_local)
                T.copy(C, C_local)

                for i, j in T.Parallel(2, 512):
                    if C_local[i] >= 0:
                        B_local[i, j] = A_local[i, j]

                T.copy(B_local, B)

        return main

    mod = kernel.compile()
    runtime_source = mod.get_kernel_source()
    assert re.search(r"\bB_local\[[^\n]+\]\s*=\s*A_local", runtime_source)
    assert re.search(r"\bif\s*\(", runtime_source)

    # Keep the exact outer-if/inner-loop source oracle on a layout that Tang
    # preserves instead of flattening during the final kernel emission.
    source = _issue_1734_layout_kernel.compile().get_kernel_source()
    assignment = re.search(r"\bB_local\[[^\n]+\]\s*=\s*A_local", source)
    assert assignment is not None, "Expected a generated B_local <- A_local assignment"
    assignment_pos = assignment.start()
    if_positions = [match.start() for match in re.finditer(r"\bif\s*\(", source)]
    for_positions = [match.start() for match in re.finditer(r"\bfor\s*\(", source)]
    if_pos = max((position for position in if_positions if position < assignment_pos), default=-1)
    inner_for_pos = max(
        (position for position in for_positions if if_pos < position < assignment_pos),
        default=-1,
    )
    outer_for_pos = max((position for position in for_positions if position < if_pos), default=-1)
    assert -1 not in (outer_for_pos, if_pos, inner_for_pos), source
    assert outer_for_pos < if_pos < inner_for_pos < assignment_pos, "Grouped loop condition should be hoisted outside the inner loop"

    device = get_current_device()
    input_a = torch.arange(2 * 512, dtype=torch.float32, device=device).reshape(2, 512)
    input_c = torch.arange(2, dtype=torch.float32, device=device) * 2 - 1
    output_b = torch.zeros_like(input_a)
    mod(input_a, output_b, input_c)
    if device.type == "ptpu":
        torch.ptpu.synchronize()
    expected = torch.where(input_c[:, None] >= 0, input_a, torch.zeros_like(input_a))
    torch.testing.assert_close(output_b.cpu(), expected.cpu())


if __name__ == "__main__":
    tilelang.testing.main()
