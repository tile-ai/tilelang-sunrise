"""
End-to-end test: verify that pts_store_async's global destination parameter
is NOT const-qualified in the generated TANG C code, while the source
(input) parameter IS const-qualified.

Background: AsyncLoadParamCollector in codegen_tang.cc handles
pts_load_async (global->shared) and pts_store_async (shared->global).
Each builtin has a different operand order, and the *destination* is the
operand that must not be const-qualified:

  pts_load_async  (global->shared): args = {dst_shared, src_global, bytes}
                  collector reads args[0] as the destination.
  pts_store_async (shared->global): args = {src_shared, dst_global, bytes}
                  collector reads args[1] as the destination -- note the
                  order is the reverse of pts_load_async.

Getting either backwards const-qualifies a written buffer. ptcc silently
accepts a const destination -- no compile error, just wrong results -- so
missing a parameter here is a silent correctness bug.

This test:
  1. Constructs a kernel whose output buffer C is ONLY written via
     pts_store_async (never loaded from global), so the collector MUST
     catch it through the pts_store_async branch alone.
  2. Checks the generated function signature: input A has const,
     output C does NOT have const.
  3. Runs an end-to-end correctness check.
"""

import re
import torch
import tilelang
import tilelang.language as T


@tilelang.jit
def elementwise_scale(A, block_M, block_N, in_dtype, out_dtype, threads):
    """Scale input A by 2.0.

    Output C is only written via T.copy (shared -> global), which lowers to
    pts_store_async. C is never loaded from global, so pts_store_async is the
    sole code path that can detect it as needing non-const qualification.
    """
    M, N = T.const("M, N")

    A: T.Tensor((M, N), in_dtype)
    C = T.empty((M, N), out_dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_N), in_dtype)
        C_local = T.alloc_fragment((block_M, block_N), out_dtype)
        C_shared = T.alloc_shared((block_M, block_N), out_dtype)

        # A: global -> shared (pts_load_async, args[0]=dst_shared)
        T.copy(A[by * block_M, bx * block_N], A_shared)
        # Compute: C = 2.0 * A
        for local_y, local_x in T.Parallel(block_M, block_N):
            C_local[local_y, local_x] = T.cast(A_shared[local_y, local_x] * T.cast(2.0, in_dtype), out_dtype)
        # C: local -> shared -> global (pts_store_async, args[1]=dst_global)
        T.copy(C_local, C_shared)
        T.copy(C_shared, C[by * block_M, bx * block_N])

    return C


def ref_scale(x):
    return x * 2.0


def check_function_signature(source_code: str) -> dict:
    """Parse the generated TANG C function signature.

    Returns a dict mapping parameter name to its const qualifier status.
    """
    sig_match = re.search(r"void\s+\w+\s*\(([^)]*)\)", source_code)
    if not sig_match:
        return {"error": "could not parse function signature"}

    params_str = sig_match.group(1)
    result = {}
    for param_decl in params_str.split(","):
        param_decl = param_decl.strip()
        has_const = param_decl.startswith("const ")
        name_match = re.search(r"\b(\w+)\s*$", param_decl)
        if name_match:
            result[name_match.group(1)] = {
                "has_const": has_const,
                "declaration": param_decl,
            }
    return result


def main():
    M, N = 256, 256
    device = "ptpu" if hasattr(torch, "ptpu") and torch.ptpu.is_available() else "cpu"

    # Compile and get generated source
    kernel = elementwise_scale.compile(
        M=M,
        N=N,
        block_M=64,
        block_N=64,
        in_dtype="float16",
        out_dtype="float16",
        threads=128,
    )
    source = kernel.get_kernel_source()

    # --- Check 1: function signature const qualifiers ---
    sig_info = check_function_signature(source)
    assert "error" not in sig_info, f"Failed to parse signature: {sig_info.get('error')}"

    print("=== Generated function signature ===")
    for name, info in sig_info.items():
        status = "const" if info["has_const"] else "NON-const"
        print(f"  {name}: {status}  ({info['declaration']})")

    # Identify input (A) and output (C) parameters by name prefix
    a_names = [n for n in sig_info if n.lower().startswith("a") or "A" in n]
    c_names = [n for n in sig_info if n.lower().startswith("c") or "C" in n]

    assert a_names, "No input parameter (A) found in function signature"
    assert c_names, "No output parameter (C) found in function signature"

    for name in a_names:
        assert sig_info[name]["has_const"], f"FAIL: input '{name}' should have const but doesn't: {sig_info[name]['declaration']}"
        print(f"  PASS: input '{name}' correctly const-qualified")

    for name in c_names:
        assert not sig_info[name]["has_const"], (
            f"FAIL: output '{name}' should be NON-const but has const: {sig_info[name]['declaration']}\n"
            f"This means pts_store_async's destination was NOT caught by AsyncLoadParamCollector."
        )
        print(f"  PASS: output '{name}' correctly NON-const (pts_store_async branch active)")

    # --- Check 2: end-to-end correctness ---
    a = torch.randn(M, N, dtype=torch.float16, device=device)
    out = elementwise_scale(a, block_M=64, block_N=64, threads=128, in_dtype="float16", out_dtype="float16")
    expected = ref_scale(a)
    if out.device.type == "ptpu":
        torch.ptpu.synchronize()
        out_cpu = out.cpu()
        expected_cpu = expected.cpu()
    else:
        out_cpu = out
        expected_cpu = expected

    # Use explicit max-diff check instead of torch.testing.assert_close so we
    # can print diagnostic values on failure.
    diff = (out_cpu.float() - expected_cpu.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    assert max_diff < 0.1, f"FAIL: max_diff={max_diff:.6f} exceeds 0.1 threshold\n  mean_diff={mean_diff:.6f}"
    print(f"  PASS: end-to-end correctness ({M}x{N}, max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f})")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
