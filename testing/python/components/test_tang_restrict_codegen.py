import tilelang
import tilelang.language as T
import tilelang.testing


def _get_global_void_lines(code: str) -> list[str]:
    # Collect every `extern "C" __global__ void` line in generated TANG code.
    # TANG emits the kernel twice: a forward declaration (ending in `;`) and the
    # `__launch_bounds__(...)` definition (ending in `{`).  The restrict bug this
    # test guards against was that the *definition* kept `__restrict__` even when
    # the forward declaration dropped it, so we must check both, not just the
    # first match.
    lines = [line.strip() for line in code.splitlines() if line.strip().startswith('extern "C" __global__ void')]
    assert lines, "Kernel signature not found in generated code"
    return lines


def test_tang_restrict_default_has_restrict():
    N = 128

    @T.prim_func
    def kernel(x: T.Tensor((N,), T.float32), y: T.Tensor((N,), T.float32)):
        with T.Kernel(N, threads=32) as pid:
            y[pid] = x[pid] + 1.0

    artifact = tilelang.lower(kernel, target="tang")
    sigs = _get_global_void_lines(artifact.kernel_source)
    # By default, kNoAlias is set and both pointers are restrict-qualified in
    # BOTH the forward declaration and the definition.
    assert all("__restrict__" in sig for sig in sigs), sigs


def test_tang_restrict_annotation_removes_restrict():
    N = 128

    @T.prim_func
    def kernel_body_annot(x: T.Tensor((N,), T.float32), y: T.Tensor((N,), T.float32)):
        # Explicitly mark buffers that may alias as non-restrict
        with T.Kernel(N, threads=32) as pid:
            T.annotate_restrict_buffers(x, y)
            y[pid] = x[pid] + 1.0

    art1 = tilelang.lower(kernel_body_annot, target="tang")
    sigs1 = _get_global_void_lines(art1.kernel_source)
    # No parameter should be emitted with __restrict__ -- in either the forward
    # declaration or the definition.
    assert all("__restrict__" not in sig for sig in sigs1), sigs1


if __name__ == "__main__":
    tilelang.testing.main()
