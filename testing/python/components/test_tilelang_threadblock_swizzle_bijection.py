"""Exhaustive check that rasterization2DRow is a valid grid permutation.

`rasterization2DRow<panel_width>()` remaps (blockIdx.x, blockIdx.y) for L2-friendly
threadblock ordering. It must be a *bijection* onto the grid: every (col, row) it
returns has to be in range and hit exactly once. A wrong formula that lets row_idx
run past gridDim.y makes a block write its C tile out of bounds, which in this
codebase has previously landed on the input B and silently corrupted it — a failure
mode that a single-shape "does it produce the right answer" check can miss entirely.

The formula is duplicated across the cuda / hip / tang backends and has been
simplified more than once to shave integer instructions (removing runtime
divisions by exploiting gx as a common factor, and specializing panel_width == 1).
Each simplification is an opportunity to get an off-by-one in the "is this the
last, possibly short, panel" predicate. The short final panel only exists when
gridDim.y is not a multiple of panel_width, so it is easy to miss by testing only
power-of-two shapes.

Rather than re-implement the formula in Python (which would test the model, not the
code), this extracts each backend's function from its header, compiles it as host
C++ with gridDim/blockIdx stubbed, and sweeps every block of every grid in the
range.
"""

import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

import tilelang.testing

REPO = Path(__file__).resolve().parents[3]
BACKENDS = ("cuda", "hip", "tang")
PANEL_WIDTHS = (1, 2, 3, 4, 5, 8, 16)
MAX_DIM = 24

_HARNESS = textwrap.dedent(r"""
    #include <cstdio>
    struct Dim { unsigned int x, y, z; };
    static Dim gridDim, blockIdx;
    struct dim3 { unsigned int x, y, z; };
    #define TL_DEVICE inline
    namespace tl {
    inline unsigned int ceil_div(unsigned int a, unsigned int b) {
      return (a + b - 1) / b;
    }
    __IMPL__
    }  // namespace tl

    template <int PW> int check(unsigned int max_dim) {
      int bad = 0;
      static bool seen[64 * 64];
      for (unsigned int gx = 1; gx <= max_dim; gx++) {
        for (unsigned int gy = 1; gy <= max_dim; gy++) {
          gridDim = {gx, gy, 1};
          for (unsigned int i = 0; i < gx * gy; i++) seen[i] = false;
          for (unsigned int by = 0; by < gy; by++) {
            for (unsigned int bx = 0; bx < gx; bx++) {
              blockIdx = {bx, by, 0};
              dim3 g = tl::rasterization2DRow<PW>();
              if (g.x >= gx || g.y >= gy) {
                if (bad < 4)
                  printf("OOB pw=%d grid=(%u,%u) blk=(%u,%u) -> (%u,%u)\n",
                         PW, gx, gy, bx, by, g.x, g.y);
                bad++;
                continue;
              }
              unsigned int slot = g.y * gx + g.x;
              if (seen[slot]) {
                if (bad < 4)
                  printf("COLLISION pw=%d grid=(%u,%u) blk=(%u,%u) -> (%u,%u)\n",
                         PW, gx, gy, bx, by, g.x, g.y);
                bad++;
              }
              seen[slot] = true;
            }
          }
        }
      }
      return bad;
    }

    int main() {
      int bad = 0;
    __CALLS__
      printf(bad ? "FAIL %d\n" : "OK\n", bad);
      return bad ? 1 : 0;
    }
""")


def _extract_fn(header_text, fn="rasterization2DRow"):
    """Pull one template function, brace-matched, out of a header."""
    m = re.search(r"template <int panel_width> TL_DEVICE dim3 " + fn + r"\(\) \{", header_text)
    assert m, f"{fn} not found"
    start = m.start()
    i = header_text.index("{", start)
    depth = 0
    for j in range(i, len(header_text)):
        if header_text[j] == "{":
            depth += 1
        elif header_text[j] == "}":
            depth -= 1
            if depth == 0:
                return header_text[start : j + 1]
    raise AssertionError("unbalanced braces")


@pytest.mark.skipif(shutil.which("g++") is None, reason="needs a host C++ compiler")
@pytest.mark.parametrize("backend", BACKENDS)
def test_rasterization2drow_is_a_grid_bijection(backend, tmp_path):
    header = REPO / "src" / "tl_templates" / backend / "threadblock_swizzle.h"
    impl = _extract_fn(header.read_text())
    calls = "\n".join(f"  bad += check<{pw}>({MAX_DIM});" for pw in PANEL_WIDTHS)
    source = _HARNESS.replace("__IMPL__", impl).replace("__CALLS__", calls)

    cpp = tmp_path / f"raster_{backend}.cpp"
    cpp.write_text(source)
    exe = tmp_path / f"raster_{backend}"
    build = subprocess.run(["g++", "-std=c++17", "-O1", "-o", str(exe), str(cpp)], capture_output=True, text=True)
    assert build.returncode == 0, f"harness build failed:\n{build.stderr[-2000:]}"

    run = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run.returncode == 0, (
        f"{backend} rasterization2DRow is not a valid grid permutation (out-of-range or duplicated block mapping):\n{run.stdout[-2000:]}"
    )


if __name__ == "__main__":
    tilelang.testing.main()
