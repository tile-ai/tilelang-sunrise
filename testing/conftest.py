import importlib.util
import os
import random
import sys

import pytest

os.environ["PYTHONHASHSEED"] = "0"

# Source-tree developer runs should import the in-tree package.  Sunrise wheel
# CI opts out explicitly so the checkout cannot shadow the artifact under test.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SOURCE_PACKAGE_ROOT = os.path.join(REPO_ROOT, "tilelang")
if os.environ.get("TILELANG_TEST_INSTALLED_WHEEL") == "1":
    sys.path[:] = [entry for entry in sys.path if os.path.abspath(entry or os.getcwd()) != REPO_ROOT]
    tilelang_spec = importlib.util.find_spec("tilelang")
    if tilelang_spec is None or tilelang_spec.origin is None:
        raise RuntimeError("Sunrise CI could not resolve the installed TileLang wheel")
    if os.path.commonpath((SOURCE_PACKAGE_ROOT, os.path.abspath(tilelang_spec.origin))) == SOURCE_PACKAGE_ROOT:
        raise RuntimeError(f"Sunrise CI resolved TileLang from the checkout: {tilelang_spec.origin}")
elif REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _configure_torch_extensions_dir():
    cache_dir = os.environ.get("TILELANG_CACHE_DIR", os.path.expanduser("~/.tilelang/cache"))
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    path = os.path.join(cache_dir, "torch_extension", f"{worker}-{os.getpid()}")
    os.makedirs(path, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = path


_configure_torch_extensions_dir()

random.seed(0)

try:
    import torch
except ImportError:
    pass
else:
    torch.manual_seed(0)
    # Workaround: hipBLASLt on ROCm 7.1 nightly has a bug with certain matmul shapes
    if hasattr(torch.version, "hip") and torch.version.hip:
        torch.backends.cuda.preferred_blas_library("hipblas")

try:
    import numpy as np
except ImportError:
    pass
else:
    np.random.seed(0)


def pytest_addoption(parser):
    parser.addoption(
        "--run-perf",
        action="store_true",
        default=False,
        help="run performance and benchmark-oriented tests",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-perf"):
        config._perf_items_filtered = 0
        return

    perf_skip = pytest.mark.skip(reason="performance test skipped by default; pass --run-perf to include it")
    perf_items_filtered = 0
    for item in items:
        if item.get_closest_marker("perf") is not None:
            item.add_marker(perf_skip)
            perf_items_filtered += 1
    config._perf_items_filtered = perf_items_filtered


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Ensure that at least one test is collected. Error out if all tests are skipped."""
    known_types = {"failed", "passed", "skipped", "deselected", "xfailed", "xpassed", "warnings", "error"}
    executed_count = sum(len(terminalreporter.stats.get(k, [])) for k in known_types.difference({"skipped", "deselected"}))
    if executed_count == 0 and getattr(config, "_perf_items_filtered", 0) > 0:
        terminalreporter.write_sep(
            "-",
            f"Skipped {config._perf_items_filtered} perf test(s). Re-run with --run-perf to include them.",
        )
        return
    if executed_count == 0:
        terminalreporter.write_sep(
            "!",
            (f"Error: No tests were collected. {dict(sorted((k, len(v)) for k, v in terminalreporter.stats.items()))}"),
        )
        pytest.exit("No tests were collected.", returncode=5)
