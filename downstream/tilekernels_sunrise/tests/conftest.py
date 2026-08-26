# Root-level conftest
#
# Loads the benchmark plugin (CLI options, markers, fixtures).
# The plugin lives in a file deliberately NOT named conftest.py to
# avoid pluggy's duplicate-registration error.
import pytest
import torch

pytest_plugins = [
    'tests.pytest_random_plugin',
    'tests.pytest_benchmark_plugin',
]


@pytest.fixture(autouse=True)
def stream_setup() -> None:
    import tvm_ffi
    strm = torch.ptpu.current_stream()
    tvm_ffi.core._env_set_current_stream(20, strm.device.index, strm.ptpu_stream)
    print('setting current ptpu stream...')
