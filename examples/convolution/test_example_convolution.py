import pytest
import tilelang.testing
from tilelang.utils.device import is_ptpu_available

import example_convolution
import example_convolution_autotune


# TODO(@cy): TMA with convolution must be fixed in future.
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_le(8, 9)
def test_example_convolution():
    example_convolution.main([])


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_le(8, 9)
def test_example_convolution_autotune():
    example_convolution_autotune.main()


@pytest.mark.skipif(not is_ptpu_available(), reason="Tang target required")
def test_example_convolution_tang():
    example_convolution.main([])


@pytest.mark.skipif(not is_ptpu_available(), reason="Tang target required")
def test_example_convolution_autotune_tang():
    example_convolution_autotune.main()


if __name__ == "__main__":
    tilelang.testing.main()
