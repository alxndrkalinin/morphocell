"""Shared pytest fixtures for the test suite."""

import pytest

from cubic.cuda import CUDAManager


@pytest.fixture(scope="session")
def gpu_available() -> bool:
    """Return True if a CUDA GPU is available."""
    return CUDAManager().get_num_gpus() > 0
