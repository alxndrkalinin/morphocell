"""Tests for CUDA helper utilities."""

import numpy as np
import pytest

from morphocell.cuda import (
    CUDAManager,
    ascupy,
    asnumpy,
    to_device,
    check_same_device,
)


def _gpu_available() -> bool:
    return CUDAManager().get_num_gpus() > 0


def test_to_device_roundtrip() -> None:
    """Move array to GPU and back if available."""
    arr = np.ones((2, 2), dtype=np.float32)
    cpu_arr = to_device(arr, "CPU")
    assert isinstance(cpu_arr, np.ndarray)

    if _gpu_available():
        gpu_arr = to_device(arr, "GPU")
        assert np.allclose(asnumpy(gpu_arr), arr)
    else:
        gpu_arr = to_device(arr, "CPU")
        assert np.allclose(gpu_arr, arr)


def test_check_same_device() -> None:
    """Ensure mismatched devices trigger an error."""
    arr = np.ones((2, 2), dtype=np.float32)
    if _gpu_available():
        gpu_arr = ascupy(arr)
        with pytest.raises(ValueError):
            check_same_device(arr, gpu_arr)
    else:
        check_same_device(arr, arr)
