"""Tests for CUDA helper utilities."""

import numpy as np
import pytest

from cubic.cuda import (
    ascupy,
    asnumpy,
    to_device,
    check_same_device,
)


@pytest.mark.parametrize("device", ["CPU", "GPU"])
def test_to_device_roundtrip(device: str, gpu_available: bool) -> None:
    """Move array to the specified device and verify roundtrip."""
    if device == "GPU" and not gpu_available:
        pytest.skip("GPU not available")

    arr = np.ones((2, 2), dtype=np.float32)
    res = to_device(arr, device)
    if device == "GPU":
        assert np.allclose(asnumpy(res), arr)
    else:
        assert np.allclose(res, arr)


def test_to_device_invalid_device_raises() -> None:
    """Ensure to_device raises ``ValueError`` for an invalid device string."""
    arr = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        to_device(arr, "INVALID_DEVICE")


def test_check_same_device(gpu_available: bool) -> None:
    """Ensure mismatched devices trigger an error when GPU is present."""
    arr = np.ones((2, 2), dtype=np.float32)
    if gpu_available:
        gpu_arr = ascupy(arr)
        with pytest.raises(ValueError):
            check_same_device(arr, gpu_arr)
    else:
        check_same_device(arr, arr)
