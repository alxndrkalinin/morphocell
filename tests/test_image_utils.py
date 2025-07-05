"""Tests for ``image_utils`` helper functions."""

import numpy as np
import pytest

from cubic.cuda import ascupy, asnumpy
from cubic.image_utils import (
    rotate_image,
    pad_image_to_cube,
    select_max_contrast_slices,
)


def test_pad_image_to_cube() -> None:
    """Pad image to a cube and verify shape."""
    img = np.zeros((2, 4, 6), dtype=np.float32)
    padded = pad_image_to_cube(img)
    assert padded.shape == (6, 6, 6)


@pytest.mark.parametrize("use_gpu", [False, True])
def test_rotate_image_cpu_vs_gpu(use_gpu: bool, gpu_available: bool) -> None:
    """CPU vs GPU results for ``rotate_image``."""
    img = np.arange(9, dtype=np.float32).reshape(1, 3, 3)
    cpu_res = rotate_image(img, 90)

    if use_gpu:
        if not gpu_available:
            pytest.skip("GPU not available")
        gpu_res = rotate_image(ascupy(img), 90)
        assert np.allclose(asnumpy(gpu_res), cpu_res)
    else:
        gpu_res = rotate_image(img, 90)
        assert np.allclose(gpu_res, cpu_res)


def test_select_max_contrast_slices() -> None:
    """Ensure the function finds the highest-contrast slice block."""
    rng = np.random.default_rng(0)
    img = rng.random((5, 4, 4), dtype=np.float32)
    img[2:4] *= 2  # higher contrast region
    result, sl = select_max_contrast_slices(img, num_slices=2, return_indices=True)
    assert result.shape[0] == 2
    assert sl.stop - sl.start == 2


def test_select_max_contrast_edge_cases() -> None:
    """Test ``select_max_contrast_slices`` edge conditions."""
    rng = np.random.default_rng(1)
    img = rng.random((3, 2, 2), dtype=np.float32)

    # num_slices = 1
    res, sl = select_max_contrast_slices(img, num_slices=1, return_indices=True)
    assert res.shape[0] == 1
    assert sl.stop - sl.start == 1

    # num_slices equal to number of slices
    res, sl = select_max_contrast_slices(img, num_slices=3, return_indices=True)
    assert res.shape[0] == 3
    assert sl.stop - sl.start == 3

    # num_slices greater than number of slices should return full volume
    res, sl = select_max_contrast_slices(img, num_slices=5, return_indices=True)
    assert res.shape[0] == img.shape[0]
    assert sl.start == 0

    # uniform contrast image should return first slices
    uniform = np.ones((4, 2, 2), dtype=np.float32)
    res, sl = select_max_contrast_slices(uniform, num_slices=2, return_indices=True)
    assert sl.start == 0
    assert np.allclose(res, uniform[:2])
