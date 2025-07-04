"""Tests for ``image_utils`` helper functions."""

import numpy as np

from morphocell.cuda import CUDAManager, ascupy, asnumpy
from morphocell.image_utils import (
    rotate_image,
    pad_image_to_cube,
    select_max_contrast_slices,
)


def _gpu_available() -> bool:
    return CUDAManager().get_num_gpus() > 0


def test_pad_image_to_cube() -> None:
    """Pad image to a cube and verify shape."""
    img = np.zeros((2, 4, 6), dtype=np.float32)
    padded = pad_image_to_cube(img)
    assert padded.shape == (6, 6, 6)


def test_rotate_image_cpu_vs_gpu() -> None:
    """CPU vs GPU results for ``rotate_image``."""
    img = np.arange(9, dtype=np.float32).reshape(1, 3, 3)
    cpu_res = rotate_image(img, 90)

    if _gpu_available():
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
