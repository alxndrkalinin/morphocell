"""Tests for Richardson-Lucy deconvolution wrappers."""

import numpy as np

from cubic.preprocessing.deconvolution import richardson_lucy_iter


def test_richardson_lucy_iter() -> None:
    """Ensure both implementations run and preserve shape."""
    image = np.zeros((3, 3, 3), dtype=np.float32)
    image[1, 1, 1] = 1.0
    psf = np.ones((3, 3, 3), dtype=np.float32) / 27.0
    res_xp = richardson_lucy_iter(image, psf, n_iter=1, implementation="xp")
    res_sk = richardson_lucy_iter(image, psf, n_iter=1, implementation="skimage")
    assert res_xp.shape == image.shape
    assert res_sk.shape == image.shape
