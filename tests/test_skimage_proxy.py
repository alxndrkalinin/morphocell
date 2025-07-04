"""Tests for the skimage proxy module."""

import numpy as np
import pytest

import morphocell.skimage as mc_skimage
from morphocell.cuda import CUDAManager, ascupy, asnumpy


@pytest.mark.parametrize("use_gpu", [False, True])
def test_filters_gaussian_cpu_vs_gpu(use_gpu: bool, gpu_available: bool) -> None:
    """Compare Gaussian filtering on CPU and GPU."""
    img = np.random.random((5, 5)).astype(np.float32)
    cpu_res = mc_skimage.filters.gaussian(img, sigma=1.0, preserve_range=True)

    if use_gpu:
        if not gpu_available:
            pytest.skip("GPU not available")
        cp = CUDAManager().get_cp()
        gpu_res = mc_skimage.filters.gaussian(
            cp.asarray(img), sigma=1.0, preserve_range=True
        )
        assert np.allclose(asnumpy(gpu_res), cpu_res, atol=1e-6)
    else:
        gpu_res = mc_skimage.filters.gaussian(img, sigma=1.0, preserve_range=True)
        assert np.allclose(gpu_res, cpu_res)
