"""Tests for the SciPy proxy implementation."""

import numpy as np
import pytest

import morphocell.scipy as mc_scipy
from morphocell.cuda import ascupy, asnumpy


@pytest.mark.parametrize("use_gpu", [False, True])
def test_ndimage_laplace_cpu_vs_gpu(use_gpu: bool, gpu_available: bool) -> None:
    """Compare Laplace filter on CPU and GPU."""
    a = np.asarray([1, 2, 3], dtype=float)
    cpu_res = mc_scipy.ndimage.laplace(a)

    if use_gpu:
        if not gpu_available:
            pytest.skip("GPU not available")
        gpu_res = mc_scipy.ndimage.laplace(ascupy(a))
        assert np.allclose(cpu_res, asnumpy(gpu_res))
    else:
        gpu_res = mc_scipy.ndimage.laplace(a)
        assert np.allclose(cpu_res, gpu_res)
