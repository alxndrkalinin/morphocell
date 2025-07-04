"""Tests for image quality metrics."""

import numpy as np

from morphocell.metrics.skimage_metrics import psnr, ssim, nrmse


def test_nrmse_scale_invariant() -> None:
    """Verify scale invariant NRMSE yields zero for scaled images."""
    a = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    b = 2 * a
    err = nrmse(a, b, scale_invariant=True)
    assert np.isclose(err, 0.0)


def test_psnr_and_ssim() -> None:
    """PSNR and SSIM should be ideal for identical images."""
    a = np.zeros((4, 4), dtype=float)
    b = np.zeros_like(a)
    assert psnr(a, b, data_range=1.0) == float("inf")
    ssim_val = ssim(a, b, data_range=1.0, win_size=3)
    assert np.isclose(ssim_val, 1.0)
