"""Tests for image quality metrics."""

import numpy as np

from cubic.metrics.skimage_metrics import psnr, ssim, nrmse


def test_nrmse_scale_invariant() -> None:
    """Verify scale invariant NRMSE yields zero for scaled images."""
    a = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    b = 2 * a
    err = nrmse(a, b, scale_invariant=True)
    assert np.isclose(err, 0.0)


def test_nrmse_non_scale_invariant() -> None:
    """Verify non-scale-invariant NRMSE is not zero for scaled images."""
    a = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    b = 2 * a
    err = nrmse(a, b, scale_invariant=False)
    assert not np.isclose(err, 0.0)


def test_nrmse_identical_arrays() -> None:
    """NRMSE should be zero for identical arrays."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    b = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    err = nrmse(a, b)
    assert np.isclose(err, 0.0)


def test_nrmse_completely_different_arrays() -> None:
    """NRMSE should be maximal for completely different arrays (ones vs zeros)."""
    a = np.zeros((2, 2), dtype=float)
    b = np.ones((2, 2), dtype=float)
    err = nrmse(a, b)
    # With zero data range the metric returns infinity
    assert np.isinf(err)


def test_nrmse_with_nans() -> None:
    """NRMSE should return nan or raise when arrays contain NaNs."""
    a = np.array([[0.0, np.nan], [1.0, 0.0]], dtype=float)
    b = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    err = nrmse(a, b)
    assert np.isnan(err)


def test_nrmse_with_infs() -> None:
    """NRMSE should return inf or nan when arrays contain Infs."""
    a = np.array([[0.0, np.inf], [1.0, 0.0]], dtype=float)
    b = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    err = nrmse(a, b)
    assert np.isinf(err) or np.isnan(err)


def test_psnr_and_ssim() -> None:
    """PSNR and SSIM should be ideal for identical images."""
    a = np.zeros((4, 4), dtype=float)
    b = np.zeros_like(a)
    assert psnr(a, b, data_range=1.0) == float("inf")
    ssim_val = ssim(a, b, data_range=1.0, win_size=3)
    assert np.isclose(ssim_val, 1.0)


def test_psnr_ssim_non_identical() -> None:
    """PSNR and SSIM for completely different images."""
    a = np.zeros((4, 4), dtype=float)
    b = np.ones_like(a)
    assert np.isclose(psnr(a, b, data_range=1.0), 0.0)
    assert ssim(a, b, data_range=1.0, win_size=3) < 1.0


def test_psnr_ssim_constant_images() -> None:
    """Constant images should yield perfect scores."""
    a = np.full((4, 4), 5.0)
    b = np.full_like(a, 5.0)
    assert psnr(a, b, data_range=10.0) == float("inf")
    assert np.isclose(ssim(a, b, data_range=10.0, win_size=3), 1.0)


def test_psnr_ssim_with_nans() -> None:
    """NaNs should propagate through the metrics."""
    a = np.zeros((8, 8), dtype=float)
    b = a.copy()
    b[0, 0] = np.nan
    assert np.isnan(psnr(a, b, data_range=1.0))
    assert np.isnan(ssim(a, b, data_range=1.0))


def test_psnr_ssim_varying_range() -> None:
    """Larger data ranges should yield higher PSNR."""
    a = np.zeros((2, 2), dtype=float)
    b = np.ones_like(a)
    small = psnr(a, b, data_range=1.0)
    large = psnr(a, b, data_range=255.0)
    assert large > small
