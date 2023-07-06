"""Implements GPU-compatible metrics from scikit-image."""

from typing import Optional
import numpy.typing as npt

from .gpu import get_device, get_image_method


def psnr(image_true: npt.ArrayLike, image_test: npt.ArrayLike, data_range: Optional[int] = None):
    """Compute the peak signal to noise ratio (PSNR) for an image."""
    assert get_device(image_true) == get_device(image_test), "Images must be on same device."
    skimage_rpsnr = get_image_method(image_true, "skimage.metrics.peak_signal_noise_ratio")
    return skimage_rpsnr(image_true, image_test, data_range=data_range)


def ssim(
    im1: npt.ArrayLike,
    im2: npt.ArrayLike,
    win_size: Optional[int] = None,
    gradient: Optional[bool] = False,
    data_range: Optional[float] = None,
    channel_axis: Optional[int] = None,
    gaussian_weights: Optional[bool] = False,
    full: Optional[bool] = False,
    **kwargs
):
    """Compute the mean structural similarity index between two images."""
    assert get_device(im1) == get_device(im2), "Images must be on same device."
    skimage_rssim = get_image_method(im1, "skimage.metrics.structural_similarity")
    return skimage_rssim(
        im1,
        im2,
        win_size=win_size,
        gradient=gradient,
        data_range=data_range,
        channel_axis=channel_axis,
        gaussian_weights=gaussian_weights,
        full=full,
        **kwargs,
    )
