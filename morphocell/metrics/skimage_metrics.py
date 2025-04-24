"""Implements GPU-compatible metrics from scikit-image."""

from typing import Optional
from functools import wraps

import numpy.typing as npt

from ..cuda import get_device
from ..skimage import metrics


def scale_invariant(fn):
    """Decorate a function to make it scale invariant."""

    @wraps(fn)
    def wrapped(image_true, image_test, *args, scale_invariant=False, **kwargs):
        """Transform input images to be scale invariant."""
        assert get_device(image_true) == get_device(image_test)
        if not scale_invariant:
            return fn(image_true, image_test, *args, **kwargs)

        gt_zero = image_true - image_true.mean()
        gt_norm = gt_zero / image_true.std()

        pred_zero = image_test - image_test.mean()
        alpha = (gt_norm * pred_zero).sum() / (pred_zero * pred_zero).sum()

        pred_scaled = pred_zero * alpha
        range_param = (image_true.max() - image_true.min()) / image_true.std()

        return fn(gt_norm, pred_scaled, *args, **{**kwargs, "data_range": range_param})

    return wrapped


@scale_invariant
def nrmse(
    image_true: npt.ArrayLike,
    image_test: npt.ArrayLike,
    normalization: Optional[str] = None,
    data_range: Optional[float] = None,
):
    """Compute the normalized root mean squared error (NRMSE) between two images."""
    assert get_device(image_true) == get_device(image_test), "Images must be on same device."

    if data_range is not None:
        mse = metrics.mean_squared_error(image_true, image_test)
        return (mse**0.5) / data_range
    elif normalization is not None:
        return metrics.normalized_root_mse(image_true, image_test, normalization=normalization)
    else:
        return metrics.normalized_root_mse(image_true, image_test)


@scale_invariant
def psnr(image_true: npt.ArrayLike, image_test: npt.ArrayLike, data_range: Optional[int] = None):
    """Compute the peak signal to noise ratio (PSNR) between two images."""
    assert get_device(image_true) == get_device(image_test), "Images must be on same device."
    return metrics.peak_signal_noise_ratio(image_true, image_test, data_range=data_range)


@scale_invariant
def ssim(
    im1: npt.ArrayLike,
    im2: npt.ArrayLike,
    win_size: Optional[int] = None,
    gradient: Optional[bool] = False,
    data_range: Optional[float] = None,
    channel_axis: Optional[int] = None,
    gaussian_weights: Optional[bool] = False,
    full: Optional[bool] = False,
    **kwargs,
):
    """Compute the mean structural similarity index between two images."""
    assert get_device(im1) == get_device(im2), "Images must be on same device."
    return metrics.structural_similarity(
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
