"""Implements GPU-compatible metrics from scikit-image."""

from typing import Any
from functools import wraps
from collections.abc import Callable

import numpy as np

from ..cuda import check_same_device
from ..skimage import metrics


def scale_invariant(fn: Callable) -> Callable:
    """Decorate a function to make it scale invariant."""

    @wraps(fn)
    def wrapped(
        image_true: np.ndarray,
        image_test: np.ndarray,
        *args: Any,
        scale_invariant: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Transform input images to be scale invariant."""
        check_same_device(image_true, image_test)
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
    image_true: np.ndarray,
    image_test: np.ndarray,
    normalization: str | None = None,
    data_range: float | None = None,
):
    """Compute the normalized root mean squared error (NRMSE) between two images."""
    if data_range is not None:
        mse = metrics.mean_squared_error(image_true, image_test)
        return (mse**0.5) / data_range
    elif normalization is not None:
        return metrics.normalized_root_mse(
            image_true, image_test, normalization=normalization
        )
    else:
        return metrics.normalized_root_mse(image_true, image_test)


@scale_invariant
def psnr(
    image_true: np.ndarray,
    image_test: np.ndarray,
    data_range: int | None = None,
):
    """Compute the peak signal to noise ratio (PSNR) between two images."""
    return metrics.peak_signal_noise_ratio(
        image_true, image_test, data_range=data_range
    )


@scale_invariant
def ssim(
    im1: np.ndarray,
    im2: np.ndarray,
    win_size: int | None = None,
    gradient: bool | None = False,
    data_range: float | None = None,
    channel_axis: int | None = None,
    gaussian_weights: bool | None = False,
    full: bool | None = False,
    **kwargs,
):
    """Compute the mean structural similarity index between two images."""
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
