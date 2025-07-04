"""Utility functions for 3D image deconvolution."""

from functools import partial
from pathlib import Path
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from skimage import io
from morphocell.skimage import restoration, util  # noqa: F401
import math

from morphocell.image_utils import pad_image
from morphocell.cuda import (
    asnumpy,
    ascupy,
    check_same_device,
    get_array_module,
    to_same_device,
)


def richardson_lucy_skimage(
    image: npt.ArrayLike,
    psf: npt.ArrayLike,
    n_iter: int = 10,
    observer_fn: Callable | None = None,
    clip: bool = True,
    filter_epsilon: float | None = None,
) -> np.ndarray:
    """Lucy-Richardson deconvolution using :mod:`morphocell.skimage`."""

    rl_partial = partial(
        restoration.richardson_lucy,
        psf=psf,
        clip=clip,
        filter_epsilon=filter_epsilon,
    )

    if observer_fn is None:
        return rl_partial(image, num_iter=n_iter)

    estimate = image.copy()
    for i in range(1, n_iter + 1):
        estimate = rl_partial(estimate, num_iter=1)
        observer_fn(estimate, i)

    return estimate


def decon_skimage(
    image: np.ndarray,
    psf: np.ndarray,
    n_iter: int = 1,
    pad_psf: bool = False,
    pad_size_z: int = 0,
    observer_fn: Callable | None = None,
    clip: bool = True,
    filter_epsilon: float | None = None,
) -> np.ndarray:
    """Perform scikit-image deconvolution with image padding."""
    check_same_device(image, psf)

    padded_img = pad_image(image, pad_size_z, mode="reflect")
    padded_psf = pad_image(psf, pad_size_z, mode="reflect") if pad_psf else psf

    # Create wrapper observer that handles postprocessing
    wrapper_observer = None
    if observer_fn is not None:

        def wrapper_observer(restored_image, i, *args):
            # Apply same postprocessing as final result
            processed_image = restored_image[
                pad_size_z : image.shape[0] + pad_size_z, :, :
            ]
            observer_fn(processed_image, i, *args)

    decon_image = richardson_lucy_skimage(
        padded_img,
        padded_psf,
        n_iter=n_iter,
        observer_fn=wrapper_observer,
        clip=clip,
        filter_epsilon=filter_epsilon,
    )

    decon_image = decon_image[pad_size_z : image.shape[0] + pad_size_z, :, :]
    return decon_image


def _pad_nd(
    img: npt.ArrayLike,
    padded_size: tuple[int, ...],
    mode: str,
    xp: Any,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    """Pad array to ``padded_size`` using ``xp.pad``."""
    padding = tuple(
        (math.ceil((i - j) / 2), math.floor((i - j) / 2))
        for i, j in zip(padded_size, img.shape)
    )
    return xp.pad(img, padding, mode), padding


def _unpad_nd(padded: npt.ArrayLike, img_size: tuple[int, ...], xp: Any) -> np.ndarray:
    """Crop padded array back to ``img_size``."""
    padding = tuple(
        (math.ceil((i - j) / 2), math.floor((i - j) / 2))
        for i, j in zip(padded.shape, img_size)
    )
    slices = tuple(slice(p[0], p[0] + s) for p, s in zip(padding, img_size))
    return padded[slices]


def richardson_lucy_xp(
    image: npt.ArrayLike,
    psf: npt.ArrayLike,
    n_iter: int = 10,
    *,
    noncirc: bool = False,
    mask: npt.ArrayLike | None = None,
    observer_fn: Callable | None = None,
) -> np.ndarray:
    """Lucy-Richardson deconvolution implemented with NumPy or CuPy."""
    xp = get_array_module(image)

    image = util.img_as_float(image)
    psf = util.img_as_float(psf)

    if not noncirc and image.shape != psf.shape:
        psf, _ = _pad_nd(psf, image.shape, "constant", xp)

    mask_values = None
    if mask is not None:
        mask = util.img_as_float(mask)
        mask_values = image * (1 - mask)
        image *= mask

    if noncirc:
        orig_size = image.shape
        ext_size = [image.shape[i] + psf.shape[i] - 1 for i in range(image.ndim)]
        psf, _ = _pad_nd(psf, ext_size, "constant", xp)

    psf = xp.fft.fftn(xp.fft.ifftshift(psf))
    otf_conj = xp.conjugate(psf)

    if noncirc:
        image, _ = _pad_nd(image, ext_size, "constant", xp)
        estimate = xp.full_like(image, xp.mean(image))
    else:
        estimate = image

    if mask is not None:
        htones = xp.ones_like(image) * mask
    else:
        htones = xp.ones_like(image)

    htones = xp.real(xp.fft.ifftn(xp.fft.fftn(htones) * otf_conj))
    htones[htones < 1e-6] = 1

    for i in range(1, n_iter + 1):
        reblurred = xp.real(xp.fft.ifftn(xp.fft.fftn(estimate) * psf))
        ratio = image / (reblurred + 1e-6)
        correction = xp.real(xp.fft.ifftn(xp.fft.fftn(ratio) * otf_conj))

        correction[correction < 0] = 0
        estimate = estimate * correction / htones

        if observer_fn is not None:
            if noncirc:
                unpadded_estimate = _unpad_nd(estimate, orig_size, xp)
                observer_fn(unpadded_estimate, i)
            else:
                observer_fn(estimate, i)

    del psf, otf_conj, htones

    if noncirc:
        estimate = _unpad_nd(estimate, orig_size, xp)

    if mask is not None:
        estimate = estimate * mask + mask_values

    return estimate


def decon_xpy(
    image: np.ndarray,
    psf: np.ndarray,
    n_iter: int = 1,
    pad_psf: bool = False,
    pad_size_z: int = 0,
    *,
    noncirc: bool = False,
    mask: npt.ArrayLike | None = None,
    observer_fn: Callable | None = None,
) -> np.ndarray:
    """Perform NumPy-based deconvolution with optional non-circulant edges."""
    check_same_device(image, psf)

    padded_img = pad_image(image, pad_size_z, mode="reflect")
    padded_psf = pad_image(psf, pad_size_z, mode="reflect") if pad_psf else psf

    # Create wrapper observer that handles postprocessing
    wrapper_observer = None
    if observer_fn is not None:

        def wrapper_observer(restored_image, i, *args):
            # Apply same postprocessing as final result
            processed_image = restored_image[
                pad_size_z : image.shape[0] + pad_size_z, :, :
            ]
            observer_fn(processed_image, i, *args)

    decon_image = richardson_lucy_xp(
        padded_img,
        padded_psf,
        n_iter=n_iter,
        noncirc=noncirc,
        mask=mask,
        observer_fn=wrapper_observer,
    )

    decon_image = decon_image[pad_size_z : image.shape[0] + pad_size_z, :, :]
    return decon_image


def deconv_iter_num_finder(
    image: str | Path | npt.ArrayLike,
    psf: str | Path | npt.ArrayLike,
    metric_fn: Callable,
    metric_threshold: int | float,
    metric_kwargs: dict[str, Any] | None = None,
    max_iter: int = 25,
    pad_size_z: int = 1,
    verbose: bool = False,
    subprocess_cuda: bool = False,
    implementation: str = "skimage",
    noncirc: bool = False,
    use_gpu: bool = False,
) -> tuple[int, list[dict[str, int | float | np.ndarray]]]:
    """Find number of LR deconvolution iterations using an image similarity metric.

    Parameters
    ----------
    implementation : str, optional
        Which LR implementation to use, ``"skimage"`` (default) or ``"xpy"``.
    noncirc : bool, optional
        When ``implementation='xpy'``, enable non-circulant edge handling.
    """
    verboseprint = print if verbose else lambda *a, **k: None

    if isinstance(image, (str, Path)):
        image = io.imread(str(image))
    if isinstance(psf, (str, Path)):
        psf = io.imread(str(psf))

    if use_gpu:
        image = ascupy(image)
        psf = ascupy(psf)

    image = util.img_as_float(image)
    psf = util.img_as_float(psf)

    if metric_kwargs is None:
        metric_kwargs = {}

    thresh_iter = 0
    results = [{"metric_gain": metric_threshold, "iter_image": asnumpy(image)}]

    def get_decon_observer(metric_fn, metric_kwargs):
        nonlocal thresh_iter

        def decon_observer(restored_image, i, *args):
            nonlocal thresh_iter

            if thresh_iter > 0:
                return

            # No postprocessing needed - deconvolution functions handle it
            curr_image = restored_image

            prev_iter_image = to_same_device(results[-1]["iter_image"], curr_image)
            metric_result = metric_fn(prev_iter_image, curr_image, **metric_kwargs)

            metric_gain = (
                metric_result[0] if isinstance(metric_result, tuple) else metric_result
            )

            results.append(
                {
                    "metric_gain": float(metric_gain),
                    "iter_image": asnumpy(curr_image),
                    "metric_result": metric_result,
                }
            )
            verboseprint(f"Iteration {i}: improvement {metric_gain:.8f}")

            if (i > 1) and (metric_gain > metric_threshold):
                thresh_iter = i
                metric_gain_total = metric_fn(
                    to_same_device(results[0]["iter_image"], curr_image),
                    curr_image,
                    **metric_kwargs,
                )
                if isinstance(metric_gain_total, tuple):
                    metric_gain_total = metric_gain_total[0]

                verboseprint(
                    f"\nThreshold {metric_threshold} reached at iteration {i}"
                    f" with improvement: {metric_gain:.8f}.\n"
                    f"Metric between original and restored images: {metric_gain_total:.8f}.\n"
                )

        return decon_observer

    observer = get_decon_observer(metric_fn=metric_fn, metric_kwargs=metric_kwargs)
    if implementation == "skimage":
        _ = decon_skimage(
            image,
            psf,
            n_iter=max_iter,
            observer_fn=observer,
            pad_size_z=pad_size_z,
        )
    elif implementation == "xpy":
        _ = decon_xpy(
            image,
            psf,
            n_iter=max_iter,
            noncirc=noncirc,
            observer_fn=observer,
            pad_size_z=pad_size_z,
        )
    else:
        raise ValueError(f"Unknown implementation: {implementation}")

    for i, result in enumerate(results):
        results[i]["iter_image"] = asnumpy(result["iter_image"])

    return (thresh_iter, results)
