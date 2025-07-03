"""Implement 3D image deconvolution using DeconvolutionLab2 or FlowDec."""

import subprocess
import warnings
import numpy as np
import numpy.typing as npt
from typing import Union, Tuple, Optional, Callable, List, Dict, Any
from pathlib import Path
from functools import partial

from skimage import io
from morphocell.skimage import restoration, util  # noqa: F401
import math

from morphocell.image_utils import pad_image
from morphocell.cuda import (
    RunAsCUDASubprocess,
    get_array_module,
    asnumpy,
    ascupy,
    to_same_device,
    check_same_device,
)

try:
    from flowdec import data as fd_data
    from flowdec import restoration as fd_restoration

    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False
    warnings.warn(
        "FlowDec / TensorFlow are not available. FlowDec deconvolution will not work."
    )

try:
    from pyvirtualdisplay import Display

    _IS_XVBF_AVAILABLE = True
except Exception:
    _IS_XVBF_AVAILABLE = False
    warnings.warn(
        "pyvirtualdisplay is not available. DeconvolutionLab2 deconvolution will not work."
    )


def check_tf_available():
    """Check if TensorFlow is available."""
    if not _TF_AVAILABLE:
        raise ImportError(
            "FlowDec / TensorFlow are required for this function, but not available. "
            "Try re-installing with `pip install morphocell[decon]`."
        )


def richardson_lucy_dl2(
    image: Union[str, Path, npt.ArrayLike],
    psf: Union[str, Path, npt.ArrayLike],
    n_iter: int = 10,
    dl2_path: str = "DeconvolutionLab2-0.1.0-SNAPSHOT-jar-with-dependencies.jar",
    tmp_dir: Union[Path, Optional[str]] = None,
    verbose: bool = False,
) -> Union[int, np.ndarray]:
    """Perform GPU-accelerated (optional) Lucy-Richardson deconvolution using DeconvolutionLab2."""
    verboseprint = print if verbose else lambda *a, **k: None

    if not _IS_XVBF_AVAILABLE:
        raise ImportError(
            "pyvirtualdisplay is required for this function, but not available. "
            "Try re-installing with `pip install morphocell[decon]`."
        )

    tmp_dir = Path(tmp_dir) if tmp_dir is not None else Path.cwd()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(image, np.ndarray):
        verboseprint(f"Saving original image to {tmp_dir}/tmp_image.tif")  # type: ignore[operator]
        io.imsave(str(tmp_dir / "tmp_image.tif"), image)
        image = tmp_dir / "tmp_image.tif"
    if isinstance(psf, np.ndarray):
        verboseprint(f"Saving PSF to {tmp_dir}/tmp_psf.tif")  # type: ignore[operator]
        io.imsave(str(tmp_dir / "tmp_psf.tif"), psf)
        psf = tmp_dir / "tmp_psf.tif"

    # run DeconvolutionLab2
    # create virtual display
    with Display(backend="xvfb", visible=False) as disp:
        print(disp)
        # display is active
        result = subprocess.run(
            [
                "java",
                "-Xmx24G",
                "-cp",
                f"{dl2_path}",
                "DeconvolutionLab2",
                "Run",
                "-image",
                "file",
                str(image),
                "-psf",
                "file",
                str(psf),
                "-algorithm",
                "RL",
                f"{n_iter}",
                "-out",
                "stack",
                f"tmp_image_RL_{n_iter}",
                "noshow",
                "-path",
                str(tmp_dir),
            ],
            capture_output=verbose,
        )
        verboseprint(result)  # type: ignore[operator]

    # read zstack again
    if result.returncode == 0:
        verboseprint(
            f"Reading deconvolved image from {str(tmp_dir)}/tmp_image_RL_{n_iter}.tif"
        )  # type: ignore[operator]
        decon_image = io.imread(str(tmp_dir / f"tmp_image_RL_{n_iter}.tif"))
        verboseprint(f"Deconvolved image shape: {decon_image.shape}")  # type: ignore[operator]
        return decon_image
    else:
        print("Deconvolution failed.")
        return -1


@RunAsCUDASubprocess(num_gpus=1, memory_fraction=0.8)
def _run_rl_flowdec_subprocess(image, psf, n_iter, start_mode, observer_fn, device):
    algo = fd_restoration.RichardsonLucyDeconvolver(
        image.ndim,
        pad_mode="none",
        start_mode=start_mode,
        observer_fn=observer_fn,
        device=device,
    ).initialize()
    data = fd_data.Acquisition(data=image, kernel=psf)
    return algo.run(data, niter=n_iter)


def _run_rl_flowdec(image, psf, n_iter, start_mode, observer_fn, device):
    algo = fd_restoration.RichardsonLucyDeconvolver(
        image.ndim,
        pad_mode="none",
        start_mode=start_mode,
        observer_fn=observer_fn,
        device=device,
    ).initialize()
    data = fd_data.Acquisition(data=image, kernel=psf)
    return algo.run(data, niter=n_iter)


def richardson_lucy_flowdec(
    image: Union[str, Path, npt.ArrayLike],
    psf: Union[str, Path, npt.ArrayLike],
    n_iter: int = 10,
    start_mode: str = "input",
    observer_fn: Optional[Callable] = None,
    device: Optional[str] = None,
    verbose: bool = False,
    subprocess_cuda: bool = False,
) -> np.ndarray:
    """Perform GPU-accelerated Lucy-Richardson deconvolution using FlowDec (TensorFlow)."""
    check_tf_available()
    verboseprint = print if verbose else lambda *a, **k: None

    if isinstance(image, (str, Path)):
        image = io.imread(str(image))
    else:
        image = np.asarray(image)
    if isinstance(psf, (str, Path)):
        psf = io.imread(str(psf))
    else:
        psf = np.asarray(psf)

    verboseprint(
        f"Deconvolving image shape {image.shape} with psf shape {psf.shape} for {n_iter} iterations."
    )  # type: ignore[operator]

    if subprocess_cuda:
        res = _run_rl_flowdec_subprocess(
            image, psf, n_iter, start_mode, observer_fn, device
        )
    else:
        res = _run_rl_flowdec(image, psf, n_iter, start_mode, observer_fn, device)

    verboseprint(f"\nDeconvolution info: {res.info}")  # type: ignore[operator]

    return res.data


def decon_flowdec(
    image: np.ndarray,
    psf: np.ndarray,
    n_iter: int = 1,
    pad_psf: bool = False,
    pad_size_z: int = 1,
    start_mode: str = "input",
    observer_fn: Optional[Callable] = None,
    device: Optional[str] = None,
    verbose: bool = False,
    subprocess_cuda: bool = False,
) -> np.ndarray:
    """Perform FlowDec deconvolution with image padding and rescaling."""
    check_tf_available()

    padded_img = pad_image(image, pad_size_z, mode="reflect")
    padded_psf = pad_image(psf, pad_size_z, mode="reflect") if pad_psf else psf

    # Create wrapper observer that handles postprocessing
    wrapper_observer = None
    if observer_fn is not None:
        def wrapper_observer(restored_image, i, *args):
            # Apply same postprocessing as final result
            processed_image = restored_image[pad_size_z : image.shape[0] + pad_size_z, :, :].astype(np.uint16)
            observer_fn(processed_image, i, *args)

    fl_decon_image = richardson_lucy_flowdec(
        padded_img,
        padded_psf,
        n_iter=n_iter,
        start_mode=start_mode,
        observer_fn=wrapper_observer,
        device=device,
        verbose=verbose,
        subprocess_cuda=subprocess_cuda,
    )

    fl_decon_image = fl_decon_image[pad_size_z : image.shape[0] + pad_size_z, :, :]

    return fl_decon_image.astype(np.uint16)


def richardson_lucy_skimage(
    image: npt.ArrayLike,
    psf: npt.ArrayLike,
    n_iter: int = 10,
    observer_fn: Optional[Callable] = None,
    clip: bool = True,
    filter_epsilon: Optional[float] = None,
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
    observer_fn: Optional[Callable] = None,
    clip: bool = True,
    filter_epsilon: Optional[float] = None,
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
            processed_image = restored_image[pad_size_z : image.shape[0] + pad_size_z, :, :]
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
    padded_size: Tuple[int, ...],
    mode: str,
    xp: Any,
) -> Tuple[np.ndarray, Tuple[Tuple[int, int], ...]]:
    """Pad array to ``padded_size`` using ``xp.pad``."""
    padding = tuple(
        (math.ceil((i - j) / 2), math.floor((i - j) / 2))
        for i, j in zip(padded_size, img.shape)
    )
    return xp.pad(img, padding, mode), padding


def _unpad_nd(padded: npt.ArrayLike, img_size: Tuple[int, ...], xp: Any) -> np.ndarray:
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
    mask: Optional[npt.ArrayLike] = None,
    observer_fn: Optional[Callable] = None,
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
        ext_size = [
            image.shape[i] + psf.shape[i] - 1 for i in range(image.ndim)
        ]
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
    mask: Optional[npt.ArrayLike] = None,
    observer_fn: Optional[Callable] = None,
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
            processed_image = restored_image[pad_size_z : image.shape[0] + pad_size_z, :, :]
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
    image: Union[str, Path, npt.ArrayLike],
    psf: Union[str, Path, npt.ArrayLike],
    metric_fn: Callable,
    metric_threshold: Union[int, float],
    metric_kwargs: Optional[Dict[str, Any]] = None,
    max_iter: int = 25,
    pad_size_z: int = 1,
    verbose: bool = False,
    subprocess_cuda: bool = False,
    implementation: str = "flowdec",
    noncirc: bool = False,
    use_gpu: bool = False,
) -> Tuple[int, List[Dict[str, Union[int, float, np.ndarray]]]]:
    """Find number of LR deconvolution iterations using an image similarity metric.

    Parameters
    ----------
    implementation : str, optional
        Which LR implementation to use, ``"flowdec"`` (default), ``"skimage"``,
        or ``"numpy"``.
    noncirc : bool, optional
        When ``implementation='numpy'``, enable non-circulant edge handling.
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
                metric_result[0]
                if isinstance(metric_result, tuple)
                else metric_result
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
    if implementation == "flowdec":
        _ = decon_flowdec(
            image,
            psf,
            n_iter=max_iter,
            observer_fn=observer,
            verbose=verbose,
            subprocess_cuda=subprocess_cuda,
            pad_size_z=pad_size_z,
        )
    elif implementation == "skimage":
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
