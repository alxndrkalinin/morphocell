"""Implement 3D image deconvolution using DeconvolutionLab2 or FlowDec."""

import subprocess
import warnings
import numpy as np
import numpy.typing as npt
from typing import Union, Tuple, Optional, Callable, List, Dict, Any
from pathlib import Path

from skimage import io

from ..image_utils import pad_image
from ..cuda import RunAsCUDASubprocess

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
    image: Union[str, npt.ArrayLike],
    psf: Union[str, npt.ArrayLike],
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
    image: Union[str, npt.ArrayLike],
    psf: Union[str, npt.ArrayLike],
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

    image = image if isinstance(image, np.ndarray) else io.imread(image)
    psf = psf if isinstance(psf, np.ndarray) else io.imread(psf)

    # assert image.shape == psf.shape
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
    image: Union[str, npt.ArrayLike],
    psf: Union[str, npt.ArrayLike],
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

    if isinstance(image, str):
        image = io.imread(str(image))
    if isinstance(psf, str):
        psf = io.imread(str(psf))

    # assert image.shape == psf.shape
    padded_img = pad_image(image, pad_size_z, mode="reflect")
    padded_psf = pad_image(psf, pad_size_z, mode="reflect") if pad_psf else psf
    # assert padded_img.shape == padded_psf.shape

    fl_decon_image = richardson_lucy_flowdec(
        padded_img,
        padded_psf,
        n_iter=n_iter,
        start_mode=start_mode,
        observer_fn=observer_fn,
        device=device,
        verbose=verbose,
        subprocess_cuda=subprocess_cuda,
    )

    fl_decon_image = fl_decon_image[pad_size_z : image.shape[0] + pad_size_z, :, :]

    return fl_decon_image.astype(np.uint16)


def decon_iter_num_finder(
    image: Union[str, npt.ArrayLike],
    psf: Union[str, npt.ArrayLike],
    metric_fn: Callable,
    metric_threshold: Union[int, float],
    metric_kwargs: Optional[Dict[str, Any]] = None,
    max_iter: int = 25,
    pad_size_z: int = 1,
    verbose: bool = False,
    subprocess_cuda: bool = False,
) -> Tuple[int, List[Dict[str, Union[int, float, np.ndarray]]]]:
    """Find number of LR deconvolution iterations using an image similarity metric."""
    check_tf_available()
    verboseprint = print if verbose else lambda *a, **k: None

    if isinstance(image, str):
        image = io.imread(str(image))
    if isinstance(psf, str):
        psf = io.imread(str(psf))
    if metric_kwargs is None:
        metric_kwargs = {}

    thresh_iter = 0
    results = [{"metric_gain": metric_threshold, "iter_image": image}]

    def get_decon_observer(metric_fn, metric_kwargs):
        nonlocal thresh_iter

        def decon_observer(restored_image, i, *args):
            nonlocal thresh_iter

            # stop metric calculations after reaching threshold to save time
            if thresh_iter == 0:  # threshold not reached
                prev_image = results[i - 1]["iter_image"]
                curr_image = restored_image[
                    pad_size_z : prev_image.shape[0] + pad_size_z, :, :
                ].astype(np.uint16)
                metric_result = metric_fn(prev_image, curr_image, **metric_kwargs)

                metric_gain = (
                    metric_result[0]
                    if isinstance(metric_result, tuple)
                    else metric_result
                )
                results.append(
                    {
                        "metric_gain": metric_gain,
                        "iter_image": curr_image,
                        "metric_result": metric_result,
                    }
                )
                verboseprint(f"Iteration {i}: improvement {metric_gain:.8f}")

                if (i > 1) and (metric_gain > metric_threshold):  # threshold reached
                    thresh_iter = i
                    metric_gain_total = metric_fn(
                        results[0]["iter_image"],
                        results[-1]["iter_image"],
                        **metric_kwargs,
                    )
                    metric_gain_total = (
                        metric_gain_total[0]
                        if isinstance(metric_gain_total, tuple)
                        else metric_gain_total
                    )

                    verboseprint(
                        f"\nThreshold {metric_threshold} reached at iteration {i}"
                        f" with improvement: {metric_gain:.8f}.\n"
                        f"Metric between original and restored images: {metric_gain_total:.8f}.\n"
                    )

        return decon_observer

    observer = get_decon_observer(metric_fn=metric_fn, metric_kwargs=metric_kwargs)
    _ = decon_flowdec(
        image,
        psf,
        n_iter=max_iter,
        observer_fn=observer,
        verbose=verbose,
        subprocess_cuda=subprocess_cuda,
    )

    return (thresh_iter, results)
