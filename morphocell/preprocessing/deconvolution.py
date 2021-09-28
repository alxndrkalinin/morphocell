"""Implements 3D image deconvolution using DeconvolutionLab2 or FlowDec"""

import subprocess

import numpy as np
import numpy.typing as npt
from typing import Union, Tuple, Optional, Callable, List, Dict, Sequence
from pathlib import Path

from skimage import io
from pyvirtualdisplay import Display

from flowdec import data as fd_data
from flowdec import restoration as fd_restoration

from image_utils import pad_image, rescale_isotropic
from frc import grid_crop_resolution
from utils import print_mean_std


def richardson_lucy_dl2(
    image: Union[str, npt.ArrayLike],
    psf: Union[str, npt.ArrayLike],
    n_iter: int = 10,
    dl2_path: str = "DeconcolutionLab2_-0.1.0-SNAPSHOT-jar-with-dependencies.jar",
    tmp_dir: Union[Path, Optional[str]] = None,
    verbose: bool = False,
) -> Union[int, np.ndarray]:
    """GPU-accelerated Lucy-Richardson deconvolution using DeconvoltuionLab2"""

    verboseprint = print if verbose else lambda *a, **k: None

    tmp_dir = Path(tmp_dir) if tmp_dir is not None else Path.cwd()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(image, np.ndarray):
        verboseprint(f"Saving original image to {tmp_dir}/tmp_image.tif")
        io.imsave(str(tmp_dir / "tmp_image.tif"), image)
        image = tmp_dir / "tmp_image.tif"
    if isinstance(psf, np.ndarray):
        verboseprint(f"Saving PSF to {tmp_dir}/tmp_psf.tif")
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
        verboseprint(result)

    # read zstack again
    if result.returncode == 0:
        verboseprint(f"Reading deconvolved image from {str(tmp_dir)}/tmp_image_RL_{n_iter}.tif")
        decon_image = io.imread(str(tmp_dir / f"tmp_image_RL_{n_iter}.tif"))
        verboseprint(f"Deconvolved image shape: {decon_image.shape}")
        return decon_image
    else:
        print("Deconvolution failed.")
        return -1


def richardson_lucy_flowdec(
    image: Union[str, npt.ArrayLike],
    psf: Union[str, npt.ArrayLike],
    n_iter: int = 10,
    start_mode: str = "input",
    observer_fn: Optional[Callable] = None,
    device: Optional[str] = None,
    verbose: bool = False,
) -> np.ndarray:
    """GPU-accelerated Lucy-Richardson deconvolution using FlowDec (TensorFlow)"""

    verboseprint = print if verbose else lambda *a, **k: None

    image = image if isinstance(image, np.ndarray) else io.imread(image)
    psf = psf if isinstance(psf, np.ndarray) else io.imread(psf)

    assert image.shape == psf.shape
    verboseprint(f"Deconvolving {image.shape=} with {psf.shape=}")

    algo = fd_restoration.RichardsonLucyDeconvolver(
        image.ndim,
        pad_mode="none",
        start_mode=start_mode,
        observer_fn=observer_fn,
        device=device,
    ).initialize()
    res = algo.run(fd_data.Acquisition(data=image, kernel=psf), niter=n_iter)
    verboseprint(f"\nDeconvolution info: {res.info}")

    return res.data


def decon_flowdec(
    image: Union[str, npt.ArrayLike],
    psf: Union[str, npt.ArrayLike],
    n_iter: int = 1,
    pad_size_z: int = 1,
    start_mode: str = "input",
    voxel_sizes: Union[int, float, Tuple[int, ...], Tuple[float, ...]] = 1.0,
    observer_fn: Optional[Callable] = None,
    device: Optional[str] = None,
    verbose: bool = False,
) -> np.ndarray:

    if isinstance(image, str):
        image = io.imread(str(image))
    if isinstance(psf, str):
        psf = io.imread(str(psf))
    if isinstance(voxel_sizes, int) or isinstance(voxel_sizes, float):
        voxel_sizes = (voxel_sizes, voxel_sizes, voxel_sizes)

    assert image.shape == psf.shape
    padded_img = pad_image(image, pad_size_z, mode="reflect")
    padded_psf = pad_image(psf, pad_size_z, mode="reflect")
    assert padded_img.shape == padded_psf.shape

    fl_decon_image = richardson_lucy_flowdec(
        padded_img,
        padded_psf,
        n_iter=n_iter,
        start_mode=start_mode,
        observer_fn=observer_fn,
        device=device,
        verbose=verbose,
    )
    if (voxel_sizes is not None) and (np.all(voxel_sizes != 1.0)):
        fl_decon_image = rescale_isotropic(
            fl_decon_image[pad_size_z : psf.shape[0] + pad_size_z, :, :],
            voxel_sizes=voxel_sizes,
        )
    else:
        fl_decon_image = fl_decon_image[pad_size_z : psf.shape[0] + pad_size_z, :, :]
    return fl_decon_image.astype(np.uint16)


def decon_iter_finder_frc(
    image: Union[str, npt.ArrayLike],
    psf: Union[str, npt.ArrayLike],
    max_iter: int = 25,
    frc_threshold: float = -5.0,
    frc_axis: str = "xy",
    frc_bin_delta: int = 3,
    scales: Union[int, float, Tuple[int, ...], Tuple[float, ...]] = 1.0,
    verbose: bool = False,
) -> Tuple[int, np.ndarray, List[Dict[str, List[float]]]]:
    """finds numer of LR decon iterations using FRC resolution"""

    verboseprint = print if verbose else lambda *a, **k: None
    if isinstance(image, str):
        image = io.imread(str(image))
    if isinstance(psf, str):
        psf = io.imread(str(psf))

    resolutions = [
        grid_crop_resolution(image, bin_delta=frc_bin_delta, scales=scales, aggregate=None, verbose=verbose)
    ]
    thresh_iter = 0
    thresh_iter_image = None

    def get_decon_observer(frc_bin_delta=frc_bin_delta, scales=scales):
        nonlocal resolutions, thresh_iter, thresh_iter_image

        def decon_observer(image, i, *args):
            nonlocal frc_bin_delta, scales, resolutions, thresh_iter, thresh_iter_image

            # stop computing FRC after reaching threshold to save time
            if thresh_iter == 0:
                resolution = grid_crop_resolution(
                    image, bin_delta=frc_bin_delta, scales=scales, aggregate=None, verbose=verbose
                )
                resolutions.append(resolution)

                res_prev = np.mean(resolutions[i - 1][frc_axis])
                res_curr = np.mean(resolutions[i][frc_axis])
                res_diff = res_curr - res_prev

                verboseprint(
                    f"Iteration {i}, {frc_axis} resolution improvement: from {res_prev:.4f}"
                    f" to {res_curr:.4f}, improvement {res_diff*1000:.2f} nm it^−1"
                )

                if (i > 1) and (res_diff * 1000 > frc_threshold):
                    thresh_iter = i
                    thresh_iter_image = image
                    verboseprint(
                        f"\n{frc_threshold} nm it^−1 threshold reached at iteration {i} for {frc_axis} axis\n"
                    )

        return decon_observer

    observer = get_decon_observer(frc_bin_delta=frc_bin_delta, scales=scales)
    _ = decon_flowdec(image, psf, n_iter=max_iter, voxel_sizes=scales, observer_fn=observer, verbose=verbose)

    return (thresh_iter, thresh_iter_image, resolutions)
