"""Implements 2D/3D Fourier Ring/Shell Correlation."""

from typing import Union, Sequence, Callable, Dict, Optional, Tuple
import numpy.typing as npt
from argparse import Namespace

import miplib.data.iterators.fourier_ring_iterators as iterators
from miplib.data.containers.fourier_correlation_data import FourierCorrelationData, FourierCorrelationDataCollection
import miplib.analysis.resolution.analysis as fsc_analysis
import miplib.analysis.resolution.fourier_shell_correlation as fsc
import miplib.ui.cli.miplib_entry_point_options as options

from ..image import Image
from ..gpu import asnumpy, get_array_module
from ..image_utils import (
    max_project,
    crop_tl,
    crop_bl,
    crop_tr,
    crop_br,
    crop_center,
    pad_image,
    get_xy_block_coords,
    rescale_isotropic,
    pad_image_to_cube,
    hamming_window,
    pad_images_to_matching_shape,
    checkerboard_split,
    reverse_checkerboard_split,
)

import numpy as np


def _empty_aggregate(*args: npt.ArrayLike, **kwargs) -> npt.ArrayLike:
    """Return unchaged array."""
    return args[0]


def frc_checkerboard_split(image: Image, reverse=False):
    """Split image into two by checkerboard pattern."""
    if reverse:
        image1, image2 = reverse_checkerboard_split(image.data)
    else:
        image1, image2 = checkerboard_split(image.data)
    image1 = Image(image1, spacing=image.spacing, device=image.device)
    image2 = Image(image2, spacing=image.spacing, device=image.device)
    return image1, image2


# https://github.com/sakoho81/miplib/blob/public/miplib/analysis/resolution/fourier_ring_correlation.py
class FRC(object):
    """A class for calcuating 2D Fourier ring correlation."""

    def __init__(self, image1, image2, iterator):
        """Create new FRC executable object and perform FFT on input images."""
        if image1.shape != image2.shape:
            raise ValueError("The image dimensions do not match")
        if image1.ndim != 2:
            raise ValueError("Fourier ring correlation requires 2D images.")

        # Expand to square
        image1 = pad_image_to_cube(image1, np.max(image1.shape), mode="constant")
        image2 = pad_image_to_cube(image2, np.max(image1.shape), mode="constant")

        self.iterator = iterator
        # Calculate power spectra for the input images.
        self.fft_image1 = np.fft.fftshift(np.fft.fft2(image1))
        self.fft_image2 = np.fft.fftshift(np.fft.fft2(image2))

        # Get the Nyquist frequency
        self.freq_nyq = int(np.floor(image1.shape[0] / 2.0))

    def execute(self):
        """Calculate the FRC."""
        radii = self.iterator.radii
        c1 = np.zeros(radii.shape, dtype=np.float32)
        c2 = np.zeros(radii.shape, dtype=np.float32)
        c3 = np.zeros(radii.shape, dtype=np.float32)
        points = np.zeros(radii.shape, dtype=np.float32)

        for ind_ring, idx in self.iterator:
            subset1 = self.fft_image1[ind_ring]
            subset2 = self.fft_image2[ind_ring]
            c1[idx] = np.sum(subset1 * np.conjugate(subset2)).real
            c2[idx] = np.sum(np.abs(subset1) ** 2)
            c3[idx] = np.sum(np.abs(subset2) ** 2)

            points[idx] = len(subset1)

        # Calculate FRC
        spatial_freq = asnumpy(radii.astype(np.float32) / self.freq_nyq)
        c1 = asnumpy(c1)
        c2 = asnumpy(c2)
        c3 = asnumpy(c3)
        n_points = asnumpy(points)

        with np.errstate(divide="ignore", invalid="ignore"):
            frc = np.abs(c1) / np.sqrt(c2 * c3)
            frc[frc == np.inf] = 0.0
            frc = np.nan_to_num(frc)

        data_set = FourierCorrelationData()
        data_set.correlation["correlation"] = frc
        data_set.correlation["frequency"] = spatial_freq
        data_set.correlation["points-x-bin"] = n_points

        return data_set


# https://github.com/sakoho81/miplib/blob/public/miplib/analysis/resolution/fourier_ring_correlation.py
def calculate_single_image_frc(
    image: Image, args: Namespace, average: bool = True, trim: bool = True, z_correction: int = 1
):
    """Calculate a regular FRC with a single image input.

    :param image: the image as an Image object
    :param args:  the parameters for the FRC calculation. See *miplib.ui.frc_options*
                  for details
    :return:      returns the FRC result as a FourierCorrelationData object
    """
    assert isinstance(image, Image)

    frc_data = FourierCorrelationDataCollection()

    # Hamming Windowing
    if not args.disable_hamming:
        spacing = image.spacing
        device = image.device
        image = Image(hamming_window(image.data), spacing, device=device)

    # Split and make sure that the images are the same size
    image1, image2 = frc_checkerboard_split(image)
    # image1, image2 = imops.reverse_checkerboard_split(image)
    image1.data, image2.data = pad_images_to_matching_shape(image1.data, image2.data)

    assert tuple(image1.shape) == tuple(image2.shape)
    assert tuple(image1.spacing) == tuple(image2.spacing)

    # Run FRC
    iterator = iterators.FourierRingIterator(image1.shape, args.d_bin)
    frc_task = FRC(image1.data, image2.data, iterator)
    frc_data[0] = frc_task.execute()

    if average:
        # Split and make sure that the images are the same size
        image1, image2 = frc_checkerboard_split(image, reverse=True)
        image1, image2 = pad_images_to_matching_shape(image1, image2)
        iterator = iterators.FourierRingIterator(image1.shape, args.d_bin)
        frc_task = FRC(image1.data, image2.data, iterator)

        frc_data[0].correlation["correlation"] *= 0.5
        frc_data[0].correlation["correlation"] += 0.5 * frc_task.execute().correlation["correlation"]

    def func(x, a, b, c, d):
        return a * np.exp(c * (x - b)) + d

    params = [0.95988146, 0.97979108, 13.90441896, 0.55146136]

    # Analyze results
    analyzer = fsc_analysis.FourierCorrelationAnalysis(frc_data, image1.spacing[0], args)

    result = analyzer.execute(z_correction=z_correction)[0]
    point = result.resolution["resolution-point"][1]

    cut_off_correction = func(point, *params)
    result.resolution["spacing"] /= cut_off_correction
    result.resolution["resolution"] /= cut_off_correction

    return result


def calculate_frc(
    image: Union[npt.ArrayLike, Image],
    bin_delta: int = 1,
    scales: Union[int, float, Sequence] = 1.0,
    return_resolution: bool = True,
    verbose: bool = False,
) -> Union[Sequence, float]:
    """Calculate FRC-based 2D image resolution."""
    verboseprint = print if verbose else lambda *a, **k: None

    if not isinstance(image, Image):
        xp = get_array_module(image)
        ndarray = getattr(xp, "ndarray")  # avoid mypy complains
        if not isinstance(image, ndarray):
            raise ValueError("FRC: incorrect input, should be 2D Image, Numpy or CuPy array.")

        if isinstance(scales, (int, float)):
            scales = [scales, scales]
        image = Image(image, scales)
    assert image.shape[0] == image.shape[1]
    assert len(image.spacing) == 2
    verboseprint(f"The image dimensions are {image.shape} and spacing {image.spacing} um.")  # type: ignore[operator]

    args_list = (
        f"None --bin-delta={bin_delta} --frc-curve-fit-type=smooth-spline " " --resolution-threshold-criterion=fixed"
    ).split()
    args = options.get_frc_script_options(args_list)
    frc_result = calculate_single_image_frc(image, args)

    frc_result = frc_result.resolution["resolution"] if return_resolution else frc_result
    return frc_result


def calculate_fsc(
    img_cube: npt.ArrayLike,
    bin_delta: int = 10,
    scales: Union[int, float, Sequence] = 1.0,
    z_correction: float = 1.0,
    verbose: bool = False,
):
    """Calculate FSC-based 3D image resolution."""
    verboseprint = print if verbose else lambda *a, **k: None

    if isinstance(scales, (int, float)):
        scales = [scales, scales, scales]

    assert img_cube.shape[0] == img_cube.shape[1] == img_cube.shape[2]
    miplib_img = Image(img_cube, scales)
    verboseprint(f"The image dimensions are {miplib_img.shape} and spacing {miplib_img.spacing} um.")  # type: ignore[operator]

    args_list = [
        None,
        f"--bin-delta={bin_delta}",
        "--resolution-threshold-criterion=snr",
        "--resolution-snr-value=0.5",
        "--angle-delta=15",
        "--enable-hollow-iterator",
        "--extract-angle-delta=.1",
        "--resolution-point-sigma=0.01",
        "--frc-curve-fit-type=spline",
    ]
    args = options.get_frc_script_options(args_list)
    return fsc.calculate_one_image_sectioned_fsc(miplib_img, args, z_correction=z_correction)


def grid_crop_resolution(
    image: npt.ArrayLike,
    bin_delta: int = 1,
    scales: Union[int, float, Sequence[Union[int, float]]] = 1.0,
    crop_size: int = 512,
    pad_mode: str = "reflect",
    return_resolution: bool = True,
    aggregate: Optional[Callable] = np.median,
    verbose: bool = False,
) -> Dict[str, npt.ArrayLike]:
    """Calculate FRC-based 3D image resolution by tiling and taking 2D slices along XY and XZ."""
    if not return_resolution or aggregate is None:
        aggregate_fn = _empty_aggregate
    else:
        aggregate_fn = aggregate

    if not isinstance(image, Image):
        xp = get_array_module(image)
        ndarray = getattr(xp, "ndarray")  # avoid mypy complains
        if not isinstance(image, ndarray):
            raise ValueError("FRC: incorrect input, should be 2D Image, Numpy or CuPy array.")

        if isinstance(scales, (int, float)):
            scales = [scales, scales, scales]
        image = Image(image, scales)
    assert len(image.shape) == 3 and len(image.spacing) == 3
    assert image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]
    assert image.shape[1] > crop_size and image.shape[2] > crop_size

    scales_xy = (image.spacing[1], image.spacing[2])
    scales_xz = (image.spacing[0], image.spacing[2])

    locations = get_xy_block_coords(image.shape, crop_size)

    max_projection_resolutions = []
    xy_resolutions = []
    xz_resolutions = []
    for y1, y2, x1, x2 in locations:

        loc_image = image.data[:, y1:y2, x1:x2]
        max_projection_resolution = calculate_frc(
            max_project(loc_image), bin_delta, scales_xy, return_resolution, verbose
        )
        max_projection_resolutions.append(max_projection_resolution)

        xy_slice_resolutions = []
        xz_slice_resolutions = []
        xz_slices = np.linspace(0, crop_size - 1, num=loc_image.shape[0], dtype=int)

        for slice_idx in range(loc_image.shape[0]):

            xy_slice_resolutions.append(
                calculate_frc(
                    loc_image[slice_idx, :, :],
                    bin_delta,
                    scales_xy,
                    return_resolution,
                    verbose,
                )
            )

            xz_slice = loc_image[:, xz_slices[slice_idx], :]

            pad_size = (xz_slice.shape[1] - xz_slice.shape[0]) // 2
            if (xz_slice.shape[1] - xz_slice.shape[0]) % 2 != 0:
                pad_size = (pad_size + 1, pad_size)

            padded_xz_slice = pad_image(xz_slice, pad_size, 0, pad_mode)

            xz_slice_resolutions.append(
                calculate_frc(
                    padded_xz_slice,
                    bin_delta,
                    scales_xz,
                    return_resolution,
                    verbose,
                )
            )

        xy_resolutions.append(xy_slice_resolutions)
        xz_resolutions.append(xz_slice_resolutions)

    return {
        "max_projection": aggregate_fn(max_projection_resolutions, axis=0),
        "xy": aggregate_fn(xy_resolutions, axis=0),
        "xz": aggregate_fn(xz_resolutions, axis=0),
    }


def five_crop_resolution(
    image: npt.ArrayLike,
    bin_delta: int = 1,
    scales: Union[int, float, Sequence] = 1.0,
    crop_size: int = 512,
    pad_mode: str = "reflect",
    return_resolution: bool = True,
    aggregate: Callable = np.median,
    verbose: bool = False,
) -> Dict[str, npt.ArrayLike]:
    """Calculate FRC-based 3D image resolution by taking 2D slices along XY and XZ at 4 corners and the center."""
    if not return_resolution or aggregate is None:
        aggregate_fn = _empty_aggregate
    else:
        aggregate_fn = aggregate

    if isinstance(scales, (int, float)):
        scales = [scales, scales, scales]

    assert len(image.shape) == 3 and len(scales) == 3
    assert image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]
    assert image.shape[1] > crop_size and image.shape[2] > crop_size

    scales_xy = (scales[1], scales[2])
    scales_xz = (scales[0], scales[2])

    locations = [crop_tl, crop_bl, crop_tr, crop_br, crop_center]
    max_projection_resolutions = []
    xy_resolutions = []
    xz_resolutions = []
    for loc in locations:
        loc_image = loc(image, crop_size)

        max_projection_resolution = calculate_frc(
            max_project(loc_image), bin_delta, scales_xy, return_resolution, verbose
        )
        max_projection_resolutions.append(max_projection_resolution)

        xy_slice_resolutions = []
        xz_slice_resolutions = []
        xz_slices = np.linspace(0, crop_size - 1, num=loc_image.shape[0], dtype=int)
        for slice_idx in range(loc_image.shape[0]):

            xy_slice_resolutions.append(
                calculate_frc(
                    loc_image[slice_idx, :, :],
                    bin_delta,
                    scales_xy,
                    return_resolution,
                    verbose,
                )
            )

            xz_slice = loc_image[:, xz_slices[slice_idx], :]

            padded_xz_slice = pad_image(xz_slice, (xz_slice.shape[1] - xz_slice.shape[0]) // 2, 0, pad_mode)

            xz_slice_resolutions.append(
                calculate_frc(padded_xz_slice, bin_delta, scales_xz, return_resolution, verbose)
            )

        xy_resolutions.append(xy_slice_resolutions)
        xz_resolutions.append(xz_slice_resolutions)

    return {
        "max_projection": aggregate_fn(max_projection_resolutions, axis=0),
        "xy": aggregate_fn(xy_resolutions, axis=0),
        "xz": aggregate_fn(xz_resolutions, axis=0),
    }


def frc_resolution_difference(
    image1: npt.ArrayLike,
    image2: npt.ArrayLike,
    scales: Union[int, float, Tuple[int, ...], Tuple[float, ...]] = 1.0,
    downscale_xy: bool = False,
    axis: str = "xy",
    frc_bin_delta: int = 3,
    aggregate: Callable = np.mean,
    verbose: bool = False,
) -> float:
    """Calculate difference between FRC-based resulutions of two images."""
    if isinstance(scales, (int, float)):
        scales = (scales, scales, scales)
    if np.any(np.asarray(scales) != 1.0):
        image1 = rescale_isotropic(image1, voxel_sizes=scales, downscale_xy=downscale_xy)
        image2 = rescale_isotropic(image2, voxel_sizes=scales, downscale_xy=downscale_xy)

    image1_res = grid_crop_resolution(
        image1,
        bin_delta=frc_bin_delta,
        scales=scales,
        aggregate=aggregate,
        verbose=verbose,
    )
    image2_res = grid_crop_resolution(
        image2,
        bin_delta=frc_bin_delta,
        scales=scales,
        aggregate=aggregate,
        verbose=verbose,
    )
    return (aggregate(image2_res[axis]) - aggregate(image1_res[axis])) * 1000  # return diff in nm
