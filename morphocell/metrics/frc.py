"""Implements 2D/3D Fourier Ring/Shell Correlation."""

from typing import Union, Sequence, Callable, Dict, Optional
import numpy.typing as npt
import numpy as np

import miplib.analysis.resolution.fourier_ring_correlation as frc
import miplib.analysis.resolution.fourier_shell_correlation as fsc
import miplib.ui.cli.miplib_entry_point_options as options
from miplib.data.containers.image import Image

from ..image_utils import (
    max_project,
    crop_tl,
    crop_bl,
    crop_tr,
    crop_br,
    crop_center,
    pad_image,
    get_xy_block_coords,
)


def _empty_aggregate(*args: npt.ArrayLike, **kwargs) -> npt.ArrayLike:
    """Return unchaged array."""
    return args[0]


def calculate_frc(
    images: Union[npt.ArrayLike, Sequence[npt.ArrayLike]],
    bin_delta: int = 1,
    scales: Union[int, float, Sequence] = 1.0,
    return_resolution: bool = True,
    verbose: bool = False,
) -> Union[Sequence, float]:
    """Calculate FRC-based 2D image resolution."""
    verboseprint = print if verbose else lambda *a, **k: None

    if isinstance(scales, int) or isinstance(scales, float):
        scales = [scales, scales]
    assert len(scales) == 2

    # check if a single image passes as an input
    if isinstance(images, np.ndarray) or (isinstance(images, Sequence) and len(images) == 1):

        image = images if isinstance(images, np.ndarray) else images[0]
        assert image.shape[0] == image.shape[1]
        miplib_img = Image(image, scales)
        verboseprint(f"The image dimensions are {miplib_img.shape} and spacing {miplib_img.spacing} um.")

        args_list = (
            f"None --bin-delta={bin_delta} --frc-curve-fit-type=smooth-spline "
            " --resolution-threshold-criterion=fixed"
        ).split()
        args = options.get_frc_script_options(args_list)
        frc_result = frc.calculate_single_image_frc(miplib_img, args)

    elif len(images) == 2:

        assert images[0].shape == images[1].shape
        miplib_img_1 = Image(images[0], scales)
        miplib_img_2 = Image(image[1], scales)
        verboseprint(
            f"The 1st image dimensions are {miplib_img_1.shape} and spacing {miplib_img_1.spacing} um."
            f"\tThe 2nd image dimensions are {miplib_img_2.shape} and spacing {miplib_img_2.spacing} um."
        )

        args_list = (
            f"None --bin-delta={bin_delta} --frc-curve-fit-type=smooth-spline "
            " --resolution-threshold-criterion=fixed"
        ).split()
        args = options.get_frc_script_options(args_list)
        frc_result = frc.calculate_two_image_frc(miplib_img_1, miplib_img_2, args)

    else:
        raise ValueError("FRC: incorrect input, should be one image or a list of two images")

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

    assert img_cube.shape[0] == img_cube.shape[1] == img_cube.shape[2]
    miplib_img = Image(img_cube, scales)
    verboseprint(f"The image dimensions are {miplib_img.shape} and spacing {miplib_img.spacing} um.")

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
    fsc_result = fsc.calculate_one_image_sectioned_fsc(miplib_img, args, z_correction=z_correction)
    return fsc_result


def grid_crop_resolution(
    images: npt.ArrayLike,
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

    # assuming we have more than 2 slices in each zstack
    if len(images) > 2:
        image = images
    elif len(images) == 2:
        image = images[0]
        image_2 = images[1]
    else:
        raise ValueError("FRC: incorrect input, should be one image or a list of two images")

    if isinstance(scales, int) or isinstance(scales, float):
        scales = [scales, scales, scales]

    assert len(image.shape) == 3 and len(scales) == 3
    assert image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]
    assert image.shape[1] > crop_size and image.shape[2] > crop_size

    scales_xy = (scales[1], scales[2])
    scales_xz = (scales[0], scales[2])

    locations = get_xy_block_coords(image.shape, crop_size)

    max_projection_resolutions = []
    xy_resolutions = []
    xz_resolutions = []
    for y1, y2, x1, x2 in locations:

        if len(image) > 2:
            loc_image = image[:, y1:y2, x1:x2]
            max_projection_resolution = calculate_frc(
                max_project(loc_image), bin_delta, scales_xy, return_resolution, verbose
            )
        else:
            loc_image = image[:, y1:y2, x1:x2]
            loc_image_2 = image_2[:, y1:y2, x1:x2]
            max_projection_resolution = calculate_frc(
                [max_project(loc_image), max_project(loc_image_2)],
                bin_delta,
                scales_xy,
                return_resolution,
                verbose,
            )

        max_projection_resolutions.append(max_projection_resolution)

        xy_slice_resolutions = []
        xz_slice_resolutions = []
        xz_slices = np.linspace(0, crop_size - 1, num=loc_image.shape[0], dtype=int)

        for slice_idx in range(loc_image.shape[0]):

            if len(image) > 2:
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

            else:
                xy_slice_resolutions.append(
                    calculate_frc(
                        [loc_image[slice_idx, :, :], loc_image_2[slice_idx, :, :]],
                        bin_delta,
                        scales_xy,
                        return_resolution,
                        verbose,
                    )
                )

                xz_slice = loc_image[:, xz_slices[slice_idx], :]
                xz_slice_2 = loc_image_2[:, xz_slices[slice_idx], :]

                pad_size = (xz_slice.shape[1] - xz_slice.shape[0]) // 2
                if (xz_slice.shape[1] - xz_slice.shape[0]) % 2 != 0:
                    pad_size = (pad_size + 1, pad_size)

                padded_xz_slice = pad_image(xz_slice, pad_size, 0, pad_mode)
                padded_xz_slice_2 = pad_image(xz_slice_2, pad_size, 0, pad_mode)

                xz_slice_resolutions.append(
                    calculate_frc(
                        [padded_xz_slice, padded_xz_slice_2],
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

    if isinstance(scales, int) or isinstance(scales, float):
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
    scales: Union[int, float, Sequence[Union[int, float]]] = 1.0,
    axis: str = "xy",
    frc_bin_delta: int = 3,
    aggregate: Callable = np.mean,
    verbose: bool = False,
) -> float:
    """Calculate difference between FRC-based resulutions of two images."""
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
    return aggregate(image2_res[axis]) - aggregate(image1_res[axis]) * 1000  # return diff in nm
