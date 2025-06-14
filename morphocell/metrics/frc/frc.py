"""Implements 2D/3D Fourier Ring/Shell Correlation."""

from typing import Union, Sequence, Callable, Dict, Optional, Tuple, Any
import numpy.typing as npt
from argparse import Namespace

from .frc_utils import (
    FourierRingIterator,
    AxialExcludeSectionedFourierShellIterator,
    FourierCorrelationData,
    FourierCorrelationDataCollection,
    FourierCorrelationAnalysis,
    get_frc_options,
)

from ...image import Image
from ...cuda import asnumpy, get_array_module
from ...image_utils import (
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


def frc_checkerboard_split(image: Image, reverse=False, disable_3d_sum=False):
    """Split image into two by checkerboard pattern."""
    if reverse:
        image1, image2 = reverse_checkerboard_split(
            image.data, disable_3d_sum=disable_3d_sum
        )
    else:
        image1, image2 = checkerboard_split(image.data, disable_3d_sum=disable_3d_sum)
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
            frc = np.exp(np.log(np.abs(c1)) - 0.5 * (np.log(c2) + np.log(c3)))
            frc[frc == np.inf] = 0.0
            frc = np.nan_to_num(frc)

        data_set = FourierCorrelationData()
        data_set.correlation["correlation"] = frc
        data_set.correlation["frequency"] = spatial_freq
        data_set.correlation["points-x-bin"] = n_points

        return data_set


# https://github.com/sakoho81/miplib/blob/public/miplib/analysis/resolution/fourier_ring_correlation.py
def calculate_single_image_frc(
    image: Image,
    args: Namespace,
    average: bool = True,
    trim: bool = True,
    z_correction: int = 1,
):
    """Calculate a regular FRC with a single image input.

    :param image: the image as an Image object
    :param args:  parameters for the FRC calculation
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
    iterator = FourierRingIterator(image1.shape, args.d_bin)
    frc_task = FRC(image1.data, image2.data, iterator)
    frc_data[0] = frc_task.execute()

    if average:
        # Split and make sure that the images are the same size
        image1, image2 = frc_checkerboard_split(image, reverse=True)
        image1, image2 = pad_images_to_matching_shape(image1, image2)
        iterator = FourierRingIterator(image1.shape, args.d_bin)
        frc_task = FRC(image1.data, image2.data, iterator)

        frc_data[0].correlation["correlation"] *= 0.5
        frc_data[0].correlation["correlation"] += (
            0.5 * frc_task.execute().correlation["correlation"]
        )

    def func(x, a, b, c, d):
        return a * np.exp(c * (x - b)) + d

    params = [0.95988146, 0.97979108, 13.90441896, 0.55146136]

    # Analyze results
    analyzer = FourierCorrelationAnalysis(frc_data, image1.spacing[0], args)

    result = analyzer.execute(z_correction=z_correction)[0]
    point = result.resolution["resolution-point"][1]

    cut_off_correction = func(point, *params)
    result.resolution["spacing"] /= cut_off_correction
    result.resolution["resolution"] /= cut_off_correction

    return result


def calculate_frc(
    image: Union[npt.ArrayLike, Image],
    bin_delta: int = 1,
    scales: Union[float, Sequence] = 1.0,
    resolution_threshold: str = "fixed",
    return_resolution: bool = True,
    verbose: bool = False,
) -> Union[Sequence, float]:
    """Calculate FRC-based 2D image resolution."""
    verboseprint = print if verbose else lambda *a, **k: None

    if not isinstance(image, Image):
        xp = get_array_module(image)
        ndarray = getattr(xp, "ndarray")  # avoid mypy complains
        if not isinstance(image, ndarray):
            raise ValueError(
                "FRC: incorrect input, should be 2D Image, Numpy or CuPy array."
            )
        if isinstance(scales, (int, float)):
            scales = [scales, scales]
        image = Image(image, scales)

    # assert image.shape[0] == image.shape[1], "FRC: input image should be square."
    assert len(image.spacing) == 2
    verboseprint(
        f"The image dimensions are {image.shape} and spacing {image.spacing} um."
    )  # type: ignore[operator]

    args = get_frc_options(
        bin_delta=bin_delta,
        curve_fit_type="smooth-spline",
        resolution_threshold=resolution_threshold,
    )
    frc_result = calculate_single_image_frc(image, args)

    frc_result = (
        frc_result.resolution["resolution"] if return_resolution else frc_result
    )
    return frc_result


def preprocess_img_cubes(
    img_cubes: Union[npt.ArrayLike, Image, Sequence[Union[npt.ArrayLike, Image]]],
    scales: Union[float, Sequence] = 1.0,
    zero_padding: bool = True,
    verbose: bool = False,
) -> Sequence[Image]:
    """Preprocess input image cubes."""
    verboseprint = print if verbose else lambda *a, **k: None

    if not isinstance(img_cubes, tuple):
        img_cubes = (img_cubes,)

    cube_shape = None
    img_cubes_processed = []
    for img_cube in img_cubes:
        if not isinstance(img_cube, Image):
            xp = get_array_module(img_cube)
            ndarray = getattr(xp, "ndarray")  # avoid mypy complains
            if not isinstance(img_cube, ndarray):
                raise ValueError(
                    "FSC: incorrect input, should be 3D Image, Numpy or CuPy array."
                )

        if isinstance(scales, (int, float)):
            scales = [scales, scales, scales]

        if len(set(img_cube.shape)) > 1 and zero_padding:
            img_cube = (
                pad_image_to_cube(img_cube.data)
                if isinstance(img_cube, Image)
                else pad_image_to_cube(img_cube)
            )
            assert len(set(img_cube.shape)) == 1, "FSC: image should be a cube."

        if cube_shape is None:
            cube_shape = img_cube.shape
        elif cube_shape != img_cube.shape:
            raise ValueError("FSC: all input images should have the same shape.")

        img_cube = Image(img_cube, scales)
        verboseprint(
            f"FSC: image dimensions are {img_cube.shape} and spacing {img_cube.spacing} um."
        )  # type: ignore[operator]
        img_cubes_processed.append(img_cube)

    return img_cubes_processed


def calculate_fsc_result(
    img_cubes_processed: Sequence[Image],
    bin_delta: int,
    resolution_threshold: str,
    z_correction: float,
    disable_3d_sum: bool,
) -> Any:
    """Calculate FSC result based on preprocessed image cubes."""
    args = get_frc_options(
        bin_delta=bin_delta,
        angle_delta=15,
        extract_angle_delta=0.1,
        resolution_threshold=resolution_threshold,
        curve_fit_type="spline",
    )

    if len(img_cubes_processed) == 1:
        return calculate_one_image_sectioned_fsc(
            img_cubes_processed[0],
            args,
            z_correction=z_correction,
            disable_3d_sum=disable_3d_sum,
        )
    elif len(img_cubes_processed) == 2:
        return calculate_two_image_sectioned_fsc(
            img_cubes_processed[0],
            img_cubes_processed[1],
            args,
            z_correction=z_correction,
        )
    else:
        raise ValueError("FSC: incorrect number of input images. Should be 1 or 2.")


def extract_resolution(fsc_result: Any, return_resolution: bool) -> Any:
    """Extract resolution from FSC result."""
    if return_resolution:
        angles = list()
        radii = list()

        for dataset in fsc_result:
            angles.append((int(dataset[0])))
            radii.append(dataset[1].resolution["resolution"])

        return {"xy": radii[0], "z": radii[angles.index(90)]}
    else:
        return fsc_result


def calculate_fsc(
    img_cubes: Union[npt.ArrayLike, Image, Sequence[Union[npt.ArrayLike, Image]]],
    bin_delta: int = 10,
    scales: Union[float, Sequence] = 1.0,
    resolution_threshold: str = "fixed",
    z_correction: float = 1.0,
    zero_padding: bool = True,
    disable_3d_sum: bool = False,
    return_resolution: bool = True,
    verbose: bool = False,
):
    """Calculate either single- or two-image FSC-based 3D image resolution."""
    img_cubes_processed = preprocess_img_cubes(img_cubes, scales, zero_padding, verbose)
    fsc_result = calculate_fsc_result(
        img_cubes_processed,
        bin_delta,
        resolution_threshold,
        z_correction,
        disable_3d_sum,
    )
    return extract_resolution(fsc_result, return_resolution)


# https://github.com/sakoho81/miplib/blob/public/miplib/analysis/resolution/fourier_shell_correlation.py
def calculate_one_image_sectioned_fsc(
    image, args, z_correction=1.0, disable_3d_sum=False
):
    """Calculate one-image sectioned FSC.

    I assume here that prior to calling the function,
    the image is going to be in a correct shape, resampled to isotropic spacing and zero padded. If the image
    dimensions are wrong (not a cube) the function will return an error.

    :param image: a 3D image, with isotropic spacing and cubic shape
    :type image: Image
    :param options: configuration parameters
    :param z_correction: correction, for anisotropic sampling. It is the ratio of axial vs. lateral spacing, defaults to 1
    :type z_correction: float, optional
    :return: the resolution measurement results organized by rotation angle
    :rtype: FourierCorrelationDataCollection object
    """
    assert isinstance(image, Image)
    assert all(s == image.shape[0] for s in image.shape)

    image1, image2 = frc_checkerboard_split(image, disable_3d_sum=disable_3d_sum)

    image1 = Image(hamming_window(image1.data), image1.spacing)
    image2 = Image(hamming_window(image2.data), image2.spacing)

    iterator = AxialExcludeSectionedFourierShellIterator(
        image1.shape, args.d_bin, args.d_angle, args.d_extract_angle
    )
    fsc_task = DirectionalFSC(image1.data, image2.data, iterator)

    data = fsc_task.execute()

    analyzer = FourierCorrelationAnalysis(data, image1.spacing[0], args)
    result = analyzer.execute(z_correction=z_correction)

    def func(x, a, b, c, d):
        return a * np.exp(c * (x - b)) + d

    params = [0.95988146, 0.97979108, 13.90441896, 0.55146136]

    for angle, dataset in result:
        point = dataset.resolution["resolution-point"][1]

        cut_off_correction = func(point, *params)
        dataset.resolution["spacing"] /= cut_off_correction
        dataset.resolution["resolution"] /= cut_off_correction

    return result


def calculate_two_image_sectioned_fsc(image1, image2, args, z_correction=1.0):
    """Calculate two-image sectioned FSC."""
    assert isinstance(image1, Image)
    assert isinstance(image2, Image)

    image1 = Image(hamming_window(image1.data), image1.spacing)
    image2 = Image(hamming_window(image2.data), image2.spacing)

    iterator = AxialExcludeSectionedFourierShellIterator(
        image1.shape, args.d_bin, args.d_angle, args.d_extract_angle
    )
    fsc_task = DirectionalFSC(image1.data, image2.data, iterator)
    data = fsc_task.execute()

    analyzer = FourierCorrelationAnalysis(data, image1.spacing[0], args)
    return analyzer.execute(z_correction=z_correction)


class DirectionalFSC(object):
    """Calculate the directional FSC between two images."""

    def __init__(self, image1, image2, iterator, normalize_power=False):
        """Initialize the directional FSC."""
        # assert isinstance(image1, Image)
        # assert isinstance(image2, Image)

        if image1.ndim != 3 or image1.shape[0] <= 1:
            raise ValueError("Image must be 3D")

        if image1.shape != image2.shape:
            raise ValueError("Image dimensions do not match")

        # Create an Iterator
        self.iterator = iterator

        # FFT transforms of the input images
        self.fft_image1 = np.fft.fftshift(np.fft.fftn(image1))
        self.fft_image2 = np.fft.fftshift(np.fft.fftn(image2))

        if normalize_power:
            pixels = image1.shape[0] ** 3
            self.fft_image1 /= np.array(pixels * np.mean(image1))
            self.fft_image2 /= np.array(pixels * np.mean(image2))

        self._result = None

        # self.pixel_size = image1.spacing[0]

    @property
    def result(self):
        """Return the FRC results."""
        if self._result is None:
            return self.execute()
        else:
            return self._result

    def execute(self):
        """
        Calculate the FRC.

        :return: Returns the FRC results. They are also saved inside the class.
                 The return value is just for convenience.
        """
        data_structure = FourierCorrelationDataCollection()
        radii, angles = self.iterator.steps
        freq_nyq = self.iterator.nyquist
        shape = (angles.shape[0], radii.shape[0])
        c1 = np.zeros(shape, dtype=np.float32)
        c2 = np.zeros(shape, dtype=np.float32)
        c3 = np.zeros(shape, dtype=np.float32)
        points = np.zeros(shape, dtype=np.float32)

        # iterate through the sphere and calculate initial values
        for ind_ring, shell_idx, rotation_idx in self.iterator:
            subset1 = self.fft_image1[ind_ring]
            subset2 = self.fft_image2[ind_ring]

            c1[rotation_idx, shell_idx] = np.sum(subset1 * np.conjugate(subset2)).real
            c2[rotation_idx, shell_idx] = np.sum(np.abs(subset1) ** 2)
            c3[rotation_idx, shell_idx] = np.sum(np.abs(subset2) ** 2)

            points[rotation_idx, shell_idx] = len(subset1)

        # finish up FRC calculation for every rotation angle and save results to the data structure.
        for i in range(angles.size):
            # calculate FRC for every orientation
            spatial_freq = asnumpy(radii.astype(np.float32) / freq_nyq)
            c1_i = asnumpy(c1[i])
            c2_i = asnumpy(c2[i])
            c3_i = asnumpy(c3[i])
            n_points = asnumpy(points[i])

            with np.errstate(divide="ignore", invalid="ignore"):
                frc = np.abs(c1_i) / np.sqrt(c2_i * c3_i)
                frc[frc == np.inf] = 0.0
                frc = np.nan_to_num(frc)

            result = FourierCorrelationData()
            result.correlation["correlation"] = frc
            result.correlation["frequency"] = spatial_freq
            result.correlation["points-x-bin"] = n_points

            data_structure[angles[i]] = result

        return data_structure


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
            raise ValueError(
                "FRC: incorrect input, should be 2D Image, Numpy or CuPy array."
            )

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
            loc_image.max(0),
            bin_delta=bin_delta,
            scales=scales_xy,
            return_resolution=return_resolution,
            verbose=verbose,
        )
        max_projection_resolutions.append(max_projection_resolution)

        xy_slice_resolutions = []
        xz_slice_resolutions = []
        xz_slices = np.linspace(0, crop_size - 1, num=loc_image.shape[0], dtype=int)

        for slice_idx in range(loc_image.shape[0]):
            xy_slice_resolutions.append(
                calculate_frc(
                    loc_image[slice_idx, :, :],
                    bin_delta=bin_delta,
                    scales=scales_xy,
                    return_resolution=return_resolution,
                    verbose=verbose,
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
                    bin_delta=bin_delta,
                    scales=scales_xz,
                    return_resolution=return_resolution,
                    verbose=verbose,
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
        loc_image = loc(image, crop_size)  # type: ignore

        max_projection_resolution = calculate_frc(
            loc_image.max(0),
            bin_delta=bin_delta,
            scales=scales_xy,
            return_resolution=return_resolution,
            verbose=verbose,
        )
        max_projection_resolutions.append(max_projection_resolution)

        xy_slice_resolutions = []
        xz_slice_resolutions = []
        xz_slices = np.linspace(0, crop_size - 1, num=loc_image.shape[0], dtype=int)
        for slice_idx in range(loc_image.shape[0]):
            xy_slice_resolutions.append(
                calculate_frc(
                    loc_image[slice_idx, :, :],
                    bin_delta=bin_delta,
                    scales=scales_xy,
                    return_resolution=return_resolution,
                    verbose=verbose,
                )
            )

            xz_slice = loc_image[:, xz_slices[slice_idx], :]

            padded_xz_slice = pad_image(
                xz_slice, (xz_slice.shape[1] - xz_slice.shape[0]) // 2, 0, pad_mode
            )

            xz_slice_resolutions.append(
                calculate_frc(
                    padded_xz_slice,
                    bin_delta=bin_delta,
                    scales=scales_xz,
                    return_resolution=return_resolution,
                    verbose=verbose,
                )
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
        image1 = rescale_isotropic(
            image1, voxel_sizes=scales, downscale_xy=downscale_xy
        )
        image2 = rescale_isotropic(
            image2, voxel_sizes=scales, downscale_xy=downscale_xy
        )

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
    return (
        aggregate(image2_res[axis]) - aggregate(image1_res[axis])
    ) * 1000  # return diff in nm
