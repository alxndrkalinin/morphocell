"""Implements 2D/3D Fourier Ring/Shell Correlation."""

from typing import Sequence, Callable

import numpy as np

from morphocell.image import Image
from morphocell.cuda import asnumpy, get_array_module
from morphocell.image_utils import (
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
    checkerboard_split,
    reverse_checkerboard_split,
)

from .frc_utils import (
    FourierRingIterator,
    AxialExcludeSectionedFourierShellIterator,
    FourierCorrelationData,
    FourierCorrelationDataCollection,
    FourierCorrelationAnalysis,
)


def _empty_aggregate(*args: np.ndarray, **kwargs) -> np.ndarray:
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


def calculate_frc(
    image: np.ndarray | Image,
    bin_delta: int = 1,
    scales: float | Sequence[float] = 1.0,
    resolution_threshold: str = "fixed",
    return_resolution: bool = True,
    verbose: bool = False,
) -> Sequence | float:
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

    frc_result = calculate_single_image_frc(
        image,
        bin_delta=bin_delta,
        resolution_threshold=resolution_threshold,
        curve_fit_type="smooth-spline",
        verbose=verbose,
    )

    frc_result = (
        frc_result.resolution["resolution"] if return_resolution else frc_result
    )
    return frc_result


def calculate_single_image_frc(
    image: np.ndarray | Image,
    *,
    bin_delta: int = 1,
    resolution_threshold: str = "fixed",
    threshold_value: float = 0.143,
    snr_value: float = 7.0,
    curve_fit_type: str = "spline",
    curve_fit_degree: int = 3,
    disable_hamming: bool = False,
    verbose: bool = False,
    average: bool = True,
    z_correction: float = 1.0,
    scales: float | Sequence[float] = 1.0,
    zero_padding: bool = True,
) -> FourierCorrelationData:
    """Calculate a regular FRC with a single image input."""

    image = preprocess_images((image,), scales, zero_padding, verbose)[0]
    assert isinstance(image, Image)
    assert len(image.spacing) == 2

    frc_data = FourierCorrelationDataCollection()

    # Hamming Windowing
    if not disable_hamming:
        spacing = image.spacing
        device = image.device
        image = Image(hamming_window(image.data), spacing, device=device)

    image1, image2 = frc_checkerboard_split(image)
    assert tuple(image1.shape) == tuple(image2.shape)
    assert tuple(image1.spacing) == tuple(image2.spacing)

    # Run FRC
    iterator = FourierRingIterator(image1.shape, bin_delta)
    frc_task = FRC(image1.data, image2.data, iterator)
    frc_data[0] = frc_task.execute()

    if average:
        # Split again with reverse pattern
        image1, image2 = frc_checkerboard_split(image, reverse=True)
        iterator = FourierRingIterator(image1.shape, bin_delta)
        frc_task = FRC(image1.data, image2.data, iterator)

        frc_data[0].correlation["correlation"] *= 0.5
        frc_data[0].correlation["correlation"] += (
            0.5 * frc_task.execute().correlation["correlation"]
        )

    # Analyze results
    analyzer = FourierCorrelationAnalysis(
        frc_data,
        image1.spacing[0],
        resolution_threshold=resolution_threshold,
        threshold_value=threshold_value,
        snr_value=snr_value,
        curve_fit_type=curve_fit_type,
        curve_fit_degree=curve_fit_degree,
        verbose=verbose,
    )

    result = analyzer.execute(z_correction=z_correction)[0]
    point = result.resolution["resolution-point"][1]

    # Apply cut-off correction for single image FRC
    def func(x, a, b, c, d):
        return a * np.exp(c * (x - b)) + d

    params = [0.95988146, 0.97979108, 13.90441896, 0.55146136]
    cut_off_correction = func(point, *params)
    result.resolution["spacing"] /= cut_off_correction
    result.resolution["resolution"] /= cut_off_correction

    return result


def preprocess_images(
    images: np.ndarray | Image | Sequence[np.ndarray | Image],
    scales: float | Sequence[float] = 1.0,
    zero_padding: bool = True,
    verbose: bool = False,
) -> Sequence[Image]:
    """Preprocess input images (works for both 2D and 3D)."""
    verboseprint = print if verbose else lambda *a, **k: None

    if not isinstance(images, tuple):
        images = (images,)

    image_shape = None
    images_processed = []

    for img in images:
        if not isinstance(img, Image):
            xp = get_array_module(img)
            ndarray = getattr(xp, "ndarray")  # avoid mypy complains
            if not isinstance(img, ndarray):
                raise ValueError(
                    "Incorrect input, should be 2D/3D Image, Numpy or CuPy array."
                )

        # Get the actual array data for dimension checking
        img_data = img.data if isinstance(img, Image) else img
        img_ndim = img_data.ndim

        # Handle scaling based on image dimensions
        if isinstance(scales, (int, float)):
            scales = [scales] * img_ndim  # Scale to match image dimensions
        elif len(scales) != img_ndim:
            raise ValueError(
                f"Scales length ({len(scales)}) must match image dimensions ({img_ndim})"
            )

        # Apply cubic/square padding if requested
        if len(set(img_data.shape)) > 1 and zero_padding:
            img_data = pad_image_to_cube(img_data)

            # Verify shape after padding
            expected_msg = "cube" if img_ndim == 3 else "square"
            assert len(set(img_data.shape)) == 1, (
                f"Image should be {expected_msg} after padding."
            )

            # Update img with padded data
            if isinstance(img, Image):
                img = Image(img_data, img.spacing, device=img.device)
            else:
                img = img_data

        # Ensure all images have the same shape
        current_shape = img.shape if isinstance(img, Image) else img.shape
        if image_shape is None:
            image_shape = current_shape
        elif image_shape != current_shape:
            raise ValueError("All input images should have the same shape.")

        # Create Image object if needed
        if not isinstance(img, Image):
            img = Image(img, scales)

        dimensionality = "3D" if img_ndim == 3 else "2D"
        verboseprint(
            f"{dimensionality} image dimensions are {img.shape} and spacing {img.spacing} um."
        )
        images_processed.append(img)

    return images_processed


class DirectionalFSC(object):
    """Calculate the directional FSC between two images."""

    def __init__(self, image1, image2, iterator, normalize_power=False):
        """Initialize the directional FSC."""

        if image1.ndim != 3 or image1.shape[0] <= 1:
            raise ValueError("Image must be 3D")

        if image1.shape != image2.shape:
            raise ValueError("Image dimensions do not match")

        self.iterator = iterator
        self.fft_image1 = np.fft.fftshift(np.fft.fftn(image1))
        self.fft_image2 = np.fft.fftshift(np.fft.fftn(image2))
        if normalize_power:
            pixels = image1.shape[0] ** 3
            self.fft_image1 /= np.array(pixels * np.mean(image1))
            self.fft_image2 /= np.array(pixels * np.mean(image2))

        self._result = None

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

        # calculate FRC for every orientation
        for i in range(angles.size):
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
    image: np.ndarray,
    *,
    bin_delta: int = 1,
    scales: float | Sequence[float] = 1.0,
    crop_size: int = 512,
    pad_mode: str = "reflect",
    return_resolution: bool = True,
    aggregate: Callable | None = np.median,
    verbose: bool = False,
) -> dict[str, np.ndarray]:
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
        max_projection_resolution = fsc_resolution(
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
                fsc_resolution(
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
                fsc_resolution(
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
    image: np.ndarray,
    *,
    bin_delta: int = 1,
    scales: float | Sequence[float] = 1.0,
    crop_size: int = 512,
    pad_mode: str = "reflect",
    return_resolution: bool = True,
    aggregate: Callable = np.median,
    verbose: bool = False,
) -> dict[str, np.ndarray]:
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

        max_projection_resolution = fsc_resolution(
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
                fsc_resolution(
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
                fsc_resolution(
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
    image1: np.ndarray,
    image2: np.ndarray,
    *,
    scales: float | tuple[float, float] = 1.0,
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


def fsc_resolution(
    image1: np.ndarray | Image,
    image2: np.ndarray | Image | None = None,
    *,
    bin_delta: int = 10,
    zero_padding: bool = True,
    scales: float | Sequence[float] = 1.0,
    verbose: bool = False,
):
    """Calculate either single- or two-image FSC-based 3D image resolution."""

    fsc_result = calculate_sectioned_fsc(
        image1,
        image2,
        bin_delta=bin_delta,
        verbose=verbose,
        scales=scales,
        zero_padding=zero_padding,
    )

    angle_to_resolution = {
        int(angle): dataset.resolution["resolution"] for angle, dataset in fsc_result
    }

    return {"xy": angle_to_resolution[0], "z": angle_to_resolution[90]}


def calculate_sectioned_fsc(
    image1: np.ndarray | Image,
    image2: np.ndarray | Image | None = None,
    *,
    bin_delta: int = 10,
    angle_delta: int = 15,
    extract_angle_delta: float = 0.1,
    resolution_threshold: str = "fixed",
    threshold_value: float = 0.143,
    snr_value: float = 7.0,
    curve_fit_type: str = "spline",
    curve_fit_degree: int = 3,
    disable_hamming: bool = False,
    verbose: bool = False,
    z_correction: float = 1.0,
    disable_3d_sum: bool = False,
    scales: float | Sequence[float] = 1.0,
    zero_padding: bool = True,
) -> FourierCorrelationDataCollection:
    """Calculate sectioned FSC for one or two images."""

    single_image = image2 is None

    img_cubes = (image1,) if image2 is None else (image1, image2)
    img_cubes = preprocess_images(img_cubes, scales, zero_padding, verbose)

    image1 = img_cubes[0]
    if single_image:
        assert all(s == image1.shape[0] for s in image1.shape)
        image1, image2 = frc_checkerboard_split(image1, disable_3d_sum=disable_3d_sum)
    else:
        image2 = img_cubes[1]
        assert isinstance(image2, Image)

    if not disable_hamming:
        image1 = Image(hamming_window(image1.data), image1.spacing)
        image2 = Image(hamming_window(image2.data), image2.spacing)

    # Create iterator and calculate FSC
    iterator = AxialExcludeSectionedFourierShellIterator(
        image1.shape,
        bin_delta,
        angle_delta,
        extract_angle_delta,
    )
    fsc_task = DirectionalFSC(image1.data, image2.data, iterator)
    data = fsc_task.execute()

    analyzer = FourierCorrelationAnalysis(
        data,
        image1.spacing[0],
        resolution_threshold=resolution_threshold,
        threshold_value=threshold_value,
        snr_value=snr_value,
        curve_fit_type=curve_fit_type,
        curve_fit_degree=curve_fit_degree,
        verbose=verbose,
    )
    result = analyzer.execute(z_correction=z_correction)

    # Apply cut-off correction (only for single image case)
    if single_image:

        def func(x, a, b, c, d):
            return a * np.exp(c * (x - b)) + d

        params = [0.95988146, 0.97979108, 13.90441896, 0.55146136]

        for angle, dataset in result:
            point = dataset.resolution["resolution-point"][1]
            cut_off_correction = func(point, *params)
            dataset.resolution["spacing"] /= cut_off_correction
            dataset.resolution["resolution"] /= cut_off_correction

    return result
