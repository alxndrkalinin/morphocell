"""Implements utility functions that operate on 3D images."""

from typing import Tuple, Union, List, Dict, Sequence, Optional
import numpy.typing as npt

import numpy as np

from .cuda import get_array_module, asnumpy
from .skimage import transform, exposure, measure

# image operations assume ZYX channel order


# def skimage_func(func_name: str):
#     """Wrap skimage functions to return device-specific implementation."""

#     def wrapper(*args, **kwargs) -> npt.ArrayLike:
#         """Return device-specific implementation."""
#         conversion_func = get_image_method(args[0], f"skimage.{func_name}")
#         return conversion_func(*args, **kwargs)

#     return wrapper


def image_stats(
    img: npt.ArrayLike,
    q: Tuple[float, float] = (0.1, 99.9),
) -> Dict[str, float]:
    """Compute intensity image statistics (min, max, mean, percentiles)."""
    q_min, q_max = np.percentile(img, q=q)
    return {
        "min": np.min(img),
        "max": np.max(img),
        "mean": np.mean(img),
        "percentile_min": q_min,
        "precentile_max": q_max,
    }


def rescale_xy(image: npt.ArrayLike, scale: float = 1.0, anti_aliasing: bool = True, preserve_range: bool = False):
    """Rescale 2D image or 3D image in XY."""
    scale_by = scale if image.ndim == 2 else (1.0, scale, scale)
    return_dtype = image.dtype if preserve_range else np.float32
    return transform.rescale(image, scale_by, preserve_range=preserve_range, anti_aliasing=anti_aliasing).astype(
        return_dtype
    )


def rescale_isotropic(
    img: npt.ArrayLike,
    voxel_sizes: Union[Tuple[int, ...], Tuple[float, ...]],
    downscale_xy: bool = False,
    order: int = 3,
    preserve_range: bool = True,
    target_z_size: Optional[int] = None,
    target_z_voxel_size: Optional[float] = None,
    deps: Optional[Dict] = None,
) -> npt.ArrayLike:
    """Rescale image to isotropic voxels with arbitary Z (voxel) size."""
    if target_z_voxel_size is not None:
        target_z_size = int(round(img.shape[0] * (voxel_sizes[0] / target_z_voxel_size)))

    z_size_per_spacing = img.shape[0] * voxel_sizes[0] / np.asarray(voxel_sizes)
    if target_z_size is None:
        target_z_size = img.shape[0] if downscale_xy else np.round(z_size_per_spacing[1])
    factors = target_z_size / z_size_per_spacing
    return transform.rescale(img, factors, order=order, preserve_range=preserve_range, anti_aliasing=downscale_xy)


def normalize_min_max(
    img: npt.ArrayLike,
    q: Tuple[float, float] = (0.1, 99.9),
    deps: Optional[Dict] = None,
):
    """Normalize image intensities between percentiles."""
    vmin, vmax = np.percentile(img, q=q)
    return exposure.rescale_intensity(img, in_range=(vmin, vmax), out_range=np.float32)


def max_project(
    img: npt.ArrayLike,
    axis: int = 0,
):
    """Compute maximum intensity projection along the chosen axis."""
    return np.max(img, axis)


def img_mse(
    a,
    b,
) -> int:
    """Calculate pixel-wise MSE between two images."""
    assert len(a) == len(b)
    return np.square(a - b).mean()


def pad_image(
    img: npt.ArrayLike,
    pad_size: Union[int, Sequence[int]],
    axes: Union[int, Sequence[int]] = 0,
    mode: str = "reflect",
    deps: Optional[Dict] = None,
):
    """Pad an image."""
    npad = np.asarray([(0, 0)] * img.ndim)
    axes = [axes] if isinstance(axes, int) else axes
    for ax in axes:
        npad[ax] = [pad_size] * 2 if isinstance(pad_size, int) else [pad_size[ax]] * 2
    return np.pad(img, pad_width=npad, mode=mode)


def pad_image_to_cube(
    img: npt.ArrayLike, cube_size: Optional[int] = None, mode: str = "reflect", axes: Optional[Sequence[int]] = None
):
    """Pad all image axes up to cubic shape."""
    axes = list(range(img.ndim)) if axes is None else axes
    cube_size = cube_size if cube_size is not None else np.max(img.shape)

    pad_sizes = [(0, 0)] * img.ndim
    for ax in axes:
        dim = img.shape[ax]
        if dim < cube_size:
            pad_before = (cube_size - dim) // 2
            pad_after = cube_size - dim - pad_before
            pad_sizes[ax] = (pad_before, pad_after)

    img = np.pad(img, pad_sizes, mode=mode)
    assert np.all([img.shape[i] == cube_size for i in axes])
    return img


def pad_image_to_shape(img: npt.ArrayLike, new_shape: Sequence, mode: str = "constant"):
    """Pad all image axis up to specified shape."""
    for i, dim in enumerate(img.shape):
        if dim < new_shape[i]:
            pad_size = (new_shape[i] - dim) // 2
            img = pad_image(img, pad_size=pad_size, axes=i, mode=mode)

    assert np.all([dim == new_shape[i] for i, dim in enumerate(img.shape)])
    return img


def pad_images_to_matching_shape(image1: npt.ArrayLike, image2: npt.ArrayLike, mode: str = "constant"):
    """Apply zero padding to make the size of two Images match."""
    shape = tuple(max(x, y) for x, y in zip(image1.shape, image2.shape))

    if any(map(lambda x, y: x != y, image1.shape, shape)):
        image1 = pad_image_to_shape(image1, shape, mode=mode)
    if any(map(lambda x, y: x != y, image2.shape, shape)):
        image2 = pad_image_to_shape(image2, shape, mode=mode)

    return image1, image2


def crop_tl(
    img: npt.ArrayLike, crop_size: Union[int, Sequence[int]], axes: Optional[Sequence[int]] = None
) -> npt.ArrayLike:
    """Crop from the top-left corner."""
    return crop_corner(img, crop_size, axes, "tl")


def crop_bl(
    img: npt.ArrayLike, crop_size: Union[int, Sequence[int]], axes: Optional[Sequence[int]] = None
) -> npt.ArrayLike:
    """Crop from the bottom-left corner."""
    return crop_corner(img, crop_size, axes, "bl")


def crop_tr(
    img: npt.ArrayLike, crop_size: Union[int, Sequence[int]], axes: Optional[Sequence[int]] = None
) -> npt.ArrayLike:
    """Crop from the top-right corner."""
    return crop_corner(img, crop_size, axes, "tr")


def crop_br(
    img: npt.ArrayLike, crop_size: Union[int, Sequence[int]], axes: Optional[Sequence[int]] = None
) -> npt.ArrayLike:
    """Crop from the bottom-right corner."""
    return crop_corner(img, crop_size, axes, "br")


def crop_corner(
    img: npt.ArrayLike, crop_size: Union[int, Sequence[int]], axes: Optional[Sequence[int]] = None, corner: str = "tl"
) -> npt.ArrayLike:
    """Crop a corner from the image."""
    axes = [1, 2] if axes is None else axes
    crop_size = [crop_size] * len(axes) if isinstance(crop_size, int) else crop_size

    if len(crop_size) != len(axes):
        raise ValueError("Length of 'crop_sizes' must match the length of 'axes'.")

    slices = [slice(None)] * img.ndim

    for axis, size in zip(axes, crop_size):
        if axis >= img.ndim:
            raise ValueError("Axis index out of range for the image dimensions.")

        if "t" in corner or "l" in corner:
            start = 0
            end = size if size <= img.shape[axis] else img.shape[axis]
        else:
            start = -size if size <= img.shape[axis] else -img.shape[axis]
            end = None

        if "r" in corner and axis == axes[-1]:
            slices[axis] = slice(-end if end is not None else None, None)
        elif "b" in corner and axis == axes[-2]:
            slices[axis] = slice(-end if end is not None else None, None)
        else:
            slices[axis] = slice(start, end)

    return img[tuple(slices)]


def crop_center(
    img: npt.ArrayLike, crop_size: Optional[Union[int, Sequence[int]]], axes: Optional[Sequence[int]] = None
):
    """Crop from the center of the n-dimensional image."""
    axes = list(range(img.ndim)) if axes is None else axes
    if crop_size is None:
        crop_size = [min(img.shape[axis] for axis in axes)] * len(axes)
    elif isinstance(crop_size, int):
        crop_size = [crop_size] * len(axes)

    if len(axes) != len(crop_size):
        raise ValueError("The length of 'axes' should be the same as 'crop_size'")

    slices = []
    for axis in range(img.ndim):
        if axis in axes and img.shape[axis] > crop_size[axes.index(axis)]:
            idx = axes.index(axis)
            center = img.shape[axis] // 2
            half_crop = crop_size[idx] // 2
            start = center - half_crop
            end = center + half_crop + crop_size[idx] % 2
            slices.append(slice(start, end))
        else:
            slices.append(slice(None))

    return img[tuple(slices)]


def get_random_crop_coords(
    height: int,
    width: int,
    crop_height: int,
    crop_width: int,
    h_start: float,
    w_start: float,
) -> Tuple[int, int, int, int]:
    """Crop from a random location in the image."""
    y1 = int((height - crop_height) * h_start)
    y2 = y1 + crop_height
    x1 = int((width - crop_width) * w_start)
    x2 = x1 + crop_width
    return x1, y1, x2, y2


def random_crop(
    img: npt.ArrayLike,
    crop_hw: Union[int, Tuple[int, int]],
    return_coordinates: bool = False,
):
    """Crop from a random location in the image."""
    crop_h, crop_w = (crop_hw, crop_hw) if isinstance(crop_hw, int) else crop_hw
    height, width = img.shape[1:]
    h_start, w_start = np.random.uniform(), np.random.uniform()
    x1, y1, x2, y2 = get_random_crop_coords(height, width, crop_h, crop_w, h_start, w_start)
    if return_coordinates:
        return (img[y1:y2, x1:x2], (y1, y2, x1, x2))
    else:
        return img[y1:y2, x1:x2]


def crop_to_divisor(
    img: npt.ArrayLike,
    divisors: Union[int, Sequence[int]],
    axes: Optional[Sequence[int]] = None,
    crop_type: str = "center",
) -> npt.ArrayLike:
    """Crop image to be divisible by the given divisors along specified axes."""
    if axes is None:
        axes = [1, 2]  # default to xy axes in a 3d image

    if isinstance(divisors, int):
        divisors = [divisors] * len(axes)

    if len(axes) != len(divisors):
        raise ValueError("Length of 'axes' and 'divisors' must be the same")

    crop_size = [img.shape[axis] - (img.shape[axis] % divisor) for axis, divisor in zip(axes, divisors)]

    if crop_type == "center":
        return crop_center(img, crop_size=crop_size, axes=axes)
    elif crop_type == "tl":
        return crop_tl(img, crop_size)
    elif crop_type == "bl":
        return crop_bl(img, crop_size)
    elif crop_type == "tr":
        return crop_tr(img, crop_size)
    elif crop_type == "br":
        return crop_br(img, crop_size)
    else:
        raise ValueError("Invalid crop type specified. Choose from 'center', 'tl', 'bl', 'tr', 'br'.")


def get_xy_block_coords(image_shape: npt.ArrayLike, crop_hw: Union[int, Tuple[int, int]]):
    """Compute coordinates of non-overlapping image blocks of specified shape."""
    crop_h, crop_w = (crop_hw, crop_hw) if isinstance(crop_hw, int) else crop_hw
    height, width = image_shape[1:]

    block_coords = []  # type: List[Tuple[int, ...]]
    for y in np.arange(0, height // crop_h) * crop_h:
        block_coords.extend((y, y + crop_h, x, x + crop_w) for x in np.arange(0, width // crop_w) * crop_w)

    return np.asarray(block_coords).astype(int)


def get_xy_block(image: npt.ArrayLike, patch_coordinates: List[int]):
    """Slice subvolume of 3D image by XY coordinates."""
    return image[:, patch_coordinates[0] : patch_coordinates[1], patch_coordinates[2] : patch_coordinates[3]]


def extract_patches(image: npt.ArrayLike, patch_coordinates: List[List[int]]):
    """Extract 3D patches from image given XY coordinates."""
    return [get_xy_block(image, patch_coords) for patch_coords in patch_coordinates]


def _nd_window(data, filter_function, power_function, **kwargs):
    """
    Perform on N-dimensional spatial-domain data to mitigate boundary effects in the FFT.

    Parameters
    ----------
    data : ndarray
           Input data to be windowed, modified in place.
    filter_function : 1D window generation function
           Function should accept one argument: the window length.
           Example: scipy.signal.hamming
    """
    result = data.copy().astype(np.float32)
    for axis, axis_size in enumerate(data.shape):
        # set up shape for numpy broadcasting
        filter_shape = [
            1,
        ] * data.ndim
        filter_shape[axis] = axis_size
        window = filter_function(axis_size, **kwargs).reshape(filter_shape)
        # scale the window intensities to maintain array intensity
        power_function(window, (1.0 / data.ndim), out=window)
        result *= window
    return result


def hamming_window(data):
    """Apply Hamming window to data."""
    xp = get_array_module(data)
    return _nd_window(data, xp.hamming, xp.power)


def checkerboard_split(image, disable_3d_sum=False):
    """Split an image in two, by using a checkerboard pattern."""
    # Make an index chess board structure
    shape = image.shape
    odd_index = [np.arange(1, shape[i], 2) for i in range(len(shape))]
    even_index = [np.arange(0, shape[i], 2) for i in range(len(shape))]

    # Create the two pseudo images
    if image.ndim == 2:
        image1 = image[odd_index[0], :][:, odd_index[1]]
        image2 = image[even_index[0], :][:, even_index[1]]
    elif disable_3d_sum:
        image1 = image[odd_index[0], :, :][:, odd_index[1], :][:, :, odd_index[2]]
        image2 = image[even_index[0], :, :][:, even_index[1], :][:, :, even_index[2]]

    else:
        image1 = (
            image.astype(np.uint32)[even_index[0], :, :][:, odd_index[1], :][:, :, odd_index[2]]
            + image.astype(np.uint32)[odd_index[0], :, :][:, odd_index[1], :][:, :, odd_index[2]]
        )

        image2 = (
            image.astype(np.uint32)[even_index[0], :, :][:, even_index[1], :][:, :, even_index[2]]
            + image.astype(np.uint32)[odd_index[0], :, :][:, even_index[1], :][:, :, even_index[2]]
        )

    return image1, image2


def reverse_checkerboard_split(image, disable_3d_sum=False):
    """Split an image in two, by using a checkerboard pattern."""
    # Make an index chess board structure
    shape = image.shape
    odd_index = [np.arange(1, shape[i], 2) for i in range(len(shape))]
    even_index = [np.arange(0, shape[i], 2) for i in range(len(shape))]

    # Create the two pseudo images
    if image.ndim == 2:
        image1 = image[odd_index[0], :][:, even_index[1]]
        image2 = image[even_index[0], :][:, odd_index[1]]
    elif disable_3d_sum:
        image1 = image[odd_index[0], :, :][:, odd_index[1], :][:, :, even_index[2]]
        image2 = image[even_index[0], :, :][:, even_index[1], :][:, :, odd_index[2]]

    else:
        image1 = (
            image.astype(np.uint32)[even_index[0], :, :][:, odd_index[1], :][:, :, even_index[2]]
            + image.astype(np.uint32)[odd_index[0], :, :][:, even_index[1], :][:, :, odd_index[2]]
        )

        image2 = (
            image.astype(np.uint32)[even_index[0], :, :][:, even_index[1], :][:, :, odd_index[2]]
            + image.astype(np.uint32)[odd_index[0], :, :][:, odd_index[1], :][:, :, even_index[2]]
        )

    return image1, image2


def label(image: npt.ArrayLike, **kwargs) -> npt.ArrayLike:
    """Label image using skimage.measure.label."""
    return measure.label(image, **kwargs)


def select_max_contrast_slices(img, num_slices=128, return_indices=False):
    """
    Select num_slices consecutive Z slices with maximum contrast from a 3D volume.

    Parameters:
        volume (numpy.ndarray): The 3D image volume. Assumes ZYX format.
        num_slices (int): Number of consecutive slices to select.

    Returns:
        numpy.ndarray: The selected slices.
    """
    assert img.ndim > 2, "Image should have more than 2 dimensions."
    std_devs = asnumpy(img.std(tuple(range(1, img.ndim))))
    # calculate rolling sum of standard deviations for num_slices
    rolling_sum = np.convolve(std_devs, np.ones(num_slices), "valid")
    max_contrast_idx = np.argmax(rolling_sum)
    indices = slice(max_contrast_idx, max_contrast_idx + num_slices)
    if return_indices:
        return img[indices], indices
    return img[indices]


def distance_transform_edt(
    image,
    sampling=None,
    return_distances=True,
    return_indices=False,
    distances=None,
    indices=None,
    block_params=None,
    float64_distances=False,
):
    """Compute the Euclidean distance transform of a binary image."""
    if isinstance(image, np.ndarray):
        if block_params is not None or float64_distances:
            raise ValueError(
                "NumPy array found. 'block_params' and 'float64_distances' can only be used with CuPy arrays."
            )
        from scipy.ndimage import distance_transform_edt

        return distance_transform_edt(
            image, sampling=None, return_distances=True, return_indices=False, distances=None, indices=None
        )
    else:
        # cuCIM access interface is different from scipy.ndimage
        from cucim.core.operations.morphology import distance_transform_edt

        return distance_transform_edt(
            image,
            sampling=None,
            return_distances=True,
            return_indices=False,
            distances=None,
            indices=None,
            block_params=None,
            float64_distances=False,
        )


def clahe(img, kernel_size=(2, 3, 5), clip_limit=0.01, nbins=256):
    """Apply CLAHE to the image."""
    assert len(img.shape) == len(kernel_size)
    kernel_size = np.asarray(img.shape) // kernel_size
    img = exposure.equalize_adapthist(img, kernel_size=kernel_size, clip_limit=clip_limit, nbins=nbins)
    return img
