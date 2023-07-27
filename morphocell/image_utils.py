"""Implements utility functions that operate on 3D images."""

from typing import Tuple, Union, List, Dict, Sequence, Optional
import numpy.typing as npt

import numpy as np

from .gpu import get_image_method, get_array_module

# image operations assume ZYX channel order


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
    skimage_rescale = get_image_method(image, "skimage.transform.rescale")
    scale_by = scale if image.ndim == 2 else (1.0, scale, scale)
    return_dtype = image.dtype if preserve_range else np.float32
    return skimage_rescale(image, scale_by, preserve_range=preserve_range, anti_aliasing=anti_aliasing).astype(
        return_dtype
    )


def rescale_isotropic(
    img: npt.ArrayLike,
    voxel_sizes: Union[Tuple[int, ...], Tuple[float, ...]],
    downscale_xy: bool = False,
    order: int = 3,
    target_z_size: Optional[int] = None,
    deps: Optional[Dict] = None,
) -> npt.ArrayLike:
    """Rescale image to isotropic voxels with arbitary Z size."""
    skimage_rescale = get_image_method(img, "skimage.transform.rescale")

    z_size_per_spacing = img.shape[0] * voxel_sizes[0] / np.asarray(voxel_sizes)
    if target_z_size is None:
        target_z_size = img.shape[0] if downscale_xy else np.round(z_size_per_spacing[1])
    factors = target_z_size / z_size_per_spacing
    return skimage_rescale(img, factors, order=order, preserve_range=True, anti_aliasing=downscale_xy)


def normalize_min_max(
    img: npt.ArrayLike,
    q: Tuple[float, float] = (0.1, 99.9),
    deps: Optional[Dict] = None,
):
    """Normalize image intensities between percentiles."""
    skimage_rescale_intensity = get_image_method(img, "skimage.exposure.rescale_intensity")
    vmin, vmax = np.percentile(img, q=q)
    return skimage_rescale_intensity(img, in_range=(vmin, vmax), out_range=np.float32)


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


def crop_tl(img: npt.ArrayLike, crop_hw: Union[int, Tuple[int, int]]):
    """Crop left top corner."""
    crop_h, crop_w = (crop_hw, crop_hw) if isinstance(crop_hw, int) else crop_hw
    return img[:, :crop_h, :crop_w]


def crop_bl(img: npt.ArrayLike, crop_hw: Union[int, Tuple[int, int]]):
    """Crop left bottom corner."""
    crop_h, crop_w = (crop_hw, crop_hw) if isinstance(crop_hw, int) else crop_hw
    return img[:, -crop_h:, :crop_w]


def crop_tr(img: npt.ArrayLike, crop_hw: Union[int, Tuple[int, int]]):
    """Crop right top corner."""
    crop_h, crop_w = (crop_hw, crop_hw) if isinstance(crop_hw, int) else crop_hw
    return img[:, :crop_h, -crop_w:]


def crop_br(img: npt.ArrayLike, crop_hw: Union[int, Tuple[int, int]]):
    """Crop right bottom corner."""
    crop_h, crop_w = (crop_hw, crop_hw) if isinstance(crop_hw, int) else crop_hw
    return img[:, -crop_h:, -crop_w:]


def crop_center(
    img: npt.ArrayLike, crop_size: Optional[Union[int, Sequence[int]]] = None, axes: Optional[Sequence[int]] = None
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
