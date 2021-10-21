"""Implements utility functions that operate on 3D images."""

from typing import Tuple, Union, List, Dict, Sequence, Optional
import numpy.typing as npt

import numpy as np
from .image import Image

try:
    from cupy.cuda.runtime import getDeviceCount

    if getDeviceCount() > 0:
        device_name = "GPU"
        import cupy as xp
        from cucim.skimage.transform import rescale
        from cucim.skimage.exposure import rescale_intensity

    else:
        raise
except Exception:
    device_name = "CPU"
    xp = np
    from skimage.transform import rescale
    from skimage.exposure import rescale_intensity

# image operations assume ZYX channel order


def image_stats(img: npt.ArrayLike, q: Tuple[float, float] = (0.1, 99.9)) -> Dict[str, float]:
    """Compute intensity image statistics (min, max, mean, percentiles)."""
    q_min, q_max = xp.percentile(img, q=q)
    return {
        "min": xp.min(img),
        "max": xp.max(img),
        "mean": xp.mean(img),
        "percentile_min": q_min,
        "precentile_max": q_max,
    }


def rescale_xy(img: npt.ArrayLike, factor: float = 1.0, anti_aliasing=True) -> xp.ndarray:
    """Rescale image in XY."""
    return rescale(img, (1.0, factor, factor), preserve_range=False, anti_aliasing=anti_aliasing)


def rescale_isotropic(
    img: npt.ArrayLike,
    voxel_sizes: Union[Tuple[int, ...], Tuple[float, ...]],
    downscale_xy: bool = False,
    order: int = 3,
    target_z_size: Optional[int] = None,
) -> xp.ndarray:
    """Rescale image to isotropic voxels with arbitary Z size."""
    z_size_per_spacing = (img.shape[0] * voxel_sizes[0] / np.asarray(voxel_sizes)).astype(int)

    if target_z_size is None:
        if downscale_xy:
            target_z_size = img.shape[0]
        else:
            target_z_size = z_size_per_spacing[1]

    factors = target_z_size / z_size_per_spacing
    return rescale(img, factors, order=order, preserve_range=True, anti_aliasing=downscale_xy)


def normalize_min_max(img: npt.ArrayLike, q: Tuple[float, float] = (0.1, 99.9)) -> xp.ndarray:
    """Normalize image intensities between percentiles."""
    vmin, vmax = xp.percentile(img, q=q)
    return rescale_intensity(img, in_range=(vmin, vmax), out_range=xp.float32)


def max_project(img: npt.ArrayLike, axis: int = 0) -> xp.ndarray:
    """Compute maximum intensity projection along the chosen axis."""
    return xp.max(img, axis)


def img_mse(a, b) -> int:
    """Calculate pixel-wise MSE between two images."""
    assert len(a) == len(b)
    return xp.square(a - b).mean()


def pad_image(
    img: npt.ArrayLike,
    pad_size: Union[int, Sequence[int]],
    axis: int = 0,
    mode: str = "reflect",
) -> xp.ndarray:
    """Pad an image."""
    npad = np.asarray([(0, 0)] * img.ndim)
    npad[axis] = [pad_size] * 2 if isinstance(pad_size, int) else pad_size
    return xp.pad(img, pad_width=npad, mode=mode)


def pad_image_cpu(
    img: npt.ArrayLike,
    pad_size: Union[int, Sequence[int]],
    axis: int = 0,
    mode: str = "reflect",
) -> xp.ndarray:
    """Pad an image on a CPU."""
    npad = np.asarray([(0, 0)] * img.ndim)
    npad[axis] = [pad_size] * 2 if isinstance(pad_size, int) else pad_size
    return np.pad(img, pad_width=npad, mode=mode)


def pad_image_to_cube(img: npt.ArrayLike, cube_size: int, mode: str = "reflect") -> xp.ndarray:
    """Pad all image axis up to cubic shape."""
    for i, dim in enumerate(img.shape):
        if dim < cube_size:
            pad_size = (cube_size - dim) // 2
            img = pad_image(img, pad_size=pad_size, axis=i, mode=mode)

    assert np.all([dim == cube_size for dim in img.shape])
    return img


def pad_image_to_shape(img: npt.ArrayLike, new_shape: Sequence, mode: str = "constant") -> xp.ndarray:
    """Pad all image axis up to specified shape."""
    for i, dim in enumerate(img.shape):
        if dim < new_shape[i]:
            pad_size = (new_shape[i] - dim) // 2
            img = pad_image(img, pad_size=pad_size, axis=i, mode=mode)

    assert np.all([dim == new_shape[i] for i, dim in enumerate(img.shape)])
    return img


def pad_images_to_matching_shape(image1: npt.ArrayLike, image2: npt.ArrayLike, mode: str = "constant"):
    """Apply zero padding to make the size of two Images match."""
    assert isinstance(image1, Image)
    assert isinstance(image2, Image)

    shape = tuple(max(x, y) for x, y in zip(image1.shape, image2.shape))

    if any(map(lambda x, y: x != y, image1.shape, shape)):
        image1 = pad_image_to_shape(image1, shape, mode=mode)
    if any(map(lambda x, y: x != y, image2.shape, shape)):
        image2 = pad_image_to_shape(image2, shape, mode=mode)

    return image1, image2


def crop_tl(img: npt.ArrayLike, crop_hw: Union[int, Tuple[int, int]]) -> xp.ndarray:
    """Crop left top corner."""
    crop_h, crop_w = (crop_hw, crop_hw) if isinstance(crop_hw, int) else crop_hw
    return img[:, :crop_h, :crop_w]


def crop_bl(img: npt.ArrayLike, crop_hw: Union[int, Tuple[int, int]]) -> xp.ndarray:
    """Crop left bottom corner."""
    crop_h, crop_w = (crop_hw, crop_hw) if isinstance(crop_hw, int) else crop_hw
    return img[:, -crop_h:, :crop_w]


def crop_tr(img: npt.ArrayLike, crop_hw: Union[int, Tuple[int, int]]) -> xp.ndarray:
    """Crop right top corner."""
    crop_h, crop_w = (crop_hw, crop_hw) if isinstance(crop_hw, int) else crop_hw
    return img[:, :crop_h, -crop_w:]


def crop_br(img: npt.ArrayLike, crop_hw: Union[int, Tuple[int, int]]) -> xp.ndarray:
    """Crop right bottom corner."""
    crop_h, crop_w = (crop_hw, crop_hw) if isinstance(crop_hw, int) else crop_hw
    return img[:, -crop_h:, -crop_w:]


def crop_center(img: npt.ArrayLike, crop_hw: Union[int, Tuple[int, int]]) -> xp.ndarray:
    """Crop from the center of the image."""
    crop_h, crop_w = (crop_hw, crop_hw) if isinstance(crop_hw, int) else crop_hw

    center_h = img.shape[1] // 2
    center_w = img.shape[2] // 2
    half_crop_h = crop_h // 2
    half_crop_w = crop_w // 2

    y_min = center_h - half_crop_h
    y_max = center_h + half_crop_h + crop_h % 2
    x_min = center_w - half_crop_w
    x_max = center_w + half_crop_w + crop_w % 2

    return img[:, y_min:y_max, x_min:x_max]


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
) -> xp.ndarray:
    """Crop from a random location in the image."""
    crop_h, crop_w = (crop_hw, crop_hw) if isinstance(crop_hw, int) else crop_hw
    height, width = img.shape[1:]
    h_start, w_start = np.random.uniform(), np.random.uniform()
    x1, y1, x2, y2 = get_random_crop_coords(height, width, crop_h, crop_w, h_start, w_start)
    if return_coordinates:
        return (img[y1:y2, x1:x2], (y1, y2, x1, x2))
    else:
        return img[y1:y2, x1:x2]


def get_xy_block_coords(image_shape: npt.ArrayLike, crop_hw: Union[int, Tuple[int, int]]) -> xp.ndarray:
    """Compute coordinates of non-overlapping image blocks of specified shape."""
    crop_h, crop_w = (crop_hw, crop_hw) if isinstance(crop_hw, int) else crop_hw
    height, width = image_shape[1:]

    block_coords = []
    for y in np.arange(0, height // crop_h) * crop_h:
        for x in np.arange(0, width // crop_w) * crop_w:
            block_coords.append((y, y + crop_h, x, x + crop_w))

    return np.asarray(block_coords).astype(int)


def get_xy_block(image: npt.ArrayLike, patch_coordinates: List[int]) -> xp.ndarray:
    """Slice subvolume of 3D image by XY coordinates."""
    return image[:, patch_coordinates[0] : patch_coordinates[1], patch_coordinates[2] : patch_coordinates[3]]


def extract_patches(image: npt.ArrayLike, patch_coordinates: List[List[int]]) -> List[xp.ndarray]:
    """Extract 3D patches from image given XY coordinates."""
    patches = []
    for patch_coords in patch_coordinates:
        patches.append(get_xy_block(image, patch_coords))
    return patches


def _nd_window(data, filter_function, **kwargs):
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
        xp.power(window, (1.0 / data.ndim), out=window)
        result *= window
    return result


def hamming_window(data):
    """Apply Hamming window to data."""
    assert issubclass(data.__class__, xp.ndarray)

    return _nd_window(data, xp.hamming)
