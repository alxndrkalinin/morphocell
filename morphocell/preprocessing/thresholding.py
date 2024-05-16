"""Implement 3D image thresholding."""

from typing import List, Optional
import numpy.typing as npt

import numpy as np

from ..image_utils import get_xy_block_coords, get_xy_block

from ..skimage import filters, util


def get_threshold_otsu(image: npt.ArrayLike, blur_sigma=5):
    """Perform Otsu's thresholding with Gaussian blur."""
    image = filters.gaussian(image, sigma=blur_sigma)
    return filters.threshold_otsu(image)


def select_nonempty_patches(
    image: npt.ArrayLike,
    patch_size: int = 512,
    min_nonzeros: float = 0.02,
    threshold: Optional[float] = None,
    verbose: bool = False,
) -> List[List[int]]:
    """Select XY patches from 3D image by percent of nonzero voxels."""
    verboseprint = print if verbose else lambda *a, **k: None

    selected_patches = []

    if threshold is None:
        threshold = get_threshold_otsu(image)

    binary_image = (util.img_as_float32(image) > threshold).astype(np.uint8)

    patch_coordinates = get_xy_block_coords(image.shape, patch_size)
    verboseprint(f"Nonzero pixels in the image: {binary_image.mean()}")  # type: ignore[operator]

    for single_patch_coords in patch_coordinates:
        binary_tile = get_xy_block(binary_image, single_patch_coords)
        patch_nonzero = np.count_nonzero(binary_tile) / binary_tile.size

        if patch_nonzero >= min_nonzeros:
            selected_patches.append(single_patch_coords)

    return selected_patches
