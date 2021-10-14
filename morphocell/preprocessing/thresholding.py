"""Implement 3D image thresholding."""

from typing import List
import numpy.typing as npt

import numpy as np
from skimage.filters import gaussian, threshold_otsu

from ..image_utils import get_xy_block_coords, get_xy_block


def threshold(image: npt.ArrayLike, blur_sigma=5) -> npt.ArrayLike:
    """Perform Otsu's thresholding with Gaussian blur."""
    image = gaussian(image, sigma=blur_sigma)
    thresh = threshold_otsu(image)
    binary = image > thresh
    return binary


def select_nonempty_patches(
    image: npt.ArrayLike,
    patch_size: int = 512,
    min_nonzeros: float = 0.02,
    verbose: bool = False,
) -> List[List[int]]:
    """Select XY patches from 3D image by percent of nonzero voxels."""
    verboseprint = print if verbose else lambda *a, **k: None

    selected_patches = []
    binary_image = threshold(image)
    patch_coordinates = get_xy_block_coords(image.shape, patch_size)
    verboseprint(f"Nonzero pixels in the image: {np.count_nonzero(binary_image) / binary_image.size}")

    for single_patch_coords in patch_coordinates:
        binary_tile = get_xy_block(binary_image, single_patch_coords)
        patch_nonzero = np.count_nonzero(binary_tile) / binary_tile.size

        if patch_nonzero >= min_nonzeros:
            selected_patches.append(single_patch_coords)

    return selected_patches
