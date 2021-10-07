"""Implement 3D image thresholding."""

from typing import Tuple, Optional, List, Dict, Any, Union
import numpy.typing as npt

import numpy as np
from skimage.filters import gaussian, threshold_otsu

from ..image_utils import get_xy_block_coords, get_xy_block


def threshold(image: npt.ArrayLike, blur_sigma=5) -> npt.ArrayLike:
    """Perform Otsu's thresholding with Gaussian blur."""
    image = gaussian(image, sigma=blur_sigma)
    thresh = threshold_otsu(image)
    binary = image > thresh
    return binary.astype


def select_binary_patches(
    image: npt.ArrayLike,
    scales: Tuple[float, ...],
    patch_coordinates: Optional[List[List[int]]] = None,
    min_nonzeros: float = 0.02,
    patch_size: int = 512,
    num_z_slices: int = 32,
    verbose: bool = False,
) -> List[Dict[Union[List[int], Any], Tuple[Any, Any]]]:
    """Select XY patches from 3D image by percent of nonzero voxels."""
    verboseprint = print if verbose else lambda *a, **k: None

    if patch_coordinates is None:
        z_rescale_factor = num_z_slices / image.shape[0]
        xy_rescale_factor = z_rescale_factor * (scales[1] / scales[0])
        selected_patches = []

        binary_image = threshold(image)
        verboseprint(f"Nonzero pixels in the whole FoV image: {np.count_nonzero(binary_image)/binary_image.size}")

        patch_coordinates = get_xy_block_coords(image.shape, patch_size * xy_rescale_factor)

    for patch_coords in patch_coordinates:
        image_tile = get_xy_block(image, patch_coords)
        binary_tile = get_xy_block(binary_image, patch_coords)
        patch_nonzero = np.count_nonzero(binary_tile) / binary_tile.size

        if patch_nonzero >= min_nonzeros:
            selected_patches.append({patch_coords: (image_tile, binary_tile)})

    return selected_patches
