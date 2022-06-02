"""Implement pre- and post-processing for segmentation."""

from typing import Optional
import numpy.typing as npt

import numpy as np

from ._clear_border import clear_border

from ..gpu import get_image_method
from ..image_utils import pad_image


def downscale_and_filter(image: npt.ArrayLike, downscale_factor: float = 0.5, filter_size: int = 3) -> npt.ArrayLike:
    """Subsample and filter image prior to segmentiation."""
    skimage_rescale = get_image_method(image, "skimage.transform.rescale")
    # cuCIM does not yet support rank-based median filter that is faster on integer values
    skimage_median = get_image_method(image, "skimage.filters.median")
    skimage_cube = get_image_method(image, "skimage.morphology.cube")

    if downscale_factor < 1.0:
        image = skimage_rescale(image, downscale_factor, order=3, anti_aliasing=True)

    return skimage_median(image, footprint=skimage_cube(filter_size))


def cleanup_segmentation(
    image: npt.ArrayLike,
    min_obj_size: Optional[int] = None,
    border_buffer_size: Optional[int] = None,
    max_hole_size: Optional[int] = None,
) -> npt.ArrayLike:
    """Clean up segmented image by removing small objects, clearing borders, and closing holes."""
    label = get_image_method(image.data, "skimage.measure.label")

    # both transforms preserve labels
    if min_obj_size is not None:
        remove_small_objects = get_image_method(image, "skimage.morphology.remove_small_objects")
        image = remove_small_objects(image, min_size=min_obj_size)

    if border_buffer_size is not None:
        image = clear_xy_borders(image, buffer_size=border_buffer_size)

    # returns boolean array
    if max_hole_size is not None:
        remove_holes = get_image_method(image, "skimage.morphology.remove_small_holes")
        image = remove_holes(image, area_threshold=max_hole_size)

    return label(image).astype(np.uint8)


def remove_small_objects(label_image: npt.ArrayLike, min_size: int = 500) -> npt.ArrayLike:
    """Remove objects with volume below specified threshold."""
    remove_small_objects = get_image_method(label_image, "skimage.morphology.remove_small_objects")
    label_image = remove_small_objects(label_image, min_size=min_size)
    return label_image


def clear_xy_borders(label_image: npt.ArrayLike, buffer_size: int = 0) -> npt.ArrayLike:
    """Remove masks that touch XY borders."""
    label_image = pad_image(label_image, (buffer_size + 1, buffer_size + 1), mode="constant")
    label_image = clear_border(label_image, buffer_size=buffer_size)
    return label_image[buffer_size + 1 : -(buffer_size + 1), :, :]


def remove_touching_objects(label_image: npt.ArrayLike, border_value: int = 100) -> npt.ArrayLike:
    """Find labelled masks that overlap and remove from the image."""
    skimage_binary_dilation = get_image_method(label_image, "skimage.morphology.binary_dilation")
    skimage_cube = get_image_method(label_image, "skimage.morphology.cube")
    exclude_masks = []
    for mask_idx in np.unique(label_image)[1:]:
        if mask_idx not in exclude_masks:
            binary_mask = label_image == mask_idx
            dilated_mask = skimage_binary_dilation(binary_mask, skimage_cube(3))
            mask_outline = dilated_mask & ~binary_mask

            masks_copy = label_image.copy()
            masks_copy[mask_outline] += border_value

            if masks_copy[masks_copy > border_value].sum() > 0:
                overlap_masks = np.unique(masks_copy[masks_copy > border_value]) - border_value
                exclude_masks += [mask_idx] + list(overlap_masks)

    for exclude_mask in exclude_masks:
        label_image[label_image == exclude_mask] = 0

    return label_image
