"""Implement pre- and post-processing for segmentation."""

import numpy.typing as npt

import numpy as np
from ..gpu import get_image_method
from ..image_utils import pad_image


def downsample_and_filter(image: npt.ArrayLike, downscale_factor: float = 0.5) -> npt.ArrayLike:
    """Subsample and filter image before segmenting."""
    skimage_rescale = get_image_method(image, "skimage.transform.rescale")
    skimage_median = get_image_method(image, "skimage.filters.median")
    image = skimage_rescale(image, downscale_factor, order=3, preserve_range=True, anti_aliasing=True).astype(
        np.uint16
    )
    image = skimage_median(image)
    return image


def remove_small_objects(label_image: npt.ArrayLike, min_size: int = 500) -> npt.ArrayLike:
    """Remove objects with volume below specified threshold."""
    for mask_idx in np.unique(label_image)[1:]:
        if (label_image == mask_idx).sum() < min_size:
            label_image[label_image == mask_idx] = 0
    return label_image


def clear_xy_borders(label_image: npt.ArrayLike) -> npt.ArrayLike:
    """Remove masks that touch XY borders."""
    skimage_clear_border = get_image_method(label_image, "skimage.segmentation.clear_border")
    label_image = pad_image(label_image, (1, 1), mode="constant")
    label_image = skimage_clear_border(label_image)[1:-1, :, :]
    return label_image


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
