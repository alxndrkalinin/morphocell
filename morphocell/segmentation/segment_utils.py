"""Implement pre- and post-processing for segmentation."""

import numpy.typing as npt

import numpy as np
from .gpu import get_image_method


def downsample_and_filter(image: npt.ArrayLike, downscale_factor: float = 0.5) -> npt.ArrayLike:
    """Subsample and filter image before segmenting."""
    skimage_rescale = get_image_method(image, "skimage.transform.rescale")
    skimage_median = get_image_method(image, "skimage.filter.median")
    image = skimage_rescale(image, downscale_factor, order=3, preserve_range=True, anti_aliasing=True).astype(
        np.uint16
    )
    image = skimage_median(image)
    return image


def exclude_touching_objects(label_image: npt.ArrayLike, border_value: int = 100) -> npt.ArrayLike:
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
