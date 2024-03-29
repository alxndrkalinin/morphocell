"""Implement pre- and post-processing for segmentation."""

import warnings

from typing import Optional
import numpy.typing as npt

import numpy as np
from skimage.segmentation import watershed

from ._clear_border import clear_border

from ..gpu import get_device, to_device, get_image_method
from ..image_utils import pad_image, label, distance_transform_edt


def downscale_and_filter(image: npt.ArrayLike, downscale_factor: float = 0.5, filter_size: int = 3) -> npt.ArrayLike:
    """Subsample and filter image prior to segmentiation.

    Parameters
    ----------
    image : npt.ArrayLike
        Image to be downsampled and filtered.
    downscale_factor : float, optional
        Factor by which to downscale the image, by default 0.5.
    filter_size : int, optional
        Size of median filter kernel, by default 3.

    Returns
    -------
    npt.ArrayLike
        Filtered and downsampled image.
    """
    skimage_rescale = get_image_method(image, "skimage.transform.rescale")
    # cuCIM does not yet support rank-based median filter that is faster on integer values
    skimage_median = get_image_method(image, "skimage.filters.median")
    if image.ndim == 2:
        skimage_footprint = get_image_method(image, "skimage.morphology.square")
    elif image.ndim == 3:
        skimage_footprint = get_image_method(image, "skimage.morphology.cube")
    else:
        raise ValueError("Image must be 2D or 3D.")

    if downscale_factor < 1.0:
        image = skimage_rescale(image, downscale_factor, order=3, anti_aliasing=True)

    return skimage_median(image, footprint=skimage_footprint(filter_size))


def check_labeled_binary(image):
    """
    Check if the given image is a labeled image.

    Parameters
    ----------
    image : ndarray
        The image to be checked.

    Returns
    -------
    None
    """
    assert np.issubdtype(image.dtype, np.integer), "Image must be of integer type."

    unique_values = np.unique(image)
    assert len(unique_values) > 1, "Image is constant."
    if len(unique_values) == 2:
        warnings.warn("Only one label was provided in the image.")


def cleanup_segmentation(
    label_image: npt.ArrayLike,
    min_obj_size: Optional[int] = None,
    max_obj_size: Optional[int] = None,
    border_buffer_size: Optional[int] = None,
    max_hole_size: Optional[int] = None,
) -> npt.ArrayLike:
    """Clean up segmented image by removing small objects, clearing borders, and closing holes."""
    check_labeled_binary(label_image)

    # first 3 transforms preserve labels
    if min_obj_size is not None:
        # min_obj_size = to_device(min_obj_size, get_device(label_image))
        remove_small_objects = get_image_method(label_image, "skimage.morphology.remove_small_objects")
        label_image = remove_small_objects(label_image, min_size=min_obj_size)

    if max_obj_size is not None:
        label_image = remove_large_objects(label_image, max_size=max_obj_size)

    if border_buffer_size is not None:
        label_image = clear_xy_borders(label_image, buffer_size=border_buffer_size)

    # returns boolean array
    if max_hole_size is not None:
        remove_holes = get_image_method(label_image, "skimage.morphology.remove_small_holes")
        label_image = remove_holes(label_image, area_threshold=max_hole_size)

    return label(label_image).astype(np.uint8)


def remove_large_objects(label_image: npt.ArrayLike, max_size: int = 100000) -> npt.ArrayLike:
    """Remove objects with volume above specified threshold."""
    check_labeled_binary(label_image)
    label_volumes = np.bincount(label_image.ravel())
    too_large = label_volumes > max_size
    too_large_mask = too_large[label_image]
    label_image[too_large_mask] = 0
    return label_image


def remove_small_objects(label_image: npt.ArrayLike, min_size: int = 500) -> npt.ArrayLike:
    """Remove objects with volume below specified threshold."""
    check_labeled_binary(label_image)
    remove_small_objects = get_image_method(label_image, "skimage.morphology.remove_small_objects")
    label_image = remove_small_objects(label_image, min_size=min_size)
    return label_image


def clear_xy_borders(label_image: npt.ArrayLike, buffer_size: int = 0) -> npt.ArrayLike:
    """Remove masks that touch XY borders."""
    check_labeled_binary(label_image)
    if label_image.ndim == 2:
        return clear_border(label_image, buffer_size=buffer_size)
    label_image = pad_image(label_image, (buffer_size + 1, buffer_size + 1), mode="constant")
    label_image = clear_border(label_image, buffer_size=buffer_size)
    return label_image[buffer_size + 1 : -(buffer_size + 1), :, :]


def remove_touching_objects(label_image: npt.ArrayLike, border_value: int = 100) -> npt.ArrayLike:
    """Find labelled masks that overlap and remove from the image."""
    check_labeled_binary(label_image)
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


def segment_watershed(image, ball_size=15):
    """Segment image using watershed algorithm."""
    device = get_device(image)
    skimage_ball = get_image_method(image, "skimage.morphology.ball")
    skimage_peak_local_max = get_image_method(image, "skimage.feature.peak_local_max")

    distance = distance_transform_edt(image)
    coords = skimage_peak_local_max(distance, footprint=skimage_ball(ball_size), labels=image).get()

    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers = label(mask)

    labels = watershed(-distance.get(), markers, mask=image.get())
    # return in the format and on the same device as input
    return to_device(labels, device)
