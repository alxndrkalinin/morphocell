"""Implement pre- and post-processing for segmentation."""

import warnings

from typing import Optional, Sequence
import numpy.typing as npt

import numpy as np
from skimage.segmentation import watershed

from ._clear_border import clear_border

from ..gpu import get_device, to_device, get_image_method, asnumpy
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
        warnings.warn("Only one label was provided in the image. Make sure to label components first.")


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

        for label_id in np.unique(label_image)[1:]:
            mask = label_image == label_id
            filled_mask = remove_holes(mask, area_threshold=max_hole_size)
            label_image[filled_mask] = label_id

    return label(label_image).astype(np.uint8)


def find_objects(label_image, max_label=None):
    """
    Find objects in a labeled nD array.

    Parameters
    ----------
    label_image : cupy.ndarray
        nD array containing objects defined by different labels. Labels with
        value 0 are ignored.
    max_label : int, optional
        Maximum label to be searched for in `input`. If max_label is not
        specified, the positions of all objects up to the highest label are returned.

    Returns
    -------
    object_slices : list of tuples
        A list of tuples, with each tuple containing N slices (with N the
        dimension of the input array). Slices correspond to the minimal
        parallelepiped that contains the object. If a number is missing,
        None is returned instead of a slice. The label `l` corresponds to
        the index `l-1` in the returned list.
    """
    if max_label is None:
        max_label = int(np.max(label_image))

    object_slices = [None] * max_label

    for label_idx in range(1, max_label + 1):
        mask = label_image == label_idx
        if not mask.any():
            continue

        slices = []
        for dim in range(mask.ndim):
            axis_indices = np.any(mask, axis=tuple(range(mask.ndim))[:dim] + tuple(range(mask.ndim))[dim + 1 :])
            if not axis_indices.any():
                slices.append(None)
                continue
            min_idx = int(np.where(axis_indices)[0].min())
            max_idx = int(np.where(axis_indices)[0].max()) + 1
            slices.append(slice(min_idx, max_idx))

        object_slices[label_idx - 1] = tuple(slices)

    return object_slices


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
    return label(label_image[buffer_size + 1 : -(buffer_size + 1), :, :])


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


def remove_thin_objects(label_image, min_z=2):
    """Remove objects thinner than a specified minimum value in Z."""
    unique_labels = [regionlabel for regionlabel in np.unique(label_image) if regionlabel != 0]
    for regionlabel in unique_labels:
        mask = label_image == regionlabel

        maskz = np.any(mask, axis=(1, 2))
        z1 = np.argmax(maskz)
        z2 = len(maskz) - np.argmax(maskz[::-1])
        size_z = abs(z2 - z1)

        if size_z <= min_z:
            label_image[mask] = 0

    return label_image


def segment_watershed(image, ball_size=15):
    """Segment image using watershed algorithm."""
    device = get_device(image)
    skimage_ball = get_image_method(image, "skimage.morphology.ball")
    skimage_peak_local_max = get_image_method(image, "skimage.feature.peak_local_max")

    distance = distance_transform_edt(image)
    coords = skimage_peak_local_max(distance, footprint=skimage_ball(ball_size), labels=image)

    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(asnumpy(coords.T))] = True
    markers = label(mask)

    labels = watershed(-asnumpy(distance), markers, mask=asnumpy(image))
    # return in the format and on the same device as input
    return to_device(labels, device)


def _binary_fill_holes(image):
    """Fill holes in binary objects."""
    if get_device(image) == "GPU":
        from cupyx.scipy.ndimage import binary_fill_holes
    elif get_device(image) == "CPU":
        from scipy.ndimage import binary_fill_holes
    else:
        raise ValueError("Unknown device.")

    return binary_fill_holes(image)


def fill_label_holes(lbl_img, **binary_fill_holes_kwargs):
    """
    Fill small holes in label image.

    Inspired by: https://github.com/stardist/stardist/blob/master/stardist/utils.py
    """

    def grow(sl, interior):
        return tuple(slice(s.start - int(w[0]), s.stop + int(w[1])) for s, w in zip(sl, interior))

    def shrink(interior):
        return tuple(slice(int(w[0]), (-1 if w[1] else None)) for w in interior)

    objects = find_objects(lbl_img)
    lbl_img_filled = np.zeros_like(lbl_img)

    for i, sl in enumerate(objects, 1):
        if sl is None:
            continue
        interior = [(s.start > 0, s.stop < sz) for s, sz in zip(sl, lbl_img.shape)]
        shrink_slice = shrink(interior)
        grown_mask = lbl_img[grow(sl, interior)] == i
        mask_filled = _binary_fill_holes(grown_mask, **binary_fill_holes_kwargs)[shrink_slice]
        lbl_img_filled[sl][mask_filled] = i

    return lbl_img_filled


def fill_holes_slicer(
    image: npt.ArrayLike, area_threshold: int = 1000, num_iterations: int = 1, axes: Optional[Sequence[int]] = None
):
    """
    Fill holes in slices of binary or labeled objects.

    Inspired by: https://github.com/True-North-Intelligent-Algorithms/tnia-python/blob/main/tnia/morphology/fill_holes.py
    """
    skimage_remove_small_holes = get_image_method(image, "skimage.morphology.remove_small_holes")
    axes = range(image.ndim) if axes is None else axes

    for label_id in np.unique(image)[1:]:
        binary = image == label_id

        for _ in range(num_iterations):
            for axis in axes:
                slicers = [slice(None)] * image.ndim
                for i in range(binary.shape[axis]):
                    slicers[axis] = slice(i, i + 1)
                    binary_slice = binary[tuple(slicers)].squeeze()
                    filled_slice = skimage_remove_small_holes(binary_slice, area_threshold)
                    binary[tuple(slicers)] = filled_slice.reshape(binary_slice.shape)

        image[binary] = label_id

    return image
