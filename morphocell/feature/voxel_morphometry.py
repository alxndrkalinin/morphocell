"""Implement extraction of 3D voxel-based morphological features."""

from typing import List, Optional
import numpy.typing as npt

from ..gpu import get_image_method


def regionprops(image_bin: npt.ArrayLike, image: npt.ArrayLike, features: Optional[List] = None):
    """Extract region-based morphological features."""
    features = [] if features is None else features

    skimage_label = get_image_method(image, "skimage.measure.label")
    skimage_regionprops_table = get_image_method(image, "skimage.measure.regionprops_table")

    label_image_bin = skimage_label(image_bin)

    # this will fail on GPU if any properties return objects, see https://github.com/rapidsai/cucim/pull/272
    return skimage_regionprops_table(label_image_bin, image, properties=["label"] + features)
