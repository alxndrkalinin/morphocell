"""Implement extraction of 3D voxel-based morphological features."""

from typing import List, Optional
import numpy.typing as npt

from ..gpu import get_image_method


def regionprops(label_image: npt.ArrayLike, image: Optional[npt.ArrayLike] = None, features: Optional[List] = None):
    """Extract region-based morphological features."""
    features = [] if features is None else features
    skimage_regionprops_table = get_image_method(label_image, "skimage.measure.regionprops_table")

    # this will return results on the same device as the input label_image
    return skimage_regionprops_table(label_image, image, properties=["label"] + features)
