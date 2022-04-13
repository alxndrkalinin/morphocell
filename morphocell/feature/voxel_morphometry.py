"""Implement extraction of 3D voxel-based morphological features."""

from typing import List, Optional
import numpy.typing as npt

from ..gpu import get_image_method


def regionprops(image_bin: npt.ArrayLike, image: npt.ArrayLike, features: Optional[List] = None):
    """Perform Otsu's thresholding with Gaussian blur."""
    features = [] if features is None else features

    skimage_label = get_image_method(image, "skimage.measure.label")
    skimage_regionprops_table = get_image_method(image, "skimage.measure.regionprops_table")

    label_image_bin = skimage_label(image_bin)
    return skimage_regionprops_table(label_image_bin, image, properties=["label"] + features)
