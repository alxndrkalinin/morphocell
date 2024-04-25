"""Implement extraction of 3D voxel-based morphological features.

Executing regionprops on GPU currently does not provide performance gains, see:
https://github.com/rapidsai/cucim/issues/241
"""

from typing import List, Tuple, Optional
import numpy.typing as npt

import numpy as np

from ..gpu import get_image_method


def regionprops(
    label_image: npt.ArrayLike, intensity_image: Optional[npt.ArrayLike] = None, spacing: Optional[List] = None
):
    """Extract region-based morphological features."""
    skimage_regionprops = get_image_method(label_image, "skimage.measure.regionprops")
    return skimage_regionprops(label_image, intensity_image, spacing=spacing)


def regionprops_table(
    label_image: npt.ArrayLike,
    intensity_image: Optional[npt.ArrayLike] = None,
    properties: Optional[List] = None,
    spacing: Optional[List] = None,
):
    """Extract region-based morphological features and return in pandas-compatible format."""
    if properties is not None:
        properties = properties if "label" in properties else properties + ["label"]
    else:
        properties = []
    skimage_regionprops_table = get_image_method(label_image, "skimage.measure.regionprops_table")
    return skimage_regionprops_table(label_image, intensity_image, properties=properties)


def calculate_feature_percentiles(
    regionprops_dict: dict,
    percentiles: Tuple = (0.01, 0.99),
    features_include: Optional[list] = None,
    features_exclude: Optional[list] = None,
):
    """Calculate percentiles of features in regionprops."""
    percentiles = np.asarray(percentiles) * 100
    if features_include is not None:
        features = [f for f in features_include if f in regionprops_dict]
    else:
        features = list(regionprops_dict.keys())
    if features_exclude is not None:
        features = [f for f in features if f not in features_exclude]
    return {f: np.percentile(regionprops_dict[f], percentiles) for f in features if f in regionprops_dict}
