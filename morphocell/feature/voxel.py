"""Implement extraction of 3D voxel-based morphological features.

Executing regionprops on GPU currently does not provide performance gains, see:
https://github.com/rapidsai/cucim/issues/241
"""

from typing import List, Tuple, Optional
import numpy.typing as npt

import numpy as np

from ..cuda import asnumpy
from ..skimage import measure


def regionprops(
    label_image: npt.ArrayLike, intensity_image: Optional[npt.ArrayLike] = None, spacing: Optional[List] = None
):
    """Extract region-based morphological features."""
    return measure.regionprops(label_image, intensity_image, spacing=spacing)


def regionprops_table(
    label_image: npt.ArrayLike,
    intensity_image: Optional[npt.ArrayLike] = None,
    properties: Optional[List] = None,
    spacing: Optional[List] = None,
):
    """Extract region-based morphological features and return in pandas-compatible format."""
    if properties is not None:
        properties = properties if "label" in properties else np.append(properties, "label")
    else:
        properties = []
    return measure.regionprops_table(label_image, intensity_image, properties=properties)


def extract_features(label_image, features, feature_ranges=None):
    """Extract features from a label image using regionprops."""
    numeric_feature_cols = [feat for feat in features if feat != "label"]
    image_props = regionprops_table(asnumpy(label_image), properties=features)
    selected_features = [feat for feat in image_props.keys() if feat.split("-")[0] in numeric_feature_cols]
    feature_values = np.column_stack([image_props[feat] for feat in selected_features])
    if feature_ranges is not None:
        feature_values = norm_features_by_range(feature_values, selected_features, feature_ranges)
    labels = image_props["label"]
    return labels, feature_values


def norm_features_by_range(feature_values, features, min_max_ranges):
    """Normalize feature values using min-max scaling."""
    mins = np.asarray([min_max_ranges[feature][0] for feature in features])
    ranges = np.asarray([min_max_ranges[feature][1] - min_max_ranges[feature][0] for feature in features])
    ranges[ranges == 0] = np.finfo(np.float32).eps
    return (feature_values - mins) / ranges


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
