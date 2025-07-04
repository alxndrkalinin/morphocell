"""Tests for the voxel feature module."""

import numpy as np

from morphocell.feature import voxel


def test_regionprops_extract_features() -> None:
    """Extract simple features and check results."""
    labels = np.array([[0, 1], [1, 1]], dtype=np.int32)
    props = voxel.regionprops_table(labels, properties=["area"])
    assert props["label"].tolist() == [1]
    assert int(props["area"][0]) == 3

    labels_out, feature_values = voxel.extract_features(labels, ["area"])
    assert labels_out.tolist() == [1]
    assert feature_values.shape == (1, 1)
    assert feature_values[0, 0] == 3


def test_regionprops_multiple_labels() -> None:
    """Extract multiple features from a multi-label image."""
    labels = np.array(
        [
            [0, 1, 1],
            [2, 2, 0],
        ],
        dtype=np.int32,
    )

    props = voxel.regionprops_table(labels, properties=["area", "centroid", "bbox"])
    assert sorted(props["label"].tolist()) == [1, 2]
    assert props["area"].tolist() == [2, 2]
    assert "centroid-0" in props and "centroid-1" in props
    assert "bbox-0" in props and "bbox-3" in props

    labels_out, feats = voxel.extract_features(labels, ["area", "centroid"])
    assert labels_out.tolist() == [1, 2]
    assert feats.shape == (2, 3)
