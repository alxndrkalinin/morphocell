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
