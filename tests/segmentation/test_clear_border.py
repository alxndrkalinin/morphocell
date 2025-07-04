"""Tests for border-clearing segmentation helper."""

import numpy as np

from morphocell.segmentation._clear_border import clear_border


def test_clear_border_simple() -> None:
    """Remove objects touching the image border."""
    labels = np.array(
        [
            [1, 1, 0],
            [0, 2, 0],
            [0, 0, 0],
        ],
        dtype=int,
    )
    result = clear_border(labels.copy())
    assert 1 not in np.unique(result)
    assert 2 in np.unique(result)
