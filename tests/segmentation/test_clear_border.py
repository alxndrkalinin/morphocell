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


def test_clear_border_3d() -> None:
    """Handle 3D labeled volumes."""
    labels = np.zeros((3, 3, 3), dtype=int)
    labels[1, 1, 1] = 1  # interior object
    labels[0, 0, 0] = 2  # border object
    result = clear_border(labels.copy())
    assert 2 not in np.unique(result)
    assert 1 in np.unique(result)


def test_clear_border_all_border_objects() -> None:
    """All objects removed when touching border."""
    labels = np.array([[1, 0], [0, 2]], dtype=int)
    result = clear_border(labels.copy())
    assert np.all(result == 0)
