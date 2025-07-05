"""Tests for the Cellpose segmentation wrapper."""

import numpy as np
import pytest

from cubic.segmentation.cellpose import cellpose_segment


def test_cellpose_import_error() -> None:
    """Cellpose should raise if the library is unavailable."""
    image = np.zeros((1, 32, 32), dtype=np.float32)
    with pytest.raises(ImportError):
        cellpose_segment(image)
