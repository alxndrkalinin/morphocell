"""Tests for detection metrics based on IoU."""

import numpy as np

from morphocell.metrics.average_precision import compute_matches, average_precision


def test_average_precision_perfect_match() -> None:
    """Perfect overlap should yield perfect precision."""
    mask_true = np.array([[0, 1], [0, 2]], dtype=np.int32)
    mask_pred = mask_true.copy()
    thresholds = [0.5]
    matches, _ = compute_matches(mask_true, mask_pred, thresholds, return_iou=True)
    ap, tp, fp, fn = average_precision(mask_true, mask_pred, thresholds, matches)
    assert ap[0] == 1
    assert tp[0] == 2
    assert fp[0] == 0
    assert fn[0] == 0
