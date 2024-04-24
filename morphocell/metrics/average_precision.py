"""Implements device-agnostic average precision metric for segmentation."""
# Modified from StarDist/Cellpose with added support for CUDA GPUs by Alexandr Kalinin.
#
# Copyright (c) 2018-2024, Uwe Schmidt, Martin Weigert
# https://github.com/stardist/stardist/blob/cc5f412d6bacfa08e138d7c097e98f507df46b51/stardist/matching.py
# Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
# https://github.com/MouseLand/cellpose/blob/509ffca33737058b0b4e2e96d506514e10620eb3/cellpose/metrics.py

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..gpu import get_device, asnumpy, ascupy


def _label_overlap(x, y):
    """Measure label overlap, device agnostic."""
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1 + int(x.max()), 1 + int(y.max())), dtype=np.uint)

    if get_device(x) == "CPU" and get_device(y) == "CPU":
        from numba import jit

        @jit(nopython=True)
        def _label_overlap_cpu(x, y, overlap):
            for i in range(len(x)):
                overlap[x[i], y[i]] += 1
            return overlap

        return _label_overlap_cpu(x, y, overlap)

    elif get_device(x) == "GPU" and get_device(y) == "GPU":
        from cupyx import jit

        @jit.rawkernel()
        def label_overlap_kernel(x, y, overlap, N):
            idx = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
            if idx < N:
                jit.atomic_add(overlap, (x[idx], y[idx]), 1)

        N = x.size
        threads_per_block = 128
        overlap = ascupy(overlap)
        blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
        label_overlap_kernel[blocks_per_grid, threads_per_block](x, y, overlap, N)
        return overlap

    else:
        raise ValueError("x and y should be on the same device.")


def _intersection_over_union(masks_true, masks_pred):
    """Calculate the intersection over union of all mask pairs, device agnostic."""
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = overlap.sum(axis=0, keepdims=True)
    n_pixels_true = overlap.sum(axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    return iou


def _matches_at_threshold(iou, th):
    """Identify matches based on IoU and threshold."""
    n_min = min(iou.shape[0], iou.shape[1])
    assert n_min > 0, "No masks to match"
    costs = -(iou >= th).astype(float) - iou / (2 * n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = iou[true_ind, pred_ind] >= th
    return true_ind[match_ok], pred_ind[match_ok]


def compute_matches(mask_true, mask_pred, thresholds, return_iou=False):
    """Compute and store IoU and matching indices for various thresholds."""
    iou = _intersection_over_union(mask_true, mask_pred)[1:, 1:]
    iou = np.nan_to_num(asnumpy(iou))
    matches = {th: _matches_at_threshold(iou, th) for th in thresholds}
    return (matches, iou) if return_iou else matches


def average_precision(masks_true, masks_pred, thresholds, matches_per_threshold=None):
    """Calculate average precision and other metrics for a single pair of mask images with pre-computed matches."""
    if matches_per_threshold is None:
        matches_per_threshold = compute_matches(masks_true, masks_pred, thresholds)

    tp = np.asarray([len(matches_per_threshold[th][0]) for th in thresholds])
    fp = asnumpy(masks_pred.max()) - tp
    fn = asnumpy(masks_true.max()) - tp

    sum_counts = tp + fp + fn
    ap = np.where(sum_counts > 0, tp / sum_counts, 0)

    return ap, tp, fp, fn
