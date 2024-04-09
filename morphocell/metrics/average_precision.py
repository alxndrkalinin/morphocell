"""Implements device-agnostic average precision metric for segmentation."""
# Modified from Cellpose with added support for CUDA GPUs by Alexandr Kalinin.
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
    # iou = xp.nan_to_num(iou)  # do outside to avoid using xp
    return iou


def _true_positive(iou, th):
    """Calculate the true positive at threshold th."""
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= th).astype(float) - iou / (2 * n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = iou[true_ind, pred_ind] >= th
    tp = match_ok.sum()
    return tp


def average_precision(masks_true, masks_pred, threshold=[0.5, 0.75, 0.9]):
    """Calculate average precision for masks_true and masks_pred, device agnostic."""
    not_list = False
    if not isinstance(masks_true, list):
        masks_true, masks_pred = [masks_true], [masks_pred]
        not_list = True

    if not isinstance(threshold, (list, np.ndarray)):
        threshold = [threshold]

    if len(masks_true) != len(masks_pred):
        raise ValueError("masks_true and masks_pred must have the same length")

    ap, tp, fp, fn = np.zeros((4, len(masks_true), len(threshold)), np.float32)
    n_true = np.asarray([asnumpy(mt.max()) for mt in masks_true])
    n_pred = np.asarray([asnumpy(mp.max()) for mp in masks_pred])

    for n in range(len(masks_true)):
        if n_pred[n] > 0:
            iou = _intersection_over_union(masks_true[n], masks_pred[n])[1:, 1:]
            iou = np.nan_to_num(asnumpy(iou))
            for k, th in enumerate(threshold):
                tp[n, k] = _true_positive(iou, th)
        fp[n, :] = n_pred[n] - tp[n, :]
        fn[n, :] = n_true[n] - tp[n, :]
        ap[n, :] = np.divide(
            tp[n, :],
            tp[n, :] + fp[n, :] + fn[n, :],
            out=np.zeros_like(tp[n, :]),
            where=tp[n, :] + fp[n, :] + fn[n, :] != 0,
        )

    if not_list:
        ap, tp, fp, fn = ap[0], tp[0], fp[0], fn[0]

    return ap, tp, fp, fn
