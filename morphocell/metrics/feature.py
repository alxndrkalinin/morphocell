"""Calculate feature-based metrics for morphology comparison.

Only matching currently supports GPU, regionprops are forced to execute CPU, see:
https://github.com/rapidsai/cucim/issues/241
"""

import numpy as np

from .average_precision import compute_matches
from ..feature.voxel import extract_features


def _calculate_cosine(true_features, pred_features):
    """Calculate cosine distance between true and pred features."""
    norm_true = np.linalg.norm(true_features)
    norm_pred = np.linalg.norm(pred_features)
    if norm_true == 0 or norm_pred == 0:
        return 1
    return 1 - np.dot(true_features, pred_features) / (norm_true * norm_pred)


def cosine_median(
    label_image_true,
    label_image_pred,
    features,
    thresholds=None,
    matches_per_threshold=None,
    feature_ranges=None,
    return_features=False,
):
    """Calculate cosine distance between median features of true and pred masks."""
    true_labels, true_features = extract_features(label_image_true, features, feature_ranges)
    pred_labels, pred_features = extract_features(label_image_pred, features, feature_ranges)

    if thresholds is None:
        return _calculate_cosine(np.median(true_features, axis=0), np.median(pred_features, axis=0))

    if matches_per_threshold is None:
        matches_per_threshold = compute_matches(label_image_true, label_image_pred, thresholds)

    distances = {}
    for th in thresholds:
        true_ind, pred_ind = matches_per_threshold[th]
        filtered_true_feats = true_features[np.isin(true_labels, true_ind)]
        filtered_pred_feats = pred_features[np.isin(pred_labels, pred_ind)]
        distances[th] = _calculate_cosine(
            np.median(filtered_true_feats, axis=0), np.median(filtered_pred_feats, axis=0)
        )

    if not return_features:
        return distances
    else:
        return distances, true_features, pred_features


def morphology_correlations(
    label_image_true, label_image_pred, features, thresholds, matches_per_threshold, min_max_ranges=None
):
    """Calculate morphology correlations for matched masks using precomputed matches."""
    true_labels, true_features = extract_features(label_image_true, features, min_max_ranges)
    pred_labels, pred_features = extract_features(label_image_pred, features, min_max_ranges)

    correlations = {}
    for th in thresholds:
        true_ind, pred_ind = matches_per_threshold[th]
        true_indices = np.isin(true_labels, true_ind)
        pred_indices = np.isin(pred_labels, pred_ind)
        if true_indices.any() and pred_indices.any():
            filtered_true_feats = true_features[true_indices]
            filtered_pred_feats = pred_features[pred_indices]
            if filtered_true_feats.shape[0] > 1 and filtered_true_feats.shape == filtered_pred_feats.shape:
                correlations[th] = _calculate_correlations(filtered_true_feats, filtered_pred_feats, features)
            else:
                correlations[th] = dict(zip(features, [np.nan] * len(features)))
        else:
            correlations[th] = dict(zip(features, [np.nan] * len(features)))

    return correlations


def _calculate_correlations(true_matrix, pred_matrix, features):
    """Calculate Pearson correlation between true and pred features."""
    if true_matrix.size > 0 and pred_matrix.size > 0:
        correlation_matrix = np.corrcoef(true_matrix.T, pred_matrix.T)[: len(features), len(features) :]
        return dict(zip(features, np.diag(correlation_matrix)))
    else:
        return dict(zip(features, [np.nan] * len(features)))
