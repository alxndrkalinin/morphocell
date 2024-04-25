"""Calculate feature-based metrics for morphology comparison."""

import numpy as np

from .average_precision import compute_matches
from ..feature.voxel_morphometry import regionprops_table


def _norm_min_max(feature_values, features, min_max_ranges):
    """Normalize feature values using min-max scaling."""
    mins = np.asarray([min_max_ranges[feature][0] for feature in features])
    ranges = np.asarray([min_max_ranges[feature][1] - min_max_ranges[feature][0] for feature in features])
    ranges[ranges == 0] = np.finfo(np.float32).eps
    return (feature_values - mins) / ranges


def _calculate_cosine(true_features, pred_features):
    """Calculate cosine distance between true and pred features."""
    norm_true = np.linalg.norm(true_features)
    norm_pred = np.linalg.norm(pred_features)
    if norm_true == 0 or norm_pred == 0:
        return 1
    return 1 - np.dot(true_features, pred_features) / (norm_true * norm_pred)


def cosine_median(label_true, label_pred, features, thresholds=None, matches_per_threshold=None, min_max_ranges=None):
    """Calculate cosine distance between median features of true and pred masks."""
    true_props = regionprops_table(label_true, properties=features)
    pred_props = regionprops_table(label_pred, properties=features)
    features = [feat for feat in features if feat != "label"]
    true_features = np.column_stack([true_props[feat] for feat in features])
    pred_features = np.column_stack([pred_props[feat] for feat in features])

    if min_max_ranges:
        true_features = _norm_min_max(true_features, features, min_max_ranges)
        pred_features = _norm_min_max(pred_features, features, min_max_ranges)

    if thresholds is None:
        return _calculate_cosine(np.median(true_features, axis=0), np.median(pred_features, axis=0))

    if matches_per_threshold is None:
        matches_per_threshold = compute_matches(label_true, label_pred, thresholds)

    distances = {}
    for th in thresholds:
        true_ind, pred_ind = matches_per_threshold[th]
        filtered_true_feats = true_features[np.isin(true_props["label"], true_ind)]
        filtered_pred_feats = pred_features[np.isin(pred_props["label"], pred_ind)]
        distances[th] = _calculate_cosine(
            np.median(filtered_true_feats, axis=0), np.median(filtered_pred_feats, axis=0)
        )
    return distances


def calculate_morphology_correlations(masks_true, masks_pred, features, thresholds, matches_per_threshold):
    """Calculate morphology correlations for matched masks using precomputed matches."""
    true_props = regionprops_table(masks_true, properties=features)
    pred_props = regionprops_table(masks_pred, properties=features)
    features = [feat for feat in features if feat != "label"]

    correlations = {}
    for th in thresholds:
        true_ind, pred_ind = matches_per_threshold[th]
        true_indices = np.isin(true_props["label"], true_ind)
        pred_indices = np.isin(pred_props["label"], pred_ind)

        if true_indices.any() and pred_indices.any():
            true_matrix = np.column_stack([true_props[feat][true_indices] for feat in features if feat != "label"])
            pred_matrix = np.column_stack([pred_props[feat][pred_indices] for feat in features if feat != "label"])
            if true_matrix.shape[0] > 1 and true_matrix.shape == pred_matrix.shape:
                correlations[th] = _calculate_correlations(true_matrix, pred_matrix, features)
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
