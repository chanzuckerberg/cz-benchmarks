"""
Implementation of metric functions and registration with the registry.

This file defines and registers various metrics with a global `MetricRegistry`.
Metrics are categorized into the following types:
- Clustering metrics (e.g., Adjusted Rand Index, Normalized Mutual Information)
- Embedding quality metrics (e.g., Silhouette Score)
- Integration metrics (e.g., Entropy Per Cell, Batch Silhouette)
- Perturbation metrics (e.g., Mean Squared Error, Pearson Correlation)
- Label prediction metrics (e.g., Mean Fold Accuracy, Mean Fold F1 Score)

Each metric is registered with:
- A unique `MetricType` identifier.
- The function implementing the metric.
- Required arguments for the metric function.
- A description of the metric's purpose.
- Tags categorizing the metric.
"""

import numpy as np
from scib_metrics import silhouette_batch, silhouette_label
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
)
from scipy.stats import pearsonr, spearmanr
from .utils import compute_entropy_per_cell, mean_fold_metric, jaccard_score

from .types import MetricRegistry, MetricType


def spearman_correlation(a, b):
    """Wrapper for spearmanr that returns only the correlation coefficient."""
    result = spearmanr(a, b)
    # Handle both old and new scipy versions
    if hasattr(result, "correlation"):
        value = result.correlation
        return 0 if np.isnan(value) else value
    elif hasattr(result, "statistic"):
        value = result.statistic
        return 0 if np.isnan(value) else value
    else:
        # Fallback for very old versions that return (correlation, pvalue)
        return result[0]


# Create the global metric registry
metrics_registry = MetricRegistry()

# Register clustering metrics
metrics_registry.register(
    MetricType.ADJUSTED_RAND_INDEX,
    func=adjusted_rand_score,
    required_args={"labels_true", "labels_pred"},
    description="Adjusted Rand index between two clusterings",
    tags={"clustering"},
)

metrics_registry.register(
    MetricType.NORMALIZED_MUTUAL_INFO,
    func=normalized_mutual_info_score,
    required_args={"labels_true", "labels_pred"},
    description="Normalized mutual information between two clusterings",
    tags={"clustering"},
)

# Register embedding quality metrics
metrics_registry.register(
    MetricType.SILHOUETTE_SCORE,
    func=silhouette_label,
    required_args={"X", "labels"},
    description="Silhouette score for clustering evaluation",
    tags={"embedding"},
)

# Register integration metrics
metrics_registry.register(
    MetricType.ENTROPY_PER_CELL,
    func=compute_entropy_per_cell,
    required_args={"X", "labels"},
    description=(
        "Computes entropy of batch labels in local neighborhoods. "
        "Higher values indicate better batch mixing."
    ),
    tags={"integration"},
)

metrics_registry.register(
    MetricType.BATCH_SILHOUETTE,
    func=silhouette_batch,
    required_args={"X", "labels", "batch"},
    description=(
        "Batch-aware silhouette score that measures how well cells "
        "cluster across batches."
    ),
    tags={"integration"},
)

# Perturbation metrics
metrics_registry.register(
    MetricType.MEAN_SQUARED_ERROR,
    func=mean_squared_error,
    required_args={"y_true", "y_pred"},
    description="Mean squared error between true and predicted values",
    tags={"perturbation"},
)

metrics_registry.register(
    MetricType.PEARSON_CORRELATION,
    func=pearsonr,
    required_args={"x", "y"},
    description="Pearson correlation between true and predicted values",
    tags={"perturbation"},
)

metrics_registry.register(
    MetricType.JACCARD,
    func=jaccard_score,
    required_args={"y_true", "y_pred"},
    description="Jaccard similarity between true and predicted values",
    tags={"perturbation"},
)

# Register cross-validation classification metrics
metrics_registry.register(
    MetricType.MEAN_FOLD_ACCURACY,
    func=mean_fold_metric,
    required_args={"results_df"},
    default_params={"metric": "accuracy", "classifier": None},
    tags={"label_prediction"},
)

metrics_registry.register(
    MetricType.MEAN_FOLD_F1_SCORE,
    func=mean_fold_metric,
    required_args={"results_df"},
    default_params={"metric": "f1", "classifier": None},
    tags={"label_prediction"},
)

metrics_registry.register(
    MetricType.MEAN_FOLD_PRECISION,
    func=mean_fold_metric,
    required_args={"results_df"},
    default_params={"metric": "precision", "classifier": None},
    tags={"label_prediction"},
)

metrics_registry.register(
    MetricType.MEAN_FOLD_RECALL,
    func=mean_fold_metric,
    required_args={"results_df"},
    default_params={"metric": "recall", "classifier": None},
    tags={"label_prediction"},
)

metrics_registry.register(
    MetricType.MEAN_FOLD_AUROC,
    func=mean_fold_metric,
    required_args={"results_df"},
    default_params={"metric": "auroc", "classifier": None},
    tags={"label_prediction"},
)

metrics_registry.register(
    MetricType.ACCURACY,
    func=accuracy_score,
    required_args={"y_true", "y_pred"},
    description="Accuracy between true and predicted values",
    tags={"label_prediction"},
)

metrics_registry.register(
    MetricType.PRECISION,
    func=precision_score,
    required_args={"y_true", "y_pred"},
    description="Precision between true and predicted values",
    tags={"label_prediction"},
)

metrics_registry.register(
    MetricType.RECALL,
    func=recall_score,
    required_args={"y_true", "y_pred"},
    description="Recall between true and predicted values",
    tags={"label_prediction"},
)

metrics_registry.register(
    MetricType.F1,
    func=f1_score,
    required_args={"y_true", "y_pred"},
    description="F1 score between true and predicted values",
    tags={"label_prediction"},
)

metrics_registry.register(
    MetricType.SPEARMAN_CORRELATION,
    func=spearman_correlation,
    required_args={"a", "b"},
    description="Spearman correlation between true and predicted values",
    tags={"label_prediction"},
)
