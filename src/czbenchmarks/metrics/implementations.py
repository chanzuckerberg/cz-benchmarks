"""Implementation of metric functions and registration with the registry."""

from scib_metrics import silhouette_batch, silhouette_label
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    r2_score,
    mean_squared_error,
)
from .utils import compute_entropy_per_cell, mean_fold_metric, jaccard_score

from .types import MetricRegistry, MetricType


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
    default_params={"metric": "euclidean"},
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
    MetricType.R2_SCORE,
    func=r2_score,
    required_args={"y_true", "y_pred"},
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
