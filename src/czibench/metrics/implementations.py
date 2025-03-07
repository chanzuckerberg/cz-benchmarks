"""Implementation of metric functions and registration with the registry."""

from scib_metrics import silhouette_batch
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from ..tasks.utils import compute_entropy_per_cell
from .types import MetricRegistry, MetricType

# Create the global metric registry
metrics = MetricRegistry()

# Register clustering metrics
metrics.register(
    MetricType.ADJUSTED_RAND_INDEX,
    func=adjusted_rand_score,
    required_args={"labels_true", "labels_pred"},
    description="Adjusted Rand index between two clusterings",
    tags={"clustering"},
)

metrics.register(
    MetricType.NORMALIZED_MUTUAL_INFO,
    func=normalized_mutual_info_score,
    required_args={"labels_true", "labels_pred"},
    description="Normalized mutual information between two clusterings",
    tags={"clustering"},
)

# Register embedding quality metrics
metrics.register(
    MetricType.SILHOUETTE_SCORE,
    func=silhouette_score,
    required_args={"X", "labels"},
    default_params={"metric": "euclidean"},
    description="Silhouette score for clustering evaluation",
    tags={"embedding"},
)

# Register integration metrics
metrics.register(
    MetricType.ENTROPY_PER_CELL,
    func=compute_entropy_per_cell,
    required_args={"X", "labels"},
    description=(
        "Computes entropy of batch labels in local neighborhoods. "
        "Higher values indicate better batch mixing."
    ),
    tags={"integration"},
)

metrics.register(
    MetricType.BATCH_SILHOUETTE,
    func=silhouette_batch,
    required_args={"X", "labels", "batch_labels"},
    description=(
        "Batch-aware silhouette score that measures how well cells "
        "cluster across batches."
    ),
    tags={"integration"},
)


def mean_fold_metric(results_df, metric="accuracy", classifier=None):
    """Compute mean of a metric across folds.

    Args:
        results_df: DataFrame containing cross-validation results. Must have columns:
            - "classifier": Name of the classifier (e.g., "lr", "knn")
            - One of the following metric columns:
                - "accuracy": For accuracy scores
                - "f1": For F1 scores
                - "precision": For precision scores
                - "recall": For recall scores
        metric: Name of metric column to average ("accuracy", "f1", etc.)
        classifier: Optional classifier name to filter results

    Returns:
        Mean value of the metric across folds

    Raises:
        KeyError: If the specified metric column is not present in results_df
    """
    if classifier:
        df = results_df[results_df["classifier"] == classifier]
    else:
        df = results_df
    return df[metric].mean()


# Register cross-validation classification metrics
metrics.register(
    MetricType.MEAN_FOLD_ACCURACY,
    func=mean_fold_metric,
    required_args={"results_df"},
    default_params={"metric": "accuracy", "classifier": None},
    tags={"label_prediction"},
)

metrics.register(
    MetricType.MEAN_FOLD_F1_SCORE,
    func=mean_fold_metric,
    required_args={"results_df"},
    default_params={"metric": "f1", "classifier": None},
    tags={"label_prediction"},
)

metrics.register(
    MetricType.MEAN_FOLD_PRECISION,
    func=mean_fold_metric,
    required_args={"results_df"},
    default_params={"metric": "precision", "classifier": None},
    tags={"label_prediction"},
)

metrics.register(
    MetricType.MEAN_FOLD_RECALL,
    func=mean_fold_metric,
    required_args={"results_df"},
    default_params={"metric": "recall", "classifier": None},
    tags={"label_prediction"},
)
