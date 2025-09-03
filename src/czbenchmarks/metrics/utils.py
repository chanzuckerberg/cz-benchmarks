import collections
import logging
import statistics
from typing import Iterable, Literal, Union

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from ..constants import RANDOM_SEED
from .types import AggregatedMetricResult, MetricResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _safelog(a: np.ndarray) -> np.ndarray:
    """Compute safe log that handles zeros by returning 0.

    Args:
        a: Input array

    Returns:
        Array with log values, with 0s where input was 0
    """
    a = a.astype("float")
    return np.log(a, out=np.zeros_like(a), where=(a != 0))


def nearest_neighbors_hnsw(
    data: np.ndarray,
    expansion_factor: int = 200,
    max_links: int = 48,
    n_neighbors: int = 100,
    random_seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Find nearest neighbors using HNSW algorithm.

    Args:
        data: Input data matrix of shape (n_samples, n_features)
        expansion_factor: Size of dynamic candidate list for search
        max_links: Number of bi-directional links created for every new element
        n_neighbors: Number of nearest neighbors to find

    Returns:
        Tuple containing:
            - Indices array of shape (n_samples, n_neighbors)
            - Distances array of shape (n_samples, n_neighbors)
    """
    import hnswlib

    if n_neighbors > data.shape[0]:
        raise ValueError(
            f"n_neighbors ({n_neighbors}) must be less than or equal to the number of samples: {data.shape[0]}"
        )
    sample_indices = np.arange(data.shape[0])
    index = hnswlib.Index(space="l2", dim=data.shape[1])
    index.init_index(
        max_elements=data.shape[0],
        ef_construction=expansion_factor,
        M=max_links,
        random_seed=random_seed,
    )
    index.add_items(data, sample_indices)
    index.set_ef(expansion_factor)
    neighbor_indices, distances = index.knn_query(data, k=n_neighbors)
    return neighbor_indices, distances


def compute_entropy_per_cell(
    X: np.ndarray,
    labels: Union[pd.Categorical, pd.Series, np.ndarray],
    n_neighbors: int = 200,
    random_seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Compute entropy of batch labels in local neighborhoods.

    For each cell, finds nearest neighbors and computes entropy of
    batch label distribution in that neighborhood.

    Args:
        X: Cell Embedding matrix of shape (n_cells, n_features)
        labels: Series containing batch labels for each cell
        n_neighbors: Number of nearest neighbors to consider
        random_seed: Random seed for reproducibility

    Returns:
        Array of entropy values for each cell, normalized by log of number of batches
    """
    if n_neighbors > X.shape[0]:
        n_neighbors = X.shape[0]
        logger.warning(
            f"n_neighbors ({n_neighbors}) is greater than the number of samples ({X.shape[0]}). Setting n_neighbors to {n_neighbors}."
        )

    indices, _ = nearest_neighbors_hnsw(
        X, n_neighbors=n_neighbors, random_seed=random_seed
    )
    labels = np.array(list(labels))
    unique_batch_labels = np.unique(labels)
    indices_batch = labels[indices]

    label_counts_per_cell = np.vstack([(indices_batch == label).sum(1) for label in unique_batch_labels]).T
    label_counts_per_cell_normed = label_counts_per_cell / label_counts_per_cell.sum(1)[:, None]
    return (
        (-label_counts_per_cell_normed * _safelog(label_counts_per_cell_normed)).sum(1)
        / _safelog(np.array([len(unique_batch_labels)]))
    ).mean()


def jaccard_score(y_true: set[str], y_pred: set[str]):
    """Compute Jaccard similarity between true and predicted values.

    Args:
        y_true: True values
        y_pred: Predicted values
    """
    return len(y_true.intersection(y_pred)) / len(y_true.union(y_pred))


def mean_fold_metric(results_df, metric="accuracy", classifier=None):
    """Compute mean of a metric across folds.

    Args:
        results_df: DataFrame containing cross-validation results. Must have columns:
            - "classifier": Name of the classifier (e.g., "lr", "knn")
            And one of the following metric columns:
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


def aggregate_results(results: Iterable[MetricResult]) -> list[AggregatedMetricResult]:
    """aggregate a collection of MetricResults by their type and parameters"""
    grouped_results = collections.defaultdict(list)
    for result in results:
        grouped_results[result.aggregation_key].append(result)

    aggregated = []
    for results_to_agg in grouped_results.values():
        values_raw = [result.value for result in results_to_agg]
        value_mean = statistics.mean(values_raw)
        try:
            value_std_dev = statistics.stdev(values_raw, xbar=value_mean)
        except statistics.StatisticsError:
            # we only had one result so we can't compute it
            value_std_dev = None
        aggregated.append(
            AggregatedMetricResult(
                metric_type=results_to_agg[0].metric_type,
                params=results_to_agg[0].params,
                value=value_mean,
                value_std_dev=value_std_dev,
                values_raw=values_raw,
                n_values=len(values_raw),
            )
        )
    return aggregated


def temporal_smoothness(
    X: np.ndarray, time_labels: np.ndarray, k: int = 10, normalize: bool = True, adaptive_k: bool = False
) -> float:
    """
    Measure how temporally close neighbors are in Embedding space.
    Parameters:
    -----------
    X : np.ndarray
        Embedding matrix
    time_labels : np.ndarray
        Timepoint labels
    k : int
        Number of neighbors to consider
    normalize : bool
        Whether to normalize score to [0,1] range
    adaptive_k : bool
        Use adaptive k based on local density
    Returns:
    --------
    float: Smoothness score
    """
    # Convert to numpy array for consistent indexing
    X = np.asarray(X)
    time_labels = np.asarray(time_labels)

    if adaptive_k:
        # Estimate local density and adjust k
        distances, _ = NearestNeighbors(n_neighbors=30).fit(X).kneighbors(X)  # Fixed: n_neighbors not n*neighbors
        mean_distances = distances.mean(axis=1)

        # Scale k based on relative density (inverse of mean distance)
        relative_density = mean_distances.max() / (mean_distances + 1e-10)
        k_values = np.maximum(5, np.minimum(30, np.round(k * relative_density))).astype(int)
    else:
        # Use fixed k for all points
        k_values = np.array([k] * len(X))

    # Find neighbors for each point
    nn = NearestNeighbors(n_neighbors=max(k_values) + 1).fit(X)
    distances, indices = nn.kneighbors(X)

    # Calculate time differences for each point's neighborhood
    avg_time_diffs = []
    for i in range(len(X)):
        # Skip the first neighbor (self)
        k_i = k_values[i]
        neighbors = indices[i, 1 : k_i + 1]
        # Use numpy indexing instead of pandas .iloc
        time_diffs = np.abs(time_labels[i] - time_labels[neighbors]).mean()
        avg_time_diffs.append(time_diffs)

    avg_time_diffs = np.mean(avg_time_diffs)

    if normalize:
        # Compute baseline by comparing with random points
        n_samples = min(100, len(time_labels))  # Handle small datasets
        random_indices = np.random.choice(len(time_labels), n_samples, replace=False)

        baseline_diffs = []
        for i in random_indices:
            random_neighbors = np.random.choice(len(time_labels), n_samples, replace=False)
            baseline_diffs.extend(np.abs(time_labels[i] - time_labels[random_neighbors]))

        baseline = np.mean(baseline_diffs)

        # Normalize: 1 = perfect temporal consistency, 0 = random
        avg_time_diffs_norm = 1 - (avg_time_diffs / baseline)
        avg_time_diffs_norm = float(np.clip(avg_time_diffs_norm, 0, 1))

        return avg_time_diffs_norm

    return avg_time_diffs


def temporal_silhouette(
    X: np.ndarray,
    time_labels: np.ndarray,
    normalize: bool = True,
    distance_metric: Literal["euclidean", "cosine"] = "euclidean",
) -> float:
    """
    Fixed temporal silhouette score measuring whether points are closer
    to their own time point than to other time points.

    Similar to regular silhouette analysis, but using time points as clusters.

    Parameters
    ----------
    X : np.ndarray
        Array of shape (n_cells, n_features) - Embeddings
    time_labels : np.ndarray
        Array of shape (n_cells,) representing time point labels
    normalize : bool
        Whether to normalize score to [0, 1] range
    distance_metric : str
        The distance metric to use ('euclidean' or 'cosine')

    Returns
    -------
    float
        Temporal silhouette score (-1 to 1, or 0 to 1 if normalized)
        Higher values indicate better temporal clustering
    """
    X = np.asarray(X)
    time_labels = np.asarray(time_labels)

    unique_times = np.unique(time_labels)

    # Need at least 2 time points
    if len(unique_times) < 2:
        return 0.0

    # Compute all pairwise distances once
    distances = pairwise_distances(X, metric=distance_metric)

    silhouette_scores = []

    for i in range(len(X)):
        current_time = time_labels[i]

        # a(i): average distance to points in same time cluster
        same_time_mask = (time_labels == current_time) & (np.arange(len(X)) != i)

        if np.sum(same_time_mask) == 0:
            # Only one point at this time - skip or set to 0
            continue

        a_i = np.mean(distances[i, same_time_mask])

        # b(i): minimum average distance to points in other time clusters
        min_avg_distance = float("inf")

        for other_time in unique_times:
            if other_time == current_time:
                continue

            other_time_mask = time_labels == other_time

            if np.sum(other_time_mask) == 0:
                continue

            avg_distance_to_other = np.mean(distances[i, other_time_mask])
            min_avg_distance = min(min_avg_distance, avg_distance_to_other)

        # Standard silhouette formula
        if min_avg_distance == float("inf"):
            # Only one time cluster
            silhouette_i = 0.0
        else:
            b_i = min_avg_distance
            if max(a_i, b_i) > 0:
                silhouette_i = (b_i - a_i) / max(a_i, b_i)
            else:
                silhouette_i = 0.0

        silhouette_scores.append(silhouette_i)

    if len(silhouette_scores) == 0:
        return 0.0

    avg_silhouette = np.mean(silhouette_scores)

    if normalize:
        # Convert from [-1, 1] to [0, 1]
        avg_silhouette = (avg_silhouette + 1) / 2

    return avg_silhouette
