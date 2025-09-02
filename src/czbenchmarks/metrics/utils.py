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
        X: Cell Xedding matrix of shape (n_cells, n_features)
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
    Measure how temporally close neighbors are in Xedding space.

    Parameters:
    -----------
    X : array-like
        Xedding matrix
    time_labels : array-like
        Timepoint labels (numpy array, pandas Series, or list)
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
        distances, _ = NearestNeighbors(n_neighbors=30).fit(X).kneighbors(X)
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
        avg_time_diffs = 1 - (avg_time_diffs / baseline)
        avg_time_diffs = np.clip(avg_time_diffs, 0, 1)

    return avg_time_diffs


def temporal_silhouette(
    X: np.ndarray,
    time_labels: np.ndarray,
    rescale: bool = True,
    chunk_size: int = 256,
    metric: Literal["euclidean", "cosine"] = "euclidean",
) -> float:
    """
    Temporal silhouette score measuring whether points at time tn are closer
    to points at time tn+1 than to points at other time points.

    Parameters
    ----------
    X : np.ndarray
        Array of shape (n_cells, n_features) - Xeddings
    time_labels : np.ndarray
        Array of shape (n_cells,) representing time point labels (e.g., 32, 35, 67, 79)
    rescale : bool
        Scale score into the range [0, 1]
    chunk_size : int
        Size of chunks to process at a time for distance computation
    metric : str
        The distance metric to use ('euclidean' or 'cosine')

    Returns
    -------
    float
        Temporal silhouette score
    """
    unique_times = np.unique(time_labels)
    unique_times = np.sort(unique_times)

    # Only consider consecutive time pairs
    valid_pairs = []
    for i in range(len(unique_times) - 1):
        tn = unique_times[i]
        tn_plus_1 = unique_times[i + 1]
        valid_pairs.append((tn, tn_plus_1))

    if len(valid_pairs) == 0:
        return 0.0

    temporal_silhouette_scores = []

    # Process in chunks to manage memory
    for start_idx in range(0, len(X), chunk_size):
        end_idx = min(start_idx + chunk_size, len(X))
        chunk_X = X[start_idx:end_idx]
        chunk_labels = time_labels[start_idx:end_idx]

        chunk_scores = []

        for i, (current_time, current_label) in enumerate(zip(chunk_X, chunk_labels)):
            # Only compute score for points that have a next time point
            has_next_time = False
            next_time = None

            for tn, tn_plus_1 in valid_pairs:
                if current_label == tn:
                    next_time = tn_plus_1
                    has_next_time = True
                    break

            if not has_next_time:
                continue

            # Compute distances from current point to all other points
            distances = pairwise_distances(current_time.reshape(1, -1), X, metric=metric).flatten()

            # Get distances to next time points (tn+1)
            next_time_mask = time_labels == next_time
            if np.sum(next_time_mask) == 0:
                continue
            distances_to_next = distances[next_time_mask]
            a_i = np.mean(distances_to_next)  # Average distance to tn+1

            # Get distances to all other time points (not tn+1 and not current time)
            other_times_mask = (time_labels != next_time) & (time_labels != current_label)
            if np.sum(other_times_mask) == 0:
                continue
            distances_to_others = distances[other_times_mask]
            b_i = np.mean(distances_to_others)  # Average distance to other times

            # Compute silhouette-like score for this point
            # We want a_i (distance to next time) to be small relative to b_i (distance to others)
            if max(a_i, b_i) > 0:
                silhouette_i = (b_i - a_i) / max(a_i, b_i)
            else:
                silhouette_i = 0.0

            chunk_scores.append(silhouette_i)

        temporal_silhouette_scores.extend(chunk_scores)

    if len(temporal_silhouette_scores) == 0:
        return 0.0

    avg_score = np.mean(temporal_silhouette_scores)

    if rescale:
        avg_score = (avg_score + 1) / 2

    return avg_score
