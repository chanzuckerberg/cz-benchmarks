import collections
import logging
import statistics
from typing import Iterable, Union

import numpy as np
import pandas as pd
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

    label_counts_per_cell = np.vstack(
        [(indices_batch == label).sum(1) for label in unique_batch_labels]
    ).T
    label_counts_per_cell_normed = (
        label_counts_per_cell / label_counts_per_cell.sum(1)[:, None]
    )
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


def _validate_sequential_labels(labels: np.ndarray) -> np.ndarray:
    """
    Validate that labels are numeric or can be converted to numeric.
    Raises error for string/character labels that can't be ordered.
    """
    labels = np.asarray(labels)

    # Check if labels are strings/characters
    if labels.dtype.kind in ["U", "S", "O"]:  # Unicode, byte string, or object
        # Try to convert to numeric
        try:
            labels = labels.astype(float)
        except (ValueError, TypeError):
            raise ValueError(
                "Labels must be numeric or convertible to numeric. "
                "String/character labels are not supported as they don't have inherent ordering. "
                f"Got labels with dtype: {labels.dtype}"
            )

    # Ensure numeric type
    if not np.issubdtype(labels.dtype, np.number):
        try:
            labels = labels.astype(float)
        except (ValueError, TypeError):
            raise ValueError(
                f"Cannot convert labels to numeric type. Got dtype: {labels.dtype}"
            )

    return labels


def sequential_alignment(
    X: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
    normalize: bool = True,
    adaptive_k: bool = False,
) -> float:
    """
    Measure how sequentially close neighbors are in embedding space.

    Works with UNSORTED data - does not assume X and labels are pre-sorted.

    Parameters:
    -----------
    X : np.ndarray
        Embedding matrix of shape (n_samples, n_features) (can be unsorted)
    labels : np.ndarray
        Sequential labels of shape (n_samples,) (can be unsorted)
        Must be numeric or convertible to numeric. String labels will raise error.
    k : int
        Number of neighbors to consider
    normalize : bool
        Whether to normalize score to [0,1] range
    adaptive_k : bool
        Use adaptive k based on local density

    Returns:
    --------
    float: Sequential alignment score (higher = better sequential consistency)
    """
    X = np.asarray(X)
    labels = _validate_sequential_labels(labels)

    if len(X) != len(labels):
        raise ValueError("X and labels must have same length")

    if len(X) < k + 1:
        raise ValueError(f"Need at least {k + 1} samples for k={k}")

    # Handle edge case: all labels the same
    if len(np.unique(labels)) == 1:
        return 1.0 if normalize else 0.0

    if adaptive_k:
        k_values = _compute_adaptive_k(X, k)
    else:
        k_values = np.array([k] * len(X))

    # Find neighbors for each point
    max_k = max(k_values)
    nn = NearestNeighbors(n_neighbors=max_k + 1).fit(X)
    distances, indices = nn.kneighbors(X)

    # Calculate sequential distances for each point's neighborhood
    sequential_distances = []
    for i in range(len(X)):
        k_i = k_values[i]
        # Skip self (index 0)
        neighbor_indices = indices[i, 1 : k_i + 1]
        neighbor_labels = labels[neighbor_indices]

        # Mean absolute sequential distance to k nearest neighbors
        sequential_dist = np.mean(np.abs(labels[i] - neighbor_labels))
        sequential_distances.append(sequential_dist)

    mean_sequential_distance = np.mean(sequential_distances)

    if not normalize:
        return mean_sequential_distance

    # Compare against expected random sequential distance
    baseline = _compute_random_baseline(labels, k)

    # Normalize: 1 = perfect sequential consistency, 0 = random
    if baseline > 0:
        normalized_score = 1 - (mean_sequential_distance / baseline)
        normalized_score = float(np.clip(normalized_score, 0, 1))
    else:
        normalized_score = 1.0

    return normalized_score


def _compute_adaptive_k(X: np.ndarray, base_k: int) -> np.ndarray:
    """Compute adaptive k values based on local density."""
    # Choose a neighborhood size for density estimation
    # We want a neighborhood larger than base_k (so density reflects a wider local area),
    # avoid n_neighbors==1 (which returns self-distance==0) and ensure we don't exceed
    # the available samples.
    suggested = max(2, base_k * 3, len(X) // 4)
    max_allowed = max(1, len(X) - 1)
    density_k = int(min(30, suggested, max_allowed))
    # Fall back to at least 2 neighbors if dataset is very small
    density_k = max(2, density_k)
    nn_density = NearestNeighbors(n_neighbors=density_k).fit(X)
    distances, _ = nn_density.kneighbors(X)

    mean_distances = distances[:, -1]
    densities = 1 / (mean_distances + 1e-10)

    min_density, max_density = np.percentile(densities, [10, 90])
    normalized_densities = np.clip(
        (densities - min_density) / (max_density - min_density + 1e-10), 0, 1
    )

    k_scale = 0.5 + 1.5 * (1 - normalized_densities)
    k_values = np.round(base_k * k_scale).astype(int)
    upper_bound = min(50, len(X) // 2)
    lower_bound = 3
    if upper_bound < lower_bound:
        k_values = np.full_like(k_values, lower_bound)
    else:
        k_values = np.clip(k_values, lower_bound, upper_bound)

    return k_values


def _compute_random_baseline(labels: np.ndarray, k: int) -> float:
    """Compute expected sequential distance for random neighbors."""
    unique_labels = np.unique(labels)

    if len(unique_labels) == 1:
        return 0.0

    n = len(labels)
    # ensure k does not exceed available neighbors and is at least 1
    k = max(1, min(k, n - 1))
    n_samples = min(10000, n * 10)

    random_diffs = []
    for _ in range(n_samples):
        # pick a random reference index
        i = np.random.randint(0, n)
        # sample k distinct neighbor indices excluding i
        if k == n - 1:
            neighbors = np.delete(np.arange(n), i)
        else:
            # sample from range [0, n-2] then map to [0, n-1] skipping i
            choices = np.random.choice(n - 1, size=k, replace=False)
            neighbors = choices + (choices >= i).astype(int)

        # mean absolute difference between label[i] and its k random neighbors
        random_diffs.append(np.mean(np.abs(labels[i] - labels[neighbors])))

    return float(np.mean(random_diffs))
