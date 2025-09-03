import collections
import logging
import statistics
from typing import Iterable, Literal, Union

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
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


def _validate_labels(labels: np.ndarray) -> np.ndarray:
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
            raise ValueError(f"Cannot convert labels to numeric type. Got dtype: {labels.dtype}")

    return labels


def sequential_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    normalize: bool = True,
    distance_metric: Literal["euclidean", "cosine"] = "euclidean",
    use_centroids: bool = False,
) -> float:
    """
    Sequential silhouette score measuring whether consecutive timepoints
    are closer to each other than to non-consecutive timepoints.

    Works with UNSORTED data - does not assume X and labels are pre-sorted.

    Parameters
    ----------
    X : np.ndarray
        Array of shape (n_cells, n_features) - Embeddings (can be unsorted)
    labels : np.ndarray
        Array of shape (n_cells,) representing sequential time labels (can be unsorted)
        Must be numeric or convertible to numeric. String labels will raise error.
    normalize : bool
        Whether to normalize score to [0, 1] range
    distance_metric : str
        The distance metric to use ('euclidean' or 'cosine')
    use_centroids : bool
        If True, compare distances between centroids.
        If False, use point-to-point distances.

    Returns
    -------
    float
        Sequential silhouette score (-1 to 1, or 0 to 1 if normalized)
        Higher values indicate consecutive timepoints are closer than non-consecutive ones
    """
    X = np.asarray(X)
    labels = _validate_labels(labels)

    if len(X) != len(labels):
        raise ValueError("X and labels must have same length")

    # Get unique labels in sorted order (this creates the time ordering)
    unique_labels = np.unique(labels)

    if len(unique_labels) < 3:
        return 0.0  # Need at least 3 timepoints to compare consecutive vs non-consecutive

    if use_centroids:
        return _sequential_silhouette_centroids(X, labels, unique_labels, normalize, distance_metric)
    else:
        return _sequential_silhouette_points(X, labels, unique_labels, normalize, distance_metric)


def _sequential_silhouette_centroids(X, labels, unique_labels, normalize, distance_metric):
    """Compare centroid distances: consecutive vs non-consecutive timepoints."""
    # Calculate centroids for each label
    centroids = {}
    for label in unique_labels:
        mask = labels == label
        if np.sum(mask) > 0:
            centroids[label] = np.mean(X[mask], axis=0)

    silhouette_scores = []

    for i, point in enumerate(X):
        current_label = labels[i]
        current_label_idx = np.where(unique_labels == current_label)[0][0]

        # a(i): distance to consecutive timepoint centroid(s)
        consecutive_distances = []

        # Previous consecutive label
        if current_label_idx > 0:
            prev_label = unique_labels[current_label_idx - 1]
            if prev_label in centroids:
                if distance_metric == "euclidean":
                    dist = np.linalg.norm(point - centroids[prev_label])
                elif distance_metric == "cosine":
                    from sklearn.metrics.pairwise import cosine_distances

                    dist = cosine_distances([point], [centroids[prev_label]])[0, 0]
                consecutive_distances.append(dist)

        # Next consecutive label
        if current_label_idx < len(unique_labels) - 1:
            next_label = unique_labels[current_label_idx + 1]
            if next_label in centroids:
                if distance_metric == "euclidean":
                    dist = np.linalg.norm(point - centroids[next_label])
                elif distance_metric == "cosine":
                    from sklearn.metrics.pairwise import cosine_distances

                    dist = cosine_distances([point], [centroids[next_label]])[0, 0]
                consecutive_distances.append(dist)

        if len(consecutive_distances) == 0:
            continue

        a_i = min(consecutive_distances)  # Closest consecutive timepoint

        # b(i): minimum distance to non-consecutive timepoint centroids
        non_consecutive_distances = []

        for other_label_idx, other_label in enumerate(unique_labels):
            if other_label == current_label:
                continue

            # Skip if this is a consecutive timepoint
            if abs(other_label_idx - current_label_idx) == 1:
                continue

            if other_label in centroids:
                if distance_metric == "euclidean":
                    dist = np.linalg.norm(point - centroids[other_label])
                elif distance_metric == "cosine":
                    dist = cosine_distances([point], [centroids[other_label]])[0, 0]
                non_consecutive_distances.append(dist)

        if len(non_consecutive_distances) == 0:
            continue

        b_i = np.mean(non_consecutive_distances)  # Average distance to non-consecutive timepoints

        # Silhouette formula: we want consecutive to be closer, so higher is better
        if max(a_i, b_i) > 0:
            silhouette_i = (b_i - a_i) / max(a_i, b_i)
        else:
            silhouette_i = 0.0

        silhouette_scores.append(silhouette_i)

    if len(silhouette_scores) == 0:
        return 0.0

    avg_silhouette = np.mean(silhouette_scores)

    if normalize:
        avg_silhouette = (avg_silhouette + 1) / 2

    return avg_silhouette


def _sequential_silhouette_points(X, labels, unique_labels, normalize, distance_metric):
    """Compare point distances: consecutive vs non-consecutive timepoints."""
    distances = pairwise_distances(X, metric=distance_metric)
    silhouette_scores = []

    for i in range(len(X)):
        current_label = labels[i]
        current_label_idx = np.where(unique_labels == current_label)[0][0]

        # a(i): minimum average distance to points in consecutive timepoint labels
        consecutive_distances = []

        # Previous consecutive label
        if current_label_idx > 0:
            prev_label = unique_labels[current_label_idx - 1]
            prev_label_mask = labels == prev_label
            if np.sum(prev_label_mask) > 0:
                avg_dist = np.mean(distances[i, prev_label_mask])
                consecutive_distances.append(avg_dist)

        # Next consecutive label
        if current_label_idx < len(unique_labels) - 1:
            next_label = unique_labels[current_label_idx + 1]
            next_label_mask = labels == next_label
            if np.sum(next_label_mask) > 0:
                avg_dist = np.mean(distances[i, next_label_mask])
                consecutive_distances.append(avg_dist)

        if len(consecutive_distances) == 0:
            continue

        a_i = min(consecutive_distances)  # Closest consecutive timepoint

        # b(i): minimum average distance to points in non-consecutive timepoint labels
        non_consecutive_distances = []

        for other_label_idx, other_label in enumerate(unique_labels):
            if other_label == current_label:
                continue

            # Skip if this is a consecutive timepoint
            if abs(other_label_idx - current_label_idx) == 1:
                continue

            other_label_mask = labels == other_label
            if np.sum(other_label_mask) > 0:
                avg_dist = np.mean(distances[i, other_label_mask])
                non_consecutive_distances.append(avg_dist)

        if len(non_consecutive_distances) == 0:
            continue

        b_i = min(non_consecutive_distances)  # Closest non-consecutive timepoint

        # Silhouette formula: we want consecutive to be closer (smaller a_i), so higher is better
        if max(a_i, b_i) > 0:
            silhouette_i = (b_i - a_i) / max(a_i, b_i)
        else:
            silhouette_i = 0.0

        silhouette_scores.append(silhouette_i)

    if len(silhouette_scores) == 0:
        return 0.0

    avg_silhouette = np.mean(silhouette_scores)

    if normalize:
        avg_silhouette = (avg_silhouette + 1) / 2

    return avg_silhouette


def sequential_alignment(
    X: np.ndarray, labels: np.ndarray, k: int = 10, normalize: bool = True, adaptive_k: bool = False
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
    labels = _validate_labels(labels)

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
    density_k = min(30, len(X) // 4)
    nn_density = NearestNeighbors(n_neighbors=density_k).fit(X)
    distances, _ = nn_density.kneighbors(X)

    mean_distances = distances[:, -1]
    densities = 1 / (mean_distances + 1e-10)

    min_density, max_density = np.percentile(densities, [10, 90])
    normalized_densities = np.clip((densities - min_density) / (max_density - min_density + 1e-10), 0, 1)

    k_scale = 0.5 + 1.5 * (1 - normalized_densities)
    k_values = np.round(base_k * k_scale).astype(int)
    k_values = np.clip(k_values, 3, min(50, len(X) // 2))

    return k_values


def _compute_random_baseline(labels: np.ndarray, k: int) -> float:
    """Compute expected sequential distance for random neighbors."""
    unique_labels = np.unique(labels)

    if len(unique_labels) == 1:
        return 0.0

    n_samples = min(10000, len(labels) * 10)

    random_diffs = []
    for _ in range(n_samples):
        idx1, idx2 = np.random.choice(len(labels), 2, replace=False)
        random_diffs.append(abs(labels[idx1] - labels[idx2]))

    return np.mean(random_diffs)
