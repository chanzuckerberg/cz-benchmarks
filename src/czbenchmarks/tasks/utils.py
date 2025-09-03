import logging
from typing import List, Literal

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from ..constants import RANDOM_SEED
from ..tasks.types import CellRepresentation
from .constants import FLAVOR, KEY_ADDED, OBSM_KEY

logger = logging.getLogger(__name__)

MULTI_DATASET_TASK_NAMES = frozenset(["cross_species"])

TASK_NAMES = frozenset(
    {
        "clustering",
        "embedding",
        "sequential",
        "label_prediction",
        "integration",
        "perturbation",
    }.union(MULTI_DATASET_TASK_NAMES)
)


def print_metrics_summary(metrics_list):
    """Print a nice summary table of all metrics.

    Args:
        metrics_list: List of MetricResult objects or dict with metric lists
    """
    # Handle both list and dict inputs for backward compatibility
    if isinstance(metrics_list, dict):
        # Convert dict format to flat list
        all_metrics = []
        for metric_results in metrics_list.values():
            all_metrics.extend(metric_results)
        metrics_list = all_metrics

    if not metrics_list:
        print("No metrics to display.")
        return

    # Group metrics by type
    from collections import defaultdict

    grouped_metrics = defaultdict(list)

    for metric in metrics_list:
        metric_name = (
            metric.metric_type.value
            if hasattr(metric.metric_type, "value")
            else str(metric.metric_type)
        )
        grouped_metrics[metric_name].append(metric)

    # Determine grouping strategy based on available parameters
    sample_metric = metrics_list[0]
    grouping_keys = list(sample_metric.params.keys()) if sample_metric.params else []

    print("\n=== Metrics Summary ===")

    if "condition" in grouping_keys:
        # Group by condition (perturbation-style)
        _print_condition_grouped_metrics(grouped_metrics)
    elif "classifier" in grouping_keys:
        # Group by classifier (label prediction-style)
        _print_classifier_grouped_metrics(grouped_metrics)
    else:
        # Simple metric listing
        _print_simple_metrics(grouped_metrics)

    # Overall statistics
    print("\nOverall Statistics:")
    for metric_name, results in grouped_metrics.items():
        values = [r.value for r in results if not np.isnan(r.value)]
        if values:
            print(
                f"{metric_name.replace('_', ' ').title()}: "
                f"mean={np.mean(values):.4f}, std={np.std(values):.4f}, "
                f"count={len(values)}"
            )


def _print_condition_grouped_metrics(grouped_metrics):
    """Print metrics grouped by condition."""
    # Extract all unique conditions
    all_conditions = set()
    for results in grouped_metrics.values():
        for result in results:
            if "condition" in result.params:
                all_conditions.add(result.params["condition"])

    all_conditions = sorted(all_conditions)

    if not all_conditions:
        _print_simple_metrics(grouped_metrics)
        return

    # Create summary table
    summary_data = []
    for condition in all_conditions:
        row = {"condition": condition}

        for metric_name, results in grouped_metrics.items():
            # Find result for this condition
            condition_result = next(
                (r for r in results if r.params.get("condition") == condition), None
            )
            if condition_result:
                row[metric_name] = f"{condition_result.value:.4f}"
            else:
                row[metric_name] = "N/A"

        summary_data.append(row)

    # Print table
    if summary_data:
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False))
        print(f"\nResults across {len(all_conditions)} conditions")


def _print_classifier_grouped_metrics(grouped_metrics):
    """Print metrics grouped by classifier."""
    # Extract all unique classifiers
    all_classifiers = set()
    for results in grouped_metrics.values():
        for result in results:
            if "classifier" in result.params:
                all_classifiers.add(result.params["classifier"])

    all_classifiers = sorted(all_classifiers)

    # Create summary table
    summary_data = []
    for classifier in all_classifiers:
        row = {"classifier": classifier}

        for metric_name, results in grouped_metrics.items():
            # Find result for this classifier
            classifier_result = next(
                (r for r in results if r.params.get("classifier") == classifier), None
            )
            if classifier_result:
                row[metric_name] = f"{classifier_result.value:.4f}"
            else:
                row[metric_name] = "N/A"

        summary_data.append(row)

    # Print table
    if summary_data:
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False))
        print(f"\nResults across {len(all_classifiers)} classifiers")


def _print_simple_metrics(grouped_metrics):
    """Print simple metric listing without grouping."""
    for metric_name, results in grouped_metrics.items():
        print(f"\n{metric_name.replace('_', ' ').title()}:")
        for i, result in enumerate(results):
            params_str = (
                ", ".join([f"{k}={v}" for k, v in result.params.items()])
                if result.params
                else ""
            )
            params_display = f" ({params_str})" if params_str else ""
            print(f"  {i + 1}: {result.value:.4f}{params_display}")


def binarize_values(y_true: np.ndarray, y_pred: np.ndarray):
    """Convert continuous values to binary classification.

    Filters out NaN and infinite values, then converts values to binary
    using a threshold of 0 (positive values become 1, others become 0).

    Args:
        y_true: True continuous values
        y_pred: Predicted continuous values

    Returns:
        tuple: (true_binary, pred_binary) - binary arrays for classification metrics
    """
    ids = np.where(~np.isnan(y_true) & ~np.isinf(y_true))[0]
    y_true = y_true[ids]
    y_pred = y_pred[ids]
    pred_binary = (y_pred > 0).astype(int)
    true_binary = (y_true > 0).astype(int)
    return true_binary, pred_binary


def cluster_embedding(
    adata: AnnData,
    n_iterations: int = 2,
    flavor: Literal["leidenalg", "igraph"] = FLAVOR,
    use_rep: str = "X",
    key_added: str = KEY_ADDED,
    *,
    random_seed: int = RANDOM_SEED,
) -> List[int]:
    """Cluster cells in embedding space using the Leiden algorithm.

    Computes nearest neighbors in the embedding space and runs the Leiden
    community detection algorithm to identify clusters.

    Args:
        adata: AnnData object containing the embedding
        n_iterations: Number of iterations for the Leiden algorithm
        flavor: Flavor of the Leiden algorithm
        use_rep: Key in adata.obsm containing the embedding coordinates
                  If None, embedding is assumed to be in adata.X
        key_added: Key in adata.obs to store the cluster assignments
        random_seed (int): Random seed for reproducibility
    Returns:
        List of cluster assignments as integers
    """
    sc.pp.neighbors(adata, use_rep=use_rep, random_state=random_seed)
    sc.tl.leiden(
        adata,
        key_added=key_added,
        flavor=flavor,
        n_iterations=n_iterations,
        random_state=random_seed,
    )
    return list(adata.obs[key_added])


def filter_minimum_class(
    features: np.ndarray,
    labels: np.ndarray | pd.Series,
    min_class_size: int = 10,
) -> tuple[np.ndarray, np.ndarray | pd.Series]:
    """Filter data to remove classes with too few samples.

    Removes classes that have fewer samples than the minimum threshold.
    Useful for ensuring enough samples per class for ML tasks.

    Args:
        features: Feature matrix of shape (n_samples, n_features)
        labels: Labels array of shape (n_samples,)
        min_class_size: Minimum number of samples required per class

    Returns:
        Tuple containing:
            - Filtered feature matrix
            - Filtered labels as categorical data
    """
    label_name = labels.name if hasattr(labels, "name") else "unknown"
    logger.info(f"Label composition ({label_name}):")

    class_counts = pd.Series(labels).value_counts()
    logger.info(f"Total classes before filtering: {len(class_counts)}")

    filtered_counts = class_counts[class_counts >= min_class_size]
    logger.info(f"Total classes after filtering (min_class_size={min_class_size}): {len(filtered_counts)}")

    labels = pd.Series(labels) if isinstance(labels, np.ndarray) else labels
    class_counts = labels.value_counts()

    valid_classes = class_counts[class_counts >= min_class_size].index
    valid_indices = labels.isin(valid_classes)

    features_filtered = features[valid_indices]
    labels_filtered = labels[valid_indices]

    return features_filtered, pd.Categorical(labels_filtered)


def run_standard_scrna_workflow(
    adata: AnnData,
    n_top_genes: int = 3000,
    n_pcs: int = 50,
    obsm_key: str = OBSM_KEY,
    random_state: int = RANDOM_SEED,
) -> CellRepresentation:
    """Run a standard preprocessing workflow for single-cell RNA-seq data.


    This function performs common preprocessing steps for scRNA-seq analysis:
    1. Normalization of counts per cell
    2. Log transformation
    3. Identification of highly variable genes
    4. Subsetting to highly variable genes
    5. Principal component analysis

    Args:
        adata: AnnData object containing the raw count data
        n_top_genes: Number of highly variable genes to select
        n_pcs: Number of principal components to compute
        random_state: Random seed for reproducibility
    """
    adata = adata.copy()

    # Standard preprocessing steps for single-cell data
    sc.pp.normalize_total(adata)  # Normalize counts per cell
    sc.pp.log1p(adata)  # Log-transform the data

    # Identify highly variable genes using Seurat method
    # FIXME: should n_top_genes be set to min(n_top_genes, n_genes)?
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)

    # Subset to only highly variable genes to reduce noise
    adata = adata[:, adata.var["highly_variable"]].copy()

    # Run PCA for dimensionality reduction
    sc.pp.pca(adata, n_comps=n_pcs, key_added=obsm_key, random_state=random_state)

    return adata.obsm[obsm_key]
