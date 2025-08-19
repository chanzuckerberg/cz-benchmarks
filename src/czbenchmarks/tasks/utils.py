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
        "label_prediction",
        "integration",
        "perturbation",
    }.union(MULTI_DATASET_TASK_NAMES)
)


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
    return list(adata.obs["leiden"])


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
    logger.info(
        f"Total classes after filtering "
        f"(min_class_size={min_class_size}): {len(filtered_counts)}"
    )

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