import logging
from typing import List

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

logger = logging.getLogger(__name__)


# TODO: Later we can add cluster parameters as kwargs here and add them
# to the task config
def cluster_embedding(adata: AnnData, obsm_key: str = "emb") -> List[int]:
    """Cluster cells in embedding space using the Leiden algorithm.

    Computes nearest neighbors in the embedding space and runs the Leiden
    community detection algorithm to identify clusters.

    Args:
        adata: AnnData object containing the embedding
        obsm_key: Key in adata.obsm containing the embedding coordinates

    Returns:
        List of cluster assignments as integers
    """
    sc.pp.neighbors(adata, use_rep=obsm_key)
    sc.tl.leiden(adata, key_added="leiden", flavor="igraph", n_iterations=2)
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
    adata: AnnData, n_top_genes: int = 3000, n_pcs: int = 50, random_state: int = 42
) -> AnnData:
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

    # Identify highly variable genes using Seurat v3 method
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)

    # Subset to only highly variable genes to reduce noise
    adata = adata[:, adata.var["highly_variable"]].copy()

    # Run PCA for dimensionality reduction
    sc.pp.pca(adata, n_comps=n_pcs, random_state=random_state)

    return adata
