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
    X: np.ndarray, y: np.ndarray | pd.Series, min_class_size: int = 10
) -> tuple[np.ndarray, np.ndarray | pd.Series]:
    """Filter data to remove classes with too few samples.

    Removes classes that have fewer samples than the minimum threshold.
    Useful for ensuring enough samples per class for ML tasks.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Labels array of shape (n_samples,)
        min_class_size: Minimum number of samples required per class

    Returns:
        Tuple containing:
            - Filtered feature matrix
            - Filtered labels as categorical data
    """
    label_name = y.name if hasattr(y, "name") else "unknown"
    logger.info(f"Label composition ({label_name}):")

    value_counts = pd.Series(y).value_counts()
    logger.info(f"Total classes before filtering: {len(value_counts)}")

    filtered_counts = value_counts[value_counts >= min_class_size]
    logger.info(
        f"Total classes after filtering "
        f"(min_class_size={min_class_size}): {len(filtered_counts)}"
    )

    y = pd.Series(y) if isinstance(y, np.ndarray) else y
    class_counts = y.value_counts()

    valid_classes = class_counts[class_counts >= min_class_size].index
    valid_indices = y.isin(valid_classes)

    X_filtered = X[valid_indices]
    y_filtered = y[valid_indices]

    return X_filtered, pd.Categorical(y_filtered)
