from typing import List

import pandas as pd
import numpy as np
import scanpy as sc
from anndata import AnnData


import logging

logger = logging.getLogger(__name__)


# TODO: Later we can add cluster parameters as kwargs here and add them
# to the task config
def cluster_embedding(adata: AnnData, obsm_key: str = "emb") -> List[int]:
    """Cluster the embedding using the Leiden algorithm"""
    sc.pp.neighbors(adata, use_rep=obsm_key)
    sc.tl.leiden(adata, key_added="leiden", flavor="igraph", n_iterations=2)
    return list(adata.obs["leiden"])


def filter_minimum_class(
    X: np.ndarray, y: np.ndarray | pd.Series, min_class_size: int = 10
) -> tuple[np.ndarray, np.ndarray | pd.Series]:
    logger.info(f"Label composition ({y.name if hasattr(y, 'name') else 'unknown'}):")
    value_counts = pd.Series(y).value_counts()
    logger.info(f"Total classes before filtering: {len(value_counts)}")

    filtered_counts = value_counts[value_counts >= min_class_size]
    logger.info(
        f"Total classes after filtering (min_class_size={min_class_size}): {len(filtered_counts)}"
    )

    y = pd.Series(y) if isinstance(y, np.ndarray) else y
    class_counts = y.value_counts()

    valid_classes = class_counts[class_counts >= min_class_size].index
    valid_indices = y.isin(valid_classes)

    X_filtered = X[valid_indices]
    y_filtered = y[valid_indices]

    return X_filtered, pd.Categorical(y_filtered)
