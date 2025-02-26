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


def _safelog(a):
    return np.log(a, out=np.zeros_like(a), where=(a != 0))


def nearest_neighbors_hnsw(x, ef=200, M=48, n_neighbors=100):
    import hnswlib

    labels = np.arange(x.shape[0])
    p = hnswlib.Index(space="l2", dim=x.shape[1])
    p.init_index(max_elements=x.shape[0], ef_construction=ef, M=M)
    p.add_items(x, labels)
    p.set_ef(ef)
    idx, dist = p.knn_query(x, k=n_neighbors)
    return idx, dist


def compute_entropy_per_cell(embedding, batch_labels):
    indices, dist = nearest_neighbors_hnsw(embedding, n_neighbors=200)
    unique_batch_labels = np.unique(batch_labels)
    indices_batch = batch_labels.values[indices]

    label_counts_per_cell = np.vstack(
        [(indices_batch == label).sum(1) for label in unique_batch_labels]
    ).T
    label_counts_per_cell_normed = (
        label_counts_per_cell / label_counts_per_cell.sum(1)[:, None]
    )
    return (-label_counts_per_cell_normed * _safelog(label_counts_per_cell_normed)).sum(
        1
    ) / _safelog(len(unique_batch_labels))
