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
    features: np.ndarray, labels: np.ndarray | pd.Series, min_class_size: int = 10
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


def _safelog(a):
    a_float = np.asarray(a, dtype=np.float64)
    return np.log(a_float, out=np.zeros_like(a_float), where=(a_float != 0))


def nearest_neighbors_hnsw(x, ef=200, M=48, n_neighbors=100):
    import hnswlib

    labels = np.arange(x.shape[0])
    p = hnswlib.Index(space="l2", dim=x.shape[1])
    p.init_index(max_elements=x.shape[0], ef_construction=ef, M=M)
    p.add_items(x, labels)
    p.set_ef(ef)
    idx, dist = p.knn_query(x, k=n_neighbors)
    return idx, dist


def compute_entropy_per_cell(embedding: np.ndarray, batch_labels: np.ndarray):
    indices, dist = nearest_neighbors_hnsw(embedding, n_neighbors=200)
    unique_batch_labels = np.unique(batch_labels)
    indices_batch = batch_labels[indices]

    label_counts_per_cell = np.vstack(
        [(indices_batch == label).sum(1) for label in unique_batch_labels]
    ).T
    label_counts_per_cell_normed = (
        label_counts_per_cell / label_counts_per_cell.sum(1)[:, None]
    )
    return (-label_counts_per_cell_normed * _safelog(label_counts_per_cell_normed)).sum(
        1
    ) / _safelog(len(unique_batch_labels))
