from sklearn.metrics import silhouette_score as sklearn_silhouette_score
import numpy as np

def _safelog(a):
    return np.log(a, out=np.zeros_like(a), where=(a != 0))

def silhouette_score(embedding, labels):
    return sklearn_silhouette_score(
        embedding,
        labels
    )
    
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

    label_counts_per_cell = np.vstack([
        (indices_batch == label).sum(1) for label in unique_batch_labels
    ]).T
    label_counts_per_cell_normed = label_counts_per_cell / label_counts_per_cell.sum(1)[:, None]
    return (-label_counts_per_cell_normed * _safelog(label_counts_per_cell_normed)).sum(1) / _safelog(len(unique_batch_labels))
