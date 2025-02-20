import numpy as np
import pytest
from czibench.datasets.sc import SingleCellDataset
from czibench.datasets.types import DataType, Organism
from czibench.tasks.sc import ClusteringTask, EmbeddingTask, MetadataLabelPredictionTask


@pytest.fixture
def sc_dataset():
    d = SingleCellDataset(
        path="tests/assets/example-small.h5ad", organism=Organism.HUMAN
    )
    d.load_data()
    n = d.adata.n_obs
    d.adata.obs["cell_type"] = ["type1" if i < n / 2 else "type2" for i in range(n)]
    d.adata.obs["batch"] = ["batchA" if i < n / 2 else "batchB" for i in range(n)]
    emb = np.random.randn(n, 10)
    d.set_output(DataType.EMBEDDING, emb)
    return d


def test_clustering_task_run_task(sc_dataset):
    t = ClusteringTask(label_key="cell_type")
    sc_dataset = t._run_task(sc_dataset)
    assert hasattr(t, "input_labels")
    assert hasattr(t, "predicted_labels")


def test_clustering_task_compute_metrics(sc_dataset):
    t = ClusteringTask(label_key="cell_type")
    t._run_task(sc_dataset)
    m = t._compute_metrics()
    assert "adjusted_rand_index" in m
    assert "normalized_mutual_info" in m


def test_embedding_task_run_task(sc_dataset):
    t = EmbeddingTask(label_key="cell_type")
    sc_dataset = t._run_task(sc_dataset)
    assert hasattr(t, "embedding")
    assert hasattr(t, "input_labels")


def test_embedding_task_compute_metrics(sc_dataset):
    t = EmbeddingTask(label_key="cell_type")
    t._run_task(sc_dataset)
    m = t._compute_metrics()
    assert "silhouette_score" in m


def test_metadatalabel_prediction_task_run_task(sc_dataset):
    t = MetadataLabelPredictionTask(
        label_key="cell_type", n_folds=2, generate_predictions=True
    )
    sc_dataset = t._run_task(sc_dataset)
    assert hasattr(t, "results")
    assert hasattr(t, "predictions")


def test_metadatalabel_prediction_task_compute_metrics(sc_dataset):
    t = MetadataLabelPredictionTask(
        label_key="cell_type", n_folds=2, generate_predictions=True
    )
    t._run_task(sc_dataset)
    m = t._compute_metrics()
    assert "mean_accuracy" in m
    assert "mean_f1" in m
    assert "lr_mean_accuracy" in m
    assert "knn_mean_accuracy" in m
    assert "predictions" in m
