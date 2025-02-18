# NOT_YET_READY # This file is not yet ready for use. It is a work in progress and may contain incomplete or untested code.

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock


from czibench.datasets.sc import SingleCellDataset
from czibench.tasks.sc import (
    ClusteringTask,
    EmbeddingTask,
    BatchIntegrationTask,
    MetadataLabelPredictionTask,
)
from czibench.tasks.utils import cluster_embedding, filter_minimum_class
from czibench.metrics.clustering import adjusted_rand_index, normalized_mutual_info
from czibench.metrics.embedding import silhouette_score, compute_entropy_per_cell


import anndata as ad

@pytest.fixture
def sample_single_cell_dataset():
    file_path = "tests/assets/example-small.h5ad"
    
    # Load the actual dataset
    adata = ad.read_h5ad(file_path)
    
    # Ensure the dataset contains expected fields
    assert "X" in dir(adata), "Dataset does not contain expression data."

    # Ensure embedding is available
    if "X_pca" in adata.obsm:
        embedding = adata.obsm["X_pca"]
    else:
        # Generate mock PCA embedding if missing
        embedding = np.random.rand(adata.n_obs, 10)
        adata.obsm["X_pca"] = embedding  # Save into AnnData object

    # Ensure 'batch' column exists, if missing create a mock one
    if "batch" not in adata.obs.columns:
        adata.obs["batch"] = np.random.choice(["Batch1", "Batch2"], adata.n_obs)

    # Create dataset
    data = SingleCellDataset(path=file_path, organism="homo_sapiens")
    data.adata = adata  # Assign AnnData object
    data.output_embedding = embedding  # Ensure valid embedding
    data.sample_metadata = adata.obs.copy()  # Copy metadata
    
    return data


def test_clustering_task(sample_single_cell_dataset):
    task = ClusteringTask(label_key="cell_type")
    
    assert task.validate(sample_single_cell_dataset), "Dataset validation failed."
    
    sample_single_cell_dataset = task._run_task(sample_single_cell_dataset)
    metrics = task._compute_metrics()
    
    assert "adjusted_rand_index" in metrics
    assert "normalized_mutual_info" in metrics
    assert isinstance(metrics["adjusted_rand_index"], float)
    assert isinstance(metrics["normalized_mutual_info"], float)



def test_embedding_task(sample_single_cell_dataset):
    task = EmbeddingTask(label_key="cell_type")
    
    assert task.validate(sample_single_cell_dataset), "Dataset validation failed."
    
    sample_single_cell_dataset = task._run_task(sample_single_cell_dataset)
    metrics = task._compute_metrics()
    
    assert "silhouette_score" in metrics
    assert isinstance(metrics["silhouette_score"], float)


# TODO - debug
# def test_batch_integration_task(sample_single_cell_dataset):
#     """Test the BatchIntegrationTask execution and metric computation."""
#     task = BatchIntegrationTask(label_key="cell_type", batch_key="batch")

#     # Ensure batch key exists in sample metadata
#     assert "batch" in sample_single_cell_dataset.sample_metadata.columns, "Missing 'batch' column in dataset."

#     assert task.validate(sample_single_cell_dataset), "Dataset validation failed."
    
#     sample_single_cell_dataset = task._run_task(sample_single_cell_dataset)
#     metrics = task._compute_metrics()
    
#     assert "entropy_per_cell" in metrics
#     assert "silhouette_score" in metrics
#     assert isinstance(metrics["entropy_per_cell"], float)
#     assert isinstance(metrics["silhouette_score"], float)




def test_metadata_label_prediction_task(sample_single_cell_dataset):
    """Test the MetadataLabelPredictionTask execution and metric computation."""
    task = MetadataLabelPredictionTask(label_key="cell_type", generate_predictions=True)
    
    assert task.validate(sample_single_cell_dataset), "Dataset validation failed."
    
    sample_single_cell_dataset = task._run_task(sample_single_cell_dataset)
    metrics = task._compute_metrics()
    
    assert "mean_accuracy" in metrics
    assert "mean_f1" in metrics
    assert "mean_precision" in metrics
    assert "mean_recall" in metrics
    assert isinstance(metrics["mean_accuracy"], float)
    assert isinstance(metrics["mean_f1"], float)
    assert isinstance(metrics["mean_precision"], float)
    assert isinstance(metrics["mean_recall"], float)

    # Check predictions are generated
    assert "predictions" in metrics
    assert isinstance(metrics["predictions"], pd.DataFrame)



def test_cluster_embedding(sample_single_cell_dataset):
    """Test cluster_embedding function using a real AnnData object."""
    adata = sample_single_cell_dataset.adata
    
    # Ensure embedding is available
    assert "X_pca" in adata.obsm or "emb" in adata.obsm, "No embedding found in the dataset."

    # Assign valid embedding
    embedding_key = "X_pca" if "X_pca" in adata.obsm else "emb"
    adata.obsm["emb"] = adata.obsm[embedding_key]  

    labels = cluster_embedding(adata, obsm_key="emb")

    assert isinstance(labels, list)
    assert len(labels) == adata.n_obs




def test_filter_minimum_class():
    """Test filtering of classes with minimum sample size."""
    X = np.random.rand(100, 10)
    y = np.array(["A"] * 30 + ["B"] * 50 + ["C"] * 20)
    
    X_filtered, y_filtered = filter_minimum_class(X, y, min_class_size=25)
    
    assert X_filtered.shape[0] == 80  # Class "C" should be removed
    assert set(y_filtered) == {"A", "B"}

