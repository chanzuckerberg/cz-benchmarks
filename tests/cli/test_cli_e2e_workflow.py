import random
import numpy as np
import subprocess
from pytest_mock import MockFixture
from czbenchmarks import runner
from czbenchmarks.models.utils import list_available_models
from czbenchmarks.datasets.utils import load_dataset, list_available_datasets
from czbenchmarks.tasks import ClusteringTask, EmbeddingTask, MetadataLabelPredictionTask
from czbenchmarks.models.types import ModelType
from czbenchmarks.datasets import DataType
from czbenchmarks.metrics.types import MetricResult

def has_nvidia_cli():
    """Check if NVIDIA container CLI is available.
    """
    try:
        subprocess.run(["nvidia-container-cli", "info"], 
                      check=True,
                      stdout=subprocess.PIPE,
                      stderr=subprocess.PIPE)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def test_cli_e2e_workflow(mocker: MockFixture):
    """
    VC-2361: E2E test
    Test end-to-end workflow using CLI with model and dataset.
    
    This test verifies that the complete code path to run a benchmark works,
    focusing on a single dataset/model/task combination. Specifically, it:

    1. Loads a dataset
    2. Selects a model
    3. Runs inference:
       - Uses mock runner with dummy embeddings if NVIDIA runtime is not available
    4. Runs evaluation tasks:
       - Embedding task with cell_type labels
       - Clustering task with cell_type labels
       - Metadata label prediction task with cell_type labels
    5. Verifies that each task produces valid results

    It is not intended to verify that a model's output (embeddings) 
    or that a task's metrics are being computed correctly. That will be 
    done in the unit tests.
    """
    # region: Select a dataset
    all_datasets = list_available_datasets()
    
    # Filter for spermatogenesis datasets which load quickly
    dataset_name = random.choice([d for d in all_datasets if "spermatogenesis" in d])
    
    if not dataset_name:
        # If no matching datasets are found, use a random dataset
        dataset_name = random.choice(all_datasets)
    # endregion: Select a dataset

    # region: Load the dataset
    dataset = load_dataset(dataset_name)
    dataset.load_data()
    # endregion: Load the dataset

    # region: Select a model
    models = list_available_models()
    
    # Filter for SCGPT models which load quickly
    model_name = random.choice([m for m in models if "SCGPT" == m])
    if not model_name:
        # If no matching models are found, use a random model
        model_name = random.choice(models)
    # endregion: Select a model

    # Run inference with mock or real runner based on NVIDIA CLI availability
    if not has_nvidia_cli():
        # Create mock processed data with dummy embeddings
        mock_processed_data = dataset
        n_cells = dataset.adata.n_obs
        dummy_embeddings = np.random.normal(size=(n_cells, 100))
        model_type = ModelType[model_name]
        mock_processed_data.outputs[model_type] = {DataType.EMBEDDING: dummy_embeddings}

        # Mock run_inference to return our processed data
        mock_run_inference = mocker.patch.object(
            runner, "run_inference", return_value=mock_processed_data
        )
        dataset = runner.run_inference(model_name, dataset)
        # Verify run_inference was called correctly
        mock_run_inference.assert_called_once_with(model_name, dataset)
    else:
        dataset = runner.run_inference(model_name, dataset)

    # region: Run tasks
    # region: Run embedding task
    embedding_task = EmbeddingTask(label_key="cell_type")
    embedding_results = embedding_task.run(dataset)
    
    # Verify embedding results
    assert embedding_results is not None, "Embedding task returned no results"
    assert isinstance(embedding_results, dict), "Embedding results should be a dictionary"
    assert ModelType[model_name] in embedding_results, f"Embedding results should contain {model_name}"
    embedding_model_results = embedding_results[ModelType[model_name]]
    assert isinstance(embedding_model_results, list), "Embedding model results should be a list"
    assert len(embedding_model_results) > 0, "Embedding model results should not be empty"
    assert all(isinstance(r, MetricResult) for r in embedding_model_results), "All embedding results should be MetricResult objects"
    # endregion: Run embedding task

    # region: Run clustering task
    clustering_task = ClusteringTask(label_key="cell_type")
    clustering_results = clustering_task.run(dataset)
    
    # Verify clustering results
    assert clustering_results is not None, "Clustering task returned no results"
    assert isinstance(clustering_results, dict), "Clustering results should be a dictionary"
    assert ModelType[model_name] in clustering_results, f"Clustering results should contain {model_name}"
    clustering_model_results = clustering_results[ModelType[model_name]]
    assert isinstance(clustering_model_results, list), "Clustering model results should be a list"
    assert len(clustering_model_results) > 0, "Clustering model results should not be empty"
    assert all(isinstance(r, MetricResult) for r in clustering_model_results), "All clustering results should be MetricResult objects"
    # endregion: Run clustering task

    # region: Run metadata label prediction task
    prediction_task = MetadataLabelPredictionTask(
        label_key="cell_type",
        n_folds=2,  # Use fewer folds for faster testing
        min_class_size=1  # Small classes for testing
    )
    prediction_results = prediction_task.run(dataset)
    
    # Verify prediction results
    assert prediction_results is not None, "Prediction task returned no results"
    assert isinstance(prediction_results, dict), "Prediction results should be a dictionary"
    assert ModelType[model_name] in prediction_results, f"Prediction results should contain {model_name}"
    prediction_model_results = prediction_results[ModelType[model_name]]
    assert isinstance(prediction_model_results, list), "Prediction model results should be a list"
    assert len(prediction_model_results) > 0, "Prediction model results should not be empty"
    assert all(isinstance(r, MetricResult) for r in prediction_model_results), "All prediction results should be MetricResult objects"
    # endregion: Run metadata label prediction task
    # endregion: Run tasks