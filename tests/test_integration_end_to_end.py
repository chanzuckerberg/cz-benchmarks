import json
import numpy as np
import pytest

from czbenchmarks.constants import RANDOM_SEED
from czbenchmarks.datasets.single_cell_labeled import SingleCellLabeledDataset
from czbenchmarks.datasets.utils import load_dataset
from czbenchmarks.tasks.clustering import ClusteringTask, ClusteringTaskInput
from czbenchmarks.tasks.embedding import EmbeddingTask, EmbeddingTaskInput
from czbenchmarks.tasks.label_prediction import MetadataLabelPredictionTask, MetadataLabelPredictionTaskInput
from czbenchmarks.tasks.integration import BatchIntegrationTask, BatchIntegrationTaskInput
from czbenchmarks.tasks.single_cell.cross_species import CrossSpeciesIntegrationTask, CrossSpeciesIntegrationTaskInput
from czbenchmarks.tasks.single_cell.perturbation import PerturbationTask, PerturbationTaskInput
from czbenchmarks.tasks.types import CellRepresentation


@pytest.fixture
def dataset():
    """Load the main dataset for testing."""
    return load_dataset("tsv2_prostate")


@pytest.fixture
def human_dataset():
    """Load human dataset for cross-species testing."""
    return load_dataset("human_spermatogenesis")


@pytest.fixture
def mouse_dataset():
    """Load mouse dataset for cross-species testing."""
    return load_dataset("mouse_spermatogenesis")


@pytest.mark.integration
def test_clustering_task_integration(dataset):
    """Integration test for clustering task with model and baseline embeddings."""
    # Create random model output as a stand-in for real model results
    model_output: CellRepresentation = np.random.rand(dataset.adata.shape[0], 10)
    
    # Initialize clustering task
    clustering_task = ClusteringTask(random_seed=RANDOM_SEED)
    
    # Get raw expression data for baseline computation
    expression_data = dataset.adata.X
    
    # Compute baseline embedding
    clustering_baseline = clustering_task.compute_baseline(expression_data)
    assert clustering_baseline is not None
    
    # Run clustering task with both model output and baseline
    clustering_task_input = ClusteringTaskInput(
        obs=dataset.adata.obs,
        input_labels=dataset.labels,
        use_rep="X",
    )
    clustering_results = clustering_task.run(
        cell_representation=model_output,
        task_input=clustering_task_input,
    )
    clustering_baseline_results = clustering_task.run(
        cell_representation=clustering_baseline,
        task_input=clustering_task_input,
    )
    
    # Validate results structure
    assert isinstance(clustering_results, list)
    assert len(clustering_results) > 0
    assert isinstance(clustering_baseline_results, list)
    assert len(clustering_baseline_results) > 0
    
    # Verify each result has the expected structure
    for result in clustering_results + clustering_baseline_results:
        assert hasattr(result, 'model_dump')
        result_dict = result.model_dump()
        assert isinstance(result_dict, dict)
        assert "metric_type" in result_dict
        assert "value" in result_dict
        assert "params" in result_dict
        assert isinstance(result_dict["value"], (int, float))
        assert isinstance(result_dict["params"], dict)
    
    # Test specific expectations for clustering
    clustering_model_metrics = [r.metric_type.value for r in clustering_results]
    assert "adjusted_rand_index" in clustering_model_metrics
    assert "normalized_mutual_info" in clustering_model_metrics
    
    # Test JSON serialization
    results_dict = {
        "model": [result.model_dump() for result in clustering_results],
        "baseline": [result.model_dump() for result in clustering_baseline_results],
    }
    json_output = json.dumps(results_dict, indent=2, default=str)
    assert isinstance(json_output, str)
    assert len(json_output) > 0
    
    # Verify we can parse JSON back
    parsed_results = json.loads(json_output)
    assert isinstance(parsed_results, dict)
    assert "model" in parsed_results
    assert "baseline" in parsed_results


@pytest.mark.integration
def test_embedding_task_integration(dataset):
    """Integration test for embedding task with model and baseline embeddings."""
    # Create random model output as a stand-in for real model results
    model_output: CellRepresentation = np.random.rand(dataset.adata.shape[0], 10)
    
    # Initialize embedding task
    embedding_task = EmbeddingTask(random_seed=RANDOM_SEED)
    
    # Get raw expression data for baseline computation
    expression_data = dataset.adata.X
    
    # Compute baseline embedding
    embedding_baseline = embedding_task.compute_baseline(expression_data)
    assert embedding_baseline is not None
    
    # Run embedding task with both model output and baseline
    embedding_task_input = EmbeddingTaskInput(
        input_labels=dataset.labels,
    )
    embedding_results = embedding_task.run(
        cell_representation=model_output,
        task_input=embedding_task_input,
    )
    embedding_baseline_results = embedding_task.run(
        cell_representation=embedding_baseline,
        task_input=embedding_task_input,
    )
    
    # Validate results structure
    assert isinstance(embedding_results, list)
    assert len(embedding_results) > 0
    assert isinstance(embedding_baseline_results, list)
    assert len(embedding_baseline_results) > 0
    
    # Test specific expectations for embedding
    embedding_model_metrics = [r.metric_type.value for r in embedding_results]
    assert "silhouette_score" in embedding_model_metrics


@pytest.mark.integration
def test_prediction_task_integration(dataset):
    """Integration test for prediction task with model and baseline embeddings."""
    # Create random model output as a stand-in for real model results
    model_output: CellRepresentation = np.random.rand(dataset.adata.shape[0], 10)
    
    # Initialize prediction task
    prediction_task = MetadataLabelPredictionTask(random_seed=RANDOM_SEED)
    
    # Get raw expression data for baseline computation
    expression_data = dataset.adata.X
    
    # Compute baseline embedding
    prediction_baseline = prediction_task.compute_baseline(expression_data)
    assert prediction_baseline is not None
    
    # Run prediction task with both model output and baseline
    prediction_task_input = MetadataLabelPredictionTaskInput(
        labels=dataset.labels,
    )
    prediction_results = prediction_task.run(
        cell_representation=model_output,
        task_input=prediction_task_input,
    )
    prediction_baseline_results = prediction_task.run(
        cell_representation=prediction_baseline,
        task_input=prediction_task_input,
    )
    
    # Validate results structure
    assert isinstance(prediction_results, list)
    assert len(prediction_results) > 0
    assert isinstance(prediction_baseline_results, list)
    assert len(prediction_baseline_results) > 0
    
    # Test specific expectations for prediction
    prediction_model_metrics = [r.metric_type.value for r in prediction_results]
    assert "mean_fold_accuracy" in prediction_model_metrics
    assert "mean_fold_f1" in prediction_model_metrics
    assert "mean_fold_precision" in prediction_model_metrics
    assert "mean_fold_recall" in prediction_model_metrics
    assert "mean_fold_auroc" in prediction_model_metrics


@pytest.mark.integration
def test_batch_integration_task_integration(dataset):
    """Integration test for batch integration task with model and baseline embeddings."""
    # Create random model output as a stand-in for real model results
    model_output: CellRepresentation = np.random.rand(dataset.adata.shape[0], 10)
    
    # Initialize batch integration task
    batch_integration_task = BatchIntegrationTask(random_seed=RANDOM_SEED)
    
    # Get raw expression data for baseline computation
    expression_data = dataset.adata.X
    
    # Compute baseline embedding
    batch_integration_baseline = batch_integration_task.compute_baseline(expression_data)
    assert batch_integration_baseline is not None
    
    # Run batch integration task with both model output and baseline
    # Create artificial batch labels for testing (ensure there are multiple batches)
    batch_labels = np.random.choice(["batch_1", "batch_2", "batch_3"], size=len(dataset.labels))
    batch_integration_task_input = BatchIntegrationTaskInput(
        labels=dataset.labels,
        batch_labels=batch_labels,
    )
    batch_integration_results = batch_integration_task.run(
        cell_representation=model_output,
        task_input=batch_integration_task_input,
    )
    batch_integration_baseline_results = batch_integration_task.run(
        cell_representation=batch_integration_baseline,
        task_input=batch_integration_task_input,
    )
    
    # Validate results structure
    assert isinstance(batch_integration_results, list)
    assert len(batch_integration_results) > 0
    assert isinstance(batch_integration_baseline_results, list)
    assert len(batch_integration_baseline_results) > 0
    
    # Test specific expectations for batch integration
    batch_integration_model_metrics = [r.metric_type.value for r in batch_integration_results]
    assert "entropy_per_cell" in batch_integration_model_metrics
    assert "batch_silhouette" in batch_integration_model_metrics


@pytest.mark.integration
def test_cross_species_integration_task_integration(human_dataset, mouse_dataset):
    """Integration test for cross-species integration task (model only, no baseline)."""
    # Create random multi-species model outputs for cross-species task
    human_model_output: CellRepresentation = np.random.rand(human_dataset.adata.shape[0], 10)
    mouse_model_output: CellRepresentation = np.random.rand(mouse_dataset.adata.shape[0], 10)
    multi_species_model_output = [human_model_output, mouse_model_output]
    
    # Initialize cross-species integration task
    cross_species_task = CrossSpeciesIntegrationTask(random_seed=RANDOM_SEED)
    
    # Run cross-species integration task (only with model output, no baseline)
    cross_species_task_input = CrossSpeciesIntegrationTaskInput(
        labels=[human_dataset.labels, mouse_dataset.labels],
        organism_list=[human_dataset.organism, mouse_dataset.organism],
    )
    cross_species_results = cross_species_task.run(
        cell_representation=multi_species_model_output,
        task_input=cross_species_task_input,
    )
    
    # Validate results structure
    assert isinstance(cross_species_results, list)
    assert len(cross_species_results) > 0
    
    # Test specific expectations for cross-species integration
    cross_species_model_metrics = [r.metric_type.value for r in cross_species_results]
    assert "entropy_per_cell" in cross_species_model_metrics
    assert "batch_silhouette" in cross_species_model_metrics
    
    # Verify cross-species task doesn't have baseline
    try:
        cross_species_task.compute_baseline()
        assert False, "Cross-species task should not support baseline computation"
    except NotImplementedError:
        pass  # Expected behavior


@pytest.mark.integration
@pytest.mark.skip(reason="Perturbation task has not yet been validated as being effective")
def test_perturbation_task_integration():
    """Integration test for perturbation task (skipped - task not yet validated)."""
    # This test is skipped because the perturbation task has not yet been
    # validated as being effective for benchmarking purposes
    pass