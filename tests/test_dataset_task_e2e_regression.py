import functools
import numpy as np
import pytest
from pathlib import Path
from typing import Dict, List, Any

from czbenchmarks.constants import RANDOM_SEED
from czbenchmarks.datasets.utils import load_dataset
from czbenchmarks.tasks.clustering import ClusteringTask, ClusteringTaskInput
from czbenchmarks.tasks.embedding import EmbeddingTask, EmbeddingTaskInput
from czbenchmarks.tasks.label_prediction import (
    MetadataLabelPredictionTask,
    MetadataLabelPredictionTaskInput,
)
from czbenchmarks.tasks.integration import (
    BatchIntegrationTask,
    BatchIntegrationTaskInput,
)
from czbenchmarks.tasks.single_cell.cross_species import (
    CrossSpeciesIntegrationTask,
    CrossSpeciesIntegrationTaskInput,
)
from czbenchmarks.tasks.types import CellRepresentation

"""
This test file performs regression tests on various benchmarking tasks, using past validated expected results.
It depends on real datasets and fixture embeddings hosted in the cloud.

The regression tests ensure that the tasks produce consistent results over time, validating against expected metrics
captured from previous successful runs. If a test fails, the expected metrics should be updated only after validation
by a computational biologist.

Fixtures:
- Embedding fixtures are downloaded from the cloud using `aws s3 cp --recursive s3://cz-benchmarks-results-dev/regression-test-fixtures/embeddings/ tests/fixtures/embeddings/` (currently private to CZI developers)
- Datasets are loaded using the `load_dataset` utility, which fetches real datasets hosted in the cloud.

Tasks Tested:
1. Clustering Task
2. Embedding Task
3. Metadata Label Prediction Task
4. Batch Integration Task
5. Cross-Species Integration Task

Each test validates:
- The structure of the results.
- Specific expectations for task-specific metrics.
- Regression against expected metrics within a defined tolerance.

Note:
- The perturbation task integration test is currently skipped as it has not been validated for benchmarking purposes.
"""


# Helper functions for loading fixtures
def load_embedding_fixture(dataset_name: str) -> np.ndarray:
    """Load embedding fixture from tests/fixtures/embeddings/."""
    fixtures_dir = Path(__file__).parent / "fixtures" / "embeddings"
    embedding_file = fixtures_dir / f"{dataset_name}_UCE_model_variant-4l.npy"
    if dataset_name == "tsv2_bone_marrow":
        embedding_file = (
            fixtures_dir / f"{dataset_name}_UCE_model_variant-4l-embedding.npy"
        )

    if not embedding_file.exists():
        raise FileNotFoundError(f"Embedding fixture not found: {embedding_file}")

    return np.load(embedding_file)


def assert_metrics_match_expected(
    actual_metrics: List[Any],
    expected_metrics: List[Dict[str, Any]],
    tolerance: float = 0.01,
):
    """Assert that actual metrics match expected metrics within tolerance."""
    # Convert actual metrics to comparable format
    actual_dict = {m.metric_type.value: m.value for m in actual_metrics}
    expected_dict = {m["metric_type"]: m["value"] for m in expected_metrics}

    # Check that all expected metrics are present
    for metric_type, expected_value in expected_dict.items():
        assert metric_type in actual_dict, f"Missing metric: {metric_type}"
        actual_value = actual_dict[metric_type]

        # Allow for small numerical differences
        assert abs(actual_value - expected_value) <= tolerance, (
            f"Metric {metric_type}: expected {expected_value}, got {actual_value} (tolerance: {tolerance})"
        )


@pytest.fixture
def dataset():
    """Load the main dataset for testing."""
    return load_dataset("tsv2_bone_marrow")


@pytest.fixture
def human_dataset():
    """Load human dataset for cross-species testing."""
    return load_dataset("human_spermatogenesis")


@pytest.fixture
def mouse_dataset():
    """Load mouse dataset for cross-species testing."""
    return load_dataset("mouse_spermatogenesis")


@pytest.fixture
def rhesus_dataset():
    """Load rhesus macaque dataset for cross-species testing."""
    return load_dataset("rhesus_macaque_spermatogenesis")


@pytest.mark.integration
def test_clustering_task_regression(dataset):
    """Regression test for clustering task using fixture embeddings and expected results."""
    # Load fixture embedding
    model_output: CellRepresentation = load_embedding_fixture("tsv2_bone_marrow")

    # Initialize clustering task
    clustering_task = ClusteringTask(random_seed=RANDOM_SEED)

    # Get raw expression data for baseline computation
    expression_data = dataset.adata.X

    # Compute baseline embedding
    clustering_baseline = clustering_task.compute_baseline(expression_data)
    assert clustering_baseline is not None

    # Run clustering task with fixture embedding
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
        assert hasattr(result, "model_dump")
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

    # Regression test: Compare against expected results
    # Expected results (captured from CZI Virtual Cell Platform benchmarking results at s3://cz-benchmarks-results-dev/v0.10.0/results/20250529_004446-f1736d11.json)
    # If this test fails, update expected_metrics with new values from a successful run AFTER a computational biologist has validated the new results.
    expected_metrics = [
        {"metric_type": "adjusted_rand_index", "value": 0.39411565248721414},
        {"metric_type": "normalized_mutual_info", "value": 0.6947935391845789},
    ]
    assert_metrics_match_expected(clustering_results, expected_metrics, tolerance=0.01)


@pytest.mark.integration
def test_embedding_task_regression(dataset):
    """Regression test for embedding task using fixture embeddings and expected results."""
    # Load fixture embedding
    model_output: CellRepresentation = load_embedding_fixture("tsv2_bone_marrow")

    # Initialize embedding task
    embedding_task = EmbeddingTask(random_seed=RANDOM_SEED)

    # Get raw expression data for baseline computation
    expression_data = dataset.adata.X

    # Compute baseline embedding
    embedding_baseline = embedding_task.compute_baseline(expression_data)
    assert embedding_baseline is not None

    # Run embedding task with fixture embedding
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

    # Regression test: Compare against expected results
    # Expected results (captured from CZI Virtual Cell Platform benchmarking results at s3://cz-benchmarks-results-dev/v0.10.0/results/20250529_004446-f1736d11.json)
    # If this test fails, update expected_metrics with new values from a successful run AFTER a computational biologist has validated the new results.
    expected_metrics = [
        {"metric_type": "silhouette_score", "value": 0.5428805351257324}
    ]
    assert_metrics_match_expected(embedding_results, expected_metrics, tolerance=0.01)


@pytest.mark.integration
def test_metadata_label_prediction_task_regression(dataset):
    """Regression test for metadata label prediction task using fixture embeddings and expected results."""
    # Load fixture embedding
    model_output: CellRepresentation = load_embedding_fixture("tsv2_bone_marrow")

    # Initialize metadata label prediction task
    metadata_label_prediction_task = MetadataLabelPredictionTask(
        random_seed=RANDOM_SEED
    )

    # Get raw expression data for baseline computation
    expression_data = dataset.adata.X

    # Compute baseline embedding
    metadata_label_prediction_baseline = (
        metadata_label_prediction_task.compute_baseline(expression_data)
    )
    assert metadata_label_prediction_baseline is not None

    # Run metadata label prediction task with fixture embedding
    metadata_label_prediction_task_input = MetadataLabelPredictionTaskInput(
        labels=dataset.labels,
    )
    metadata_label_prediction_results = metadata_label_prediction_task.run(
        cell_representation=model_output,
        task_input=metadata_label_prediction_task_input,
    )
    metadata_label_prediction_baseline_results = metadata_label_prediction_task.run(
        cell_representation=metadata_label_prediction_baseline,
        task_input=metadata_label_prediction_task_input,
    )

    # Validate results structure
    assert isinstance(metadata_label_prediction_results, list)
    assert len(metadata_label_prediction_results) > 0
    assert isinstance(metadata_label_prediction_baseline_results, list)
    assert len(metadata_label_prediction_baseline_results) > 0

    # Test specific expectations for metadata label prediction
    metadata_label_prediction_model_metric_names = {
        r.metric_type.value for r in metadata_label_prediction_results 
    }

    assert "mean_fold_accuracy" in metadata_label_prediction_model_metric_names
    assert "mean_fold_f1" in metadata_label_prediction_model_metric_names
    assert "mean_fold_precision" in metadata_label_prediction_model_metric_names
    assert "mean_fold_recall" in metadata_label_prediction_model_metric_names
    assert "mean_fold_auroc" in metadata_label_prediction_model_metric_names

    # Regression test: Compare against expected results
    # Expected results (captured from CZI Virtual Cell Platform benchmarking results at s3://cz-benchmarks-results-dev/v0.10.0/results/20250529_004446-f1736d11.json)
    # If this test fails, update expected_metrics with new values from a successful run AFTER a computational biologist has validated the new results.
    print(metadata_label_prediction_results)
    metadata_label_prediction_results_filtered = [
        r for r in metadata_label_prediction_results if r.params.get("classifier") == "MEAN(knn,lr,rf)"
    ]
    print(metadata_label_prediction_results_filtered)

    expected_metrics = [
        {"metric_type": "mean_fold_accuracy", "value": 0.8498314736043509},
        {"metric_type": "mean_fold_f1", "value": 0.6800767942482768},
        {"metric_type": "mean_fold_precision", "value": 0.7103881595031953},
        {"metric_type": "mean_fold_recall", "value": 0.6713332088941601},
        {"metric_type": "mean_fold_auroc", "value": 0.9864645320019907},
    ]
    # TODO: Set tolerance per metric, if needed
    assert_metrics_match_expected(
        metadata_label_prediction_results_filtered, expected_metrics, tolerance=0.03
    )


@pytest.mark.integration
def test_batch_integration_task_regression(dataset):
    """Regression test for batch integration task using fixture embeddings and expected results."""
    # Load fixture embedding
    model_output: CellRepresentation = load_embedding_fixture("tsv2_bone_marrow")

    # Expected results (captured from test run on 2025-01-18)
    # If this test fails, update expected_metrics with new values from a successful run AFTER a computational biologist has validated the new results.
    # TODO: THESE RESULTS NEED TO BE VALIDATED BY A COMPUTATIONAL BIOLOGIST
    expected_metrics = [
        {"metric_type": "entropy_per_cell", "value": 0.5016479710268167},
        {"metric_type": "batch_silhouette", "value": 0.8620882630348206},
    ]

    # Initialize batch integration task
    batch_integration_task = BatchIntegrationTask(random_seed=RANDOM_SEED)

    # Get raw expression data for baseline computation
    expression_data = dataset.adata.X

    # Compute baseline embedding
    batch_integration_baseline = batch_integration_task.compute_baseline(
        expression_data
    )
    assert batch_integration_baseline is not None

    # Create batch labels from dataset metadata
    batch_columns = ["dataset_id", "assay", "suspension_type", "donor_id"]
    batch_labels = functools.reduce(
        lambda a, b: a + b, [dataset.adata.obs[c].astype(str) for c in batch_columns]
    )

    # Run batch integration task with fixture embedding
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
    batch_integration_model_metrics = [
        r.metric_type.value for r in batch_integration_results
    ]
    assert "entropy_per_cell" in batch_integration_model_metrics
    assert "batch_silhouette" in batch_integration_model_metrics

    # Regression test: Compare against expected results
    assert_metrics_match_expected(
        batch_integration_results, expected_metrics, tolerance=0.01
    )


@pytest.mark.integration
def test_cross_species_integration_task_regression(
    human_dataset, mouse_dataset, rhesus_dataset
):
    """Regression test for cross-species integration task using fixture embeddings and expected results."""
    # Load fixture embeddings for all 3 species
    human_model_output: CellRepresentation = load_embedding_fixture(
        "human_spermatogenesis"
    )
    mouse_model_output: CellRepresentation = load_embedding_fixture(
        "mouse_spermatogenesis"
    )
    rhesus_model_output: CellRepresentation = load_embedding_fixture(
        "rhesus_macaque_spermatogenesis"
    )
    multi_species_model_output = [
        human_model_output,
        mouse_model_output,
        rhesus_model_output,
    ]

    # Initialize cross-species integration task
    cross_species_task = CrossSpeciesIntegrationTask(random_seed=RANDOM_SEED)

    # Run cross-species integration task with fixture embeddings for all 3 species
    cross_species_task_input = CrossSpeciesIntegrationTaskInput(
        labels=[human_dataset.labels, mouse_dataset.labels, rhesus_dataset.labels],
        organism_list=[
            human_dataset.organism,
            mouse_dataset.organism,
            rhesus_dataset.organism,
        ],
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

    # Regression test: Compare against expected results
    # Expected results (captured from CZI Virtual Cell Platform benchmarking results at s3://cz-benchmarks-results-dev/v0.10.0/results/20250529_115809-1e669592.json)
    # If this test fails, update expected_metrics with new values from a successful run AFTER a computational biologist has validated the new results.
    expected_metrics = [
        {"metric_type": "entropy_per_cell", "value": 0.11187911057813454},
        {"metric_type": "batch_silhouette", "value": 0.8448842167854309},
    ]
    assert_metrics_match_expected(
        cross_species_results, expected_metrics, tolerance=0.01
    )


@pytest.mark.integration
@pytest.mark.skip(
    reason="Perturbation task has not yet been validated as being effective"
)
def test_perturbation_task_integration():
    """Integration test for perturbation task (skipped - task not yet validated)."""
    # This test is skipped because the perturbation task has not yet been
    # validated as being effective for benchmarking purposes
    pass
