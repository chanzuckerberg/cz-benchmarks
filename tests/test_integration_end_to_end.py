import json
import numpy as np
import pytest

from czbenchmarks.constants import RANDOM_SEED
from czbenchmarks.datasets.single_cell_labeled import SingleCellLabeledDataset
from czbenchmarks.datasets import SingleCellPerturbationDataset
from czbenchmarks.datasets.utils import load_dataset
from czbenchmarks.tasks.clustering import ClusteringTaskInput
from czbenchmarks.tasks.embedding import EmbeddingTaskInput
from czbenchmarks.tasks.label_prediction import (
    MetadataLabelPredictionTaskInput,
)
from czbenchmarks.tasks.sequential import SequentialOrganizationTaskInput
from czbenchmarks.tasks.types import CellRepresentation
from czbenchmarks.tasks import (
    ClusteringTask,
    EmbeddingTask,
    MetadataLabelPredictionTask,
    SequentialOrganizationTask,
)
from czbenchmarks.tasks.single_cell import (
    PerturbationExpressionPredictionTask,
    PerturbationExpressionPredictionTaskInput,
)


@pytest.mark.integration
def test_end_to_end_task_execution_predictive_tasks():
    """Integration test that runs all tasks with model and baseline embeddings.

    This test verifies the complete workflow from loading data to generating
    results, ensuring the output JSON structure is correct. It uses real-world
    data from the cloud and is marked as an integration test. It does not test the correctness
    of the task result values, which is handled by `tests/test_dataset_task_e2e_regression.py.
    """
    # Load dataset (requires cloud access)
    dataset: SingleCellLabeledDataset = load_dataset("tsv2_prostate")

    # Create random model output as a stand-in for real model results
    model_output: CellRepresentation = np.random.rand(dataset.adata.shape[0], 10)

    # Initialize all tasks
    clustering_task = ClusteringTask(random_seed=RANDOM_SEED)
    embedding_task = EmbeddingTask(random_seed=RANDOM_SEED)
    prediction_task = MetadataLabelPredictionTask(random_seed=RANDOM_SEED)
    sequential_task = SequentialOrganizationTask(random_seed=RANDOM_SEED)

    # Get raw expression data for baseline computation
    expression_data = dataset.adata.X

    # Compute baseline embeddings for each task
    clustering_baseline = clustering_task.compute_baseline(expression_data)
    embedding_baseline = embedding_task.compute_baseline(expression_data)
    prediction_baseline = prediction_task.compute_baseline(expression_data)
    sequential_baseline = sequential_task.compute_baseline(expression_data)

    # Verify baselines are returned
    assert clustering_baseline is not None
    assert embedding_baseline is not None
    assert prediction_baseline is not None
    assert sequential_baseline is not None

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

    # Run sequential organization task with both model output and baseline
    sequential_task_input = SequentialOrganizationTaskInput(
        obs=dataset.adata.obs,
        input_labels=dataset.labels,
        k=15,
        normalize=True,
        adaptive_k=False,
    )
    sequential_results = sequential_task.run(
        cell_representation=model_output,
        task_input=sequential_task_input,
    )
    sequential_baseline_results = sequential_task.run(
        cell_representation=sequential_baseline,
        task_input=sequential_task_input,
    )

    # Combine all results into a single dictionary
    all_results = {
        "clustering": {
            "model": [result.model_dump() for result in clustering_results],
            "baseline": [result.model_dump() for result in clustering_baseline_results],
        },
        "embedding": {
            "model": [result.model_dump() for result in embedding_results],
            "baseline": [result.model_dump() for result in embedding_baseline_results],
        },
        "prediction": {
            "model": [result.model_dump() for result in prediction_results],
            "baseline": [result.model_dump() for result in prediction_baseline_results],
        },
        "sequential": {
            "model": [result.model_dump() for result in sequential_results],
            "baseline": [result.model_dump() for result in sequential_baseline_results],
        },
    }

    # Validate the overall structure
    assert isinstance(all_results, dict)
    assert len(all_results) == 4
    assert "clustering" in all_results
    assert "embedding" in all_results
    assert "prediction" in all_results
    assert "sequential" in all_results

    # Validate each task has both model and baseline results
    for task_name in ["clustering", "embedding", "prediction", "sequential"]:
        task_results = all_results[task_name]
        assert isinstance(task_results, dict)
        assert "model" in task_results
        assert "baseline" in task_results
        assert isinstance(task_results["model"], list)
        assert isinstance(task_results["baseline"], list)

        # Verify results are not empty
        assert len(task_results["model"]) > 0
        assert len(task_results["baseline"]) > 0

        # Verify each result has the expected structure
        for result_type in ["model", "baseline"]:
            for result in task_results[result_type]:
                assert isinstance(result, dict)
                assert "metric_type" in result
                assert "value" in result
                assert "params" in result
                assert isinstance(result["value"], (int, float))
                assert isinstance(result["params"], dict)

    # Verify JSON serialization works correctly
    json_output = json.dumps(all_results, indent=2, default=str)
    assert isinstance(json_output, str)
    assert len(json_output) > 0

    # Verify we can parse the JSON back (note: enums become strings)
    parsed_results = json.loads(json_output)
    assert isinstance(parsed_results, dict)
    assert len(parsed_results) == 4

    # Verify the parsed structure matches (enums will be strings now)
    for task_name in ["clustering", "embedding", "prediction", "sequential"]:
        assert task_name in parsed_results
        assert "model" in parsed_results[task_name]
        assert "baseline" in parsed_results[task_name]

    # Test specific task expectations

    # Clustering should have ARI and NMI metrics
    clustering_model_metrics = [
        r["metric_type"].value for r in all_results["clustering"]["model"]
    ]
    assert "adjusted_rand_index" in clustering_model_metrics
    assert "normalized_mutual_info" in clustering_model_metrics

    # Embedding should have silhouette score
    embedding_model_metrics = [
        r["metric_type"].value for r in all_results["embedding"]["model"]
    ]
    assert "silhouette_score" in embedding_model_metrics

    # Prediction should have multiple classification metrics
    prediction_model_metrics = [
        r["metric_type"].value for r in all_results["prediction"]["model"]
    ]
    assert "mean_fold_accuracy" in prediction_model_metrics
    assert "mean_fold_f1" in prediction_model_metrics
    assert "mean_fold_precision" in prediction_model_metrics
    assert "mean_fold_recall" in prediction_model_metrics
    assert "mean_fold_auroc" in prediction_model_metrics

    # Sequential organization should have silhouette score and ARI
    sequential_model_metrics = [
        r["metric_type"].value for r in all_results["sequential"]["model"]
    ]
    assert "silhouette_score" in sequential_model_metrics
    assert "adjusted_rand_index" in sequential_model_metrics


@pytest.mark.integration
def test_end_to_end_perturbation_expression_prediction():
    """Integration test for perturbation expression prediction task.

    Loads a perturbation dataset, builds task inputs following the example,
    runs the task on a random model output and a baseline, and verifies result
    structure and JSON serialization.
    """
    # Load dataset (requires cloud access)
    dataset: SingleCellPerturbationDataset = load_dataset(
        "replogle_k562_essential_perturbpredict"
    )

    # Build task input directly from dataset
    task_input = PerturbationExpressionPredictionTaskInput(
        de_results=dataset.de_results,
        var_index=dataset.control_matched_adata.var.index,
        masked_adata_obs=dataset.control_matched_adata.obs,
        target_conditions_to_save=dataset.target_conditions_to_save,
        row_index=dataset.adata.obs.index,
    )

    # Create random model output matching dataset dimensions
    model_output: CellRepresentation = np.random.rand(
        dataset.adata.shape[0], dataset.adata.shape[1]
    )

    # Initialize task
    task = PerturbationExpressionPredictionTask()

    # Run task with model output
    model_results = task.run(model_output, task_input)

    # Compute and run baseline
    baseline_embedding = task.compute_baseline(dataset.adata.X, baseline_type="median")
    baseline_results = task.run(baseline_embedding, task_input)

    # Validate results structure
    for results in [model_results, baseline_results]:
        assert isinstance(results, list)
        assert len(results) > 0
        for result in results:
            assert hasattr(result, "metric_type")
            assert hasattr(result, "value")
            assert hasattr(result, "params")

    # Expect presence of required metric types in model results
    model_metric_types = {r.metric_type.value for r in model_results}
    for required_metric in {
        "accuracy_calculation",
        "precision_calculation",
        "recall_calculation",
        "f1_calculation",
        "spearman_correlation_calculation",
    }:
        assert required_metric in model_metric_types

    # Combine results for JSON validation
    model_serialized = [r.model_dump() for r in model_results]
    baseline_serialized = [r.model_dump() for r in baseline_results]
    all_results = {
        "perturbation": {
            "model": model_serialized,
            "baseline": baseline_serialized,
        }
    }

    # Validate combined structure
    assert "perturbation" in all_results
    assert "model" in all_results["perturbation"]
    assert "baseline" in all_results["perturbation"]
    assert isinstance(all_results["perturbation"]["model"], list)
    assert isinstance(all_results["perturbation"]["baseline"], list)
    assert len(all_results["perturbation"]["model"]) > 0
    assert len(all_results["perturbation"]["baseline"]) > 0

    # Verify each serialized result has expected keys/types
    for result in (
        all_results["perturbation"]["model"] + all_results["perturbation"]["baseline"]
    ):
        assert isinstance(result, dict)
        assert "metric_type" in result
        assert "value" in result
        assert "params" in result
        assert isinstance(result["params"], dict)

    # Verify JSON serialization and parsing
    json_output = json.dumps(all_results, indent=2, default=str)
    assert isinstance(json_output, str)
    parsed = json.loads(json_output)
    assert "perturbation" in parsed
    assert "model" in parsed["perturbation"]
    assert "baseline" in parsed["perturbation"]
