from czbenchmarks.tasks import (
    EmbeddingTask,
)
from czbenchmarks.metrics.types import MetricResult, MetricType
from czbenchmarks.cli.cli_run import run_with_inference, ModelArgs, TaskArgs
from czbenchmarks.datasets.utils import load_dataset
from czbenchmarks.models.types import ModelType
from czbenchmarks.datasets.types import DataType
import numpy as np
from unittest.mock import patch, MagicMock


class SimpleModel:
    """A simple model that generates random embeddings.

    This is a dummy implementation that doesn't do any real model inference.
    It just generates random embeddings of the correct shape for testing purposes.
    """

    def __init__(self):
        self.model_type = ModelType.SCGPT

    def run_inference(self, dataset):
        """Generate random embeddings for the dataset.

        Args:
            dataset: The dataset to generate embeddings for

        Returns:
            The dataset with random embeddings added to its outputs
        """
        mock_processed_data = dataset
        n_cells = dataset.adata.n_obs
        dummy_embeddings = np.random.normal(size=(n_cells, 100))
        model_type = self.model_type
        mock_processed_data.outputs[model_type] = {DataType.EMBEDDING: dummy_embeddings}
        return mock_processed_data


@patch("czbenchmarks.runner.ContainerRunner")
def test_cli_e2e_workflow(mock_runner):
    """
    Test end-to-end workflow using CLI with model and dataset.

    This test verifies that the complete code path to run a benchmark works,
    focusing on a single dataset/model/task combination. Specifically using
    the CLI "entrypoint" method czbenchmarks.cli.cli_run.run_with_inference.

    Note: This test uses random embeddings and is not meant to verify model output
    correctness. Its purpose is to verify that the framework components work together
    correctly in an end-to-end workflow. Model output and metric correctness will be
    verified by separate model regression tests.
    """
    # region: Setup dataset, model, and task arguments
    dataset_name = "chicken_spermatogenesis"
    dataset = load_dataset(dataset_name)
    dataset.load_data()

    # Create and run simple model
    model = SimpleModel()
    dataset = model.run_inference(dataset)

    # Mock the ContainerRunner instance to avoid a nvidia runtime error at run_with_inference
    mock_runner_instance = MagicMock()
    mock_runner_instance.run.return_value = dataset
    mock_runner.return_value = mock_runner_instance

    task_name = "embedding"
    model_name = "SCGPT"
    model_type = ModelType.SCGPT
    model_args = [
        ModelArgs(name=model_name, args={}),
    ]
    task_args = [
        TaskArgs(
            name=task_name,
            task=EmbeddingTask(label_key="cell_type"),
            set_baseline=False,
        ),
    ]
    # endregion: Setup dataset, model, and task arguments

    # region: Run with inference
    # spermatogenesis datasets load quickly
    task_results = run_with_inference(
        dataset_names=[dataset_name],
        model_args=model_args,
        task_args=task_args,
    )
    # endregion: Run with inference

    # region: Verify task results
    # Verify we got results for the task
    assert len(task_results) == 1, "Expected results for embedding task"

    # Verify task result
    task_result = task_results[0]

    # Verify basic task result fields
    assert task_result.task_name == task_name
    assert task_result.model_type == model_type
    assert task_result.dataset_name == dataset_name
    assert task_result.model_args == {}, "Expected empty model args"

    # Verify metrics
    assert isinstance(task_result.metrics, list)
    assert len(task_result.metrics) > 0
    assert all(isinstance(r, MetricResult) for r in task_result.metrics)

    # Verify specific metric values
    metric = task_result.metrics[0]
    assert metric.metric_type == MetricType.SILHOUETTE_SCORE
    assert isinstance(metric.value, float)
    assert metric.params == {}, "Expected empty metric params"
    # endregion: Verify task results
