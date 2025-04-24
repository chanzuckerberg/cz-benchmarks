from czbenchmarks.tasks import (
    EmbeddingTask,
)
from pytest_mock import MockFixture
from unittest.mock import MagicMock
from czbenchmarks.metrics.types import MetricResult, MetricType
from czbenchmarks.cli.cli_run import run_with_inference, ModelArgs, TaskArgs
from czbenchmarks.datasets import utils as dataset_utils
from czbenchmarks import runner


def test_cli_e2e_workflow(mocker: MockFixture):
    """
    Test end-to-end workflow using CLI with model and dataset.

    This test verifies that the complete code path to run a benchmark works,
    focusing on a single dataset/model/task combination. Specifically using
    the CLI "entrypoint" method czbenchmarks.cli.cli_run.run_with_inference

    It is not intended to verify that a model's output (embeddings)
    or that a task's metrics are being computed correctly. That will be
    verified by separate model regression tests.
    """
    # region: Setup model and task arguments
    mock_processed_data = MagicMock()
    mocker.patch.object(dataset_utils, "load_dataset", return_value=mock_processed_data)
    mocker.patch.object(runner, "run_inference", return_value=mock_processed_data)
    mock_task_results = [MagicMock()]
    mock_task_results[0].task_name = "embedding"
    mock_task_results[0].model_type = "SCGPT"
    mock_task_results[0].dataset_name = "chicken_spermatogenesis"
    mock_task_results[0].model_args = {}
    mock_task_results[0].metrics = [
        MetricResult(metric_type=MetricType.SILHOUETTE_SCORE, value=0.5, params={})
    ]
    mocker.patch("czbenchmarks.cli.cli_run.run_task", return_value=mock_task_results)
    model_args = [
        ModelArgs(name="SCGPT", args={}),
        ModelArgs(
            name="SCVI",
            args={"model_variant": ["homo_sapiens", "mus_musculus"]},
        ),
    ]
    task_args = [
        TaskArgs(
            name="embedding",
            task=EmbeddingTask(label_key="cell_type"),
            set_baseline=False,
        ),
    ]
    # endregion: Setup model and task arguments

    # region: Run inference and tasks
    # spermatogenesis datasets load quickly
    dataset_name = "chicken_spermatogenesis"
    task_results = run_with_inference(
        dataset_names=[dataset_name],
        model_args=model_args,
        task_args=task_args,
    )
    # endregion: Run inference and tasks

    # region: Verify results
    # Verify we got results for the task
    assert len(task_results) == 3, "Expected results for embedding task"

    # Verify task result
    task_result = task_results[0]
    print(f"task_result: {task_result}")

    # Verify basic task result fields
    assert task_result.task_name == "embedding"
    assert task_result.model_type == "SCGPT"
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
    # endregion: Verify results
