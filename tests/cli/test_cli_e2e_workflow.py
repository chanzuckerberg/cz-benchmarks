from czbenchmarks.tasks import (
    EmbeddingTask,
)
from czbenchmarks.metrics.types import MetricResult, MetricType
from czbenchmarks.cli.cli_run import (
    run_with_inference,
    CacheOptions,
    ModelArgs,
    TaskArgs,
)
from czbenchmarks.datasets.utils import load_dataset
from czbenchmarks.models.types import ModelType
from unittest.mock import patch, MagicMock
from tests.simple_models.simple_model import SimpleModel


@patch("czbenchmarks.runner.ContainerRunner")
def test_cli_e2e_workflow(mock_runner):
    """
    Test end-to-end workflow using CLI with model and dataset.

    This test verifies that the complete code path to run a benchmark works,
    focusing on a single dataset/model/task combination. Specifically using
    the CLI "entrypoint" method czbenchmarks.cli.cli_run.run_with_inference.

    Note: This test uses a simple model that generates embeddings on the dataset outputs
    and is not meant to verify model output correctness. Its purpose is to verify that
    the framework components work together correctly in an end-to-end workflow. Model
    output and metric correctness will be verified by separate model regression tests.
    """
    # region: Setup dataset, model, and task arguments
    dataset_name = "chicken_spermatogenesis"
    dataset = load_dataset(dataset_name)
    dataset.load_data()

    # Create and run simple model to produce the embedding output that will be returned by the mocked ContainerRunner.
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
        ModelArgs(name=model_name, args={"model_variant": ["human"]}),
    ]
    task_args = [
        TaskArgs(
            name=task_name,
            task=EmbeddingTask(label_key="cell_type"),
            set_baseline=False,
            baseline_args={},
        ),
    ]
    cache_options = CacheOptions(
        use_remote_cache=False,
        upload_strategy="never",
        remote_cache_url="",
        upload_results=False,
    )
    # endregion: Setup dataset, model, and task arguments

    # region: Run with inference
    # spermatogenesis datasets load quickly
    task_results = run_with_inference(
        dataset_names=[dataset_name],
        model_args=model_args,
        task_args=task_args,
        cache_options=cache_options,
    )
    # endregion: Run with inference

    # region: Verify task results
    # Verify we got results for the task
    assert len(task_results) == 1, "Expected results for embedding task"

    # Verify task result
    task_result = task_results[0]

    # Verify basic task result fields
    assert task_result.task_name == task_name
    assert task_result.task_name_display == "embedding"
    assert task_result.model_type == model_type
    assert task_result.dataset_names == [dataset_name]
    assert task_result.dataset_names_display == ["Spermatogenesis - Gallus gallus"]
    assert task_result.model_args == {"model_variant": "human"}
    assert task_result.model_name_display == "scGPT - whole-human"
    assert task_result.runtime_metrics == {}, "Expected no runtime metrics"

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
