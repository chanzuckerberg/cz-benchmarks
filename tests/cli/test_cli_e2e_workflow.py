from czbenchmarks.tasks import (
    EmbeddingTask,
)
from czbenchmarks.metrics.types import MetricResult, MetricType
from czbenchmarks.cli.cli_run import run_with_inference, ModelArgs, TaskArgs

def test_cli_e2e_workflow():
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
    # Use SCPT model type
    model_args = [ModelArgs(name="SCGPT", args={})]
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
    assert len(task_results) == 1, "Expected results for embedding task"
    
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
