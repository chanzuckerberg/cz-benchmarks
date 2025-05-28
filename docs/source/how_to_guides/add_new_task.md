# Add a Custom Task

This guide explains how to create and integrate your own evaluation task into cz-benchmarks.

## Steps to Add a New Task

### 1. Create a New Task Class
- Extend the `BaseTask` class to define your custom task.
- Create a new Python file in the `czbenchmarks/tasks/` directory (or in a subdirectory if the task is modality-specific).

    Example:

    ```python
    from typing import List, Set
    from czbenchmarks.tasks.base import BaseTask
    from czbenchmarks.datasets import DataType
    from czbenchmarks.metrics.types import MetricResult
    from czbenchmarks.models.types import ModelType

    class MyTask(BaseTask):
        def __init__(self, my_param: int = 10, *, random_seed: int = 123):
            super().__init__(random_seed=random_seed)
            self.my_param = my_param

        @property
        def display_name(self) -> str:
            return "My Example Task"

        @property
        def required_inputs(self) -> Set[DataType]:
            return {DataType.METADATA}

        @property
        def required_outputs(self) -> Set[DataType]:
            return {DataType.EMBEDDING}

        def _run_task(self, data, model_type: ModelType):
            # Retrieve necessary data
            self.embedding = data.get_output(model_type, DataType.EMBEDDING)
            self.labels = data.get_input(DataType.METADATA)["cell_type"]
            # Implement your task logic here
            # e.g., calculate some predictions or transform embeddings

        def _compute_metrics(self) -> List[MetricResult]:
            # Compute and return a list of MetricResult objects
            metric_value = ...  # Replace with actual computation
            return [MetricResult(metric_type="my_metric", value=metric_value)]
    ```

- Consult `czbenchmarks/tasks/base.py` for interface details and best practices.
- Use `display_name` to provide a human-readable name for use when displaying your task
- The `random_seed` should be used for all sources of randomness so results are reproducible
- Use the `required_inputs` and `required_outputs` properties to specify the data types your task needs.
    - `required_inputs`: Data your task consumes (e.g., `DataType.METADATA`).
    - `required_outputs`: Data your task produces or depends on (e.g., `DataType.EMBEDDING`).
- Use `data.get_input()` and `data.get_output()` to retrieve the necessary inputs and outputs.
- Implement the `_run_task` method to define the core logic of your task.
- Use the `_compute_metrics` method to calculate and return metrics as a list of `MetricResult` objects. If your task requires a `Metric` that is not already supported by `cz-benchmarks`, refer to [Adding a New Metric](../how_to_guides/add_new_metric.md).

### 5. Register the Task
- In the `src.czbenchmarks.tasks.utils.TASK_NAMES` array, add a new value for your task.

## Best Practices
- **Single Responsibility:** Ensure your task has a clear and focused purpose.
- **Error Handling:** Include robust error handling and logging.
- **Code Quality:** Use type hints and follow the project's coding style.
- **Testing:** Write unit tests to validate your task's functionality.
