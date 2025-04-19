# Add a New Task

This guide explains how to create and integrate a new evaluation task into CZ Benchmarks.

## Steps to Add a New Task

### 1. Create a New Task Class
- Extend the `BaseTask` class to define your custom task.
- Create a new Python file in the `czbenchmarks/tasks/` directory (or in a subdirectory if the task is modality-specific).

#### Example:
```python
from typing import Set
from czbenchmarks.tasks.base import BaseTask
from czbenchmarks.datasets import DataType
from czbenchmarks.metrics.types import MetricResult
from czbenchmarks.models.types import ModelType

class MyTask(BaseTask):
    def __init__(self, my_param: int = 10):
        self.my_param = my_param

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

    def _compute_metrics(self) -> [MetricResult]:
        # Compute and return a list of MetricResult objects
        metric_value = ...  # Replace with actual computation
        return [MetricResult(metric_type="my_metric", value=metric_value)]
```

### 2. Understand Required Inputs and Outputs
- Use the `required_inputs` and `required_outputs` properties to specify the data types your task needs.
- Example:
  - `required_inputs`: Data your task consumes (e.g., `DataType.METADATA`).
  - `required_outputs`: Data your task produces or depends on (e.g., `DataType.EMBEDDING`).

### 3. Implement Task Logic
- Use `data.get_input()` and `data.get_output()` to retrieve the necessary inputs and outputs.
- Implement the `_run_task` method to define the core logic of your task.
- Use the `_compute_metrics` method to calculate and return metrics as a list of `MetricResult` objects.

### 4. Reference Base Files
- Consult `czbenchmarks/tasks/base.py` for interface details and best practices.
- Ensure your implementation adheres to the expected structure and conventions.

## Best Practices
- **Single Responsibility:** Ensure your task has a clear and focused purpose.
- **Error Handling:** Include robust error handling and logging.
- **Code Quality:** Use type hints and follow the project's coding style.
- **Testing:** Write unit tests to validate your task's functionality.
