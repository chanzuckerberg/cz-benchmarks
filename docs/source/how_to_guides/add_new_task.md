# Add a Custom Task

This guide explains how to create and integrate your own evaluation task into cz-benchmarks.

## Steps to Add a New Task

### 1. Define Task-Specific Input and Output Classes

- Create Pydantic models that inherit from `TaskInput` and `TaskOutput` to define the data your task needs and produces.
    

### 2. Create a New Task Class

- Extend the `Task` class to define your custom task.
- Create a new Python file in the `czbenchmarks/tasks/` directory.
    
    Example:
    
    ```python
    from typing import List
    from czbenchmarks.tasks import Task, TaskInput, TaskOutput
    from czbenchmarks.metrics.types import MetricResult, MetricType
    from czbenchmarks.metrics import metrics_registry
    from czbenchmarks.types import ListLike
    from czbenchmarks.tasks.types import CellRepresentation
    
    # 1. Define Input and Output data structures
    class MyTaskInput(TaskInput):
        ground_truth_labels: ListLike
    
    class MyTaskOutput(TaskOutput):
        predictions: List[float]
    
    # 2. Implement the Task class
    class MyTask(Task):
        display_name = "My Example Task"
    
        def __init__(self, my_param: int = 10, *, random_seed: int = 123):
            super().__init__(random_seed=random_seed)
            self.my_param = my_param
    
        def _run_task(
            self,
            cell_representation: CellRepresentation,
            task_input: MyTaskInput
        ) -> MyTaskOutput:
            # Implement your task logic here.
            # For example, use the cell_representation to make predictions.
            # This is a placeholder for your actual logic.
            predictions = list(range(len(task_input.ground_truth_labels)))
    
            # Return the results in your output model
            return MyTaskOutput(predictions=predictions)
    
        def _compute_metrics(
            self,
            task_input: MyTaskInput,
            task_output: MyTaskOutput
        ) -> List[MetricResult]:
            # Compute and return a list of MetricResult objects
            # using the metric registry.
            ari_score = metrics_registry.compute(
                MetricType.ADJUSTED_RAND_INDEX,
                labels_true=task_input.ground_truth_labels,
                labels_pred=task_output.predictions
            )
            return [
                MetricResult(
                    metric_type=MetricType.ADJUSTED_RAND_INDEX,
                    value=ari_score
                )
            ]
    
    ```
    
- Consult `czbenchmarks/tasks/task.py` for interface details.
- Use `display_name` to provide a human-readable name for your task.
- The `random_seed` should be used for all sources of randomness so results are reproducible.
- Implement the `_run_task` method to define the core logic of your task. It takes a `cell_representation` (e.g., a model's embedding) and your custom `TaskInput` model. It should return an instance of your custom `TaskOutput` model.
- Implement the `_compute_metrics` method to calculate and return metrics as a list of `MetricResult` objects. If your task requires a metric that is not already supported, refer to [Adding a New Metric](./add_new_metric.md).
    

### 3. Register the Task

- Import and add your new task classes to the `__all__` list in `czbenchmarks/tasks/__init__.py` to make them easily accessible.
    
    ```python
    # In czbenchmarks/tasks/__init__.py
    ...
    from .my_task_file import MyTask, MyTaskInput, MyTaskOutput
    
    __all__ = [
        ...
        "MyTask",
        "MyTaskInput",
        "MyTaskOutput",
    ]
    
    ```

## Best Practices
- **Single Responsibility:** Ensure your task has a clear and focused purpose.
- **Error Handling:** Include robust error handling and logging.
- **Code Quality:** Use type hints and follow the project's coding style.
- **Testing:** Write unit tests to validate your task's functionality.
