# Add a Custom Task

This guide explains how to create and integrate your own evaluation task into cz-benchmarks.


## Overview of Task System

### Automatic Task Registration and Discovery

- All `Task` subclasses are automatically registered in a global registry (`TASK_REGISTRY`) when defined. You do **not** need to manually register your task for it to be discoverable by the framework.
- The registry supports listing all available tasks, and introspecting their parameters.

### Task Metadata: `display_name`, `description`, and `input_model`

- Each task **must** define a `display_name` (human-readable name) and should provide a `description` (short summary of the task's purpose).
- Each task **must** set an `input_model` class attribute pointing to its Pydantic input model. This enables automatic parameter introspection and validation.

### Parameter Introspection and Validation

- Task parameters are automatically discovered from the fields of the `input_model` (a Pydantic model). Types, defaults, and required/optional status are extracted and validated at runtime.
- Baseline parameters (if supported) are also introspected from the `compute_baseline` method signature.
- Users can programmatically or via CLI list available tasks and their parameters, including help text and defaults.

### Arbitrary Types in Input/Output

- The Pydantic `TaskInput` and `TaskOutput` models support arbitrary types (e.g., DataFrames, arrays) via `model_config = {"arbitrary_types_allowed": True}`. You can use complex data structures as needed for your task.

### Multi-Dataset Tasks

- If your task requires multiple datasets (e.g., for integration or cross-species tasks), set `self.requires_multiple_datasets = True` in your task's `__init__` method. The framework will enforce correct input types at runtime.

### Baseline Computation

- Tasks may optionally implement a `compute_baseline` method to provide a baseline embedding or prediction for comparison. If not applicable, this method can raise `NotImplementedError`.
- Baseline parameters (if any) should be documented in the method signature.

### Task Help and Discovery

- The registry provides methods to list all available tasks, introspect their parameters, and generate help text for each task.


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

### 4. Update the Documentation

- **After adding your new Task, update the documentation:**
    - Edit `docs/source/developer_guides/tasks.md` and add your new Task to the **Available Tasks** section, with a short description and a link to its API docs if available.


## Task Execution Flow

The `run_task()` function will automatically:
- Look up your task in the registry
- Instantiate it with the provided random seed
- Create the appropriate `TaskInput` from your `task_params`
- Execute the task and compute metrics
- Return serialized results as dictionaries


## Best Practices

- Ensure each task has a clear and focused purpose.
- Document the input and output requirements for your task.
- Follow established patterns and conventions from existing tasks for consistency.
- Use type hints to improve code readability and maintainability.
- Add logging for key steps in the task lifecycle to aid debugging and monitoring.
- Include robust error handling to make your task resilient.
- Write unit tests to validate your task's functionality.

