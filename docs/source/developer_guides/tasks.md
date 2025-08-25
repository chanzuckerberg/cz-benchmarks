# Tasks 

The `czbenchmarks.tasks` module defines **benchmarking tasks** that evaluate the performance of models based on their outputs. Tasks take a model's output (e.g., a cell embedding) and task-specific inputs (e.g., ground-truth labels from a dataset) and compute relevant evaluation metrics.

## Core Concepts

- **[`Task`](../autoapi/czbenchmarks/tasks/task/index)** The abstract base class for all tasks. It defines the standard lifecycle:
    
    1. **Execution** via the `_run_task()` method, which performs the core computation.
    2. **Metric Computation** via the `_compute_metrics()` method.
        
    It also supports multi-dataset operations (`requires_multiple_datasets`) and setting **baseline embeddings** for comparison with model outputs.
    
- **[`TaskInput`](../autoapi/czbenchmarks/tasks/task/index)** and **[`TaskOutput`](../autoapi/czbenchmarks/tasks/task/index)** Pydantic base classes used to define structured inputs and outputs for each task, ensuring type safety and clarity. Each task defines its own subclasses of `TaskInput` and `TaskOutput`.

- **[`run_task()`](../autoapi/czbenchmarks/tasks/runner/index)** The primary programmatic API endpoint for executing benchmark tasks. This function provides a simplified interface for running any registered task with standardized parameters, automatic baseline computation, and serialized results.
    

## Task Organization

Tasks in the `czbenchmarks.tasks` module are organized based on their scope and applicability:

- **Generic Tasks**: Tasks that can be applied across multiple modalities (e.g., embedding evaluation, clustering, label prediction) are placed directly in the `tasks/` directory. Each task is implemented in its own file (e.g., `embedding.py`, `clustering.py`).
    
- **Specialized Tasks**: Tasks designed for specific modalities are placed in dedicated subdirectories. For example:
    
    - `single_cell/` for single-cell-specific tasks like perturbation prediction or cross-species integration.
        
    New subdirectories can be created as needed for other modalities.
    

## Available Tasks

Each task class implements a specific evaluation goal. All tasks are located under the `czbenchmarks.tasks` namespace or its submodules.

- [`EmbeddingTask`](../autoapi/czbenchmarks/tasks/embedding/index): Computes embedding quality using the Silhouette Score based on known cell-type annotations.
    
- [`ClusteringTask`](../autoapi/czbenchmarks/tasks/clustering/index): Performs Leiden clustering on an embedding and compares it to ground-truth labels using metrics like Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI).
    
- [`MetadataLabelPredictionTask`](../autoapi/czbenchmarks/tasks/label_prediction/index): Performs k-fold cross-validation using multiple classifiers (logistic regression, KNN, random forest) on model embeddings to predict metadata labels. Evaluates metrics like accuracy, F1, precision, recall, and AUROC.
    
- [`BatchIntegrationTask`](../autoapi/czbenchmarks/tasks/integration/index): Evaluates how well a model integrates data from different batches using entropy per cell and batch-aware Silhouette scores.
    
- [`CrossSpeciesIntegrationTask`](../autoapi/czbenchmarks/tasks/single_cell/cross_species/index): A multi-dataset task that evaluates how well models embed cells from different species into a shared space, using metrics like entropy per cell and species-aware silhouette scores.
    
- [`PerturbationTask`](../autoapi/czbenchmarks/tasks/single_cell/perturbation/index): Designed for gene perturbation models. Compares predicted gene expression shifts to ground truth using metrics like mean squared error, Pearson correlation, and Jaccard similarity for DE genes.
    

## Programmatic API

The [`run_task()`](../autoapi/czbenchmarks/tasks/runner/index) function provides a programmatic interface for executing tasks without needing to instantiate task classes manually. This is the recommended approach for external library usage and automation.

### Basic Usage

```python
from czbenchmarks.tasks.runner import run_task
import numpy as np
import pandas as pd

# Prepare input data for Task (random data generated here as an example)
embedding = np.random.randn(100, 50)  # 100 cells, 50 dimensions
labels = np.random.choice(['TypeA', 'TypeB', 'TypeC'], size=100)
obs_data = pd.DataFrame({'cell_type': labels})

# Run a clustering task
results = run_task(
    task_name='clustering',
    cell_representation=embedding,
    task_params={
        'input_labels': labels,
        'obs': obs_data,
    },
    random_seed=42
)

# Results are returned as a list of dictionaries
for result in results:
    print(f"Metric: {result['metric_type']}, Value: {result['value']}")
```

### With Baseline Computation

```python
# Run with baseline computation (uses raw expression data)
raw_expression = np.random.poisson(lam=5, size=(100, 1000))  # Raw counts

results = run_task(
    task_name='clustering',
    cell_representation=raw_expression,
    run_baseline=True,
    baseline_params={'n_pcs': 50},
    task_params={
        'input_labels': labels,
        'obs': obs_data,
    },
    random_seed=42
)
```

### Parameters

- `task_name`: The normalized name of the task (e.g., 'clustering', 'embedding', 'batch_integration', 'label_prediction')
- `cell_representation`: Model embedding or raw expression data (numpy array or list for multi-dataset tasks)
- `run_baseline`: Whether to compute baseline embedding from raw data
- `baseline_params`: Parameters for the `compute_baseline()` method
- `task_params`: Parameters for the task's `TaskInput` model
- `random_seed`: Random seed for reproducibility

### Error Handling

The function includes comprehensive error handling:
- Invalid task names raise `ValueError`
- Invalid task parameters raise `ValueError` with detailed messages
- Baseline computation errors are logged and re-raised (except `NotImplementedError`)

## Extending Tasks

To define a new evaluation task:

1. **Define Input/Output Models**: Create subclasses of `TaskInput` and `TaskOutput` to structure the data your task will use.
    
2. **Inherit from [`Task`](../autoapi/czbenchmarks/tasks/task/index)**: Create your task class.
    
3. **Choose the Right Location**:
    
    - If the task is generic, add it to the `tasks/` directory.
    - If it's modality-specific, add it to a subdirectory like `single_cell/`.
        
4. **Implement Required Methods**:
    
    - `_run_task(self, cell_representation, task_input)`: Executes the task's logic using the model output (`cell_representation`) and structured `task_input`. This method should return an instance of your custom `TaskOutput` model.
    - `_compute_metrics(self, task_input, task_output)`: Returns a list of `MetricResult` objects.
        
5. **Update `__init__.py`**:
    
    - For generic tasks, add the new task to `tasks/__init__.py`.
    - For specialized tasks, add it to the `__init__.py` in the corresponding subdirectory.
        
6. **Optional Features**:
    
    - Set `requires_multiple_datasets = True` if your task operates on a list of `CellRepresentation` inputs.
    - Implement `compute_baseline(self, ...)` to define a simple baseline for comparison (e.g., PCA or using raw data).
    - If your task is stochastic, use `self.random_seed` to seed sources of randomness for reproducibility.
        
7. **Example Skeleton**:
    
    ```python
    from czbenchmarks.tasks import Task, TaskInput, TaskOutput
    from czbenchmarks.tasks.runner import run_task  # For programmatic execution
    from czbenchmarks.metrics.types import MetricResult, MetricType
    from czbenchmarks.metrics import metrics_registry
    from czbenchmarks.tasks.types import CellRepresentation
    from czbenchmarks.types import ListLike
    
    class MyTaskInput(TaskInput):
        labels: ListLike
    
    class MyTaskOutput(TaskOutput):
        processed_data: list
    
    class MyNewTask(Task):
        display_name = "My New Task"
    
        def _run_task(
            self,
            cell_representation: CellRepresentation,
            task_input: MyTaskInput
        ) -> MyTaskOutput:
            # Your logic here
            processed_data = [i * 2 for i in range(cell_representation.shape[0])]
            return MyTaskOutput(processed_data=processed_data)
    
        def _compute_metrics(
            self,
            task_input: MyTaskInput,
            task_output: MyTaskOutput
        ) -> list[MetricResult]:
            # Dummy metric computation
            metric_val = len(task_input.labels) + len(task_output.processed_data)
            return [MetricResult(metric_type=MetricType.ADJUSTED_RAND_INDEX, value=float(metric_val))]
    ```
    

## Best Practices  

- Keep tasks focused and single-purpose to ensure clarity and maintainability.
- Clearly document the input and output requirements for each task.
- Follow the patterns and conventions established in existing tasks for consistency.
- Use type hints to improve code readability and clarity.
- Add logging for key steps in the task lifecycle to facilitate debugging and monitoring.
- Test your tasks using both the direct class interface and the `run_task()` programmatic API to ensure compatibility.
