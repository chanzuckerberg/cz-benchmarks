# Add a Custom Task

This guide explains how to create and integrate your own evaluation task into cz-benchmarks.

## Overview of the Task System

### Required Components

| Component | Description | Example |
|-----------|-------------|---------|
| **`display_name`** | Human-readable name for the task | `"Cell Clustering"` |
| **`input_model`** | Pydantic class defining task input schema | `ClusteringTaskInput` |
| **`_run_task()`** | Core task implementation method | Returns `TaskOutput` instance |
| **`_compute_metrics()`** | Metric computation method | Returns `List[MetricResult]` |

### Optional Components

| Component | Description | Default |
|-----------|-------------|---------|
| **`description`** | One-sentence task description | Extracted from docstring |
| **`compute_baseline()`** | Baseline implementation for comparison | Raises `NotImplementedError` |
| **`requires_multiple_datasets`** | Flag for multi-dataset tasks | `False` |

### Key Features

- **Automatic Registration**: Tasks are automatically registered by the Task base class (in  `__init_subclass__`)
- **Arbitrary Types**: Input/Output models support complex objects (DataFrames, arrays) via `model_config = {"arbitrary_types_allowed": True}`
- **Type Safety**: Full Pydantic validation for all inputs

## Step-by-Step Implementation

### 1. Define Input and Output Models

Create Pydantic models that inherit from `TaskInput` and `TaskOutput`:

```python
from czbenchmarks.tasks import TaskInput, TaskOutput
from czbenchmarks.types import ListLike
from typing import List
import pandas as pd

class MyTaskInput(TaskInput):
    """Input model for MyTask."""
    ground_truth_labels: ListLike
    metadata: pd.DataFrame  # Example of arbitrary type support

class MyTaskOutput(TaskOutput):
    """Output model for MyTask."""
    predictions: List[float]
    confidence_scores: List[float]
```

### 2. Implement the Task Class

Create a new file in `src/czbenchmarks/tasks/` (e.g., `my_task.py`):

```python
import logging
from typing import List
import numpy as np

from ..constants import RANDOM_SEED
from ..metrics.types import MetricResult, MetricType
from ..metrics import metrics_registry
from .task import Task, TaskInput, TaskOutput
from .types import CellRepresentation

logger = logging.getLogger(__name__)

class MyTaskInput(TaskInput):
    """Input model for MyTask."""
    ground_truth_labels: ListLike

class MyTaskOutput(TaskOutput):
    """Output model for MyTask."""
    predictions: List[float]

class MyTask(Task):
    """Example task that demonstrates the basic task implementation pattern.
    
    This task performs a simple prediction based on cell embeddings
    and evaluates against ground truth labels.
    """
    
    # REQUIRED: Class attributes for task metadata
    display_name = "My Example Task"
    description = "Predicts numeric labels from cell embeddings using a simple algorithm."
    input_model = MyTaskInput
    
    def __init__(self, my_param: int = 10, *, random_seed: int = RANDOM_SEED):
        """Initialize the task with custom parameters.
        
        Args:
            my_param: Custom parameter for the task
            random_seed: Random seed for reproducibility
        """
        super().__init__(random_seed=random_seed)
        self.my_param = my_param
        logger.info(f"Initialized {self.display_name} with my_param={my_param}")

    def _run_task(
        self,
        cell_representation: CellRepresentation,
        task_input: MyTaskInput,
    ) -> MyTaskOutput:
        """Core task implementation.
        
        Args:
            cell_representation: Cell embeddings or gene expression data
            task_input: Validated input parameters
            
        Returns:
            Task output containing predictions
        """
        logger.info(f"Running task on {len(task_input.ground_truth_labels)} samples")
        
        # Example implementation - replace with your logic
        np.random.seed(self.random_seed)
        predictions = np.random.random(len(task_input.ground_truth_labels)).tolist()
        
        return MyTaskOutput(predictions=predictions)

    def _compute_metrics(
        self,
        task_input: MyTaskInput,
        task_output: MyTaskOutput,
    ) -> List[MetricResult]:
        """Compute evaluation metrics.
        
        Args:
            task_input: Original task input
            task_output: Results from _run_task
            
        Returns:
            List of metric results
        """
        # Use metrics registry to compute standard metrics
        metrics = []
        
        # Example: Compute correlation if applicable
        if len(task_input.ground_truth_labels) == len(task_output.predictions):
            correlation = metrics_registry.compute(
                MetricType.PEARSON_CORRELATION,
                y_true=task_input.ground_truth_labels,
                y_pred=task_output.predictions,
            )
            metrics.append(
                MetricResult(
                    metric_type=MetricType.PEARSON_CORRELATION,
                    value=correlation,
                    params={"my_param": self.my_param},
                )
            )
        
        return metrics

    def compute_baseline(
        self,
        expression_data: CellRepresentation,
        **kwargs,
    ) -> CellRepresentation:
        """Optional: Compute baseline embedding using standard preprocessing.
        
        Args:
            expression_data: Raw gene expression data
            **kwargs: Additional parameters for baseline computation
            
        Returns:
            Baseline embedding for comparison
        """
        # Use the parent class implementation for PCA baseline
        return super().compute_baseline(expression_data, **kwargs)
```

### 3. Register the Task

Add your task to `src/czbenchmarks/tasks/__init__.py`:

```python
# Add these imports
from .my_task import MyTask, MyTaskInput, MyTaskOutput

# Add to __all__ list
__all__ = [
    # ... existing exports ...
    "MyTask",
    "MyTaskInput", 
    "MyTaskOutput",
]
```

**Note**: Registration happens automatically when the class is defined thanks to `__init_subclass__`. Adding to `__init__.py` makes it easily importable.

### 4. Test Your Task

```python
# Test script example
from czbenchmarks.tasks import MyTask, MyTaskInput
import numpy as np

# Create test data
cell_rep = np.random.random((100, 50))  # 100 cells, 50 features
task_input = MyTaskInput(ground_truth_labels=np.random.randint(0, 3, 100))

# Run task
task = MyTask(my_param=5, random_seed=42)
results = task.run(cell_rep, task_input)
print(f"Computed {len(results)} metrics")
```

### 5. Update Documentation

Add your task to `docs/source/developer_guides/tasks.md`:

```markdown
### Available Tasks

- **My Example Task** – Predicts numeric labels from cell embeddings using a simple algorithm.
  See: `czbenchmarks.tasks.my_task.MyTask`
```

## Advanced Features

### Multi-Dataset Tasks

For tasks requiring multiple datasets (e.g., integration tasks):

```python
def __init__(self, *, random_seed: int = RANDOM_SEED):
    super().__init__(random_seed=random_seed)
    self.requires_multiple_datasets = True  # Enable multi-dataset mode
```

### Custom Baseline Parameters

Document baseline parameters in the method signature:

```python
def compute_baseline(
    self,
    expression_data: CellRepresentation,
    n_components: int = 50,
    highly_variable_genes: bool = True,
    **kwargs,
) -> CellRepresentation:
    """Compute PCA baseline with custom parameters."""
    return super().compute_baseline(
        expression_data, 
        n_components=n_components,
        highly_variable_genes=highly_variable_genes,
        **kwargs
    )
```

## Task Discovery and CLI Integration

Tasks are automatically discovered via the `TASK_REGISTRY`:

```python
from czbenchmarks.tasks import TASK_REGISTRY

# List all available tasks
print(TASK_REGISTRY.list_tasks())

# Get task information  
info = TASK_REGISTRY.get_task_info("my_example_task")
print(f"Description: {info.description}")
print(f"Parameters: {list(info.task_params.keys())}")
```

## Tips

- ✅ **Single Responsibility**: Each task should solve one well-defined problem
- ✅ **Reproducibility**: Pass `self.random_seed` to any library function calls that have stochastic behavior
- ✅ **Type Safety**: Use explicit type hints throughout
- ✅ **Logging**: Log key steps for debugging (`logger.info`, `logger.debug`)
- ✅ **Error Handling**: Provide informative error messages
- ✅ **Documentation**: Clear docstrings for all public methods
- ✅ **Testing**: Unit tests for input validation, core logic, and metrics
- ✅ **Performance**: Consider memory usage for large datasets

