# czibench

## Installation Instructions

### From PyPI

```bash
pip install czibench
```

### From Source

```bash
git clone https://github.com/chanzuckerberg/czibench.git
cd czibench
pip install -e .
```

## Requirements

TBD

## Architecture Overview

This is a benchmarking framework for machine learning models (with a focus on single-cell data), built around Docker containers and a modular architecture. Here are the key components:

### Core components

#### Base Classes

- `czibench.datasets.base.BaseDataset`: Abstract base class for all datasets
- `czibench.models.base.BaseModel`: Abstract base class for all models
- `czibench.metrics.base.BaseMetric`: Abstract base class for metrics
- `czibench.tasks.base.BaseTask`: Abstract base class for evaluation tasks
- `czibench.runner.base.BaseModelRunner`: Handles Docker container I/O and execution

#### Container system

- Uses Docker for model execution
- Base image (`czibench-base`) provides common infrastructure
- Models are packaged as Docker containers
- Standard I/O paths for data exchange:
  - Input: `/app/input/data.dill`
  - Output: `/app/output/data.dill`

## Contributing New Components

### Adding a new model

1. Create a new directory under `docker/your_model/` with:

```markdown
docker/your_model/
├── Dockerfile
├── model.py
├── config.yaml
└── requirements.txt
```

The Dockerfile must be built from `czibench-base` and add `model.py` to the image.

```docker
FROM ghcr.io/chanzuckerberg/czibench-base:latest
COPY docker/your_model/model.py .
```

2. Implement the model container in `model.py` (see template):

```python
"""
REQUIRED MODEL IMPLEMENTATION FILE: model.py

This file MUST:
1. Define a ModelRunner class that inherits from BaseModelRunner
2. Implement the run_model() method
3. Specify the model_class class variable

Example:
"""

from czibench.models.base import BaseModelRunner, BaseModel

class ModelRunner(BaseModelRunner):
    """
    Required container class for model implementation.
    This class will be imported and run by the base image's entrypoint.py
    """
    
    # Required: Specify which model class this container uses
    model_class = BaseModel  # Replace with your model class
    
    def run_model(self):
        """
        Required: Implement your model's inference logic here.
        self.data will contain the input dataset.
        """
        raise NotImplementedError("Model implementation required")
```

3. Implement the model class:

```python
class SCVI(SingleCellModel):
    available_organisms = [Organism.HUMAN, Organism.MOUSE]
    required_batch_keys = ['dataset_id', 'assay', 'suspension_type', 'donor_id']

    def _validate_model_requirements(self, dataset: SingleCellDataset) -> bool:
        # Check if all required batch keys are present in obs
        missing_keys = [key for key in self.required_batch_keys 
                       if key not in dataset.adata.obs.columns]
        
        if missing_keys:
            logger.error(f"Missing required batch keys: {missing_keys}")
            return False
            
        return True
```

### Adding new metrics

1. Create a new metric class inheriting from `czibench.metrics.base.BaseMetric`:

```python
from czibench.metrics.base import BaseMetric

class MyNewMetric(BaseMetric):
    def validate(self, data: BaseDataset) -> bool:
        # Validate required data components
        pass
        
    def _compute_metric(self, data: BaseDataset) -> float:
        # Implement metric computation
        pass
```


### Adding New Tasks

1. Create a task class inheriting from `czibench.tasks.base.BaseTask`:

```python
from czibench.tasks.base import BaseTask

class MyNewTask(BaseTask):
    available_metrics = [MyMetric1, MyMetric2]
    
    def validate(self, data: BaseDataset) -> bool:
        # Validate task requirements
        pass
        
    def _run_task(self, data: BaseDataset) -> BaseDataset:
        # Implement task logic
        pass
```

## Running Models

Using the `ContainerRunner`:

```python
from czibench.runner.container import ContainerRunner
from czibench.datasets.load import load_dataset

# Load dataset
dataset = load_dataset("mouse_brain_atlas")

# Run model
runner = ContainerRunner(
    image="your-model-image:latest",
    gpu=True,  # Optional GPU support
    memory="8g"  # Optional memory limit
)
result = runner.run(dataset)
```

## Using Custom Data

To use your own data:

1. Prepare your data in a compatible format (AnnData for single-cell)
2. Load using the appropriate dataset type:

```python
from czibench.datasets.sc import SingleCellDataset
from czibench.datasets.types import Organism

dataset = SingleCellDataset(
    path="path/to/your/data.h5ad",
    organism=Organism.HUMAN
)
```
