# czibench

## Installation Instructions

### From Source

```bash
git clone https://github.com/chanzuckerberg/czibench.git
cd czibench
pip install -e .
```

## Requirements

Core dependencies:

- docker>=6.1.0
- pyyaml>=6.0
- boto3>=1.28.0

Data handling:

- anndata>=0.9.0
- h5py>=3.8.0
- dill>=0.3.6
- scikit-learn
- scanpy

Clustering:

- igraph>=0.11.8
- leidenalg>=0.10.2

## Architecture Overview

This is a benchmarking framework for machine learning models (with a focus on single-cell data), built around Docker containers and a modular architecture. Here are the key components:

### Core components

#### Base Classes

- `czibench.datasets.base.BaseDataset`: Abstract base class for all datasets
- `czibench.models.base.BaseModel`: Abstract base class for all models
- `czibench.tasks.base.BaseTask`: Abstract base class for evaluation tasks

#### Container system

- Uses Docker for model execution
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

Refer to the template docker directory as a starting point (`docker/template/`)!

2. Implement the model class in `model.py` (see template):

```python
"""
REQUIRED MODEL IMPLEMENTATION FILE

This file MUST:
1. Define a model class that inherits from BaseSingleCell (for single-cell data)
   or BaseModel (for other modalities)
2. Implement required class variables and methods
3. Implement the run_model() method
4. Create an instance and call run() if used as main

Example implementation for a single-cell model:
"""

from czibench.models.sc import BaseSingleCell
from czibench.datasets.types import Organism
from czibench.datasets.sc import SingleCellDataset

class ExampleModel(BaseSingleCell):
    # Required: Specify which organisms this model supports
    available_organisms = [Organism.HUMAN, Organism.MOUSE]

    # Required: Specify which metadata columns are needed for batching
    required_obs_keys = ['dataset_id', 'donor_id']  # Add required columns

    @classmethod
    def _validate_model_requirements(cls, dataset: SingleCellDataset) -> bool:
        """
        Required: Implement validation logic specific to your model
        Args:
            dataset: The input SingleCellDataset to validate
        Returns:
            bool: True if dataset meets model requirements, False otherwise
        """
        # Check if all required batch keys are present
        missing_keys = [key for key in cls.required_obs_keys
                       if key not in dataset.adata.obs.columns]

        if missing_keys:
            return False

        return True

    def run_model(self):
        """
        Required: Implement your model's inference logic here.
        Access input data via self.data.adata
        Set output embedding via self.data.output_embedding
        """
        # Add your model implementation here
        raise NotImplementedError("Model implementation required")

if __name__ == "__main__":
    ExampleModel().run()
```

### Adding new metrics

1. Create a new function that outputs a score given some input. Below are a couple example clustering metrics in `metrics/`.

```python
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

def adjusted_rand_index(original_labels, predicted_labels):
    return adjusted_rand_score(
        original_labels,
        predicted_labels
    )

def normalized_mutual_info(original_labels, predicted_labels):
    return adjusted_mutual_info_score(
        original_labels,
        predicted_labels
    )
```

### Adding New Tasks

1. Create a task class inheriting from `czibench.tasks.base.BaseTask`. For example:

```python
from czibench.tasks.base import BaseTask

class ClusteringTask(BaseTask):

    def __init__(self, label_key: str):
        self.label_key = label_key

    def validate(self, data: SingleCellDataset):
        return data.output_embedding is not None and self.label_key in data.sample_metadata.columns

    def _run_task(self, data: SingleCellDataset) -> SingleCellDataset:
        adata = data.adata
        adata.obsm["emb"] = data.output_embedding
        self.input_labels = data.sample_metadata[self.label_key]
        self.predicted_labels = your_label_prediction_function(...)
        return data

    def _compute_metrics(self) -> Dict[str, float]:
        return {
            "adjusted_rand_index": adjusted_rand_index(self.input_labels, self.predicted_labels),
            "normalized_mutual_info": normalized_mutual_info(self.input_labels, self.predicted_labels),
        }
```

## Running Models

Using the `ContainerRunner`:

```python
from czibench.runner import ContainerRunner
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

## Example Usage

```python
from czibench.datasets.sc import SingleCellDataset
from czibench.datasets.types import Organism
from czibench.runner import ContainerRunner
from czibench.tasks.sc import ClusteringTask, EmbeddingTask

dataset = SingleCellDataset(
    "example.h5ad",
    organism=Organism.HUMAN,
)

runner = ContainerRunner(
    image="czibench-scvi:latest",
    gpu=True,
)

dataset = runner.run(dataset)

task = ClusteringTask(label_key="cell_type")
dataset, clustering_results = task.run(dataset)

task = EmbeddingTask(label_key="cell_type")
dataset, embedding_results = task.run(dataset)
```
