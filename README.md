# CZ Benchmarks

**PROJECT STATUS: UNSTABLE**

**This project is under development and not yet stable. It is being actively developed, but not supported and not ready for community contribution. Things may break without notice, and it is not likely the developers will respond to requests for user support.**

CZ Benchmarks is a package for standardized evaluation and comparison of biology-oriented machine learning models (starting with single-cell transcriptomics) across various tasks and metrics. The package provides a modular architecture for running containerized models, executing evaluation tasks, and computing performance metrics.

## Installation Instructions

### From Source

```bash
git clone https://github.com/chanzuckerberg/cz-benchmarks.git
cd cz-benchmarks
pip install -e .
```

## Example Usage
* Load a dataset, run a model, run several tasks, and compute metrics

```python
from cz_benchmarks.datasets.single_cell import SingleCellDataset
from cz_benchmarks.tasks import ClusteringTask, EmbeddingTask, MetadataLabelPredictionTask
from cz_benchmarks.runner import ContainerRunner
from cz_benchmarks.datasets.types import Organism

dataset = load_dataset("example")
runner = ContainerRunner(image="cz_benchmarks-scvi:latest", gpu=True)
dataset = runner.run(dataset)

task = ClusteringTask(label_key="cell_type")
dataset, clustering_results = task.run(dataset)

task = EmbeddingTask(label_key="cell_type")
dataset, embedding_results = task.run(dataset)

task = MetadataLabelPredictionTask(label_key="cell_type")
dataset, prediction_results = task.run(dataset)
```

## Architecture Overview

This is a benchmarking framework for machine learning models (with a focus on single-cell data), built around Docker containers and a modular architecture. Here are the key components:

### Core components

#### Base Classes

- `cz_benchmarks.datasets.base.BaseDataset`: Abstract base class for all datasets
- `cz_benchmarks.models.base.BaseModel`: Abstract base class for all models
- `cz_benchmarks.tasks.base.BaseTask`: Abstract base class for evaluation tasks

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

2. Implement your base model class in `src/cz_benchmarks/models/`. This should implement the model-specific validator (`_validate_model_requirements`).

3. Implement the model class in `model.py` (see template):

```python
"""
REQUIRED MODEL IMPLEMENTATION FILE

This file MUST:
1. Define a model class that inherits from your base model class that you implemented in the benchmarking library (e.g. ScviValidator)
2. Implement the run_model() method
3. Create an instance and call run() if used as main

Example implementation for a single-cell model:
"""

from cz_benchmarks.models.single_cell import BaseYourModel

class ExampleModel(BaseYourModel):

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

1. Create a task class inheriting from `cz_benchmarks.tasks.base.BaseTask`. For example:

```python
from cz_benchmarks.tasks.base import BaseTask

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
from cz_benchmarks.runner import ContainerRunner
from cz_benchmarks.datasets.utils import load_dataset

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
from cz_benchmarks.datasets.single_cell import SingleCellDataset
from cz_benchmarks.datasets.types import Organism

dataset = SingleCellDataset(
    path="path/to/your/data.h5ad",
    organism=Organism.HUMAN
)
```

## Contributing
This project adheres to the Contributor Covenant code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to opensource@chanzuckerberg.com.

## Reporting Security Issues
Please note: If you believe you have found a security issue, please responsibly disclose by contacting us at security@chanzuckerberg.com.

## License Notice for Dependencies
This repository is licensed under the MIT License; however, it relies on certain third-party dependencies that are licensed under the GNU General Public License (GPL). Specifically:
- igraph (v0.11.8) is licensed under the GNU General Public License (GPL).
- leidenalg (v0.10.2) is licensed under the GNU General Public License v3 or later (GPLv3+).

These libraries are not included in this repository but must be installed separately by users. Please be aware that the GPL license terms apply to these dependencies, and certain uses of GPL-licensed code may have licensing implications for your own software.