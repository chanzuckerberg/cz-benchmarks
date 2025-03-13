# CZ Benchmarks

**PROJECT STATUS: UNSTABLE**

**This project is under development and not yet stable. It is being actively developed, but not supported and not ready for community contribution. Things may break without notice, and it is not likely the developers will respond to requests for user support.**

CZ Benchmarks is a package for standardized evaluation and comparison of biology-oriented machine learning models (starting with single-cell transcriptomics) across various tasks and metrics. The package provides a modular architecture for running containerized models, executing evaluation tasks, and computing performance metrics.

## Installation Instructions

### From Source

```bash
git clone https://github.com/chanzuckerberg/cz-benchmarks.git
cd cz-benchmarks
pip install .
```

## Example Usage

```python
from czbenchmarks.datasets.utils import load_dataset
from czbenchmarks.runner import ContainerRunner
from czbenchmarks.tasks import ClusteringTask, EmbeddingTask, MetadataLabelPredictionTask

# Load dataset with custom configuration
dataset = load_dataset("example", config_path="custom.yaml")

# Run model
runner = ContainerRunner(
    image="czbenchmarks-scvi:latest",
    gpu=True,
)
dataset = runner.run(dataset)

# Run evaluation tasks
task = ClusteringTask(label_key="cell_type")
clustering_results = task.run(dataset)

task = EmbeddingTask(label_key="cell_type")
embedding_results = task.run(dataset)

task = MetadataLabelPredictionTask(label_key="cell_type")
prediction_results = task.run(dataset)
```

## Architecture Overview

This is a benchmarking framework for machine learning models (with a focus on single-cell data), built around Docker containers and a modular architecture. Here are the key components:

### Core Components

#### Base Classes

- `czbenchmarks.datasets.base.BaseDataset`: Abstract base class for all datasets
- `czbenchmarks.models.implementations.base_model_implementation.BaseModelImplementation`: Abstract base class for all model implementations
- `czbenchmarks.models.validators.base_model_validator.BaseModelValidator`: Abstract base class for model validation
- `czbenchmarks.tasks.base.BaseTask`: Abstract base class for evaluation tasks

#### Container System

- Uses Docker for model execution
- Models are packaged as Docker containers with standardized interfaces
- Each model container includes:
  - Model implementation inheriting from `BaseModelImplementation`
  - Model-specific validator inheriting from `BaseModelValidator`
  - Configuration file (config.yaml)
  - Requirements file (requirements.txt)

### Available Models

The framework currently supports the following models:

- scVI: Variational inference for single-cell RNA-seq data
- scGPT: Single-cell GPT model for transcriptomics
- Geneformer: Transformer model for gene expression
- scGenePT: Single-cell Gene Prediction Transformer
- UCE: Universal Cell Embedding model

## Contributing New Components

### Adding a New Model

1. Create a new directory under `docker/your_model/` with:

```markdown
docker/your_model/
├── Dockerfile
├── model.py
├── config.yaml
└── requirements.txt
```

Refer to the template docker directory as a starting point (`docker/template/`)!

2. Implement your model validator in `src/czbenchmarks/models/validators/`. This should inherit from `BaseModelValidator` or `BaseSingleCellValidator` and implement:
   - Required data type specifications
   - Model-specific validation rules
   - Supported organisms and data requirements

3. Implement your model class in `model.py` (see template):

```python
from czbenchmarks.models.implementations.base_model_implementation import BaseModelImplementation
from czbenchmarks.models.validators.your_model import YourModelValidator

class YourModel(YourModelValidator, BaseModelImplementation):
    def get_model_weights_subdir(self, dataset: BaseDataset) -> str:
        """Specify subdirectory for model weights."""
        return dataset.organism.name

    def _download_model_weights(self, dataset: BaseDataset):
        """Download model weights from your source."""
        # Implement model weight downloading logic
        pass

    def run_model(self, dataset: BaseDataset):
        """Implement your model's inference logic."""
        # Access input data via dataset.adata
        # Set output embedding via dataset.set_output()
        pass

    def parse_args(self):
        """Parse model-specific arguments."""
        pass

if __name__ == "__main__":
    YourModel().run()
```

### Adding New Tasks

1. Create a task class inheriting from `czbenchmarks.tasks.base.BaseTask`. For example:

```python
from czbenchmarks.tasks.base import BaseTask

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

## Using Custom Data

To use your own data:

1. Prepare your data in a compatible format (AnnData for single-cell)
2. Create a custom configuration file (e.g., `custom.yaml`)
3. Load using the appropriate dataset type:

```python
from czbenchmarks.datasets.utils import load_dataset

dataset = load_dataset("your_dataset", config_path="custom.yaml")
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