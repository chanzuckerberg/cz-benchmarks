# Models

The `czbenchmarks.models` module provides the infrastructure to run models in a modular and reproducible way. It consists of two main components:

1. **Model Implementations**  
2. **Model Validators**

## Model Implementations

> **Important**  
> All model implementations must extend [BaseModelImplementation](../autoapi/czbenchmarks/models/implementations/base_model_implementation/index). This class defines the core logic for executing a model, including downloading weights, parsing arguments, validating inputs, and setting outputs.

Model implementations are defined in Docker containers and executed either programmatically or via the CLI.

> **Note:**  
> Docker is used for model implementations to ensure reproducibility, modularity, and dependency isolation. It allows developers to package models with all their dependencies, libraries, and configurations into a portable container. This ensures that the model runs consistently across different environments, eliminates compatibility issues, and simplifies deployment. Additionally, Docker enables modularity by isolating each model implementation, making it easier to manage, test, and update individual models without affecting others.

### Responsibilities of an implementation:

- Define the logic for downloading pretrained weights (`_download_model_weights`)
- Process input data as required (e.g. tokenization, filtering, transformation)
- Run inference and extract relevant outputs
- Store outputs via `dataset.set_output(...)`
- Clean up any temporary files

### Common Utilities:

- Use `sync_s3_to_local()` from `czbenchmarks.utils` to download from S3.
- Use `get_model_weights_subdir()` to route weights per variant or organism.
- Use `parse_args()` to register model-specific CLI arguments (e.g., `--model_variant`, `--gene_pert`).

### Example Implementations Include:

- **Geneformer** — tokenizes input with `TranscriptomeTokenizer` and extracts embeddings via `EmbExtractor`.
- **SCVI** — uses `scvi-tools` to load pretrained weights and extract latent representations.
- **UCE** — uses `AnndataProcessor` and custom embedding generation logic.
- **SCGPT**, **scGenePT** — transformers for transcriptomic data or perturbation prediction.

> **Note**  
> Concrete model implementations should be added to the `docker/` directory, not the `implementations/` directory. The `implementations/` directory is reserved for base classes and shared logic.

## Model Validators

Validators enforce the constraints that a dataset must satisfy to be compatible with a given model.

> **Note:**  
> A user would need to create a custom validator when the existing validators do not fully address the specific requirements of their dataset or model. Since most validators are designed to handle common scenarios, a custom validator becomes necessary for unique use cases, such as enforcing specialized constraints on dataset structure, validating custom metadata fields, or ensuring compatibility with a novel model type. Custom validators allow users to define tailored validation logic that aligns with the specific needs of their model and dataset, ensuring accurate and reliable results.

All validators must inherit from one of the following:

- [BaseModelValidator](../autoapi/czbenchmarks/models/validators/base_model_validator/index)  

    - Generic base class with support for arbitrary dataset types.

- [BaseSingleCellValidator](../autoapi/czbenchmarks/models/validators/base_single_cell_model_validator/index)  
    Provides standard checks for single-cell models such as validating:

    - `Organism` compatibility
    - Required keys in `.obs` and `.var`
    - Gene naming conventions (e.g., `ENSG` prefix for human)

Validators are mixed into the implementation class via inheritance:

```python
class MyModelValidator(BaseSingleCellValidator):
    available_organisms = [Organism.HUMAN]
    required_obs_keys = ["cell_type"]
    required_var_keys = ["feature_name"]
    model_type = ModelType.MY_MODEL

class MyModel(MyModelValidator, BaseModelImplementation):
    ...
```

### Best Practices for Validators

When creating a new validator, follow these best practices:

- **Document Validation Requirements Clearly**: Ensure that the purpose and requirements of the validator are well-documented.
- **Use Descriptive Variable Names**: Choose meaningful names for variables to improve code readability.
- **Add Logging for Validation Steps**: Include logging to track validation progress and identify issues.
- **Follow Existing Validator Patterns**: Use existing validators as a reference to maintain consistency.
- **Implement Comprehensive Validation Checks**: Ensure that all necessary checks are implemented to validate datasets thoroughly.
- **Support Multiple Organisms When Possible**: Design validators to handle datasets from multiple organisms.
- **Include Detailed Error Messages**: Provide clear and actionable error messages when validation fails.

### Example Usage of Validators

Here is an example of how to use a validator:

```python
from czbenchmarks.models.validators import YourModelValidator

validator = YourModelValidator()
try:
    validator.validate_dataset(dataset)
    print("Dataset validation passed!")
except ValueError as e:
    print(f"Validation failed: {e}")
```

## Model Directory Structure

The `models/` directory is organized as follows:

```
models/
├── __init__.py
├── README.md
├── implementations/                  # Model implementations
│   ├── __init__.py
│   ├── base_model_implementation.py  # Base implementation class
│   └── README.md
└── validators/                       # Model validators
        ├── __init__.py
        ├── base_model_validator.py       # Base validator class
        ├── base_single_cell_model_validator.py
        ├── <model-specific-validator>.py
        └── README.md
```

- **`implementations/`**: Contains model-specific implementations.
- **`validators/`**: Contains model-specific validation rules.

## Developer Guide: Writing a New Model

To add a new model:

1. **Create a Docker subdirectory** under `docker/<your_model>/` with:

     - `model.py`: Your implementation class
     - `config.yaml`: S3 URIs for weights and any variants
     - `requirements.txt`: Python dependencies
     - `Dockerfile`: Image definition (base on Python GPU image)

2. **Define a validator**: 

     - Use `BaseSingleCellValidator` or `BaseModelValidator` or a custom validator, if needed (see [Model Validators section)[#model-validators], above).
     - Set `available_organisms`, `required_obs_keys`, `required_var_keys`, and `model_type` 

3. **Define a model implementation** that: 

     - Implements `get_model_weights_subdir()` and `_download_model_weights()` 
     - Implements `run_model(dataset: BaseDataset)`  
     - Calls `dataset.set_output(model_type, DataType.XXX, value)`  
     - Parses CLI arguments if needed via `parse_args()` 

4. **Use model type enums** from [ModelType](../autoapi/czbenchmarks/models/types/index)   

     - Ensure your model is registered correctly in `ModelType`. 

5. **Configure variants in `config.yaml`**  

     - Define a top-level `models:` block that maps `model_variant` to S3 URIs for pretrained weights and tokenizer resources. 

### Example Skeleton

```python
from czbenchmarks.models.implementations.base_model_implementation import BaseModelImplementation
from czbenchmarks.models.validators.base_single_cell_model_validator import BaseSingleCellValidator
from czbenchmarks.datasets import DataType, BaseDataset
from czbenchmarks.models.types import ModelType
from czbenchmarks.utils import sync_s3_to_local
from omegaconf import OmegaConf
from pathlib import Path

class MyModelValidator(BaseSingleCellValidator):
    available_organisms = [Organism.HUMAN]
    required_obs_keys = []
    required_var_keys = ["feature_name"]
    model_type = ModelType.MYMODEL

class MyModel(MyModelValidator, BaseModelImplementation):
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_variant", type=str, default="default")
        return parser.parse_args()

    def get_model_weights_subdir(self, dataset: BaseDataset) -> str:
        return self.args.model_variant

    def _download_model_weights(self, dataset: BaseDataset):
        config = OmegaConf.load("config.yaml")
        model_uri = config.models[self.args.model_variant].model_uri
        bucket = model_uri.split("/")[2]
        key = "/".join(model_uri.split("/")[3:])
        sync_s3_to_local(bucket, key, self.model_weights_dir)

    def run_model(self, dataset: BaseDataset):
        adata = dataset.adata
        # Run inference and compute embeddings
        embeddings = ...  # np.ndarray
        dataset.set_output(self.model_type, DataType.EMBEDDING, embeddings)

    def run(self):
        super().run()  # Handles I/O, validation, and execution
```

> **Note:**  
> The example above is specific to the Single Cell Transcriptomics domain space.
