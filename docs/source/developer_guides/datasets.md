# Datasets

The `czbenchmarks.datasets` module defines the dataset abstraction used across all benchmark pipelines. It provides a uniform and type-safe way to manage dataset inputs ensuring compatibility with tasks.

## Overview

cz-benchmarks currently supports single-cell RNA-seq data stored in the [`AnnData`](https://anndata.readthedocs.io/en/stable/) H5AD format. The dataset system is extensible and can be used for other data modalities by creating new dataset types.

## Key Components

- [Dataset](../autoapi/czbenchmarks/datasets/dataset/index)  
   An abstract class that provides ensures all concrete classes provide the following functionality:

   - Loading a dataset file into memory.
   - Validation of the specified dataset file.
   - Specification of an `Organism`.
   - Performs organism-based validation using the `Organism` enum.
   - Storing task-specific outputs to disk for later use by `Task`s.

   All dataset types must inherit from `Dataset`.

- [SingleCellDataset](../autoapi/czbenchmarks/datasets/single_cell/index)  
   An abstract implementation of `Dataset` for single-cell data.

   Responsibilities:

   - Loads AnnData object from H5AD files via `anndata.read_h5ad`.
   - Stores Anndata in `adata` instance variable.
   - Validates gene name prefixes and that expression values are raw counts.

- [SingleCellLabeledDataset](../autoapi/czbenchmarks/datasets/single_cell_labeled/index)  
   Subclass of `SingleCellDataset` designed for perturbation benchmarks.

   Responsibilities:

   - Stores labels (expected prediction values) from a specified `obs` column.
   - Validates the label column exists


- [SingleCellPerturbationDataset](../autoapi/czbenchmarks/datasets/single_cell_perturbation/index)  
   Subclass of `SingleCellDataset` designed for perturbation benchmarks.

   Responsibilities:

   - Validates presence of `condition_key` and `split_key` (e.g., `condition`, `split`)
   - Stores control and perturbed cells
   - Computes and stores `DataType.PERTURBATION_TRUTH` as ground-truth reference
   - Filters `adata` to only include control cells for inference.

   Example valid perturbation formats:

   - `"ctrl"`: control
   - `"GENE+ctrl"`: single-gene perturbation
   - `"GENE1+GENE2"`: combinatorial perturbation

- [Organism](../autoapi/czbenchmarks/datasets/types/index)  
   Enum that specifies supported species (e.g., HUMAN, MOUSE) and gene prefixes (e.g., `ENSG` and `ENSMUSG`, respectively).

## Adding a New Dataset

To define a custom dataset:

1. **Inherit from `Dataset`** or one of its subclasses (e.g., `SingleCellDataset`).
2. Implement the abstract methods:
    - `load_data(self)` — Populate instance variables (e.g., `self.adata`, `self.labels`) with data loaded from `self.path`.
    - `store_task_inputs(self)` — Save processed data needed by tasks to the `task_inputs_dir`.
    - `_validate(self)` — Raise exceptions for missing or malformed data.


### Example Skeleton

```python
from czbenchmarks.datasets import SingleCellDataset
from czbenchmarks.datasets.types import Organism
import anndata as ad

class MyCustomDataset(SingleCellDataset):
    def load_data(self):
        # First, load the base data
        super().load_data()
        # Then, load any custom data. For example, a special annotation.
        if "my_custom_key" not in self.adata.obs:
            raise ValueError("Dataset is missing 'my_custom_key' in obs.")
        self.my_annotation = self.adata.obs["my_custom_key"]

    def _validate(self):
        # First, run parent validation
        super()._validate()
        # Then, add custom validation logic
        assert all(self.my_annotation.notna()), "Custom annotation has missing values!"

    def store_task_inputs(self):
        # This method would be implemented to save any derived data
        # that tasks might need.
        pass
```

## Accessing Data

Once a dataset is loaded, its data is stored in instance attributes, which can be accessed directly.

```python
# For a SingleCellLabeledDataset
dataset.load_data()
adata_object = dataset.adata
labels_series = dataset.labels

# For a SingleCellPerturbationDataset
dataset.load_data()
control_adata = dataset.adata
truth_data_dict = dataset.perturbation_truth
```

## Tips for Developers

- **AnnData Views:** Use `.copy()` when slicing to avoid "view" issues in Scanpy.

## Related References

- [Add Custom Dataset Guide](../how_to_guides/add_custom_dataset)
- [Dataset API](../autoapi/czbenchmarks/datasets/dataset/index)
- [SingleCellDataset API](../autoapi/czbenchmarks/datasets/single_cell/index)
- [Organism Enum](../autoapi/czbenchmarks/datasets/types/index)

