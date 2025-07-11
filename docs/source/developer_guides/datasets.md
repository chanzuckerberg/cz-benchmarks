# Datasets

The `czbenchmarks.datasets` module defines the dataset abstraction used across all benchmark pipelines. It provides a uniform and type-safe way to manage dataset inputs ensuring compatibility with tasks.

## Overview

cz-benchmarks currently supports single-cell RNA-seq data stored in the [`AnnData`](https://anndata.readthedocs.io/en/stable/) H5AD format. The dataset system is extensible and can be used for other data modalities by creating new dataset types.

## Key Components

-  [Dataset](../autoapi/czbenchmarks/datasets/base/index)  
   An abstract class that provides ensures all concrete classes provide the following functionality:

   - Loading a dataset file into memory.
   - Validation of the specified dataset file.
   - Specification of an `Organism`.
   - Performs organism-based validation using the `Organism` enum.
   - Storing task-specific outputs to disk for later use by `Task`s.

   All dataset types must inherit from `Dataset`.

-  [SingleCellDataset](../autoapi/czbenchmarks/datasets/single_cell/index)  
   An abstract implementation of `Dataset` for single-cell data.

   Responsibilities:

   - Loads AnnData object from H5AD files via `anndata.read_h5ad`.
   - Stores Anndata in `adata` instance variable.
   - Validates gene name prefixes and that expression values are raw counts.

-  [SingleCellLabeledDataset](../autoapi/czbenchmarks/datasets/single_cell_labeled/index)  
   Subclass of `SingleCellDataset` designed for perturbation benchmarks.

   Responsibilities:

   - Stores labels (expected prediction values) from a specified `obs` column.
   - Validates the label column exists


-  [SingleCellPerturbationDataset](../autoapi/czbenchmarks/datasets/single_cell_perturbation/index)  
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

-  [Organism](../autoapi/czbenchmarks/datasets/types/index)  
   Enum that specifies supported species (e.g., HUMAN, MOUSE) and gene prefixes (e.g., `ENSG` and `ENSMUSG`, respectively).

## Adding a New Dataset

To define a custom dataset:

1. **Inherit from `Dataset`** and implement:

   TODO: update

   - `_validate(self)` — raise exceptions for missing or malformed data
   - `load_data(self)` — populate `self.inputs` with required values



### Example Skeleton

TODO: Update

```python
from czbenchmarks.datasets.base import Dataset
from czbenchmarks.datasets.types import DataType, Organism
import anndata as ad

class MyCustomDataset(Dataset):
    def load_data(self):
        adata = ad.read_h5ad(self.path)

    def _validate(self):
        adata = self.get_input(DataType.ANNDATA)
        assert "my_custom_key" in adata.obs.columns, "Missing key!"
```

## Accessing Inputs and Outputs

Use the following methods for safe access:

```python
dataset.get_input(DataType.ANNDATA)
dataset.get_input(DataType.METADATA)
```


## Tips for Developers

TODO: Add others?

- **AnnData Views:** Use `.copy()` when slicing to avoid "view" issues in Scanpy.

## Related References

- [Add Custom Dataset Guide](../how_to_guides/add_custom_dataset)
- [Dataset API](../autoapi/czbenchmarks/datasets/base/index)
- [SingleCellDataset API](../autoapi/czbenchmarks/datasets/single_cell/index)
- [Organism Enum](../autoapi/czbenchmarks/datasets/types/index)

