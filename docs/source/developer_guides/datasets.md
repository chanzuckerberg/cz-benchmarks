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

   - Validates presence of specific AnnData features: `condition_key` in `adata.obs` column names, and keys named `control_cells_ids` and `de_results_wilcoxon` in `adata.uns`.
   - It also validates that `de_gene_col` is in the column names of the differential expression results. And that `control_name` is present in the data of condition column in `adata.obs`.
   - Matches control cells with perturbation data and determines which genes can be masked for benchmarking
   - Computes and stores `control_matched_adata` (anndata that is split into `X`, `obs`, and `var` for output), `control_cells_ids`, `de_results`, `target_genes_to_save`.

   Example valid perturbation formats:

   - ``{condition_name}`` or ``{condition_name}_{perturb}`` for matched control samples, where perturb can be any type of perturbation.
   - ``{perturb}`` for a single perturbation

- [Organism](../autoapi/czbenchmarks/datasets/types/index)  
   Enum that specifies supported species (e.g., HUMAN, MOUSE) and gene prefixes (e.g., `ENSG` and `ENSMUSG`, respectively).

## Using Available Datasets

### Listing Available Datasets

To list all datasets registered in the system:

```python
from czbenchmarks.datasets.utils import list_available_datasets
available_datasets = list_available_datasets()
```

### Loading a Dataset

To load a dataset by name, use the `load_dataset` utility. The returned object will be an instance of the appropriate dataset class, such as `SingleCellLabeledDataset` or `SingleCellPerturbationDataset`:

```python
from czbenchmarks.datasets import load_dataset, SingleCellLabeledDataset

dataset: SingleCellLabeledDataset = load_dataset("tsv2_prostate")
```

### Accessing Dataset Attributes

After loading, you can access the Dataset's attributes, which vary depending on the dataset type:

#### For `SingleCellLabeledDataset`:

```python
adata_object = dataset.adata        # AnnData object with expression data
labels_series = dataset.labels      # Labels from the specified obs column
```

#### For `SingleCellPerturbationDataset`:

```python
control_cells_ids = dataset.control_cells_ids                  # Dictionary mapping each condition to a dictionary of treatment cell barcodes to matched control cell barcodes
target_conditions_dict = dataset.target_conditions_dict        # Dictionary of masked genes for each condition
de_results = dataset.de_results                                # Differential expression results
control_matched_adata = dataset.control_matched_adata          # AnnData object for matched controls
```

Refer to the class docstrings and API documentation for more details on available attributes and methods.

## Tips for Developers

- **AnnData Views:** Use `.copy()` when slicing to avoid "view" issues in Scanpy.

## Related References

- [Add Custom Dataset Guide](../how_to_guides/add_custom_dataset)
- [Dataset API](../autoapi/czbenchmarks/datasets/dataset/index)
- [SingleCellDataset API](../autoapi/czbenchmarks/datasets/single_cell/index)
- [Organism Enum](../autoapi/czbenchmarks/datasets/types/index)
