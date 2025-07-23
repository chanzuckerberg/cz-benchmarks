# Add a Custom Dataset

This guide explains how to integrate your own dataset into cz-benchmarks.

## Requirements

For single-cell datasets:
- The dataset file must be an `.h5ad` file conforming to the [AnnData on-disk format](https://anndata.readthedocs.io/en/latest/fileformat-prose.html#on-disk-format).
- The AnnData object's `var_names` must specify the `ensembl_id` for each gene OR `var` must contain a column named `ensembl_id`.
- The AnnData object must meet the validation requirements for the specific task it will be used with. This may include:
    - `obs` must contain required metadata columns (e.g., `cell_type` for `SingleCellLabeledDataset`, or `condition` and `split` for `SingleCellPerturbationDataset`).
    - The `ensemble_id` values must be valid for the specified `Organism`.



## Steps to Add Your Dataset

### 1. Prepare Your Data

- Save your data as an AnnData object in `.h5ad` format.
- Ensure the following:
  - Metadata columns (e.g., cell type, batch) are included in `obs`.
  - Gene names are properly defined in `var`.

### 2. Update Datasets Configuration File

- Update `src/czbenchmarks/conf/datasets.yaml` by adding a new dataset entry:

```yaml
datasets:
  ...

  my_labeled_dataset:
    _target_: czbenchmarks.datasets.SingleCellLabeledDataset
    path: /path/to/your/labeled_data.h5ad
    organism: ${organism:HUMAN}
    label_column_key: "cell_type" # Column in adata.obs with labels

  my_perturbation_dataset:
    _target_: czbenchmarks.datasets.SingleCellPerturbationDataset
    path: /path/to/your/perturb_data.h5ad
    organism: ${organism:MOUSE}
    condition_key: "condition" # Column in adata.obs for 'ctrl' vs perturbation
    split_key: "split" # Column in adata.obs for 'train'/'test'/'val'
```

- **Explanation:**
  -   `datasets`: Defines the datasets to be loaded.
  -   `my_labeled_dataset`: A unique identifier for your dataset.
  -   `_target_`: Specifies the `Dataset` class to instantiate. `cz-benchmarks` supports `SingleCellLabeledDataset` (for tasks requiring ground-truth labels like clustering or classification) and `SingleCellPerturbationDataset` (for perturbation prediction tasks).
  -   `path`: Path to your `.h5ad` file. This may be a local filesystem path or an S3 URL (`s3://...`).
  -   `organism`: Specify the organism, which must be a value from the `czbenchmarks.datasets.types.Organism` enum (e.g., HUMAN, MOUSE).
  -   `label_column_key`: (For `SingleCellLabeledDataset`) The column in `.obs` containing the labels.
  -   `condition_key` / `split_key`: (For `SingleCellPerturbationDataset`) Columns in `.obs` for perturbation conditions and data splits.

  You may add multiple datasets to thie files, as children of `datasets`.

### 3. Load and Validate Your Dataset in Python

- Use the following Python code to load your dataset:

```python
from czbenchmarks.datasets import load_dataset

# Instantiate the Dataset object from your custom configuration
dataset = load_dataset(dataset_name="my_labeled_dataset")

# dataset.load_data() is called automatically by the load_dataset utility.
# You can now access the loaded data via the .adata attribute
print(dataset.adata)

# Ensure the basic requirements are met by the Dataset
dataset.validate()
print(f"Labels: {dataset.labels.head()}")
```

Fix any loading or validation errors, as needed.

## Tips for Customization

- **Preprocessing:** If your dataset requires specialized preprocessing, consider subclassing `BaseDataset` in your project.
- **Validation:** Ensure organism-specific validations (e.g. gene name prefixes) are met.
- **Test with Tasks:** Run your dataset with the intended tasks to ensure it is fully compliant with their requirements.

