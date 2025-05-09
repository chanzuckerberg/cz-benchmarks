# Add a Custom Dataset

This guide explains how to integrate your own dataset into cz-benchmarks.

## Requirements

For single-cell datasets:
- The dataset file must be an `.h5ad` file conforming to the [AnnData on-disk format](https://anndata.readthedocs.io/en/latest/fileformat-prose.html#on-disk-format).
- The AnnData object's `var_names` must specify the `ensembl_id` for each gene OR `var` must contain a column named `ensembl_id`.
- The AnnData object must meet the validation requirements of the specific models that the dataset will be used to benchmark. This means that:
    - `obs` and `var` each contain the required metadata columns, as specified by the models' `required_obs_keys` and `required_var_keys` properties, respectively.
    - The `ensemble_id` values must be valid for the models' accepted organisms, as specified by the `available_organisms` property. 


## Steps to Add Your Dataset

### 1. Prepare Your Data

- Save your data as an AnnData object in `.h5ad` format.
- Ensure the following:
  - Metadata columns (e.g., cell type, batch) are included in `obs`.
  - Gene names are properly defined in `var`.

### 2. Create a Custom Configuration File

- Update `src/czbenchmarks/conf/datasets.yaml` by adding a new dataset entry:

```yaml
datasets:
  ...

  my_dataset:
    _target_: czbenchmarks.datasets.SingleCellDataset
    path: ~/path_to_your_data/my_data.h5ad
    organism: ${organism:HUMAN}
```

- **Explanation:**
  - `datasets`: Defines the datasets to be loaded.
  - `my_dataset`: A unique identifier for your dataset.
  - `_target_`: Specifies the `Dataset` class to instantiate. Currently, `cz-benchmarks` supports `src.czbenchmarks.datasets.single_cell.SingleCellDataset` and `src.czbenchmarks.datasets.single_cell.PerturbationSingleCellDataset` Dataset types.
  - `path`: Path to your `.h5ad` file. This may be be a local filesystem path or an S3 URL (`s3://...`).
  - `organism`: Specify the organism, which must be a value from the `src.czbenchmarks.datasets.types.Organism` (e.g., HUMAN, MOUSE).

  You may add multiple datasets to thie files, as children of `datasets`.

### 3. Load and Validate Your Dataset in Python

- Use the following Python code to load your dataset:

```python
from czbenchmarks.datasets.utils import load_dataset

# Instantiate the `SingleCellDataset` object from the configuration specified in `datasets.yaml`
dataset = load_dataset("my_dataset")

# Load the H5AD file into memory as an AnnData object, storing in the `ANNDATA` input "slot" of the dataset.
dataset.load_data()

# Ensure the basic requirements are met by the Dataset
dataset.validate()
print(dataset.get_input("ANNDATA"))
```

Fix any loading or validation errors, as needed.

## Tips for Customization

- **Preprocessing:** If your dataset requires specialized preprocessing, consider subclassing `BaseDataset` in your project.
- **Validation:** Ensure organism-specific validations (e.g. gene name prefixes) are met.
- **Test with Models:** Specific models may have additional validation requirements, so you will need to invoke applicable models with your specific dataset to ensure that it is fully compliant.

