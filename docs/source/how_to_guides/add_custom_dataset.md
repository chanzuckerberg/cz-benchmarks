# Add a Custom Dataset

This guide explains how to integrate your own dataset into CZ Benchmarks.

## Requirements

- For single-cell models, the dataset must conform to AnnData standards in `.h5ad` file format.
- Ensure the dataset includes required metadata columns (e.g., `obs` for metadata, `var` for gene names, etc.).


## Steps to Add Your Dataset

### 1. Prepare Your Data

- Save your data as an AnnData object in `.h5ad` format.
- Ensure the following:
  - Metadata columns (e.g., cell type, batch) are included in `obs`.
  - Gene names are properly defined in `var`.

### 2. Create a Custom Configuration File

- Create a YAML file (e.g., `custom.yaml`) with the following structure:

```yaml
datasets:
  my_dataset:
    _target_: czbenchmarks.datasets.SingleCellDataset
    path: ~/path_to_your_data/my_data.h5ad
    organism: HUMAN
```

- **Explanation:**
  - `datasets`: Defines the datasets to be loaded.
  - `my_dataset`: A unique identifier for your dataset.
  - `_target_`: Specifies the dataset class to instantiate.
  - `path`: Path to your `.h5ad` file.
  - `organism`: Specify the organism (e.g., HUMAN, MOUSE).

### 3. Load Your Dataset in Python

- Use the following Python code to load your dataset:

```python
from czbenchmarks.datasets.utils import load_dataset

dataset = load_dataset("my_dataset", config_path="custom.yaml")
```

- This command loads and instantiates your dataset as a `SingleCellDataset`.

## Tips for Customization

- **Preprocessing:** If your dataset requires specialized preprocessing, consider subclassing `BaseDataset` in your project.
- **Validation:** Ensure organism-specific validations (e.g. gene name prefixes) are met.
- **Testing:** Verify that your dataset loads correctly and includes all required metadata.

