
# Add a Custom Dataset Type

This guide explains how to add your own single-cell dataset or define a new custom dataset type (i.e., a new Python class for a dataset) to cz-benchmarks.


## Adding Datasets for Supported Types

This section describes how to add new datasets for use with the `czbenchmarks.datasets` module, focusing on single-cell RNA-seq data in AnnData `.h5ad` format.

### Requirements

For single-cell datasets:

- The dataset file must be an `.h5ad` file conforming to the [AnnData on-disk format](https://anndata.readthedocs.io/en/latest/fileformat-prose.html#on-disk-format).
- The AnnData object's `var_names` must specify the Ensembl gene ID for each gene, **or** `var` must contain a column named `ensembl_id`.
- The AnnData object must meet the validation requirements for the specific dataset class:
  - For `SingleCellLabeledDataset`: `obs` must contain the label column (e.g., `cell_type`).
  - For `SingleCellPerturbationDataset`: `obs` must contain the `condition_key` column and the control condition value (`control_name`). The differential expression results must be present in `uns` under the specified test name (e.g., `de_results_wilcoxon` or `de_results_t_test`), and must include the `de_gene_col` column.
- The Ensembl gene IDs must be valid for the specified `Organism` (e.g., `ENSG` for human, `ENSMUSG` for mouse).


### 1. Prepare Your Data

- Save your data as an AnnData object in `.h5ad` format.
- Ensure:
  - All required metadata columns (e.g., cell type, batch, condition) are included in `obs`.
  - Gene names or Ensembl IDs are properly defined in `var` or as `var_names`.


### 2. Update Datasets Configuration File

Add your dataset to the configuration file (e.g., `src/czbenchmarks/conf/datasets.yaml`):

```yaml
datasets:
  my_labeled_dataset:
  _target_: czbenchmarks.datasets.SingleCellLabeledDataset
  path: /path/to/your/labeled_data.h5ad
  organism: ${organism:HUMAN}
  label_column_key: "cell_type" # Column in adata.obs with labels

  my_perturbation_dataset:
  _target_: czbenchmarks.datasets.SingleCellPerturbationDataset
  path: /path/to/your/perturb_data.h5ad
  organism: ${organism:MOUSE}
  condition_key: condition
  control_name: non-targeting
  de_gene_col: gene_id
  deg_test_name: wilcoxon
```

**Explanation of keys:**

- `datasets`: Top-level key for dataset definitions.
- Each child (e.g., `my_labeled_dataset`) is a unique dataset identifier.
- `_target_`: The fully qualified class name of the dataset type. Supported types include:
  - `czbenchmarks.datasets.SingleCellLabeledDataset` (for labeled single-cell data)
  - `czbenchmarks.datasets.SingleCellPerturbationDataset` (for perturbation datasets)
- `path`: Path to the `.h5ad` file (local or S3).
- `organism`: Must be a value from `czbenchmarks.datasets.types.Organism` (e.g., HUMAN, MOUSE).
- `label_column_key`: (For `SingleCellLabeledDataset`) Name of the label column in `obs`.
- `condition_key`, `control_name`, `de_gene_col`, `deg_test_name`: (For `SingleCellPerturbationDataset`) Required keys for perturbation data and DE results.

You may add multiple datasets as children of `datasets`.


### 3. Using Datasets

Datasets can be loaded in two ways:

#### a. Registering Datasets with a YAML Configuration

Define datasets in a YAML file and load them by name using `load_dataset`:

**Example: `user_dataset.yaml`**

```yaml
datasets:
  user_dataset:
  _target_: czbenchmarks.datasets.SingleCellLabeledDataset
  organism: ${organism:HUMAN}
  path: s3://<bucket name>/<path>/example-small.h5ad
```

Load the dataset in Python:

```python
from czbenchmarks.datasets.utils import load_dataset
dataset = load_dataset("user_dataset", config_path='user_dataset.yaml')
```

#### b. Loading Local Datasets Directly

For quick experiments, use `load_local_dataset` to instantiate a dataset directly:

```python
from czbenchmarks.datasets.utils import load_local_dataset
from czbenchmarks.datasets.types import Organism

dataset = load_local_dataset(
  dataset_class="czbenchmarks.datasets.SingleCellLabeledDataset",
  path="my_data.h5ad",
  organism=Organism.HUMAN
)
print(dataset.adata)
```

You can use any supported dataset class and provide additional keyword arguments as needed.

---

## Steps to Add a New Dataset Type

### 1. Define a New Dataset Class

Create a new Python class that inherits from `Dataset` or one of its subclasses (such as `SingleCellDataset`) in `czbenchmarks.datasets`. Implement the required methods:

- **`load_data(self)`**: Load your data from disk (using `self.path`) and populate instance variables (e.g., `self.adata`, `self.labels`). You can call `super().load_data()` to leverage base loading logic if using a subclass like `SingleCellDataset`.
- **`_validate(self)`**: Add custom validation logic for your dataset. Call `super()._validate()` to include base checks, then add any dataset-specific assertions or error checks.
- **`store_task_inputs(self)`**: (Optional, but recommended) Save any derived or preprocessed data needed by tasks to `self.task_inputs_dir`.

**Example:**

```python
from czbenchmarks.datasets import SingleCellDataset
import anndata as ad

class MyCustomDataset(SingleCellDataset):
  def load_data(self):
    # Load the base AnnData object using the parent method
    super().load_data()
    # Add custom loading logic
    if "my_custom_key" not in self.adata.obs:
      raise ValueError("Dataset is missing 'my_custom_key' in obs.")
    self.my_annotation = self.adata.obs["my_custom_key"]

  def _validate(self):
    # Run parent validation
    super()._validate()
    # Add custom validation logic
    assert all(self.my_annotation.notna()), "Custom annotation has missing values!"

  def store_task_inputs(self):
    # Optional: Save any derived data needed by tasks
    pass
```


### 2. Register and Use Your Dataset

- Add your new dataset class to the appropriate module in `czbenchmarks.datasets`.
- Register it in the `src/czbenchmarks/datasets/__init__.py` file.
- You can now use your new dataset type in YAML configs or with `load_local_dataset`.


### 3. Test and Validate

- Ensure your dataset loads and validates correctly.
- Test it with the intended tasks to ensure compatibility.

### Tips
- Place your new class in the appropriate module under `czbenchmarks.datasets`.
- If your dataset type is specialized (e.g., single-cell), inherit from the relevant subclass (`SingleCellDataset`).
- Refer to existing classes in `single_cell.py` or `single_cell_labeled.py` for more examples.
- Register your class in the module’s `__init__.py` if you want it to be importable directly from `czbenchmarks.datasets`.


## Related References

- [Datasets Usage Guide](../developer_guides/datasets.md)
- [Dataset API](../autoapi/czbenchmarks/datasets/dataset/index)
- [SingleCellDataset API](../autoapi/czbenchmarks/datasets/single_cell/index)
- [Organism Enum](../autoapi/czbenchmarks/datasets/types/index)

