Datasets
========

The `czbenchmarks.datasets` module defines the dataset abstraction used across all benchmark pipelines. It provides a uniform and type-safe way to manage dataset inputs and outputs, ensuring compatibility with models and tasks.

Overview
--------

CZ Benchmarks currently supports single-cell RNA-seq data stored in the [AnnData](https://anndata.readthedocs.io/en/stable/) (H5AD) format. The dataset system is extensible and can be used for other data modalities by creating new dataset types.

Key Components
--------------

-  :doc:`BaseDataset <../autoapi/czbenchmarks/datasets/base/index>`   
  An abstract class that provides methods for:
  
  - Storing typed inputs and model outputs (`set_input`, `set_output`)
  - Type validation via `DataType` enums
  - Serialization and deserialization (`.dill`-based)
  - Loading/unloading memory-intensive data

  All dataset types must inherit from `BaseDataset`.

-  :doc:`SingleCellDataset <../autoapi/czbenchmarks/datasets/single_cell/index>`   
  A concrete implementation of `BaseDataset` for single-cell data.

  Responsibilities:
  - Loads anndata files via `anndata.read_h5ad`
  - Stores `.obs` as metadata and `.X` as the expression matrix
  - Performs organism-based validation using the `Organism` enum
  - Validates gene name prefixes and presence of expected columns

  Automatically sets:
  - `DataType.ANNDATA`
  - `DataType.METADATA`
  - `DataType.ORGANISM`

-  :doc:`PerturbationSingleCellDataset <../autoapi/czbenchmarks/datasets/single_cell/index>`   
  Subclass of `SingleCellDataset` designed for perturbation benchmarks.

  Responsibilities:
  - Validates presence of `condition_key` and `split_key` (e.g., `condition`, `split`)
  - Stores control and perturbed cells
  - Computes and stores `DataType.PERTURBATION_TRUTH` as ground-truth reference

  Automatically filters `adata` to only include control cells for inference.

  Example valid perturbation formats:
  - `"ctrl"`: control
  - `"GENE+ctrl"`: single-gene perturbation
  - `"GENE1+GENE2"`: combinatorial perturbation

-  :doc:`DataType <../autoapi/czbenchmarks/datasets/types/index>`   
  Defines all valid input and output types (e.g., `ANNDATA`, `METADATA`, `EMBEDDING`, etc.)
  with expected Python types (`AnnData`, `pd.DataFrame`, `np.ndarray`, etc.)

-  :doc:`Organism <../autoapi/czbenchmarks/datasets/types/index>`   
  Enum that specifies supported species (e.g., HUMAN, MOUSE) and gene prefixes (e.g., `ENSG`).

Adding a New Dataset
---------------------

To define a custom dataset:

1. **Inherit from `BaseDataset`** and implement:

   - `_validate(self)` — raise exceptions for missing or malformed data
   - `load_data(self)` — populate `self.inputs` with required values
   - `unload_data(self)` — clear memory-heavy inputs (e.g., `adata`) before serialization

2. **Register all required inputs** using `self.set_input(data_type, value)`
3. **Store model outputs** using `self.set_output(model_type, data_type, value)`
4. **Use the `DataType` enum** to enforce type safety and input validation

Example Skeleton
^^^^^^^^^^^^^^^^

.. code-block:: python

   from czbenchmarks.datasets.base import BaseDataset
   from czbenchmarks.datasets.types import DataType, Organism
   import anndata as ad

   class MyCustomDataset(BaseDataset):
       def load_data(self):
           adata = ad.read_h5ad(self.path)
           self.set_input(DataType.ANNDATA, adata)
           self.set_input(DataType.METADATA, adata.obs)
           self.set_input(DataType.ORGANISM, Organism.HUMAN)

       def unload_data(self):
           self._inputs.pop(DataType.ANNDATA, None)
           self._inputs.pop(DataType.METADATA, None)

       def _validate(self):
           adata = self.get_input(DataType.ANNDATA)
           assert "my_custom_key" in adata.obs.columns, "Missing key!"

Accessing Inputs and Outputs
----------------------------

Use the following methods for safe access:

.. code-block:: python

   dataset.get_input(DataType.ANNDATA)
   dataset.get_input(DataType.METADATA)
   dataset.get_output(ModelType.SCVI, DataType.EMBEDDING)

Serialization Support
---------------------

Datasets can be serialized to disk after model inference. Internally, `.dill` is used to support complex Python objects like `AnnData`.

.. code-block:: python

   dataset.serialize("/tmp/my_dataset.dill")
   loaded = BaseDataset.deserialize("/tmp/my_dataset.dill")

   # Don't forget to reload memory-intensive fields
   loaded.load_data()

Tips for Developers
--------------------

- **AnnData Views:** Use `.copy()` when slicing to avoid "view" issues in Scanpy.
- **Organism Validation:** Always set `DataType.ORGANISM` and validate `var_names` with `Organism.prefix`.
- **Gene Names:** Ensure `.var` has `feature_name` or `ensembl_id` depending on model requirements.
- **Metadata Compatibility:** Validate that all label keys required by tasks (e.g., `cell_type`, `sex`, `batch`) exist in `.obs`.

Related References
------------------

- :doc:`Add Custom Dataset Guide <../how_to_guides/add_custom_dataset>`
- :doc:`BaseDataset API <../autoapi/czbenchmarks/datasets/base/index>`
- :doc:`SingleCellDataset API <../autoapi/czbenchmarks/datasets/single_cell/index>`
- :doc:`DataType Enum <../autoapi/czbenchmarks/datasets/types/index>`
- :doc:`Organism Enum <../autoapi/czbenchmarks/datasets/types/index>`

