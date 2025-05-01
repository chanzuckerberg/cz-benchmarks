API Reference
==============

The CZ Benchmarks package consists of several core modules, each designed to work independently while contributing to a cohesive benchmarking workflow. Below is an overview of these modules, along with links to their detailed documentation.

Core Modules
------------

- **Datasets** (`czbenchmarks.datasets`):  
   Contains classes for loading and validating datasets (e.g., `SingleCellDataset`), with support for AnnData and custom metadata. See the full documentation: :doc:`./autoapi/czbenchmarks/datasets/index`.

- **Models** (`czbenchmarks.models`):  
   - **Implementations:**  
      Contains the concrete model inference logic in Docker container form. The base class is `BaseModelImplementation`.  
   - **Validators:**  
      Enforces that datasets meet the requirements of particular models. Validators extend from `BaseModelValidator` or `BaseSingleCellValidator`.  See the full documentation: :doc:`./autoapi/czbenchmarks/models/index`.

- **Tasks** (`czbenchmarks.tasks`):  
   Provides evaluation tasks (e.g., clustering, embedding, perturbation prediction) by extending the `BaseTask` class. See the full documentation: :doc:`./autoapi/czbenchmarks/tasks/index`.

- **Metrics** (`czbenchmarks.metrics`):  
   Maintains a registry of metric functions through the `MetricRegistry` interface and organizes metrics into categories (clustering, embedding, etc.). See the full documentation: :doc:`./autoapi/czbenchmarks/metrics/index`.

- **Runner** (`czbenchmarks.runner`):  
   Orchestrates the overall workflow: loading datasets, running model inference, executing tasks, and serializing results. See the full documentation: :doc:`./autoapi/czbenchmarks/runner/index`.

Additional Utilities
--------------------

- **CLI** (`czbenchmarks.cli`):  
   Command-line interface for interacting with the CZ Benchmarks package. See the full documentation: :doc:`czbenchmarks.cli <./autoapi/czbenchmarks/cli/cli/index>`.

- **Utils** (`czbenchmarks.utils`):  
   Contains utility functions and helpers used across the package. See the full documentation: :doc:`./autoapi/czbenchmarks/utils/index`.


.. .. toctree::
..     :maxdepth: 1

..     ./autoapi/czbenchmarks/cli/cli/index.rst
..     ./autoapi/czbenchmarks/datasets/index.rst
..     ./autoapi/czbenchmarks/models/index.rst
..     ./autoapi/czbenchmarks/tasks/index.rst
..     ./autoapi/czbenchmarks/metrics/index.rst
..     ./autoapi/czbenchmarks/utils/index.rst
..     ./autoapi/czbenchmarks/runner/index.rst
