API Reference
==============

The cz-benchmarks package consists of several core modules, each designed to work independently while contributing to a cohesive benchmarking workflow. Below is an overview of these modules, along with links to their detailed documentation.

Core Modules
------------

- **Datasets** (`czbenchmarks.datasets`):  
   Contains classes for loading and validating datasets (e.g., `SingleCellDataset`), with support for AnnData and custom metadata. See the full documentation: :doc:`./autoapi/czbenchmarks/datasets/index`.

- **Tasks** (`czbenchmarks.tasks`):  
   Provides evaluation tasks (e.g., clustering, embedding, perturbation prediction) by extending the `BaseTask` class. See the full documentation: :doc:`./autoapi/czbenchmarks/tasks/index`.

- **Metrics** (`czbenchmarks.metrics`):  
   Maintains a registry of metric functions through the `MetricRegistry` interface and organizes metrics into categories (clustering, embedding, etc.). See the full documentation: :doc:`./autoapi/czbenchmarks/metrics/index`.

Additional Utilities
--------------------

- **Utils** (`czbenchmarks.utils`):  
   Contains utility functions and helpers used across the package. See the full documentation: :doc:`./autoapi/czbenchmarks/utils/index`.

- **CLI** (`czbenchmarks.cli`):  
   Command-line interface for interacting with the cz-benchmarks package. See the full documentation: :doc:`czbenchmarks.cli <./autoapi/czbenchmarks/cli/cli/index>`.

.. .. toctree::
..     :maxdepth: 1

..     ./autoapi/czbenchmarks/cli/cli/index.rst
..     ./autoapi/czbenchmarks/datasets/index.rst
..     ./autoapi/czbenchmarks/tasks/index.rst
..     ./autoapi/czbenchmarks/metrics/index.rst
..     ./autoapi/czbenchmarks/utils/index.rst
..     ./autoapi/czbenchmarks/runner/index.rst
