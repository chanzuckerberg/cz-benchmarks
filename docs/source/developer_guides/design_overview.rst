Design Overview
===============

cz-benchmarks is designed with modularity and reproducibility in mind. Its core components include:

- **Datasets**:  
    Manage input data (AnnData objects, metadata) and ensure data integrity through type checking with custom DataType definitions. Images are supported in the future.
    See :doc:`datasets` for more details.


- **Tasks**:  
    Define evaluation operations such as clustering, embedding evaluation, label prediction, and perturbation assessment. Tasks extend the `BaseTask` class and serve as blueprints for benchmarking.  
    See :doc:`tasks` for more details.

- **Metrics**:  
    A central `MetricRegistry` handles the registration and computation of metrics, enabling consistent and reusable evaluation criteria.  
    See :doc:`metrics` for more details.


- **Configuration Management**:  
    Uses Hydra and OmegaConf to dynamically compose configurations for datasets.


Key Design Concepts
-------------------

- **Declarative Configuration:**  
  Use Hydra and OmegaConf to centralize and manage configuration for datasets.

- **Loose Coupling:**  
  Components communicate through well-defined interfaces. This minimizes dependencies and makes testing easier.

- **Validation and Type Safety:**  
  Custom type definitions in the datasets.



Class Diagrams
----------------


.. autoclasstree::  czbenchmarks.datasets 
   :name: class-diagram-datasets
   :alt: Class diagram for cz-benchmarks Datasets
   :zoom:


.. autoclasstree:: czbenchmarks.tasks czbenchmarks.tasks.single_cell
   :name: class-diagram-tasks
   :alt: Class diagram for cz-benchmarks Tasks
   :zoom:


.. autoclasstree:: czbenchmarks.metrics.implementations czbenchmarks.metrics.types
   :name: class-diagram
   :alt: Class diagram for cz-benchmarks Metrics
   :zoom:

