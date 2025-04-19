Design Overview
===============

CZ Benchmarks is designed with modularity and reproducibility in mind. Its core components include:

- **Datasets**:  
    Manage input data (AnnData objects, metadata) and ensure data integrity through type checking with custom DataType definitions. Images are supported in the future.
    See :doc:`datasets` for more details.

- **Models**:  
    Encapsulated in Docker containers, models implement the `BaseModelImplementation` interface and are validated by corresponding validator classes.  
    See :doc:`models` for more details.

- **Tasks**:  
    Define evaluation operations such as clustering, embedding evaluation, label prediction, and perturbation assessment. Tasks extend the `BaseTask` class and serve as blueprints for benchmarking.  
    See :doc:`tasks` for more details.

- **Metrics**:  
    A central `MetricRegistry` handles the registration and computation of metrics, enabling consistent and reusable evaluation criteria.  
    See :doc:`metrics` for more details.

- **Runner**:  
    Orchestrates the workflow by handling containerized execution, automatic serialization, and seamless integration of datasets, models, and tasks.

- **Configuration Management**:  
    Uses Hydra and OmegaConf to dynamically compose configurations for datasets, models, and tasks.


Key Design Concepts
-------------------

- **Declarative Configuration:**  
  Use Hydra and OmegaConf to centralize and manage configuration for datasets, models, and tasks.

- **Loose Coupling:**  
  Components communicate through well-defined interfaces. This minimizes dependencies and makes testing easier.

- **Validation and Type Safety:**  
  Custom type definitions in the datasets and validators enforce that the data and model outputs meet expected standards.


.. raw:: html

    <br><br>


.. Flowchart
.. ----------

.. .. graphviz::
..    :class: .diagram-font

..         digraph flowchart {
..             rankdir=LR;
..             node [shape=box];
..             "User Input" -> "Hydra Config";
..             "Hydra Config" -> "Dataset Loader";
..             "Dataset Loader" -> "Container Runner";
..             "Container Runner" -> "Model Docker Image";
..             "Model Docker Image" -> "Model Outputs";
..             "Model Outputs" -> "Task";
..             "Task" -> "Metric";
..         }


Class Diagrams
----------------

.. raw:: html

    <br><br><br>


.. container:: class-diagram-container 

    .. inheritance-diagram::  czbenchmarks.datasets.types czbenchmarks.datasets.base czbenchmarks.datasets.single_cell czbenchmarks.models.types czbenchmarks.models.implementations.base_model_implementation czbenchmarks.models.validators.base_model_validator czbenchmarks.models.validators.base_single_cell_model_validator czbenchmarks.tasks.base czbenchmarks.tasks.clustering czbenchmarks.tasks.embedding czbenchmarks.tasks.integration czbenchmarks.tasks.label_prediction czbenchmarks.tasks.single_cell.cross_species czbenchmarks.tasks.single_cell.perturbation czbenchmarks.metrics.types czbenchmarks.metrics.implementations 
        :parts: -1