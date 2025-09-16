Design and Architecture Overview
================================

cz-benchmarks is built for modularity and reproducibility. This guide gives you a clear, high-level overview of the `cz-benchmarks` architecture. By understanding these core ideas, you’ll be able to use and extend the package more effectively.


Key Design Concepts
-------------------

- **Declarative Configuration:**  
  Use Hydra and OmegaConf to centralize and manage configuration for datasets.

- **Loose Coupling:**  
  Components communicate through well-defined interfaces. This minimizes dependencies and makes testing easier.

- **Validation and Type Safety:**  
  Custom type definitions in the datasets.


At its heart, the framework follows a simple principle: **separating data from evaluation**. `Dataset` objects handle and standardize biological data, making sure it’s ready for analysis. `Task` objects then run evaluations on the outputs from your models, letting you focus on your results.

Its core components include:

- **Datasets**:  
    Handle input data such as AnnData objects and metadata, making sure the data is correct and ready to use by checking types with custom DataType definitions. The Dataset component is responsible for loading, validating, and giving easy access to standardized biological data (for example, from an `.h5ad` file). It takes care of reading different data formats and checks that the data has everything needed for evaluation, like the right gene names and required metadata. Support for images will be added in the future.
    See :doc:`datasets` for more details.

- **CellRepresentation**:  
    Represents the output from your model (for example, a cell embedding as a `np.ndarray`). The framework follows a "bring-your-own-model" approach: you run your model independently and provide the resulting `CellRepresentation` to the evaluation `Task`.

- **TaskInput** and **TaskOutput**: 
    These are Pydantic models that ensure type-safe data transfer. ``TaskInput`` bundles the necessary information from a ``Dataset`` (like ground-truth labels), while ``TaskOutput`` structures the results from a task's internal computation before metrics are calculated.

- **Tasks**:  
    Define the different types of evaluations you can run, such as clustering, embedding quality, label prediction, and perturbation analysis. Think of Tasks as the "evaluation engine" of the framework. Each `Task` (like `ClusteringTask` or `PerturbationTask`) contains all the logic needed to run a specific biological benchmark. You give a Task your model’s `CellRepresentation`, and it computes the relevant performance metrics for you.  
    Tasks are built by extending the base `Task` class, making it easy to create new types of evaluations or customize existing ones.  
    See :doc:`tasks` for more details.

- **Metrics**:  
    A central `MetricRegistry` handles the registration and computation of metrics, enabling consistent and reusable evaluation criteria.  
    See :doc:`metrics` for more details.

- **MetricRegistry** and **MetricResult**: 
    The registry provides a centralized way to compute metrics (`ADJUSTED_RAND_INDEX`, `MEAN_SQUARED_ERROR`, etc.). All tasks use this registry to produce a standardized list of `MetricResult` objects.    

- **Configuration Management**:  
    Uses Hydra and OmegaConf to dynamically compose configurations for datasets.


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



The Standard Workflow
---------------------

A typical benchmarking workflow follows these steps:

1. **Load Dataset**:  
    Use ``dataset = load_dataset(...)`` to load a dataset. This gives you a ``Dataset`` object with loaded data (e.g., ``dataset.adata``) and relevant metadata (e.g., ``dataset.labels``).

2. **User Generates Model Output**:  
    Run your own ML model using the data from the ``Dataset`` object (e.g., ``dataset.adata.X``) to produce a ``CellRepresentation`` (such as a cell embedding). For example: ``embedding = my_model(dataset.adata)``. This step happens outside the ``cz-benchmarks`` package.

3. **Prepare Task Inputs**:
    Create an instance of the task-specific ``TaskInput`` class, populating it with the necessary ground-truth data from the ``Dataset`` object. For example: ``task_input = TaskInput(labels=dataset.labels)``.

4. **Instantiate and Run Task**:
    Instantiate the desired ``Task`` and call its ``.run()`` method, passing your ``CellRepresentation`` and the prepared ``TaskInput``. For example: ``results = task.run(embedding, task_input)``.

5. **Analyze Results**:
    The task returns a list of ``MetricResult`` objects, which you can then analyze, plot, or save.

.. raw:: html

   <div class="mermaid">
   graph TD
     A[Load Dataset] --> B[User Generates Model Output]
     B --> C[Prepare Task Inputs]
     C --> D[Instantiate and Run Task]
     D --> E[Analyze Results]
   </div>


This modular design allows you to evaluate any model on any compatible dataset using a standardized and reproducible set of tasks and metrics.



