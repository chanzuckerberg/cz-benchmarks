# Tasks 

The `czbenchmarks.tasks` module defines **benchmarking tasks** that evaluate the performance of models based on their outputs. Tasks take in datasets with model-generated outputs and compute metrics specific to each task type.

## Core Concepts

- [BaseTask](../autoapi/czbenchmarks/tasks/base/index)  
   All task classes inherit from this abstract base class. It defines the standard lifecycle of a task:
   
   1. **Input and Output Validation** via `required_inputs` and `required_outputs`
   2. **Execution** via the `_run_task()` method
   3. **Metric Computation** via the `_compute_metrics()` method

   It also supports multi-dataset operations (`requires_multiple_datasets`) and setting **baseline embeddings** (via PCA) for comparison with model outputs.

## Task Organization

Tasks in the `czbenchmarks.tasks` module are organized based on their scope and applicability:

- **Generic Tasks**: Tasks that can be applied across multiple modalities (e.g., embedding evaluation, clustering, label prediction) are placed directly in the `tasks/` directory. Each task is implemented in its own file (e.g., `embedding.py`, `clustering.py`) with clear and descriptive names. Generic tasks avoid dependencies specific to any particular modality.

- **Specialized Tasks**: Tasks designed for specific modalities are placed in dedicated subdirectories. For example:  
   - `single_cell/` for single-cell-specific tasks
   - `imaging/` for imaging-related tasks. **Not implemented in current release, For future imaging models**

   New subdirectories can be created as needed for other modalities.

## Available Tasks

Each task class implements a specific evaluation goal. All tasks are located under the `czbenchmarks.tasks` namespace or its submodules.

- [ClusteringTask](../autoapi/czbenchmarks/tasks/clustering/index):  Performs Leiden clustering on the embedding produced by a model and compares it to ground-truth cell-type labels using metrics like Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI).

- [EmbeddingTask](../autoapi/czbenchmarks/tasks/embedding/index):  Computes embedding quality using the Silhouette Score based on known cell-type annotations.

- [MetadataLabelPredictionTask](../autoapi/czbenchmarks/tasks/label_prediction/index):  Performs k-fold cross-validation classification using multiple classifiers (logistic regression, KNN, random forest) on the model embeddings to predict metadata labels (e.g., cell type, sex). Evaluates metrics like accuracy, F1, precision, recall, and AUROC.

- [BatchIntegrationTask](../autoapi/czbenchmarks/tasks/integration/index):  Evaluates how well the model integrates batch-specific embeddings using entropy per cell and batch-aware Silhouette scores. Assesses whether embeddings mix batches while preserving biological labels.

- [PerturbationTask](../autoapi/czbenchmarks/tasks/single_cell/perturbation/index):  Designed for gene perturbation prediction models. Compares predicted gene expression shifts to ground truth. Computes metrics such as mean squared error, Pearson R², and Jaccard overlap for DE genes.

- [CrossSpeciesIntegrationTask](../autoapi/czbenchmarks/tasks/single_cell/cross_species/index):  A multi-dataset task. Evaluates how well models embed cells from different species into a shared space using metrics like entropy per cell and silhouette scores. Requires embeddings from multiple species as input.

## Extending Tasks

To define a new evaluation task:

1. **Inherit from** [BaseTask](../autoapi/czbenchmarks/tasks/base/index)


2. **Choose the Right Location**:  
    - If the task is generic and works across multiple modalities, add it to the `tasks/` directory.
    - If the task is specific to a particular modality, add it to the appropriate subdirectory (e.g., `single_cell/`, `imaging/`).


3. **Create the Task File**:  
    Each task should be implemented in its own file. Below is an example skeleton for creating a new task:


4. **Override the following methods:**

    - `required_inputs`: a set of `DataType` values required as inputs
    - `required_outputs`: a set of `DataType` values expected as model outputs
    - `_run_task(data, model_type)`: executes task logic using input data and model outputs
    - `_compute_metrics()`: returns a list of `MetricResult` objects


5. **Update `__init__.py`**:  
    - For generic tasks, add the new task to `tasks/__init__.py`.
    - For specialized tasks, add the new task to the `__init__.py` file in the corresponding modality-specific subdirectory.


6. **Documentation**:  
    - Add detailed docstrings to your task class and methods.
    - Update the relevant documentation files to include the new task.


7. **Optional Features**:  
    - Set `requires_multiple_datasets = True` if your task operates on a list of datasets
    - Call `self.compute_baseline(dataset)` in your task to enable PCA baseline comparisons
    - if your task is stochastic, use `self.random_seed` to seed sources of randomness for reproducibilty
      and to make it easier for users to estimate the variance of the task metrics due to randomness


8. **Return Metrics**:  
    - Use the [MetricRegistry](../autoapi/czbenchmarks/metrics/types/index) to compute and return standard metrics with strong typing.


9. **Example Skeleton**:  

    ```python
    from czbenchmarks.tasks.base import BaseTask
    from czbenchmarks.datasets import DataType
    from czbenchmarks.metrics.types import MetricResult

    class MyNewTask(BaseTask):
          @property
          def required_inputs(self):
                return {DataType.METADATA}

          @property
          def required_outputs(self):
                return {DataType.EMBEDDING}

          def _run_task(self, data, TODO):
                self.embedding = data.get_output(model_type, DataType.EMBEDDING)
                self.labels = data.get_input(DataType.METADATA)["cell_type"]

          def _compute_metrics(self):
                result = ...  # your metric computation here
                return [MetricResult(metric_type="my_metric", value=result)]
    ```

## Best Practices  

- Keep tasks focused and single-purpose to ensure clarity and maintainability.
- Clearly document the input and output requirements for each task.
- Follow the patterns and conventions established in existing tasks for consistency.
- Use type hints to improve code readability and clarity.
- Add logging for key steps in the task lifecycle to facilitate debugging and monitoring.
