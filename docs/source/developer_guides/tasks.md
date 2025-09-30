# Tasks


The `czbenchmarks.tasks` module defines **benchmarking tasks** that evaluate the performance of models based on their outputs. Tasks take a model's output (e.g., a cell embedding) and task-specific inputs (e.g., ground-truth labels from a dataset) and compute relevant evaluation metrics.


## Usage

Tasks are used to evaluate model outputs in a standardized way. Each task takes a model's output (such as a cell embedding) and task-specific inputs, and computes relevant evaluation metrics.

### Core Concepts

- **[`Task`](../autoapi/czbenchmarks/tasks/task/index)**: The abstract base class for all tasks. It defines the standard lifecycle:
    1. **Execution** via the `_run_task()` method, which performs the core computation.
    2. **Metric Computation** via the `_compute_metrics()` method.
    3. Multi-dataset operations via the `requires_multiple_datasets` method.
    4. Baseline embeddings via the `compute_baseline` method.
- **[`TaskInput`](../autoapi/czbenchmarks/tasks/task/index)** and **[`TaskOutput`](../autoapi/czbenchmarks/tasks/task/index)**: Pydantic base classes used to define structured inputs and outputs for each task, ensuring type safety and clarity.

### Task Organization
Tasks in the `czbenchmarks.tasks` module are organized based on their scope and applicability:

- **Generic Tasks**: Tasks that can be applied across multiple modalities (e.g., embedding evaluation, clustering, label prediction) are placed directly in the `tasks/` directory. Each task is implemented in its own file (e.g., `embedding.py`, `clustering.py`).
- **Specialized Tasks**: Tasks designed for specific modalities are placed in dedicated subdirectories (e.g., `single_cell/`). For example:

    - `single_cell/` for single-cell-specific tasks like perturbation prediction or cross-species integration.

    New subdirectories can be created as needed for other modalities.

### Available Tasks

Each task class implements a specific evaluation goal. All tasks are located under the `czbenchmarks.tasks` namespace or its submodules.

- [`EmbeddingTask`](../autoapi/czbenchmarks/tasks/embedding/index): Computes embedding quality using the Silhouette Score based on known cell-type annotations.
- [`ClusteringTask`](../autoapi/czbenchmarks/tasks/clustering/index): Performs Leiden clustering on an embedding and compares it to ground-truth labels using metrics like Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI).
- [`MetadataLabelPredictionTask`](../autoapi/czbenchmarks/tasks/label_prediction/index): Performs k-fold cross-validation using multiple classifiers (logistic regression, KNN, random forest) on model embeddings to predict metadata labels. Evaluates metrics like accuracy, F1, precision, recall, and AUROC.
- [`BatchIntegrationTask`](../autoapi/czbenchmarks/tasks/integration/index): Evaluates how well a model integrates data from different batches using entropy per cell and batch-aware Silhouette scores.
- [`CrossSpeciesIntegrationTask`](../autoapi/czbenchmarks/tasks/single_cell/cross_species/index): A multi-dataset task that evaluates how well models embed cells from different species into a shared space, using metrics like entropy per cell and species-aware silhouette scores.
- [`CrossSpeciesLabelPredictionTask`](../autoapi/czbenchmarks/tasks/single_cell/cross_species_label_prediction/index): A multi-dataset task that evaluates how well model embeddings can be used to determine cell properties across species.
- [`PerturbationExpressionPredictionTask`](../autoapi/czbenchmarks/tasks/single_cell/perturbation_expression_prediction/index): Designed for perturbation models. Compares the model's ability to predict masked gene expression levels relative to ground truth using metrics like Spearman correlation, accuracy, F1, precision, and recall.

- [`SequentialOrganizationTask`](../autoapi/czbenchmarks/tasks/sequential/index): Evaluates sequential consistency in embeddings using time point labels. Computes metrics like silhouette score and sequential alignment to assess how well embeddings preserve sequential organization between cells.

For instructions on **adding a new custom task**, see [How to Add a Custom Task](../how_to_guides/add_new_task.md).

