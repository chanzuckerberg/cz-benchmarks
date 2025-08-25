# Metrics

The `czbenchmarks.metrics` module provides a unified and extensible framework for computing performance metrics across all evaluation tasks.

## Overview

At the core of this module is a centralized registry, `MetricRegistry`, which stores all supported metrics. Each metric is registered with a unique type, required arguments, default parameters, a description, and a set of descriptive tags.

### Purpose

- Allows tasks to declare and compute metrics in a unified, type-safe, and extensible manner.
- Ensures metrics are reproducible and callable via shared interfaces across tasks like clustering, embedding, and label prediction.

## Key Components

- [MetricRegistry](../autoapi/czbenchmarks/metrics/types/index)  
  A class that registers and manages metric functions, performs argument validation, and handles invocation.

- [MetricType](../autoapi/czbenchmarks/metrics/types/index)  
  An `Enum` defining all supported metric names. Each task refers to `MetricType` members to identify which metrics to compute.

- **Tags:**  
  Each metric is tagged with its associated category to allow filtering:

  - `clustering`: ARI, NMI
  - `embedding`: Silhouette Score
  - `integration`: Entropy per Cell, Batch Silhouette
  - `label_prediction`: Accuracy, F1, Precision, Recall, AUROC
  - `perturbation`: Spearman correlation, Accuracy, F1, Precision, Recall

## Supported Metrics

The following metrics are pre-registered:

| **Metric Type**          | **Task**         | **Description**                                                                                                  |
|--------------------------|------------------|------------------------------------------------------------------------------------------------------------------|
| `adjusted_rand_index`    | clustering       | Measures the similarity between two clusterings, adjusted for chance. A higher value indicates better alignment. |
| `normalized_mutual_info` | clustering       | Quantifies the amount of shared information between two clusterings, normalized to ensure comparability.         |
| `silhouette_score`       | embedding        | Evaluates how well-separated clusters are in an embedding space. Higher scores indicate better-defined clusters. |
| `entropy_per_cell`       | integration      | Assesses the mixing of batch labels at the single-cell level. Higher entropy indicates better integration.       |
| `batch_silhouette`       | integration      | Combines silhouette scoring with batch information to evaluate clustering quality while accounting for batch effects. |
| `spearman_correlation`   | perturbation     | Rank correlation between predicted and actual values     |
| `accuracy_score`         | perturbation     | Accuracy of binary classification predictions against ground truth labels     |
| `f1_score`               | perturbation     | F1-score balancing precision and recall for binary classification tasks     |
| `precision_score`        | perturbation     | Precision measuring the fraction of true positives among predicted positives     |
| `recall_score`           | perturbation     | Recall measuring the fraction of true positives correctly identified     |
| `mean_fold_accuracy`     | label_prediction | Average accuracy across k-fold cross-validation splits, indicating overall classification performance.           |
| `mean_fold_f1`           | label_prediction | Average F1 score across folds, balancing precision and recall for classification tasks.                          |
| `mean_fold_precision`    | label_prediction | Average precision across folds, reflecting the proportion of true positives among predicted positives.           |
| `mean_fold_recall`       | label_prediction | Average recall across folds, indicating the proportion of true positives correctly identified.                   |
| `mean_fold_auroc`        | label_prediction | Average area under the ROC curve across folds, measuring the ability to distinguish between classes.             |

## How to Compute a Metric

Use `metrics_registry.compute()` inside your task's `_compute_metrics()` method:

```python
from czbenchmarks.metrics.types import MetricType, metrics_registry

value = metrics_registry.compute(
    MetricType.ADJUSTED_RAND_INDEX,
    labels_true=true_labels,
    labels_pred=predicted_labels,
)

# Wrap in a result object
from czbenchmarks.metrics.types import MetricResult
result = MetricResult(metric_type=MetricType.ADJUSTED_RAND_INDEX, value=value)
```

## Adding a Custom Metric

To add a new metric to the registry:

1. **Add a new member to the enum:**  
   Edit `MetricType` in `czbenchmarks/metrics/types.py`:

   ```python
   class MetricType(Enum):
       ...
       MY_CUSTOM_METRIC = "my_custom_metric"
   ```

2. **Define the metric function:**

   ```python
   def my_custom_metric(y_true, y_pred):
       # return a float value
       return float(...)
   ```

3. **Register it in the registry:**  
   Add to `czbenchmarks/metrics/implementations.py`:

   ```python
   metrics_registry.register(
       MetricType.MY_CUSTOM_METRIC,
       func=my_custom_metric,
       required_args={"y_true", "y_pred"},
       default_params={"normalize": True},
       description="Description of your custom metric",
       tags={"my_category"},
   )
   ```

4. **Use in your task:**  
   Now the metric is available for any task to compute.

## Using Metric Tags

You can list metrics by category using tags:

```python
metrics_registry.list_metrics(tags={"clustering"})  # returns a set of MetricType
```

## Best Practices

When implementing or using metrics, follow these guidelines to ensure consistency and reliability:

1. **Type Safety:**  Always use the `MetricType` enum instead of string literals to refer to metrics. This ensures type safety and avoids errors due to typos.

2. **Pure Functions:**  Metrics should be **pure functions**, meaning they must not have side effects. This ensures reproducibility and consistency across computations.

3. **Return Types:**  All metric functions must return a `float` value to maintain uniformity in results.

4. **Validation:**  
   - Validate inputs manually within your metric function if there are strict assumptions about input shapes or types.
   - Include required argument validation to ensure the metric function is called with the correct parameters.

5. **Default Parameters:**  Use `default_params` only for optional keyword arguments. Avoid using them for required arguments.

6. **Tags:**  Assign appropriate tags to metrics for categorization. Tags help in filtering and organizing metrics by their use cases (e.g., `clustering`, `embedding`, `label_prediction`).

7. **Documentation:**  
   - Provide a short and clear `description` for each metric to explain its purpose and usage.
   - Document all parameters and their expected types or shapes to guide users effectively.

## Related References

- [MetricRegistry API](../autoapi/czbenchmarks/metrics/types/index)
- [Add New Metric Guide](../how_to_guides/add_new_metric)
- [ClusteringTask](../autoapi/czbenchmarks/tasks/clustering/index)
- [PerturbationTask](../autoapi/czbenchmarks/tasks/single_cell/perturbation/index)
