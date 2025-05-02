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
  - `perturbation`: MSE, R², Jaccard Similarity

## Supported Metrics

The following metrics are pre-registered:

| **Metric Type**          | **Tags**         | **Description**                                                                 |
|--------------------------|------------------|---------------------------------------------------------------------------------|
| `adjusted_rand_index`    | clustering       | Adjusted Rand Index (ARI) between predicted and true labels                     |
| `normalized_mutual_info` | clustering       | Normalized Mutual Information (NMI)                                             |
| `silhouette_score`       | embedding        | Silhouette score of clusters in embedding space                                 |
| `entropy_per_cell`       | integration      | Local entropy of batch labels (higher is better mixing)                         |
| `batch_silhouette`       | integration      | Silhouette score that accounts for batch mixing                                 |
| `mean_squared_error`     | perturbation     | Mean squared error between predicted and ground truth expressions               |
| `r2_score`               | perturbation     | Pearson R² between predicted and true gene deltas                               |
| `jaccard`                | perturbation     | Jaccard similarity between predicted and true top DE genes                      |
| `mean_fold_accuracy`     | label_prediction | Mean accuracy across k-fold cross-validation splits                             |
| `mean_fold_f1`           | label_prediction | Mean F1 score across folds                                                      |
| `mean_fold_precision`    | label_prediction | Mean precision across folds                                                     |
| `mean_fold_recall`       | label_prediction | Mean recall across folds                                                        |
| `mean_fold_auroc`        | label_prediction | Area under ROC curve, averaged across folds                                     |

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

4. **Use in your task or CLI:**

   Now the metric is available for any task to compute.

## Using Metric Tags

You can list metrics by category using tags:

```python
metrics_registry.list_metrics(tags={"clustering"})  # returns a set of MetricType
```

## Developer Tips

- Metrics should be **pure functions** (i.e., no side effects)
- Return types must always be `float`
- Use `default_params` only for optional kwargs
- Validate inputs manually in your metric if shape or type assumptions are strict
- Document your metric with a short `description`

## Related References

- [MetricRegistry API](../autoapi/czbenchmarks/metrics/types/index)
- [Add New Metric Guide](../how_to_guides/add_new_metric)
- [ClusteringTask](../autoapi/czbenchmarks/tasks/clustering/index)
- [PerturbationTask](../autoapi/czbenchmarks/tasks/single_cell/perturbation/index)


