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
  - `perturbation`: Spearman correlation

## Supported Metrics

The following metrics are pre-registered:

| **Metric Type** | **Task** | **Description** |
|--------------------------|------------------|------------------------------------------------------------------------------------------------------------------|
| `adjusted_rand_index`    | clustering       | Measures the similarity between two clusterings, adjusted for chance. A higher value indicates better alignment. |
| `normalized_mutual_info` | clustering       | Quantifies the amount of shared information between two clusterings, normalized to ensure comparability.         |
| `silhouette_score`       | embedding        | Evaluates how well-separated clusters are in an embedding space. Higher scores indicate better-defined clusters. |
| `entropy_per_cell`       | integration      | Assesses the mixing of batch labels at the single-cell level. Higher entropy indicates better integration.       |
| `batch_silhouette`       | integration      | Combines silhouette scoring with batch information to evaluate clustering quality while accounting for batch effects. |
| `spearman_correlation`   | perturbation     | Rank correlation between predicted and actual values     |
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

## Related References

- [MetricRegistry API](../autoapi/czbenchmarks/metrics/types/index)
- [Add New Metric Guide](../how_to_guides/add_new_metric)
- [ClusteringTask](../autoapi/czbenchmarks/tasks/clustering/index)
- [PerturbationExpressionPredictionTask](../autoapi/czbenchmarks/tasks/single_cell/perturbation_expression_prediction/index)
