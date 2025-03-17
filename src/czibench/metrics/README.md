# Metrics Package

This package provides a centralized registry for computing and managing evaluation metrics across the benchmark.

## Overview

The metrics package implements a type-safe registry system for computing various evaluation metrics. It supports:

- Clustering metrics (ARI, NMI)
- Embedding quality metrics (Silhouette score)
- Integration metrics (Entropy per cell, Batch silhouette)
- Cross-validation metrics (Accuracy, F1, Precision, Recall)

## Usage

```python
from czibench.metrics import MetricType, metrics

# Compute a clustering metric
ari_score = metrics.compute(
    MetricType.ADJUSTED_RAND_INDEX,
    labels_true=true_labels,
    labels_pred=predicted_labels
)

# Compute a mean fold metric across all classifiers
mean_accuracy = metrics.compute(
    MetricType.MEAN_FOLD_ACCURACY,
    results_df=results_df
)

# Compute a mean fold metric for a specific classifier
lr_accuracy = metrics.compute(
    MetricType.MEAN_FOLD_ACCURACY,
    results_df=results_df,
    classifier="lr"
)
```

## Adding New Metrics

1. **Define the Metric Type**:
```python
# In types.py
class MetricType(Enum):
    YOUR_NEW_METRIC = "your_new_metric"
```

2. **Register the Metric**:
```python
# In implementations.py
metrics.register(
    MetricType.YOUR_NEW_METRIC,
    func=your_metric_function,
    required_args={"arg1", "arg2"},
    default_params={"param1": "default"},
    description="Description of your metric",
    tags={"category"}
)
```

## Filtering Metrics by Tags

Metrics can be filtered by their associated tags to find related metrics:

```python
# Get all clustering metrics
clustering_metrics = metrics.list_metrics(tags={"clustering"})
# Returns: {MetricType.ADJUSTED_RAND_INDEX, MetricType.NORMALIZED_MUTUAL_INFO}

# Get all integration metrics
integration_metrics = metrics.list_metrics(tags={"integration"})
# Returns: {MetricType.ENTROPY_PER_CELL, MetricType.BATCH_SILHOUETTE}

# Get all metrics (no filtering)
all_metrics = metrics.list_metrics()
```

Common tag categories:
- `clustering`: Clustering quality metrics
- `embedding`: Embedding quality metrics
- `integration`: Integration quality metrics
- `label_prediction`: Cross-validation metrics

## Metric Categories

### Clustering Metrics
- `ADJUSTED_RAND_INDEX`: Measures similarity between two clusterings
- `NORMALIZED_MUTUAL_INFO`: Information-theoretic measure of clustering quality

### Embedding Quality Metrics
- `SILHOUETTE_SCORE`: Measures how well-separated clusters are in embedding space

### Integration Metrics
- `ENTROPY_PER_CELL`: Measures batch mixing quality
- `BATCH_SILHOUETTE`: Batch-aware clustering quality measure

### Cross-validation Metrics
- `MEAN_FOLD_ACCURACY`: Mean accuracy across cross-validation folds
- `MEAN_FOLD_F1_SCORE`: Mean F1 score across folds
- `MEAN_FOLD_PRECISION`: Mean precision across folds
- `MEAN_FOLD_RECALL`: Mean recall across folds

## Best Practices

1. **Type Safety**: Always use `MetricType` enum instead of string literals
2. **Documentation**: Provide clear descriptions and parameter documentation
3. **Validation**: Include required argument validation
4. **Tags**: Use appropriate tags for metric categorization