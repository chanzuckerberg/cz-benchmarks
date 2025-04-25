# Add a New Metric

This guide explains how to add a new metric to the CZ Benchmarks metrics system. Follow the steps below to implement and register your custom metric.

---

## Step-by-Step Guide

### 1. Define a New Metric Type
First, define your new metric type in the `MetricType` enum.

1. Open the file: `czbenchmarks/metrics/types.py`.
2. Add a new entry to the `MetricType` enum. For example:
    ```python
    from enum import Enum

    class MetricType(Enum):
         ...
         YOUR_NEW_METRIC = "your_new_metric"
    ```

---

### 2. Implement the Metric Function
Next, register a function that computes the value of your metric.

1. Define the function in implementation :doc:`../autoapi/czbenchmarks/metrics/index`. class. For example:
    ```python
    def your_new_metric_func(y_true, y_pred):
         """
         Computes the value of your new metric.

         Args:
              y_true (list): Ground truth values.
              y_pred (list): Predicted values.

         Returns:
              float: Computed metric value.
         """
         # Replace with your metric calculation logic
         return float(...)
    ```

2. Ensure the function:
    - Accepts required arguments (e.g., `y_true`, `y_pred`).
    - Returns a `float` value representing the metric.

---

### 3. Register the Metric
Register your metric with the `MetricRegistry` to make it available for use.

1. Open the file: `czbenchmarks/metrics/implementations.py`.
2. Add the registration code:
    ```python
    from czbenchmarks.metrics.registry import metrics_registry
    from czbenchmarks.metrics.types import MetricType

    metrics_registry.register(
         MetricType.YOUR_NEW_METRIC,
         func=your_new_metric_func,
         required_args={"y_true", "y_pred"},
         default_params={},  # Add any default parameters if needed
         description="A brief description of your new metric.",
         tags={"custom"}  # Add relevant tags for categorization
    )
    ```

---

### 4. Use the Metric
Once registered, you can compute your metric in czbenchmarks/tasks module in the codebase.

1. Import the necessary components:
    ```python
    from czbenchmarks.metrics.types import MetricType, metrics_registry
    ```

2. Compute the metric:
    ```python
    value = metrics_registry.compute(
         MetricType.YOUR_NEW_METRIC,
         y_true=true_values,
         y_pred=pred_values
    )
    ```

---

## Best Practices for Adding Metrics
- **Documentation:** Provide clear and concise documentation for your metric, including its purpose and usage.
- **Tags:** Use descriptive tags to categorize your metric (e.g., `{"custom", "classification"}`).
- **Error Handling:** Ensure your metric function handles edge cases gracefully and raises meaningful errors.
- **Testing:** Write unit tests to validate the correctness of your metric implementation.
