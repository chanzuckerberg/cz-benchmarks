import pytest
import numpy as np
from enum import Enum
from czbenchmarks.metrics.types import MetricType, MetricResult
from czbenchmarks.metrics.utils import aggregate_results
from czbenchmarks.metrics import metrics_registry


def test_register_metric_valid(dummy_metric_registry, dummy_metric_function):
    """Test that registering a metric works with valid arguments."""
    try:
        dummy_metric_registry.register(
            MetricType.ADJUSTED_RAND_INDEX,
            func=dummy_metric_function,
            required_args={"x", "y"},
            description="Test metric",
            tags={"test"},
        )

        # Verify registration
        info = dummy_metric_registry.get_info(MetricType.ADJUSTED_RAND_INDEX)
        assert info.func == dummy_metric_function
        assert info.required_args == {"x", "y"}
        assert info.description == "Test metric"
        assert info.tags == {"test"}
    except Exception as e:
        pytest.fail(f"Metric registration failed unexpectedly: {e}")


def test_compute_metric_valid(
    dummy_metric_registry, dummy_metric_function, sample_data
):
    """Test that computing a metric works with valid arguments."""
    dummy_metric_registry.register(
        MetricType.ADJUSTED_RAND_INDEX,
        func=dummy_metric_function,
        required_args={"x", "y"},
    )

    try:
        result = dummy_metric_registry.compute(
            MetricType.ADJUSTED_RAND_INDEX, x=sample_data["X"], y=sample_data["y_true"]
        )
        assert isinstance(result, float)
        assert result == 0.5  # Expected return value from dummy_metric_function
    except Exception as e:
        pytest.fail(f"Metric computation failed unexpectedly: {e}")


def test_register_metric_invalid_type(dummy_metric_registry, dummy_metric_function):
    """Test that registering a metric with invalid MetricType fails."""

    class InvalidMetricType(Enum):
        INVALID = "invalid"

    with pytest.raises(TypeError):
        dummy_metric_registry.register(
            InvalidMetricType.INVALID,
            func=dummy_metric_function,
            required_args={"x", "y"},
        )


def test_compute_metric_missing_args(dummy_metric_registry, dummy_metric_function):
    """Test that computing a metric with missing required arguments fails."""
    dummy_metric_registry.register(
        MetricType.ADJUSTED_RAND_INDEX,
        func=dummy_metric_function,
        required_args={"x", "y"},
    )

    with pytest.raises(ValueError, match="Missing required arguments"):
        dummy_metric_registry.compute(
            MetricType.ADJUSTED_RAND_INDEX,
            x=np.array([1, 2, 3]),  # Missing 'y' argument
        )


def test_compute_metric_invalid_type(dummy_metric_registry):
    """Test that computing a metric with invalid MetricType fails."""
    with pytest.raises(ValueError, match="Unknown metric type"):
        dummy_metric_registry.compute(
            "not_a_metric_type", x=np.array([1, 2, 3]), y=np.array([1, 2, 3])
        )


def test_list_metrics_with_tags(dummy_metric_registry, dummy_metric_function):
    """Test that listing metrics with tags works correctly."""
    # Register metrics with different tags
    dummy_metric_registry.register(
        MetricType.ADJUSTED_RAND_INDEX,
        func=dummy_metric_function,
        tags={"clustering", "test"},
    )
    dummy_metric_registry.register(
        MetricType.SILHOUETTE_SCORE,
        func=dummy_metric_function,
        tags={"embedding", "test"},
    )

    # Test filtering by tags
    clustering_metrics = dummy_metric_registry.list_metrics(tags={"clustering"})
    assert MetricType.ADJUSTED_RAND_INDEX in clustering_metrics
    assert MetricType.SILHOUETTE_SCORE not in clustering_metrics

    test_metrics = dummy_metric_registry.list_metrics(tags={"test"})
    assert MetricType.ADJUSTED_RAND_INDEX in test_metrics
    assert MetricType.SILHOUETTE_SCORE in test_metrics


def test_metric_default_params(dummy_metric_registry, dummy_metric_function):
    """Test that default parameters are properly handled."""
    default_params = {"metric": "euclidean", "random_state": 42}
    dummy_metric_registry.register(
        MetricType.SILHOUETTE_SCORE,
        func=dummy_metric_function,
        required_args={"x", "y"},
        default_params=default_params,
    )

    info = dummy_metric_registry.get_info(MetricType.SILHOUETTE_SCORE)
    assert info.default_params == default_params


def test_aggregate_results():
    results = [
        # aggregrate group 1
        MetricResult(
            metric_type=MetricType.SILHOUETTE_SCORE, params={"foo": 1}, value=0.30
        ),
        MetricResult(
            metric_type=MetricType.SILHOUETTE_SCORE, params={"foo": 1}, value=0.50
        ),
        # aggregrate group 2
        MetricResult(
            metric_type=MetricType.SILHOUETTE_SCORE, params={"foo": 2}, value=0.60
        ),
        MetricResult(
            metric_type=MetricType.SILHOUETTE_SCORE, params={"foo": 2}, value=0.80
        ),
        # aggregrate group 3
        MetricResult(metric_type=MetricType.ADJUSTED_RAND_INDEX, value=0.10),
        MetricResult(metric_type=MetricType.ADJUSTED_RAND_INDEX, params={}, value=0.90),
    ]
    agg_results = aggregate_results(results)
    assert len(agg_results) == 3

    # there should be two silhoutte score aggregated results since there are two different params
    try:
        ss_foo1_result = next(
            r
            for r in agg_results
            if r.metric_type == MetricType.SILHOUETTE_SCORE and r.params == {"foo": 1}
        )
    except StopIteration:
        pytest.fail(
            "No aggregated result for MetricType.SILHOUETTE_SCORE with params {'foo': 1}"
        )
    assert ss_foo1_result.value == pytest.approx(0.40, abs=1e-2)
    assert ss_foo1_result.value_std_dev == pytest.approx(0.1414, abs=1e-4)
    assert ss_foo1_result.values_raw == [0.30, 0.50]
    assert ss_foo1_result.n_values == 2

    try:
        ss_foo2_result = next(
            r
            for r in agg_results
            if r.metric_type == MetricType.SILHOUETTE_SCORE and r.params == {"foo": 2}
        )
    except StopIteration:
        pytest.fail(
            "No aggregated result for MetricType.SILHOUETTE_SCORE with params {'foo': 2}"
        )
    assert ss_foo2_result.value == pytest.approx(0.70, abs=1e-2)
    assert ss_foo2_result.value_std_dev == pytest.approx(0.1414, abs=1e-4)
    assert ss_foo2_result.values_raw == [0.60, 0.80]
    assert ss_foo2_result.n_values == 2

    # both should be aggregated together since params is empty
    try:
        ari_result = next(
            r for r in agg_results if r.metric_type == MetricType.ADJUSTED_RAND_INDEX
        )
    except StopIteration:
        pytest.fail("No aggregated result for MetricType.ADJUSTED_RAND_INDEX")
    assert not ari_result.params
    assert ari_result.value == pytest.approx(0.50, abs=1e-2)
    assert ari_result.value_std_dev == pytest.approx(0.5657, abs=1e-4)
    assert ari_result.values_raw == [0.10, 0.90]
    assert ari_result.n_values == 2


def test_aggregate_results_works_on_empty():
    assert aggregate_results([]) == []


def test_aggregate_results_handles_just_one():
    results = [
        MetricResult(
            metric_type=MetricType.SILHOUETTE_SCORE, params={"foo": 2}, value=0.42
        )
    ]
    agg_results = aggregate_results(results)

    assert len(agg_results) == 1
    agg_result = agg_results[0]

    assert agg_result.metric_type == MetricType.SILHOUETTE_SCORE
    assert agg_result.params == {"foo": 2}
    assert agg_result.value == pytest.approx(0.42)
    assert agg_result.value_std_dev is None
    assert agg_result.values_raw == [0.42]
    assert agg_result.n_values == 1


# Tests for newly registered metrics


def test_accuracy_metric():
    """Test that ACCURACY metric is registered and computes correctly."""
    # Test perfect accuracy
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 1, 0, 0, 1])

    accuracy = metrics_registry.compute(
        MetricType.ACCURACY_CALCULATION,
        y_true=y_true,
        y_pred=y_pred,
    )

    assert accuracy == 1.0

    # Test partial accuracy
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])  # 3/4 correct

    accuracy = metrics_registry.compute(
        MetricType.ACCURACY_CALCULATION,
        y_true=y_true,
        y_pred=y_pred,
    )

    assert accuracy == 0.75


def test_precision_metric():
    """Test that PRECISION metric is registered and computes correctly."""
    # True positives: 2, False positives: 1, Precision = 2/(2+1) = 0.67
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array(
        [1, 1, 1, 0, 0]
    )  # TP: positions 0,1; FP: position 2; FN: position 4

    precision = metrics_registry.compute(
        MetricType.PRECISION_CALCULATION,
        y_true=y_true,
        y_pred=y_pred,
    )

    assert precision == pytest.approx(0.6667, abs=1e-4)


def test_recall_metric():
    """Test that RECALL metric is registered and computes correctly."""
    # True positives: 2, False negatives: 1, Recall = 2/(2+1) = 0.67
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 1, 1, 0, 0])  # TP: positions 0,1; FN: position 4

    recall = metrics_registry.compute(
        MetricType.RECALL_CALCULATION,
        y_true=y_true,
        y_pred=y_pred,
    )

    assert recall == pytest.approx(0.6667, abs=1e-4)


def test_f1_metric():
    """Test that F1 metric is registered and computes correctly."""
    # F1 = 2 * (precision * recall) / (precision + recall)
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 1, 1, 0, 0])

    f1 = metrics_registry.compute(
        MetricType.F1_CALCULATION,
        y_true=y_true,
        y_pred=y_pred,
    )

    # With precision = recall = 2/3, F1 = 2/3
    assert f1 == pytest.approx(0.6667, abs=1e-4)


def test_spearman_correlation_metric():
    """Test that SPEARMAN_CORRELATION metric is registered and computes correctly."""
    # Perfect positive correlation
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([2, 4, 6, 8, 10])

    correlation = metrics_registry.compute(
        MetricType.SPEARMAN_CORRELATION_CALCULATION,
        a=a,
        b=b,
    )

    # Spearman correlation should be 1.0 for perfect monotonic relationship
    assert isinstance(correlation, (float, np.floating))
    assert correlation == pytest.approx(1.0, abs=1e-10)

    # Perfect negative correlation
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([5, 4, 3, 2, 1])

    correlation = metrics_registry.compute(
        MetricType.SPEARMAN_CORRELATION_CALCULATION,
        a=a,
        b=b,
    )

    assert isinstance(correlation, (float, np.floating))
    assert correlation == pytest.approx(-1.0, abs=1e-10)


def test_metric_registration_tags():
    """Test that new metrics are registered with correct tags."""
    # Test that accuracy is tagged as label_prediction
    accuracy_metrics = metrics_registry.list_metrics(tags={"label_prediction"})

    expected_metrics = {
        MetricType.ACCURACY_CALCULATION,
        MetricType.PRECISION_CALCULATION,
        MetricType.RECALL_CALCULATION,
        MetricType.F1_CALCULATION,
        MetricType.SPEARMAN_CORRELATION_CALCULATION,
    }

    for metric in expected_metrics:
        assert (
            metric in accuracy_metrics
        ), f"{metric} should be tagged as 'label_prediction'"


def test_metric_required_args():
    """Test that new metrics have correct required arguments."""
    # Test accuracy required args
    accuracy_info = metrics_registry.get_info(MetricType.ACCURACY_CALCULATION)
    assert accuracy_info.required_args == {"y_true", "y_pred"}

    # Test precision required args
    precision_info = metrics_registry.get_info(MetricType.PRECISION_CALCULATION)
    assert precision_info.required_args == {"y_true", "y_pred"}

    # Test recall required args
    recall_info = metrics_registry.get_info(MetricType.RECALL_CALCULATION)
    assert recall_info.required_args == {"y_true", "y_pred"}

    # Test F1 required args
    f1_info = metrics_registry.get_info(MetricType.F1_CALCULATION)
    assert f1_info.required_args == {"y_true", "y_pred"}

    # Test Spearman correlation required args
    spearman_info = metrics_registry.get_info(
        MetricType.SPEARMAN_CORRELATION_CALCULATION
    )
    assert spearman_info.required_args == {"a", "b"}


def test_metric_error_handling():
    """Test that metrics handle error cases appropriately."""
    # Test with mismatched array lengths
    y_true = np.array([1, 1, 0])
    y_pred = np.array([1, 0])  # Different length

    with pytest.raises(ValueError):
        metrics_registry.compute(
            MetricType.ACCURACY_CALCULATION,
            y_true=y_true,
            y_pred=y_pred,
        )

    # Test with empty arrays - precision with empty arrays should return 0.0 with warning, not raise
    y_true = np.array([])
    y_pred = np.array([])

    # Empty arrays should return 0.0 for precision (sklearn behavior)
    result = metrics_registry.compute(
        MetricType.PRECISION_CALCULATION,
        y_true=y_true,
        y_pred=y_pred,
    )
    assert result == 0.0

    # Test with invalid correlation inputs (should raise error)
    with pytest.raises((ValueError, TypeError)):
        metrics_registry.compute(
            MetricType.SPEARMAN_CORRELATION_CALCULATION,
            a=np.array([1, 2, 3]),
            b=np.array([]),  # Mismatched lengths
        )


def test_metrics_with_different_data_types():
    """Test that metrics work with different input data types."""
    # Test with lists instead of numpy arrays
    y_true = [1, 1, 0, 0]
    y_pred = [1, 1, 0, 1]

    accuracy = metrics_registry.compute(
        MetricType.ACCURACY_CALCULATION,
        y_true=y_true,
        y_pred=y_pred,
    )

    assert accuracy == 0.75

    # Test Spearman with lists
    a = [1, 2, 3, 4]
    b = [1, 3, 2, 4]

    correlation = metrics_registry.compute(
        MetricType.SPEARMAN_CORRELATION_CALCULATION,
        a=a,
        b=b,
    )

    assert isinstance(correlation, (float, np.floating))
