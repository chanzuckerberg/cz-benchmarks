import pytest
import numpy as np
from enum import Enum
from czbenchmarks.metrics.types import MetricType


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
