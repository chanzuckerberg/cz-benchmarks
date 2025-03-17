import pytest
import numpy as np
from czbenchmarks.metrics.types import MetricRegistry


@pytest.fixture
def dummy_metric_function():
    """Returns a dummy metric function that returns a constant value."""

    def dummy_func(x, y):
        return 0.5

    return dummy_func


@pytest.fixture
def dummy_metric_registry():
    """Returns a fresh MetricRegistry instance."""
    return MetricRegistry()


@pytest.fixture
def sample_data():
    """Returns sample data for testing metrics."""
    np.random.seed(42)
    X = np.random.normal(size=(100, 10))
    y_true = np.random.randint(0, 3, size=100)
    y_pred = np.random.randint(0, 3, size=100)
    return {"X": X, "y_true": y_true, "y_pred": y_pred}
