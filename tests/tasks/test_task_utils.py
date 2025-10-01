from io import StringIO
import sys
import numpy as np
from czbenchmarks.tasks.utils import (
    looks_like_lognorm,
    print_correlation_metrics_baseline_and_model,
)
import scipy.sparse as sp
import pandas as pd


def test_print_correlation_metrics_baseline_and_model():
    """Test the print_correlation_metrics_baseline_and_model function."""

    # Create test data
    model_values = [0.5, 0.6, 0.7, 0.8]
    baseline_values = [0.4, 0.5, 0.6, 0.7]
    metrics_df = pd.DataFrame({"Model": model_values, "Baseline": baseline_values})

    # Capture printed output
    captured_output = StringIO()
    sys.stdout = captured_output

    print_correlation_metrics_baseline_and_model(metrics_df=metrics_df)
    output = captured_output.getvalue()

    # Validate output contains expected elements
    for col in metrics_df.columns:
        assert col in output

    row_names = [
        "Number of conditions",
        "Mean correlation",
        "Standard Deviation",
        "Min correlation",
        "25th percentile",
        "Median correlation",
        "75th percentile",
        "Max correlation",
    ]
    for row in row_names:
        assert row in output


class TestLooksLikeLognorm:
    """Test suite for the looks_like_lognorm function."""

    def test_raw_count_data_returns_false(self):
        """Test that raw count data (integers) returns False."""
        # Create mock raw count data (integers)
        raw_data = np.random.randint(0, 1000, size=(100, 50))

        assert not looks_like_lognorm(raw_data, random_seed=42)

    def test_log_normalized_data_returns_true(self):
        """Test that log-normalized data (with fractional values) returns True."""
        # Create mock log-normalized data with fractional values
        log_data = np.random.lognormal(0, 1, size=(100, 50))

        assert looks_like_lognorm(log_data, random_seed=42)
        assert looks_like_lognorm(sp.csr_matrix(log_data), random_seed=42)

    def test_log_normalized_data_returns_true_sparse_data(self):
        """Test that log-normalized data (with fractional values) returns True."""
        # Create mock log-normalized data with fractional values
        log_data = np.zeros((100, 50))
        # Make 50 of the values log-normalized (fractional, non-integer)
        # We'll pick 50 random indices and set them to log-normalized values
        indices = np.random.choice(log_data.size, size=50, replace=False)
        lognorm_values = np.random.lognormal(0, 1, size=50)
        np.put(log_data, indices, lognorm_values)
        assert looks_like_lognorm(log_data, random_seed=42)
        assert looks_like_lognorm(sp.csr_matrix(log_data), random_seed=42)

    def test_normalized_non_integer_data_returns_true(self):
        """Test that any non-integer data returns True."""
        # Create data with fractional values (simulating normalized but not necessarily log-transformed)
        normalized_data = np.random.rand(100, 50) * 10  # Random floats between 0-10

        assert looks_like_lognorm(normalized_data, random_seed=42)

    def test_custom_n_cells_parameter(self):
        """Test that the n_cells parameter works correctly."""
        # Create log-normalized data
        log_data = np.random.lognormal(0, 1, size=(1000, 50))

        # Both should return True for log-normalized data
        assert looks_like_lognorm(log_data, sample_size=50, random_seed=42)
        assert looks_like_lognorm(log_data, sample_size=100, random_seed=42)

    def test_custom_epsilon_parameter(self):
        """Test that the epsilon parameter affects detection sensitivity."""
        # Create data that's almost integer but with tiny fractional parts
        almost_integer_data = np.random.randint(0, 100, size=(100, 50)) + 1e-4

        # With default epsilon (1e-2), should return False
        assert not looks_like_lognorm(almost_integer_data, random_seed=42)

        # With very small tol (1e-5), should return True
        assert looks_like_lognorm(almost_integer_data, tol=1e-5, random_seed=42)

    def test_mixed_integer_and_float_data(self):
        """Test data that's mostly integer but has some fractional values."""
        # Create mostly integer data
        mixed_data = np.random.randint(0, 100, size=(100, 50)).astype(float)
        # Add a fractional value to ensure the sum is not an integer
        mixed_data[0, 0] += 0.3  # Make first cell have fractional sum
        # Should return True since some cells have fractional sums
        assert looks_like_lognorm(mixed_data, random_seed=42)

        log_data = np.random.lognormal(0, 1, size=(100, 50))

        assert looks_like_lognorm(log_data, random_seed=42)
        assert looks_like_lognorm(sp.csr_matrix(log_data), random_seed=42)
