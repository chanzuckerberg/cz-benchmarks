import numpy as np
from czbenchmarks.tasks.utils import looks_like_lognorm


class TestLooksLikeLognorm:
    """Test suite for the looks_like_lognorm function."""

    def test_raw_count_data_returns_false(self):
        """Test that raw count data (integers) returns False."""
        # Create mock raw count data (integers)
        raw_data = np.random.randint(0, 1000, size=(100, 50))

        assert not looks_like_lognorm(raw_data)

    def test_log_normalized_data_returns_true(self):
        """Test that log-normalized data (with fractional values) returns True."""
        # Create mock log-normalized data with fractional values
        log_data = np.random.lognormal(0, 1, size=(100, 50))

        assert looks_like_lognorm(log_data)

    def test_normalized_non_integer_data_returns_true(self):
        """Test that any non-integer data returns True."""
        # Create data with fractional values (simulating normalized but not necessarily log-transformed)
        normalized_data = np.random.rand(100, 50) * 10  # Random floats between 0-10

        assert looks_like_lognorm(normalized_data)

    def test_custom_n_cells_parameter(self):
        """Test that the n_cells parameter works correctly."""
        # Create log-normalized data
        log_data = np.random.lognormal(0, 1, size=(1000, 50))

        # Both should return True for log-normalized data
        assert looks_like_lognorm(log_data, sample_size=50)
        assert looks_like_lognorm(log_data, sample_size=100)

    def test_custom_epsilon_parameter(self):
        """Test that the epsilon parameter affects detection sensitivity."""
        # Create data that's almost integer but with tiny fractional parts
        almost_integer_data = np.random.randint(0, 100, size=(100, 50)) + 1e-4

        # With default epsilon (1e-2), should return False
        assert not looks_like_lognorm(almost_integer_data)

        # With very small tol (1e-5), should return True
        assert looks_like_lognorm(almost_integer_data, tol=1e-5)

    def test_mixed_integer_and_float_data(self):
        """Test data that's mostly integer but has some fractional values."""
        # Create mostly integer data
        mixed_data = np.random.randint(0, 100, size=(100, 50)).astype(float)
        # Add a fractional value to ensure the sum is not an integer
        mixed_data[0, 0] += 0.3  # Make first cell have fractional sum
        # Should return True since some cells have fractional sums
        assert looks_like_lognorm(mixed_data)

    def test_fractional_values_integer_sums(self):
        """Test that fractional values with integer sums return False."""
        # Create data where individual values are fractional but sums are integers
        # For example: [0.5, 0.5] sums to 1.0 (integer)
        data_with_integer_sums = np.array([[0.5, 0.5], [1.5, 2.5]])  # sums: [1.0, 4.0]

        # Should return False because cell sums are integers (within epsilon)
        assert not looks_like_lognorm(data_with_integer_sums)
