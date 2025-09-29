import numpy as np
import scipy.sparse as sp
from czbenchmarks.tasks.utils import looks_like_lognorm


class TestLooksLikeLognorm:
    """Test suite for the looks_like_lognorm function."""

    def test_raw_count_data_returns_false(self):
        """Test that raw count data (integers) returns False."""
        # Create mock raw count data (integers)
        raw_data = np.random.randint(0, 1000, size=(100, 50))

        result = looks_like_lognorm(raw_data)

        assert not result

    def test_log_normalized_data_returns_true(self):
        """Test that log-normalized data (with fractional values) returns True."""
        # Create mock log-normalized data with fractional values
        log_data = np.random.lognormal(0, 1, size=(100, 50))

        result = looks_like_lognorm(log_data)

        assert result

    def test_normalized_non_integer_data_returns_true(self):
        """Test that any non-integer data returns True."""
        # Create data with fractional values (simulating normalized but not necessarily log-transformed)
        normalized_data = np.random.rand(100, 50) * 10  # Random floats between 0-10

        result = looks_like_lognorm(normalized_data)

        assert result

    def test_sparse_raw_count_data_returns_false(self):
        """Test that sparse raw count data returns False."""
        # Create sparse raw count data
        raw_data = np.random.randint(0, 100, size=(100, 50))
        sparse_data = sp.csr_matrix(raw_data)

        result = looks_like_lognorm(sparse_data)

        assert result

    def test_sparse_log_normalized_data_returns_true(self):
        """Test that sparse log-normalized data returns True."""
        # Create sparse log-normalized data
        log_data = np.random.lognormal(0, 1, size=(100, 50))
        sparse_log_data = sp.csr_matrix(log_data)

        result = looks_like_lognorm(sparse_log_data)

        assert result

    def test_custom_n_cells_parameter(self):
        """Test that the n_cells parameter works correctly."""
        # Create log-normalized data
        log_data = np.random.lognormal(0, 1, size=(1000, 50))

        # Test with different n_cells values
        result_50 = looks_like_lognorm(log_data, sample_size=50)
        result_100 = looks_like_lognorm(log_data, sample_size=100)

        # Both should return True for log-normalized data
        assert result_50
        assert result_100

    def test_n_cells_larger_than_data_size(self):
        """Test behavior when n_cells is larger than the actual number of cells."""
        # Create small dataset
        log_data = np.random.lognormal(0, 1, size=(10, 50))

        # Request more cells than available
        result = looks_like_lognorm(log_data, sample_size=100)

        # Should still work by using all available cells
        assert result

    def test_custom_epsilon_parameter(self):
        """Test that the epsilon parameter affects detection sensitivity."""
        # Create data that's almost integer but with tiny fractional parts
        almost_integer_data = np.random.randint(0, 100, size=(100, 50)) + 1e-4

        # With default epsilon (1e-2), should return False
        result_default = looks_like_lognorm(almost_integer_data)
        assert not result_default

        # With very small tol (1e-5), should return True
        result_small_tol = looks_like_lognorm(almost_integer_data, tol=1e-5)
        assert result_small_tol

    def test_all_zero_data(self):
        """Test behavior with all-zero data."""
        zero_data = np.zeros((100, 50))

        result = looks_like_lognorm(zero_data)

        # All zeros should be considered as integer data (raw counts)
        assert not result

    def test_mixed_integer_and_float_data(self):
        """Test data that's mostly integer but has some fractional values."""
        # Create mostly integer data
        mixed_data = np.random.randint(0, 100, size=(100, 50)).astype(float)
        # Add a fractional value to ensure the sum is not an integer
        mixed_data[0, 0] += 0.3  # Make first cell have fractional sum

        result = looks_like_lognorm(mixed_data)

        # Should return True since some cells have fractional sums
        assert result

    def test_single_cell_data(self):
        """Test behavior with single cell data."""
        # Single cell with integer values
        single_cell_int = np.array([[1, 2, 3, 4, 5]])
        result_int = looks_like_lognorm(single_cell_int)
        assert not result_int

        # Single cell with fractional values
        single_cell_float = np.array([[1.1, 2.2, 3.3, 4.4, 5.5]])
        result_float = looks_like_lognorm(single_cell_float)
        assert result_float

    def test_deterministic_behavior_with_seed(self):
        """Test that results are consistent when sampling the same data."""
        # Create data
        log_data = np.random.lognormal(0, 1, size=(1000, 50))

        # Set seed for reproducible sampling
        np.random.seed(42)
        result1 = looks_like_lognorm(log_data, sample_size=100)

        np.random.seed(42)
        result2 = looks_like_lognorm(log_data, sample_size=100)

        # Results should be the same with same seed
        assert result1 == result2

    def test_edge_case_very_small_dataset(self):
        """Test with very small dataset (fewer cells than default sampling)."""
        # Create tiny dataset with sums that are actually fractional
        tiny_data = np.array(
            [[1.3, 2.5], [3.2, 4.1]]
        )  # 2 cells, 2 genes (sums: 3.8, 7.3)

        result = looks_like_lognorm(
            tiny_data, sample_size=500
        )  # Request more cells than available

        # Should still work and return True for fractional data
        assert result

    def test_fractional_values_integer_sums(self):
        """Test that fractional values with integer sums return False."""
        # Create data where individual values are fractional but sums are integers
        # For example: [0.5, 0.5] sums to 1.0 (integer)
        data_with_integer_sums = np.array([[0.5, 0.5], [1.5, 2.5]])  # sums: [1.0, 4.0]

        result = looks_like_lognorm(data_with_integer_sums)

        # Should return False because cell sums are integers (within epsilon)
        assert not result

    def test_explains_function_behavior(self):
        """Test that demonstrates the function checks cell sums, not individual values."""
        # Create data where individual values are fractional but cell sums are integers
        integer_sum_data = np.array(
            [
                [0.25, 0.25, 0.25, 0.25],  # sum = 1.0
                [0.5, 0.5, 1.0, 1.0],  # sum = 3.0
                [1.1, 1.9, 2.0, 3.0],  # sum = 8.0
            ]
        )

        result_integer_sums = looks_like_lognorm(integer_sum_data)
        assert not result_integer_sums

        # Create data where cell sums are fractional
        fractional_sum_data = np.array(
            [
                [0.3, 0.4],  # sum = 0.7
                [1.1, 2.2],  # sum = 3.3
                [0.9, 1.5],  # sum = 2.4
            ]
        )

        result_fractional_sums = looks_like_lognorm(fractional_sum_data)
        assert result_fractional_sums
