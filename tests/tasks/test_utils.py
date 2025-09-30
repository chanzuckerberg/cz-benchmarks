import pytest
import numpy as np
import pandas as pd
from czbenchmarks.tasks.utils import aggregate_cells_to_samples


def test_aggregate_cells_to_samples():
    """Test cell to sample aggregation functionality."""
    # Create test data
    n_cells = 100
    n_features = 20

    embeddings = np.random.randn(n_cells, n_features)
    labels = np.random.choice(["A", "B"], size=n_cells)
    sample_ids = [f"sample_{i // 10}" for i in range(n_cells)]  # 10 samples

    # Test mean aggregation
    sample_emb, sample_labels, sample_ids_out = aggregate_cells_to_samples(
        embeddings, labels, sample_ids, aggregation_method="mean"
    )

    # Check shapes
    assert len(sample_emb) == 10  # Should have 10 samples
    assert sample_emb.shape[1] == n_features
    assert len(sample_labels) == 10
    assert len(sample_ids_out) == 10

    # Check that aggregation actually happened
    assert len(sample_emb) < len(embeddings)

    # Test median aggregation
    sample_emb_median, _, _ = aggregate_cells_to_samples(
        embeddings, labels, sample_ids, aggregation_method="median"
    )

    # Should have same shape but different values
    assert sample_emb_median.shape == sample_emb.shape
    assert not np.array_equal(sample_emb_median, sample_emb)


def test_aggregate_cells_to_samples_edge_cases():
    """Test edge cases for cell to sample aggregation."""
    # Test with single sample
    embeddings = np.random.randn(5, 3)
    labels = ["A"] * 5
    sample_ids = ["sample1"] * 5

    sample_emb, sample_labels, sample_ids_out = aggregate_cells_to_samples(
        embeddings, labels, sample_ids
    )

    assert len(sample_emb) == 1
    assert len(sample_labels) == 1
    assert sample_labels.iloc[0] == "A"
    assert sample_ids_out.iloc[0] == "sample1"

    # Test with mismatched lengths
    with pytest.raises(ValueError, match="Mismatched lengths"):
        aggregate_cells_to_samples(
            embeddings[:-1],
            labels,
            sample_ids,  # One fewer embedding
        )

    with pytest.raises(ValueError, match="Mismatched lengths"):
        aggregate_cells_to_samples(
            embeddings,
            labels[:-1],
            sample_ids,  # One fewer label
        )


def test_aggregate_cells_to_samples_types():
    """Test that the function handles different input types correctly."""
    n_cells = 20
    n_features = 5

    # Test with numpy arrays
    embeddings = np.random.randn(n_cells, n_features)
    labels = np.array(["A", "B"] * 10)
    sample_ids = np.array([f"sample_{i // 5}" for i in range(n_cells)])

    sample_emb, sample_labels, sample_ids_out = aggregate_cells_to_samples(
        embeddings, labels, sample_ids
    )

    assert isinstance(sample_emb, np.ndarray)
    assert isinstance(sample_labels, pd.Series)
    assert isinstance(sample_ids_out, pd.Series)

    # Test with pandas Series
    labels_series = pd.Series(labels)
    sample_ids_series = pd.Series(sample_ids)

    sample_emb2, sample_labels2, sample_ids_out2 = aggregate_cells_to_samples(
        embeddings, labels_series, sample_ids_series
    )

    # Results should be the same
    np.testing.assert_array_equal(sample_emb, sample_emb2)
    pd.testing.assert_series_equal(sample_labels, sample_labels2)
    pd.testing.assert_series_equal(sample_ids_out, sample_ids_out2)


def test_aggregate_cells_to_samples_label_consistency():
    """Test that each sample gets a consistent label."""
    # Create data where each sample has mixed labels initially
    embeddings = np.random.randn(30, 4)

    # Create sample IDs and labels such that first occurrence determines sample label
    sample_ids = ["s1"] * 10 + ["s2"] * 10 + ["s3"] * 10
    labels = ["A"] * 5 + ["B"] * 5 + ["B"] * 5 + ["A"] * 5 + ["A"] * 10

    sample_emb, sample_labels, sample_ids_out = aggregate_cells_to_samples(
        embeddings, labels, sample_ids
    )

    # Each sample should have a single label (first occurrence)
    assert len(sample_labels) == 3
    assert sample_labels.iloc[0] == "A"  # s1: first cell was "A"
    assert sample_labels.iloc[1] == "B"  # s2: first cell was "B"
    assert sample_labels.iloc[2] == "A"  # s3: first cell was "A"
