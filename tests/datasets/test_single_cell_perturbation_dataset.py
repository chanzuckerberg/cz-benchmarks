import pytest
from czbenchmarks.datasets.types import DataType


def test_perturbation_dataset_load_data(dummy_perturbation_anndata):
    """Tests the loading of perturbation dataset data."""
    truth = dummy_perturbation_anndata.get_input(DataType.PERTURBATION_TRUTH)
    assert "test1+ctrl" in truth
    assert "test2+ctrl" in truth
    assert dummy_perturbation_anndata.adata.shape == (2, 3)


def test_perturbation_dataset_load_data_missing_condition_key(
    perturbation_missing_condition,
):
    """Tests that loading data fails when the condition key is missing."""
    with pytest.raises(ValueError, match="Condition key .* not found in adata.obs"):
        perturbation_missing_condition.load_data()


def test_perturbation_dataset_load_data_missing_split_key(perturbation_missing_split):
    """Tests that loading data fails when the split key is missing."""
    with pytest.raises(ValueError, match="Split key .* not found in adata.obs"):
        perturbation_missing_split.load_data()


def test_perturbation_dataset_unload_data(dummy_perturbation_anndata):
    """Tests the unloading of perturbation dataset data."""
    dummy_perturbation_anndata.unload_data()
    with pytest.raises(KeyError):
        dummy_perturbation_anndata.get_input(DataType.PERTURBATION_TRUTH)


def test_perturbation_dataset_validate_invalid_split(perturbation_invalid_split):
    """Test that validation fails with invalid split values."""
    with pytest.raises(ValueError, match="Invalid split value"):
        perturbation_invalid_split.validate()


def test_perturbation_dataset_validate_invalid_condition(
    perturbation_invalid_condition,
):
    """Test that validation fails with invalid condition format."""
    with pytest.raises(ValueError, match="Invalid perturbation condition format.*"):
        perturbation_invalid_condition.validate()


def test_perturbation_dataset_validate_invalid_combo(perturbation_invalid_combo):
    """Test that validation fails with invalid combo perturbation."""
    with pytest.raises(ValueError, match="Invalid gene prefix in condition.*"):
        perturbation_invalid_combo.validate()


def test_perturbation_dataset_validate_success(perturbation_valid_conditions):
    """Test that validation succeeds with valid condition formats."""
    perturbation_valid_conditions.validate()  # Should not raise any exceptions
