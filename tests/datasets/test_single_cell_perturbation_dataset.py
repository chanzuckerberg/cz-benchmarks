import pytest
from czbenchmarks.datasets.types import Organism, DataType


def test_perturbation_dataset_load_data(dummy_perturbation_anndata):
    """Tests the loading of perturbation dataset data."""
    truth = dummy_perturbation_anndata.get_input(DataType.PERTURBATION_TRUTH)
    assert "test1" in truth
    assert "test2" in truth
    assert dummy_perturbation_anndata.adata.shape == (2, 3)


def test_perturbation_dataset_load_data_missing_condition_key(
    perturbation_missing_condition,
):
    """Tests that loading data fails when the condition key is missing."""
    with pytest.raises(AssertionError):
        perturbation_missing_condition.load_data()


def test_perturbation_dataset_load_data_missing_split_key(perturbation_missing_split):
    """Tests that loading data fails when the split key is missing."""
    with pytest.raises(AssertionError):
        perturbation_missing_split.load_data()


def test_perturbation_dataset_unload_data(dummy_perturbation_anndata):
    """Tests the unloading of perturbation dataset data."""
    dummy_perturbation_anndata.unload_data()
    with pytest.raises(KeyError):
        dummy_perturbation_anndata.get_input(DataType.PERTURBATION_TRUTH)


def test_perturbation_dataset_properties(dummy_perturbation_anndata):
    """Test that PerturbationSingleCellDataset properties are correct."""
    assert dummy_perturbation_anndata.n_cells == 6
    assert dummy_perturbation_anndata.n_genes == 3
    assert dummy_perturbation_anndata.organism == Organism.HUMAN
    assert dummy_perturbation_anndata.condition_key == "condition"
    assert dummy_perturbation_anndata.split_key == "split"


def test_perturbation_dataset_validate_missing_condition(
    perturbation_missing_condition,
):
    """Test that validation fails when condition column is missing."""
    with pytest.raises(ValueError, match="Missing required column"):
        perturbation_missing_condition.validate()


def test_perturbation_dataset_validate_missing_split(perturbation_missing_split):
    """Test that validation fails when split column is missing."""
    with pytest.raises(ValueError, match="Missing required column"):
        perturbation_missing_split.validate()


def test_perturbation_dataset_validate_invalid_split(perturbation_invalid_split):
    """Test that validation fails with invalid split values."""
    with pytest.raises(ValueError, match="Invalid split value"):
        perturbation_invalid_split.validate()


def test_perturbation_dataset_validate_invalid_condition(
    perturbation_invalid_condition,
):
    """Test that validation fails with invalid condition format."""
    with pytest.raises(ValueError, match="Invalid condition format"):
        perturbation_invalid_condition.validate()


def test_perturbation_dataset_validate_invalid_combo(perturbation_invalid_combo):
    """Test that validation fails with invalid combo perturbation."""
    with pytest.raises(ValueError, match="Invalid gene name format"):
        perturbation_invalid_combo.validate()


def test_perturbation_dataset_validate_success(perturbation_valid_conditions):
    """Test that validation succeeds with valid condition formats."""
    perturbation_valid_conditions.validate()  # Should not raise any exceptions
