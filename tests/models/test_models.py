import pytest

from czbenchmarks.datasets.types import Organism, DataType
from tests.utils import (
    create_dummy_anndata,
    DummyDataset,
    DummySingleCellModelValidator,
    DummySingleCellPerturbationModelValidator,
    DummySingleCellModelValidatorWithObsKeys,
)


# For all fully implemented single‑cell validators, dataset validation should pass
# on a valid benchmarking dataset fixture.
@pytest.mark.parametrize(
    "validator_class, dataset",
    [
        (DummySingleCellModelValidator, "dummy_single_cell_dataset"),
        (DummySingleCellPerturbationModelValidator, "dummy_perturbation_dataset"),
    ],
)
def test_valid_dataset(validator_class, dataset, request):
    """Test that validation passes for valid datasets with all required components."""
    validator = validator_class()
    dataset = request.getfixturevalue(dataset)
    try:
        validator.validate_dataset(dataset)
    except Exception as e:
        pytest.fail(f"Validation failed unexpectedly: {e}")


def test_invalid_dataset_type():
    """Test that validation fails when the dataset_type is incompatible."""
    validator = (
        DummySingleCellModelValidator()
    )  # Expects a SingleCellDataset (exact type match)
    dummy_ds = DummyDataset("dummy_path")
    # Provide required inputs to bypass missing-input errors.
    ann = create_dummy_anndata(
        obs_columns=["dataset_id", "assay", "suspension_type", "donor_id"]
    )
    dummy_ds.set_input(DataType.ANNDATA, ann)
    dummy_ds.set_input(DataType.METADATA, ann.obs)
    dummy_ds.set_input(DataType.ORGANISM, Organism.HUMAN)
    with pytest.raises(ValueError, match="Dataset type mismatch"):
        validator.validate_dataset(dummy_ds)


def test_missing_required_inputs(dummy_single_cell_dataset):
    """Test that validation fails when required inputs are missing."""
    validator = (
        DummySingleCellModelValidator()
    )  # Expects both ANNDATA and METADATA as inputs.
    # First unload the data to test missing inputs
    dummy_single_cell_dataset.unload_data()
    with pytest.raises(ValueError, match="Missing required inputs"):
        validator.validate_dataset(dummy_single_cell_dataset)


def test_incompatible_organism(dummy_mouse_dataset):
    """Test that validation fails when the organism is incompatible."""
    validator = (
        DummySingleCellModelValidator()
    )  # DummySingleCellModelValidator supports HUMAN and MOUSE only.
    # Change organism to an unsupported one
    dummy_mouse_dataset.set_input(DataType.ORGANISM, Organism.CHIMPANZEE)
    with pytest.raises(ValueError, match="Dataset organism"):
        validator.validate_dataset(dummy_mouse_dataset)


def test_missing_required_obs_keys(dummy_single_cell_dataset):
    """Test that validation fails when required obs keys are missing."""
    validator = (
        DummySingleCellModelValidatorWithObsKeys()
    )  # Requires obs keys: dataset_id, assay, suspension_type, donor_id.
    # Create new anndata missing one required obs key
    ann = create_dummy_anndata(obs_columns=["dataset_id", "assay", "suspension_type"])
    dummy_single_cell_dataset.set_input(DataType.ANNDATA, ann)
    dummy_single_cell_dataset.set_input(DataType.METADATA, ann.obs)
    with pytest.raises(ValueError, match="Missing required obs keys"):
        validator.validate_dataset(dummy_single_cell_dataset)


def test_missing_required_var_keys(dummy_single_cell_dataset):
    """Test that validation fails when required var keys are missing."""
    validator = DummySingleCellModelValidator()  # Requires var key: "feature_id".
    # Create new anndata without feature_id
    ann = create_dummy_anndata(var_columns=["some_other_feature"])
    dummy_single_cell_dataset.set_input(DataType.ANNDATA, ann)
    dummy_single_cell_dataset.set_input(DataType.METADATA, ann.obs)
    with pytest.raises(ValueError, match="Missing required var keys"):
        validator.validate_dataset(dummy_single_cell_dataset)
