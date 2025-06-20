import pytest


def test_validate_dataset_path_exists(dummy_human_anndata):
    """Test that validation succeeds when dataset path exists
    and organism is valid."""
    dummy_human_anndata.validate()  # Should not raise any exceptions


def test_validate_dataset_path_not_exists(dummy_human_anndata):
    """Test that validation fails when dataset path does not exist."""
    dummy_human_anndata.path = "non_existing_path.h5ad"
    with pytest.raises(ValueError, match="Dataset path does not exist"):
        dummy_human_anndata.validate()


def test_validate_dataset_wrong_organism_type(dummy_human_anndata):
    """Tests that dataset validation fails when the organism type is invalid."""
    dummy_human_anndata.organism = "not_an_organism"
    with pytest.raises(ValueError, match="Organism is not a valid Organism enum"):
        dummy_human_anndata.validate()


# FIXME: Implement these tests when cache PR is merged
@pytest.mark.skip(reason="Not implemented")
def test_validate_dataset_load_data(dataset_path):
    """Test that the dataset loads data correctly."""
    pass


@pytest.mark.skip(reason="Not implemented")
def test_validate_dataset_cache_data(dataset_path):
    """Test that the dataset caches data correctly."""
    pass
