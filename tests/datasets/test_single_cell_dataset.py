import pytest
from czbenchmarks.datasets.types import Organism, DataType
from czbenchmarks.datasets.single_cell import SingleCellDataset


def test_single_cell_dataset_init_load_unload(dummy_human_anndata):
    """Tests the initialization, loading, and unloading of a single-cell dataset."""
    # Dataset is already loaded, so test unload first
    dummy_human_anndata.unload_data()
    with pytest.raises(KeyError):
        dummy_human_anndata.get_input(DataType.ANNDATA)

    # Now test loading
    dummy_human_anndata.load_data()
    assert dummy_human_anndata.get_input(DataType.ANNDATA) is not None


def test_single_cell_dataset_validate_wrong_organism_type(dummy_human_anndata):
    """Tests that dataset validation fails when the organism type is invalid."""
    with pytest.raises(TypeError):
        SingleCellDataset(dummy_human_anndata.path, "not_an_organism")


def test_single_cell_dataset_validate_wrong_gene_prefix(
    dummy_human_anndata_wrong_prefix,
):
    """Tests that dataset validation fails when gene prefixes don't match organism."""
    with pytest.raises(ValueError, match="Dataset does not contain valid gene names"):
        dummy_human_anndata_wrong_prefix.validate()


def test_single_cell_dataset_validate_without_load(dummy_human_anndata):
    """Tests that dataset validation fails when load_data is not called."""
    # First unload the data since fixture provides loaded dataset
    dummy_human_anndata.unload_data()
    with pytest.raises(ValueError, match="Dataset does not contain anndata object"):
        dummy_human_anndata.validate()


def test_single_cell_dataset_properties(dummy_human_anndata):
    """Test that SingleCellDataset properties are correct."""
    assert dummy_human_anndata.n_cells == 5
    assert dummy_human_anndata.n_genes == 3
    assert dummy_human_anndata.organism == Organism.HUMAN


def test_single_cell_dataset_validate_empty(empty_anndata):
    """Test that SingleCellDataset validation fails with empty AnnData."""
    with pytest.raises(ValueError, match="AnnData object is empty"):
        empty_anndata.validate()


def test_single_cell_dataset_validate_float_counts(float_counts_anndata):
    """Test that SingleCellDataset validation fails with float counts."""
    with pytest.raises(ValueError, match="X matrix must contain integer values"):
        float_counts_anndata.validate()


def test_single_cell_dataset_validate_wrong_prefix(dummy_human_anndata_wrong_prefix):
    """Test that SingleCellDataset validation fails with wrong gene name prefixes."""
    with pytest.raises(ValueError, match="Gene names must start with"):
        dummy_human_anndata_wrong_prefix.validate()


def test_single_cell_dataset_validate_success(dummy_human_anndata):
    """Test that SingleCellDataset validation succeeds with valid data."""
    dummy_human_anndata.validate()  # Should not raise any exceptions
