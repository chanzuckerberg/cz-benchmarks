import pytest
from czbenchmarks.datasets.types import DataType
from czbenchmarks.datasets.single_cell import SingleCellDataset


def test_single_cell_dataset_init_load_unload(dummy_human_anndata):
    """Tests the initialization, loading, and unloading of a single-cell dataset."""
    # Dataset is already loaded, so test unload first
    dummy_human_anndata.unload_data()
    assert dummy_human_anndata.adata is None

    # Now test loading
    dummy_human_anndata.load_data()
    assert dummy_human_anndata.adata is not None


def test_single_cell_dataset_validate_wrong_organism_type(dummy_human_anndata):
    """Tests that dataset validation fails when the organism type is invalid."""
    with pytest.raises(TypeError):
        SingleCellDataset(path=dummy_human_anndata.path, organism="not_an_organism")
