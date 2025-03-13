import pytest
import numpy as np
import anndata as ad
from czbenchmarks.datasets.single_cell import SingleCellDataset
from czbenchmarks.datasets.types import Organism, DataType


# Tests the initialization, loading, and unloading of a single-cell dataset.
def test_single_cell_dataset_init_load_unload(dummy_human_anndata):
    ds = SingleCellDataset(dummy_human_anndata, Organism.HUMAN)
    ds.load_data()
    # We should have ANNDATA stored
    assert ds.get_input(DataType.ANNDATA) is not None
    ds.unload_data()
    with pytest.raises(KeyError):
        ds.get_input(DataType.ANNDATA)


# Tests that dataset validation fails when the dataset does not contain an AnnData object.
def test_single_cell_dataset_validate_no_anndata(tmp_path):
    path = tmp_path / "empty.h5ad"
    adata = ad.AnnData(X=np.empty((0, 0)))
    adata.write_h5ad(path)

    ds = SingleCellDataset(str(path), Organism.HUMAN)
    ds.load_data()
    with pytest.raises(ValueError, match="Dataset does not contain anndata object"):
        ds.validate()


# Tests that dataset validation fails when the organism type is invalid.
def test_single_cell_dataset_validate_wrong_organism_type(dummy_human_anndata):
    with pytest.raises(TypeError):
        SingleCellDataset(dummy_human_anndata, "not_an_organism")


# Tests the properties of the SingleCellDataset class.
def test_single_cell_dataset_properties(dummy_human_anndata):
    ds = SingleCellDataset(dummy_human_anndata, Organism.HUMAN)
    ds.load_data()
    assert ds.organism == ds.get_input(DataType.ORGANISM)
    assert ds.adata is ds.get_input(DataType.ANNDATA)
    assert ds.adata.shape == (5, 3)
