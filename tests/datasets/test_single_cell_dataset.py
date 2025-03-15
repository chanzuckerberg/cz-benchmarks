import pytest
import numpy as np
import anndata as ad
from czbenchmarks.datasets.types import Organism, DataType
from czbenchmarks.datasets.single_cell import SingleCellDataset
from tests.utils import create_dummy_anndata


def test_single_cell_dataset_init_load_unload(dummy_human_anndata):
    """Tests the initialization, loading, and unloading of a single-cell dataset."""
    ds = SingleCellDataset(dummy_human_anndata, Organism.HUMAN)
    ds.load_data()
    # We should have ANNDATA stored
    assert ds.get_input(DataType.ANNDATA) is not None
    ds.unload_data()
    with pytest.raises(KeyError):
        ds.get_input(DataType.ANNDATA)


def test_single_cell_dataset_validate_no_anndata(tmp_path):
    """Tests that dataset validation fails when the dataset does not contain an AnnData object."""
    path = tmp_path / "empty.h5ad"
    adata = ad.AnnData(X=np.empty((0, 0)))
    adata.write_h5ad(path)

    ds = SingleCellDataset(str(path), Organism.HUMAN)
    ds.load_data()
    with pytest.raises(ValueError, match="Dataset does not contain anndata object"):
        ds.validate()


def test_single_cell_dataset_validate_wrong_organism_type(dummy_human_anndata):
    """Tests that dataset validation fails when the organism type is invalid."""
    with pytest.raises(TypeError):
        SingleCellDataset(dummy_human_anndata, "not_an_organism")


def test_single_cell_dataset_validate_wrong_gene_prefix(
    dummy_human_anndata_wrong_prefix,
):
    """Tests that dataset validation fails when gene prefixes don't match organism."""
    ds = SingleCellDataset(dummy_human_anndata_wrong_prefix, Organism.HUMAN)
    ds.load_data()
    with pytest.raises(ValueError, match="Dataset does not contain valid gene names"):
        ds.validate()


def test_single_cell_dataset_validate_non_integer_x(tmp_path):
    """Tests that dataset validation fails when X matrix is not integer type."""
    path = tmp_path / "float_counts.h5ad"
    # Create data with float counts instead of integers
    adata = create_dummy_anndata(n_cells=5, n_genes=3, organism=Organism.HUMAN)
    adata.X = np.ones((5, 3), dtype=np.float32)  # Override X to be float
    adata.write_h5ad(path)

    ds = SingleCellDataset(str(path), Organism.HUMAN)
    ds.load_data()
    with pytest.raises(ValueError, match="Dataset X matrix must have integer dtype"):
        ds.validate()


def test_single_cell_dataset_validate_without_load(dummy_human_anndata):
    """Tests that dataset validation fails when load_data is not called."""
    ds = SingleCellDataset(dummy_human_anndata, Organism.HUMAN)
    with pytest.raises(ValueError, match="Dataset does not contain anndata object"):
        ds.validate()


def test_single_cell_dataset_properties(dummy_human_anndata):
    """Tests the properties of the SingleCellDataset class."""
    ds = SingleCellDataset(dummy_human_anndata, Organism.HUMAN)
    ds.load_data()
    assert ds.organism == ds.get_input(DataType.ORGANISM)
    assert ds.adata is ds.get_input(DataType.ANNDATA)
    assert ds.adata.shape == (5, 3)
