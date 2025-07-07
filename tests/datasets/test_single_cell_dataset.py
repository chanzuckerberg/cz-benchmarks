import pandas as pd
import pytest
import anndata as ad
from czbenchmarks.datasets.single_cell import SingleCellLabeledDataset


def test_single_cell_dataset_init_load(dummy_human_anndata, n_cells=5, n_genes=3):
    """Tests the initialization, loading, and unloading of a single-cell dataset."""
    dummy_human_anndata.load_data()
    assert dummy_human_anndata.adata is not None
    assert isinstance(dummy_human_anndata.adata, ad.AnnData)
    assert hasattr(dummy_human_anndata.adata, "X")
    assert dummy_human_anndata.adata.X.shape == (n_cells, n_genes)
    assert dummy_human_anndata.adata.obs.shape[0] == n_cells
    assert dummy_human_anndata.adata.var.shape[0] == n_genes


def test_single_cell_dataset_validate_wrong_organism_type(dummy_human_anndata):
    """Tests that dataset validation fails when the organism type is invalid."""
    dataset = SingleCellLabeledDataset(
        path=dummy_human_anndata.path, organism="not_an_organism"
    )
    with pytest.raises(ValueError, match="Organism is not a valid Organism enum"):
        dataset.validate()


def test_single_cell_dataset_validate_wrong_gene_prefix(
    dummy_human_anndata_wrong_prefix,
):
    """Tests that dataset validation fails when gene prefixes don't match organism."""
    with pytest.raises(ValueError, match="Dataset does not contain valid gene names"):
        dummy_human_anndata_wrong_prefix.validate()


def test_single_cell_dataset_validate_success(dummy_human_anndata):
    """Test that SingleCellDataset validation succeeds with valid data."""
    dummy_human_anndata.validate()  # Should not raise any exceptions


def test_single_cell_dataset_store_task_inputs(dummy_human_anndata):
    """Tests that the store_task_inputs method writes labels to a file."""
    dummy_human_anndata.store_task_inputs()
    output_file = dummy_human_anndata.dir / dummy_human_anndata.name / "cell_types.json"
    assert output_file.exists()
    cell_types = pd.read_json(output_file, typ="series")
    assert not cell_types.empty
    assert all(cell_type in cell_types.values for cell_type in ["type_0", "type_1", "type_2"])