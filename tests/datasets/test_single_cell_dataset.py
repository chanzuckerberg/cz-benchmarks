import pytest
import anndata as ad

from tests.datasets.test_dataset import DatasetTests

class SingleCellDatasetTests(DatasetTests):
    """Common tests for all subclasses of SingleCellDataset class."""

    def test_single_cell_dataset_init_load(self, valid_dataset, n_cells=5, n_genes=3):
        """Tests the loading of a single-cell dataset."""
        valid_dataset.load_data()

        assert valid_dataset.adata is not None
        assert isinstance(valid_dataset.adata, ad.AnnData)
        assert hasattr(valid_dataset.adata, "X")
        # assert valid_dataset.adata.X.shape == (n_cells, n_genes)
        # assert valid_dataset.adata.obs.shape[0] == n_cells
        # assert valid_dataset.adata.var.shape[0] == n_genes


    def test_single_cell_dataset_validate_wrong_organism_type(self, valid_dataset):
        """Tests that dataset validation fails when the organism type is invalid."""
        valid_dataset.load_data()
        
        valid_dataset.organism = "invalid_organism"
        invalid_dataset = valid_dataset
        
        with pytest.raises(ValueError, match="Organism is not a valid Organism enum"):
            invalid_dataset.validate()


    def test_single_cell_dataset_validate_wrong_gene_prefix(self, valid_dataset):
        """Tests that dataset validation fails when gene prefixes don't match organism."""
        valid_dataset.load_data()
        
        valid_dataset.adata.var_names = [f"BAD{i}" for i in range(1, valid_dataset.adata.n_vars + 1)]
        invalid_dataset = valid_dataset

        with pytest.raises(ValueError, match="Dataset does not contain valid gene names"):
            invalid_dataset.validate()

    @pytest.mark.skip("Skipping test for raw counts validation as it is cannot be enforced until all datasets are verified to have integer raw counts and validation warning is converted to an error.")
    def test_single_cell_dataset_validate_bad_raw_counts(self, valid_dataset):
        """Tests that dataset validation fails when raw counts are not valid."""
        valid_dataset.load_data()

        valid_dataset.adata.X = valid_dataset.adata.X.toarray().astype(float) + 0.5
        invalid_dataset = valid_dataset

        with pytest.raises(ValueError, match="Dataset X matrix does not contain raw counts"):
            invalid_dataset.validate()


