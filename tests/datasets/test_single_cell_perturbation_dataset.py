from pathlib import Path
import re
import pandas as pd
import pytest

from czbenchmarks.datasets.single_cell_perturbation import SingleCellPerturbationDataset
from czbenchmarks.datasets.types import Organism
from tests.datasets.test_single_cell_dataset import SingleCellDatasetTests
from tests.utils import create_dummy_anndata

class TestSingleCellPerturbationDataset(SingleCellDatasetTests):
    """Tests for the SingleCellPerturbationDataset class."""

    @pytest.fixture
    def valid_dataset(self, tmp_path) -> SingleCellPerturbationDataset:
        """Fixture to provide a valid SingleCellPerturbationDataset H5AD file."""
        return SingleCellPerturbationDataset(
            path=self.valid_dataset_file(tmp_path),
            organism=Organism.HUMAN,
            condition_key="condition",
            split_key="split",
        )

    def valid_dataset_file(self, tmp_path) -> Path:
        """Creates a valid SingleCellPerturbationDataset H5AD file."""
        file_path = tmp_path / "dummy_perturbation.h5ad"
        adata = create_dummy_anndata(
            n_cells=6,
            n_genes=3,
            obs_columns=["condition", "split"],
            organism=Organism.HUMAN,
        )
        adata.obs["condition"] = ["ctrl", "ctrl", "test1+ctrl", "test1+ctrl", "test2+ctrl", "test2+ctrl"]
        adata.obs["split"] = ["train", "train", "test", "test", "test", "test"]
        adata.write_h5ad(file_path)
        
        return file_path

    @pytest.fixture
    def perturbation_missing_split_column_h5ad(self, tmp_path) -> Path:
        """Creates a PerturbationSingleCellDataset missing the split column."""
        file_path = tmp_path / "perturbation_missing_split.h5ad"
        adata = create_dummy_anndata(
            n_cells=6, n_genes=3, obs_columns=["condition"], organism=Organism.HUMAN
        )
        adata.obs["condition"] = ["ctrl", "ctrl", "test1", "test1", "test2", "test2"]
        adata.write_h5ad(file_path)

        return file_path


    @pytest.fixture
    def perturbation_invalid_split_h5ad(self, tmp_path) -> Path:
        """Creates a PerturbationSingleCellDataset with invalid split values."""
        file_path = tmp_path / "perturbation_invalid_split.h5ad"
        adata = create_dummy_anndata(
            n_cells=6,
            n_genes=3,
            obs_columns=["condition", "split"],
            organism=Organism.HUMAN,
        )
        adata.obs["condition"] = ["ctrl", "ctrl", "test1", "test1", "test2", "test2"]
        adata.obs["split"] = ["BAD", "train", "test", "test", "test", "test"]
        adata.write_h5ad(file_path)

        return file_path

    @pytest.fixture
    def perturbation_missing_condition_column_h5ad(self, tmp_path) -> Path:
        """Creates a PerturbationSingleCellDataset with invalid condition format."""
        file_path = tmp_path / "perturbation_invalid_condition.h5ad"
        adata = create_dummy_anndata(
            n_cells=6,
            n_genes=3,
            obs_columns=["split"],
            organism=Organism.HUMAN,
        )
        adata.obs["split"] = ["train", "train", "test", "test", "test", "test"]
        adata.write_h5ad(file_path)

        return file_path

    @pytest.fixture
    def perturbation_invalid_condition_h5ad(self, tmp_path) -> Path:
        """Creates a PerturbationSingleCellDataset with invalid condition format."""
        file_path = tmp_path / "perturbation_invalid_condition.h5ad"
        adata = create_dummy_anndata(
            n_cells=6,
            n_genes=3,
            obs_columns=["condition", "split"],
            organism=Organism.HUMAN,
        )
        adata.obs["condition"] = [
            "BADctrl",  
            "BADctrl",
            "test1",
            "test1",
            "test2",
            "test2",
        ]
        adata.obs["split"] = ["train", "train", "test", "test", "test", "test"]
        adata.write_h5ad(file_path)

        return file_path


    @pytest.fixture
    def perturbation_valid_conditions_h5ad(self, tmp_path) -> Path:
        """Creates a PerturbationSingleCellDataset with all valid condition formats."""
        file_path = tmp_path / "perturbation_valid.h5ad"
        adata = create_dummy_anndata(
            n_cells=9,
            n_genes=3,
            obs_columns=["condition", "split"],
            organism=Organism.HUMAN,
        )
        adata.obs["condition"] = [
            "ctrl",
            "ctrl",
            "ENSG00000123456+ctrl",  # Single gene perturbation
            "ENSG00000123456+ctrl",
            "ENSG00000123456+ENSG00000789012",  # Combinatorial perturbation
            "ENSG00000123456+ENSG00000789012",
            "ENSG00000111111+ctrl",
            "ENSG00000111111+ctrl",
            "ctrl",
        ]
        adata.obs["split"] = ["train"] * 3 + ["test"] * 6
        adata.write_h5ad(file_path)

        return file_path


    def test_perturbation_dataset_load_data(self, valid_dataset):
        """Tests the loading of perturbation dataset data."""
        
        valid_dataset.load_data()

        truth = valid_dataset.perturbation_truth
        assert "test1+ctrl" in truth
        assert "test2+ctrl" in truth
        assert valid_dataset.adata.shape == (2, 3)


    def test_perturbation_dataset_load_data_missing_condition_key(self, perturbation_missing_condition_column_h5ad):
        """Tests that loading data fails when the condition column is missing."""
        invalid_dataset = SingleCellPerturbationDataset(
            perturbation_missing_condition_column_h5ad,
            organism=Organism.HUMAN,
            condition_key="condition",
            split_key="split",
        )

        with pytest.raises(
            ValueError, match="Condition key 'condition' not found in adata.obs"
        ):
            invalid_dataset.load_data()
        


    def test_perturbation_dataset_load_data_missing_split_key(self, perturbation_missing_split_column_h5ad):
        """Tests that loading data fails when the split column is missing."""
        invalid_dataset = SingleCellPerturbationDataset(
            perturbation_missing_split_column_h5ad,
            organism=Organism.HUMAN,
            condition_key="condition",
            split_key="split",
        )

        with pytest.raises(ValueError, match="Split key 'split' not found in adata.obs"):
            invalid_dataset.load_data()


    def test_perturbation_dataset_validate_invalid_split(self, perturbation_invalid_split_h5ad):
        """Test that validation fails with invalid split values."""
        dataset = SingleCellPerturbationDataset(
            perturbation_invalid_split_h5ad,
            organism=Organism.HUMAN,
            condition_key="condition",
            split_key="split",
        )
        dataset.load_data()
        
        with pytest.raises(ValueError, match=re.escape("Invalid split value(s): {'BAD'}")):
            dataset.validate()


    def test_perturbation_dataset_validate_invalid_condition(self,
        perturbation_invalid_condition_h5ad,
    ):
        """Test that validation fails with invalid condition format."""
        dataset = SingleCellPerturbationDataset(
            perturbation_invalid_condition_h5ad,
            organism=Organism.HUMAN,
            condition_key="condition",
            split_key="split",
        )
        dataset.load_data()

        with pytest.raises(ValueError, match=""):
            dataset.validate()


    def test_perturbation_dataset_store_task_inputs(self, tmp_path, valid_dataset: SingleCellPerturbationDataset):
        """Tests that the store_task_inputs method writes labels to a file."""
        valid_dataset.load_data()
        
        valid_dataset.store_task_inputs()
        # TODO: Assert that multiple files are created for each condition. For now, we will just check one condition
        output_file = tmp_path / "dummy_perturbation_task_inputs" / "single_cell_perturbation" / "perturbation_truths" / "test1+ctrl.json"
        assert output_file.exists()
        truth_df = pd.read_json(output_file)
        assert not truth_df.empty
        assert ["cell_2", "cell_3"] == truth_df.index.tolist()
        assert ["ENSG00000000000", "ENSG00000000001", "ENSG00000000002"] == truth_df.columns.tolist()

