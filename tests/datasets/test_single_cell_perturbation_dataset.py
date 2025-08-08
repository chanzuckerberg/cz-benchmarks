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
            control_name="ctrl",
        )

    def valid_dataset_file(self, tmp_path) -> Path:
        """Creates a valid SingleCellPerturbationDataset H5AD file."""
        file_path = tmp_path / "dummy_perturbation.h5ad"
        adata = create_dummy_anndata(
            n_cells=6,
            n_genes=3,
            obs_columns=["condition"],
            organism=Organism.HUMAN,
        )
        adata.obs["condition"] = [
            "ctrl",
            "ctrl",
            "test1",
            "test1",
            "test2",
            "test2",
        ]
        adata.uns["control_cells_ids"] = {
            "test1": ["cell_1", "cell_2"],
            "test2": ["cell_1", "cell_2"],
        }
        # Provide sufficient DE results to pass internal filtering and sampling
        de_conditions = ["test1"] * 10 + ["test2"] * 10
        de_genes = [
            f"ENSG000000000{str(i).zfill(2)}" for i in range(20)
        ]
        adata.uns["de_results_wilcoxon"] = pd.DataFrame(
            {
                "condition": de_conditions,
                "gene": de_genes,
                "pval": [1e-6] * 20,
                "logfoldchange": [2.0] * 20,
            }
        )
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
    def perturbation_missing_condition_column_h5ad(self, tmp_path) -> Path:
        """Creates a PerturbationSingleCellDataset with invalid condition format."""
        file_path = tmp_path / "perturbation_invalid_condition.h5ad"
        adata = create_dummy_anndata(
            n_cells=6,
            n_genes=3,
            obs_columns=[],
            organism=Organism.HUMAN,
        )
        adata.write_h5ad(file_path)

        return file_path

    @pytest.fixture
    def perturbation_invalid_condition_h5ad(self, tmp_path) -> Path:
        """Creates a PerturbationSingleCellDataset with invalid condition format."""
        file_path = tmp_path / "perturbation_invalid_condition.h5ad"
        adata = create_dummy_anndata(
            n_cells=6,
            n_genes=3,
            obs_columns=["condition"],
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
        adata.write_h5ad(file_path)

        return file_path

    @pytest.fixture
    def perturbation_valid_conditions_h5ad(self, tmp_path) -> Path:
        """Creates a PerturbationSingleCellDataset with all valid condition formats."""
        file_path = tmp_path / "perturbation_valid.h5ad"
        adata = create_dummy_anndata(
            n_cells=9,
            n_genes=3,
            obs_columns=["condition"],
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
        adata.write_h5ad(file_path)

        return file_path

    def test_perturbation_dataset_load_data(self, valid_dataset):
        """Tests the loading of perturbation dataset data."""

        valid_dataset.load_data()

        # After loading, data should be created for each perturbation with matched controls
        # Expect 2 conditions (test1, test2), each with 2 perturbed + 2 control cells -> 8 total
        assert valid_dataset.adata.shape == (8, 3)
        # Target genes should be stored per cell (for each unique cell index)
        assert hasattr(valid_dataset, "target_genes_to_save")
        unique_obs_count = len(set(valid_dataset.adata.obs.index.tolist()))
        assert len(valid_dataset.target_genes_to_save) == unique_obs_count
        # Each cell should have 5 sampled genes (50% of 10 per condition)
        sampled_lengths = {len(v) for v in valid_dataset.target_genes_to_save.values()}
        assert sampled_lengths == {5}

    def test_perturbation_dataset_load_data_missing_condition_key(
        self, perturbation_missing_condition_column_h5ad
    ):
        """Tests that loading data fails when the condition column is missing."""
        invalid_dataset = SingleCellPerturbationDataset(
            perturbation_missing_condition_column_h5ad,
            organism=Organism.HUMAN,
            condition_key="condition",
        )

        with pytest.raises(
            ValueError, match="Condition key 'condition' not found in adata.obs"
        ):
            invalid_dataset.load_data()

    def test_perturbation_dataset_validate_invalid_condition(
        self,
        perturbation_invalid_condition_h5ad,
    ):
        """Test that validation fails with invalid condition format."""
        dataset = SingleCellPerturbationDataset(
            perturbation_invalid_condition_h5ad,
            organism=Organism.HUMAN,
            condition_key="condition",
        )
        dataset.load_data()

        with pytest.raises(ValueError, match=""):
            dataset.validate()

    def test_perturbation_dataset_store_task_inputs(
        self, tmp_path, valid_dataset: SingleCellPerturbationDataset
    ):
        """Tests that the store_task_inputs method writes labels to a file."""
        valid_dataset.load_data()

        valid_dataset.store_task_inputs()
        # TODO: Assert that multiple files are created for each condition. For now, we will just check one condition
        output_file = (
            tmp_path
            / "dummy_perturbation_task_inputs"
            / "single_cell_perturbation"
            / "perturbation_truths"
            / "test1+ctrl.json"
        )
        assert output_file.exists()
        truth_df = pd.read_json(output_file)
        assert not truth_df.empty
        assert ["cell_2", "cell_3"] == truth_df.index.tolist()
        assert [
            "ENSG00000000000",
            "ENSG00000000001",
            "ENSG00000000002",
        ] == truth_df.columns.tolist()
