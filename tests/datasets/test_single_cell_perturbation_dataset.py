from pathlib import Path
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
        # Set indices so that splitting on '_' and taking token [1] yields the condition
        adata.obs_names = [
            "ctrl_test1_a",  # control cell 1
            "ctrl_test2_b",  # control cell 2
            "cond_test1_a",
            "cond_test1_b",
            "cond_test2_a",
            "cond_test2_b",
        ]
        # Provide matched control cell IDs per condition using the two control cells above
        adata.uns["control_cells_ids"] = {
            "test1": ["ctrl_test1_a", "ctrl_test2_b"],
            "test2": ["ctrl_test1_a", "ctrl_test2_b"],
        }
        # Provide sufficient DE results to pass internal filtering and sampling
        de_conditions = ["test1"] * 10 + ["test2"] * 10
        de_genes = [f"ENSG000000000{str(i).zfill(2)}" for i in range(20)]
        adata.uns["de_results_wilcoxon"] = pd.DataFrame(
            {
                "condition": de_conditions,
                "gene": de_genes,
                "pval_adj": [1e-6] * 20,
                "logfoldchange": [2.0] * 20,
            }
        )
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
        # Ensure required uns keys exist so load_data() succeeds, and failure occurs at validate()
        adata.uns["control_cells_ids"] = {
            "test1": ["cell_0", "cell_1"],
            "test2": ["cell_0", "cell_1"],
        }
        de_conditions = ["test1"] * 10 + ["test2"] * 10
        de_genes = [f"ENSG000000000{str(i).zfill(2)}" for i in range(20)]
        adata.uns["de_results_wilcoxon"] = pd.DataFrame(
            {
                "condition": de_conditions,
                "gene": de_genes,
                "pval_adj": [1e-6] * 20,
                "logfoldchange": [2.0] * 20,
            }
        )
        adata.write_h5ad(file_path)

        return file_path

    def test_perturbation_dataset_load_data(self, valid_dataset):
        """Tests the loading of perturbation dataset data."""

        valid_dataset.load_data()

        # After loading, data should be created for each perturbation with matched controls
        # Expect 2 conditions (test1, test2), each with 2 perturbed + 2 control cells -> 8 total
        assert valid_dataset.control_matched_adata.shape == (8, 3)
        # Target genes should be stored per cell (for each unique cell index)
        assert hasattr(valid_dataset, "target_genes_to_save")
        unique_obs_count = len(
            set(valid_dataset.control_matched_adata.obs.index.tolist())
        )
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
        """Tests that the store_task_inputs method writes expected files."""
        valid_dataset.load_data()

        out_dir = valid_dataset.store_task_inputs()
        control_file = out_dir / "control_cells_ids.json"
        target_genes_file = out_dir / "target_genes_to_save.json"
        de_results_file = out_dir / "de_results.json"

        assert control_file.exists()
        assert target_genes_file.exists()
        assert de_results_file.exists()

        # Validate that DE results JSON is readable and has expected columns
        de_df = pd.read_json(de_results_file)
        assert not de_df.empty
        assert set(["condition", "gene", "pval_adj", "logfoldchange"]).issubset(
            set(de_df.columns)
        )

    def test_perturbation_dataset_load_from_task_inputs(
        self, tmp_path, valid_dataset: SingleCellPerturbationDataset
    ):
        """Tests that load_from_task_inputs restores dataset state saved by store_task_inputs."""
        # Prepare and store inputs from a loaded dataset
        valid_dataset.load_data()
        out_dir = valid_dataset.store_task_inputs()

        # Create a fresh dataset instance and load from stored inputs
        reloaded = SingleCellPerturbationDataset(
            path=valid_dataset.path,
            organism=Organism.HUMAN,
            condition_key="condition",
            control_name="ctrl",
        )
        reloaded.load_from_task_inputs(out_dir)

        # Verify core attributes are restored
        assert reloaded.control_cells_ids == valid_dataset.control_cells_ids
        assert reloaded.target_genes_to_save == valid_dataset.target_genes_to_save

        # DE results structure should match
        assert isinstance(reloaded.de_results, pd.DataFrame)
        assert list(reloaded.de_results.columns) == list(
            valid_dataset.de_results.columns
        )
        assert len(reloaded.de_results) == len(valid_dataset.de_results)

        # AnnData shape and annotations should match
        assert (
            reloaded.control_matched_adata.shape
            == valid_dataset.control_matched_adata.shape
        )
        assert reloaded.control_matched_adata.obs.equals(
            valid_dataset.control_matched_adata.obs
        )
        assert reloaded.control_matched_adata.var.equals(
            valid_dataset.control_matched_adata.var
        )
