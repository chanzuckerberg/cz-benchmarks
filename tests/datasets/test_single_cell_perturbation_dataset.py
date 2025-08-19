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
            deg_test_name="wilcoxon",
            percent_genes_to_mask=0.5,
            min_de_genes=5,
            pval_threshold=1e-4,
            min_logfoldchange=1.0,
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
        # Provide corresponding t-test DE results used when deg_test_name == "t-test"
        adata.uns["de_results_t_test"] = pd.DataFrame(
            {
                "condition": de_conditions,
                "gene": de_genes,
                "pval_adj": [1e-6] * 20,
                "standardized_mean_diff": [2.0] * 20,
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
        # Also include t-test results for parameterized runs with deg_test_name == "t-test"
        adata.uns["de_results_t_test"] = pd.DataFrame(
            {
                "condition": de_conditions,
                "gene": de_genes,
                "pval_adj": [1e-6] * 20,
                "standardized_mean_diff": [2.0] * 20,
            }
        )
        adata.write_h5ad(file_path)

        return file_path

    @pytest.mark.parametrize("deg_test_name", ["wilcoxon", "t-test"])
    @pytest.mark.parametrize("percent_genes_to_mask", [0.5, 1.0])
    @pytest.mark.parametrize("min_de_genes", [1, 5])
    @pytest.mark.parametrize("pval_threshold", [1e-4, 1e-2])
    def test_perturbation_dataset_load_data(
        self,
        tmp_path,
        deg_test_name,
        percent_genes_to_mask,
        min_de_genes,
        pval_threshold,
    ):
        """Tests the loading of perturbation dataset data across parameter combinations."""

        dataset = SingleCellPerturbationDataset(
            path=self.valid_dataset_file(tmp_path),
            organism=Organism.HUMAN,
            condition_key="condition",
            control_name="ctrl",
            deg_test_name=deg_test_name,
            percent_genes_to_mask=percent_genes_to_mask,
            min_de_genes=min_de_genes,
            pval_threshold=pval_threshold,
            min_logfoldchange=1.0,
        )

        dataset.load_data()

        # After loading, data should be created for each perturbation with matched controls
        # Expect 2 conditions (test1, test2), each with 2 perturbed + 2 control cells -> 8 total
        assert dataset.control_matched_adata.shape == (8, 3)
        # Target genes should be stored per cell (for each unique cell index)
        assert hasattr(dataset, "target_conditions_to_save")
        unique_obs_count = len(set(dataset.control_matched_adata.obs.index.tolist()))
        assert len(dataset.target_conditions_to_save) == unique_obs_count
        # With 10 DE genes per condition in fixtures
        expected_sampled = int(10 * percent_genes_to_mask)
        sampled_lengths = {len(v) for v in dataset.target_conditions_to_save.values()}
        assert sampled_lengths == {expected_sampled}

    def test_perturbation_dataset_load_data_missing_condition_key(
        self,
        perturbation_missing_condition_column_h5ad,
    ):
        """Tests that loading data fails when the condition column is missing."""
        invalid_dataset = SingleCellPerturbationDataset(
            perturbation_missing_condition_column_h5ad,
            organism=Organism.HUMAN,
            condition_key="condition",
            deg_test_name="wilcoxon",
            percent_genes_to_mask=0.5,
            min_de_genes=5,
            pval_threshold=1e-4,
            min_logfoldchange=1.0,
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
            deg_test_name="wilcoxon",
            percent_genes_to_mask=0.5,
            min_de_genes=5,
            pval_threshold=1e-4,
            min_logfoldchange=1.0,
        )
        dataset.load_data()
        with pytest.raises(ValueError, match=""):
            dataset.validate()

    @pytest.mark.parametrize("deg_test_name", ["wilcoxon", "t-test"])
    def test_perturbation_dataset_store_task_inputs(
        self,
        tmp_path,
        deg_test_name,
    ):
        """Tests that the store_task_inputs method writes expected files."""
        dataset = SingleCellPerturbationDataset(
            path=self.valid_dataset_file(tmp_path),
            organism=Organism.HUMAN,
            condition_key="condition",
            control_name="ctrl",
            deg_test_name=deg_test_name,
            percent_genes_to_mask=0.5,
            min_de_genes=5,
            pval_threshold=1e-4,
            min_logfoldchange=1.0,
        )
        dataset.load_data()

        out_dir = dataset.store_task_inputs()
        control_file = out_dir / "control_cells_ids.json"
        target_conditions_file = out_dir / "target_conditions_to_save.json"
        de_results_file = out_dir / "de_results.json"

        assert control_file.exists()
        assert target_conditions_file.exists()
        assert de_results_file.exists()

        # Validate that DE results JSON is readable and has expected columns
        de_df = pd.read_json(de_results_file)
        assert not de_df.empty
        base_cols = {"condition", "gene", "pval_adj"}
        assert base_cols.issubset(set(de_df.columns))
        if deg_test_name == "wilcoxon":
            assert "logfoldchange" in de_df.columns
        else:
            assert "standardized_mean_diff" in de_df.columns

    @pytest.mark.parametrize("deg_test_name", ["wilcoxon", "t-test"])
    @pytest.mark.parametrize("percent_genes_to_mask", [0.5, 1.0])
    @pytest.mark.parametrize("min_de_genes", [1, 5])
    @pytest.mark.parametrize("pval_threshold", [1e-4, 1e-2])
    def test_perturbation_dataset_load_de_results_from_csv(
        self,
        tmp_path,
        deg_test_name,
        percent_genes_to_mask,
        min_de_genes,
        pval_threshold,
    ):
        """Tests loading DE results from an external CSV via de_results_path."""
        # Create the base AnnData file using existing helper to ensure obs/uns layout
        h5ad_path = self.valid_dataset_file(tmp_path)

        # Create a DE results CSV with required columns for both tests
        # Include two conditions that match the AnnData: test1 and test2, 10 genes each
        csv_path = tmp_path / "de_results.csv"
        conditions = ["test1"] * 10 + ["test2"] * 10
        genes = [f"GENE_{i}" for i in range(20)]
        de_df = pd.DataFrame(
            {
                "condition": conditions,
                "gene": genes,
                "pval_adj": [1e-6] * 20,
                "logfoldchange": [2.0] * 20,
                "standardized_mean_diff": [2.0] * 20,
            }
        )
        de_df.to_csv(csv_path, index=False)

        # Construct dataset pointing to the CSV and with permissive thresholds
        dataset = SingleCellPerturbationDataset(
            path=h5ad_path,
            organism=Organism.HUMAN,
            condition_key="condition",
            control_name="ctrl",
            deg_test_name=deg_test_name,
            percent_genes_to_mask=percent_genes_to_mask,
            min_de_genes=min_de_genes,
            pval_threshold=pval_threshold,
            min_logfoldchange=0.0,
            de_results_path=csv_path,
        )

        dataset.load_data()

        # Expect 2 conditions (test1, test2), each with 2 perturbed + 2 control cells -> 8 total
        assert dataset.control_matched_adata.shape == (8, 3)

        # Target genes should be stored per cell (for each unique cell index)
        assert hasattr(dataset, "target_conditions_to_save")
        unique_obs_count = len(set(dataset.control_matched_adata.obs.index.tolist()))
        assert len(dataset.target_conditions_to_save) == unique_obs_count

        # With 10 genes per condition and percent as parameter
        expected_sampled = int(10 * percent_genes_to_mask)
        sampled_lengths = {len(v) for v in dataset.target_conditions_to_save.values()}
        assert sampled_lengths == {expected_sampled}
