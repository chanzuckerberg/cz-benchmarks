import json
import numpy as np
from pathlib import Path
import pandas as pd
import pytest
import anndata as ad

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
            min_de_genes_to_mask=5,
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
    @pytest.mark.parametrize("min_de_genes_to_mask", [1, 5])
    @pytest.mark.parametrize("pval_threshold", [1e-4, 1e-2])
    def test_perturbation_dataset_load_data(
        self,
        tmp_path,
        deg_test_name,
        percent_genes_to_mask,
        min_de_genes_to_mask,
        pval_threshold,
    ):
        """Tests the loading of perturbation dataset data across parameter combinations."""
        condition_key = "condition"
        dataset = SingleCellPerturbationDataset(
            path=self.valid_dataset_file(tmp_path),
            organism=Organism.HUMAN,
            condition_key=condition_key,
            control_name="ctrl",
            deg_test_name=deg_test_name,
            percent_genes_to_mask=percent_genes_to_mask,
            min_de_genes_to_mask=min_de_genes_to_mask,
            pval_threshold=pval_threshold,
            min_logfoldchange=1.0,
        )

        dataset.load_data()

        # After loading, data should be created for each perturbation with matched controls
        # Expect 2 conditions (test1, test2), each with 2 perturbed + 2 control cells -> 8 total
        assert dataset.control_matched_adata.shape == (8, 3)
        # Target genes should be stored per cell (for each unique cell index)
        assert hasattr(dataset, "target_conditions_dict")
        unique_condition_count = len(
            np.unique(
                dataset.control_matched_adata.obs[condition_key][
                    ~dataset.control_matched_adata.obs[condition_key].str.startswith(
                        "ctrl"
                    )
                ]
            )
        )

        assert len(dataset.target_conditions_dict) == unique_condition_count
        # With 10 DE genes per condition in fixtures
        expected_sampled = int(10 * percent_genes_to_mask)
        sampled_lengths = {len(v) for v in dataset.target_conditions_dict.values()}
        assert sampled_lengths == {expected_sampled}

    def test_perturbation_dataset_load_data_missing_condition_key(
        self,
        perturbation_missing_condition_column_h5ad,
    ):
        condition_key = "condition"
        """Tests that loading data fails when the condition column is missing."""
        invalid_dataset = SingleCellPerturbationDataset(
            perturbation_missing_condition_column_h5ad,
            organism=Organism.HUMAN,
            condition_key=condition_key,
            deg_test_name="wilcoxon",
            percent_genes_to_mask=0.5,
            min_de_genes_to_mask=5,
            pval_threshold=1e-4,
            min_logfoldchange=1.0,
        )

        with pytest.raises(
            ValueError, match=f"Condition key '{condition_key}' not found in adata.obs"
        ):
            invalid_dataset.load_data()

    def test_perturbation_dataset_validate_invalid_condition(
        self,
        perturbation_invalid_condition_h5ad,
    ):
        """Test that validation fails with invalid condition format."""
        condition_key = "condition"
        dataset = SingleCellPerturbationDataset(
            perturbation_invalid_condition_h5ad,
            organism=Organism.HUMAN,
            condition_key=condition_key,
            deg_test_name="wilcoxon",
            percent_genes_to_mask=0.5,
            min_de_genes_to_mask=5,
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
        """Tests that the store_task_inputs method writes expected separate files."""
        condition_key = "condition"

        dataset = SingleCellPerturbationDataset(
            path=self.valid_dataset_file(tmp_path),
            organism=Organism.HUMAN,
            condition_key=condition_key,
            control_name="ctrl",
            deg_test_name=deg_test_name,
            percent_genes_to_mask=0.5,
            min_de_genes_to_mask=5,
            pval_threshold=1e-4,
            min_logfoldchange=1.0,
        )
        dataset.load_data()

        task_inputs_dir = dataset.store_task_inputs()
        assert task_inputs_dir.exists()
        assert task_inputs_dir.is_dir()

        # Check that all required files exist
        expected_files = [
            "control_matched_adata.h5ad",
            "control_cells_ids.json",
            "target_conditions_dict.json",
            "de_results.csv",
        ]

        for filename in expected_files:
            filepath = task_inputs_dir / filename
            assert filepath.exists(), f"Expected file {filename} not found"

        # Load and validate the main AnnData file

        task_adata = ad.read_h5ad(task_inputs_dir / "control_matched_adata.h5ad")
        assert isinstance(task_adata, ad.AnnData)

        # Load and validate JSON files
        with open(task_inputs_dir / "control_cells_ids.json", "r") as f:
            control_cells_ids = json.load(f)
        assert isinstance(control_cells_ids, dict)

        with open(task_inputs_dir / "target_conditions_dict.json", "r") as f:
            target_conditions_dict = json.load(f)
        assert isinstance(target_conditions_dict, dict)

        # Load and validate DE results CSV (should only have optimized columns)
        de_df = pd.read_csv(task_inputs_dir / "de_results.csv")
        assert not de_df.empty
        # Only the necessary columns should be present
        expected_cols = {condition_key, "gene_id"}
        if deg_test_name == "wilcoxon":
            expected_cols.add("logfoldchange")
        else:
            expected_cols.add("standardized_mean_diff")
        assert set(de_df.columns) == expected_cols

        # Load and validate cell barcode index
        cell_barcode_index = task_adata.uns["cell_barcode_index"]
        assert isinstance(cell_barcode_index, np.ndarray)
        assert len(cell_barcode_index) == dataset.adata.shape[0]

    @pytest.mark.parametrize("deg_test_name", ["wilcoxon", "t-test"])
    @pytest.mark.parametrize("percent_genes_to_mask", [0.5, 1.0])
    @pytest.mark.parametrize("min_de_genes_to_mask", [1, 5])
    @pytest.mark.parametrize("pval_threshold", [1e-4, 1e-2])
    def test_perturbation_dataset_load_de_results_from_csv(
        self,
        tmp_path,
        deg_test_name,
        percent_genes_to_mask,
        min_de_genes_to_mask,
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
            min_de_genes_to_mask=min_de_genes_to_mask,
            pval_threshold=pval_threshold,
            min_logfoldchange=0.0,
            de_results_path=csv_path,
        )

        dataset.load_data()

        # Expect 2 conditions (test1, test2), each with 2 perturbed + 2 control cells -> 8 total
        assert dataset.control_matched_adata.shape == (8, 3)

        # Target genes should be stored per cell (for each unique cell index)
        assert hasattr(dataset, "target_conditions_dict")
        unique_condition_count = len(
            np.unique(
                dataset.control_matched_adata.obs["condition"][
                    ~dataset.control_matched_adata.obs["condition"].str.startswith(
                        "ctrl"
                    )
                ]
            )
        )

        assert len(dataset.target_conditions_dict) == unique_condition_count

        # With 10 genes per condition and percent as parameter
        expected_sampled = int(10 * percent_genes_to_mask)
        sampled_lengths = {len(v) for v in dataset.target_conditions_dict.values()}
        assert sampled_lengths == {expected_sampled}

    @pytest.mark.parametrize("deg_test_name", ["wilcoxon", "t-test"])
    def test_control_matched_adata_contains_task_data(self, deg_test_name, tmp_path):
        """Test that control_matched_adata contains all required task data in uns."""
        dataset = SingleCellPerturbationDataset(
            path=self.valid_dataset_file(tmp_path),
            organism=Organism.HUMAN,
            condition_key="condition",
            control_name="ctrl",
            deg_test_name=deg_test_name,
            percent_genes_to_mask=0.5,
            min_de_genes_to_mask=2,
            pval_threshold=1e-4,
            min_logfoldchange=1.0,
        )
        dataset.load_data()

        # Verify that control_matched_adata exists and has the required keys in uns
        assert hasattr(dataset, "control_matched_adata")
        assert dataset.control_matched_adata is not None

        required_uns_keys = {
            "target_conditions_dict",
            "de_results",
            "cell_barcode_index",
            "control_cells_ids",
        }

        actual_uns_keys = set(dataset.control_matched_adata.uns.keys())
        assert required_uns_keys.issubset(actual_uns_keys), (
            f"Missing required keys in control_matched_adata.uns. "
            f"Required: {required_uns_keys}, Found: {actual_uns_keys}"
        )

        # Verify the contents of each key
        uns = dataset.control_matched_adata.uns

        # Check target_conditions_dict
        assert isinstance(uns["target_conditions_dict"], dict)
        assert len(uns["target_conditions_dict"]) > 0
        assert uns["target_conditions_dict"] == dataset.target_conditions_dict

        # Check control_cells_ids
        assert isinstance(uns["control_cells_ids"], dict)
        assert len(uns["control_cells_ids"]) > 0
        assert uns["control_cells_ids"] == dataset.control_cells_ids

        # Check de_results can be reconstructed as DataFrame (should only have optimized columns)
        assert isinstance(uns["de_results"], dict)
        de_df = pd.DataFrame(uns["de_results"])
        assert not de_df.empty
        # Only the necessary columns should be present
        expected_cols = {"condition", "gene_id"}
        if deg_test_name == "wilcoxon":
            expected_cols.add("logfoldchange")
        else:
            expected_cols.add("standardized_mean_diff")
        assert set(de_df.columns) == expected_cols

        # Check cell_barcode_index
        assert isinstance(uns["cell_barcode_index"], np.ndarray)
        assert len(uns["cell_barcode_index"]) == len(dataset.adata.obs.index)
        # Should match the original dataset's observation index
        np.testing.assert_array_equal(
            uns["cell_barcode_index"], dataset.adata.obs.index.astype(str).values
        )

    @pytest.mark.parametrize("deg_test_name", ["wilcoxon", "t-test"])
    def test_task_input_creation_from_control_matched_adata(
        self, deg_test_name, tmp_path
    ):
        """Test that PerturbationExpressionPredictionTaskInput can be created directly from control_matched_adata."""
        from czbenchmarks.tasks.single_cell.perturbation_expression_prediction import (
            PerturbationExpressionPredictionTaskInput,
        )

        dataset = SingleCellPerturbationDataset(
            path=self.valid_dataset_file(tmp_path),
            organism=Organism.HUMAN,
            condition_key="condition",
            control_name="ctrl",
            deg_test_name=deg_test_name,
            percent_genes_to_mask=0.5,
            min_de_genes_to_mask=2,
            pval_threshold=1e-4,
            min_logfoldchange=1.0,
        )
        dataset.load_data()

        # This should work without any errors since control_matched_adata contains all required data
        task_input = PerturbationExpressionPredictionTaskInput(
            adata=dataset.control_matched_adata,
            target_conditions_dict=dataset.target_conditions_dict,
            de_results=dataset.de_results,
        )

        # Verify the task input was created successfully
        assert task_input is not None
        assert hasattr(task_input, "adata")
        assert hasattr(task_input, "target_conditions_dict")
        assert hasattr(task_input, "de_results")
        assert task_input.adata is not None

        # Verify that required data in uns is accessible
        required_uns_keys = {
            "cell_barcode_index",
            "control_cells_ids",
        }
        actual_uns_keys = set(task_input.adata.uns.keys())
        assert required_uns_keys.issubset(actual_uns_keys)

        # Verify data integrity - the data should match the original dataset
        assert (
            task_input.adata.uns["target_conditions_dict"]
            == dataset.target_conditions_dict
        )
        assert task_input.adata.uns["control_cells_ids"] == dataset.control_cells_ids

        # Check that DE results can be reconstructed
        de_df = pd.DataFrame(task_input.adata.uns["de_results"])
        assert len(de_df) == len(dataset.de_results)

        # Check cell barcode index
        np.testing.assert_array_equal(
            task_input.adata.uns["cell_barcode_index"],
            dataset.adata.obs.index.astype(str).values,
        )
