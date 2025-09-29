import json
import numpy as np
from pathlib import Path
import pandas as pd
import pytest
import anndata as ad

from czbenchmarks.datasets.single_cell_perturbation import SingleCellPerturbationDataset
from czbenchmarks.datasets.types import Organism
from czbenchmarks.tasks.single_cell.perturbation_expression_prediction import (
    PerturbationExpressionPredictionTaskInput,
)
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
        # Strict 1-1 control map: condition -> {treated_barcode -> control_barcode}
        adata.uns["control_cells_map"] = {
            "test1": {
                "cond_test1_a": "ctrl_test1_a",
                "cond_test1_b": "ctrl_test2_b",
            },
            "test2": {
                "cond_test2_a": "ctrl_test1_a",
                "cond_test2_b": "ctrl_test2_b",
            },
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
        # Provide a minimal strict map using default obs_names from create_dummy_anndata
        adata.uns["control_cells_map"] = {
            "test1": {"cell_2": "cell_0", "cell_3": "cell_1"},
            "test2": {"cell_4": "cell_0", "cell_5": "cell_1"},
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

    @pytest.mark.parametrize("percent_genes_to_mask", [0.5, 1.0])
    @pytest.mark.parametrize("min_de_genes_to_mask", [1, 5])
    @pytest.mark.parametrize("pval_threshold", [1e-4, 1e-2])
    def test_perturbation_dataset_load_data(
        self,
        tmp_path,
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
            percent_genes_to_mask=percent_genes_to_mask,
            min_de_genes_to_mask=min_de_genes_to_mask,
            pval_threshold=pval_threshold,
            min_logfoldchange=1.0,
        )

        dataset.load_data()

        # Verify dataset was loaded and UNS prepared for task
        assert dataset.adata.shape == (6, 3)
        # Target genes should be populated
        assert hasattr(dataset, "target_conditions_dict")
        unique_condition_count = len(
            np.unique(
                dataset.adata.obs[condition_key][
                    ~dataset.adata.obs[condition_key].str.startswith("ctrl")
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
            condition_key="condition",
            control_name="ctrl",
            percent_genes_to_mask=0.5,
            min_de_genes_to_mask=5,
            pval_threshold=1e-4,
            min_logfoldchange=1.0,
        )

        with pytest.raises(
            ValueError, match=f"Condition key '{condition_key}' not found in adata.obs"
        ):
            invalid_dataset.load_data()

    def test_perturbation_dataset_store_task_inputs(
        self,
        tmp_path,
    ):
        """Tests that the store_task_inputs method writes the task inputs h5ad."""
        condition_key = "condition"

        dataset = SingleCellPerturbationDataset(
            path=self.valid_dataset_file(tmp_path),
            organism=Organism.HUMAN,
            condition_key="condition",
            control_name="ctrl",
            percent_genes_to_mask=0.5,
            min_de_genes_to_mask=5,
            pval_threshold=1e-4,
            min_logfoldchange=1.0,
        )
        dataset.load_data()

        task_inputs_dir = dataset.store_task_inputs()
        assert task_inputs_dir.exists()
        assert task_inputs_dir.is_dir()

        # Check the single h5ad exists
        h5_path = task_inputs_dir / "perturbation_task_inputs.h5ad"
        assert h5_path.exists(), "Expected perturbation_task_inputs.h5ad to be written"

        # Load and validate the AnnData file and required UNS keys
        task_adata = ad.read_h5ad(h5_path)
        assert isinstance(task_adata, ad.AnnData)
        assert "de_results" in task_adata.uns
        assert "target_conditions_dict" in task_adata.uns
        assert "control_cells_map" in task_adata.uns

    def test_dataset_adata_contains_task_data(self, tmp_path):
        """Test that dataset.adata contains all required task data in uns."""
        dataset = SingleCellPerturbationDataset(
            path=self.valid_dataset_file(tmp_path),
            organism=Organism.HUMAN,
            condition_key="condition",
            control_name="ctrl",
            percent_genes_to_mask=0.5,
            min_de_genes_to_mask=2,
            pval_threshold=1e-4,
            min_logfoldchange=1.0,
        )
        dataset.load_data()

        # Verify that dataset.adata has the required keys in uns
        required_uns_keys = {
            "target_conditions_dict",
            "de_results",
            "control_cells_map",
        }

        actual_uns_keys = set(dataset.adata.uns.keys())
        assert required_uns_keys.issubset(actual_uns_keys), (
            f"Missing required keys in adata.uns. "
            f"Required: {required_uns_keys}, Found: {actual_uns_keys}"
        )

        # Verify the contents of each key
        uns = dataset.adata.uns

        # Check target_conditions_dict
        assert isinstance(uns["target_conditions_dict"], dict)
        assert len(uns["target_conditions_dict"]) > 0
        assert uns["target_conditions_dict"] == dataset.target_conditions_dict

        # Check control_cells_map
        assert isinstance(uns["control_cells_map"], dict)
        assert len(uns["control_cells_map"]) > 0
        assert uns["control_cells_map"] == dataset.control_cells_ids

        # Check de_results is a DataFrame with only necessary columns
        de_df = uns["de_results"]
        assert isinstance(de_df, pd.DataFrame)
        assert not de_df.empty
        expected_cols = {"condition", "gene_id", "logfoldchange"}
        assert set(de_df.columns) == expected_cols

    @pytest.mark.parametrize("percent_genes_to_mask", [0.5, 1.0])
    @pytest.mark.parametrize("min_de_genes_to_mask", [1, 5])
    @pytest.mark.parametrize("pval_threshold", [1e-4, 1e-2])
    def test_perturbation_dataset_load_de_results_from_csv(
        self,
        tmp_path,
        percent_genes_to_mask,
        min_de_genes_to_mask,
        pval_threshold,
    ):
        """Tests loading DE results from an external CSV via de_results_path."""
        # Create the base AnnData file using existing helper to ensure obs/uns layout
        self.valid_dataset_file(tmp_path)

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
            }
        )
        de_df.to_csv(csv_path, index=False)

        # Construct dataset pointing to the CSV and with permissive thresholds
        dataset = SingleCellPerturbationDataset(
            path=self.valid_dataset_file(tmp_path),
            organism=Organism.HUMAN,
            condition_key="condition",
            control_name="ctrl",
            percent_genes_to_mask=percent_genes_to_mask,
            min_de_genes_to_mask=min_de_genes_to_mask,
            pval_threshold=pval_threshold,
            min_logfoldchange=0.0,
        )
        dataset.load_data()

        # Build task input from dataset.adata
        task_input = PerturbationExpressionPredictionTaskInput(
            adata=dataset.adata,
            gene_index=dataset.adata.var.index,
            cell_index=dataset.adata.obs.index,
        )

        # Verify the task input was created successfully
        assert task_input is not None
        assert hasattr(task_input, "adata")
        assert task_input.adata is not None

        # Verify that required data in uns is accessible
        required_uns_keys = {"control_cells_map"}
        actual_uns_keys = task_input.adata.uns.keys()
        assert required_uns_keys.issubset(actual_uns_keys)

        # Verify data integrity - the data should match the original dataset
        assert (
            task_input.adata.uns["target_conditions_dict"]
            == dataset.target_conditions_dict
        )
        assert task_input.adata.uns["control_cells_map"] == dataset.control_cells_ids

        # Check that DE results can be reconstructed
        de_df = task_input.adata.uns["de_results"]
        assert len(de_df) == len(dataset.de_results)

        # Check cell barcode index
        # Cell barcode index is not persisted separately; dataset indices are used directly
