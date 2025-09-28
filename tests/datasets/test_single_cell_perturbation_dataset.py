from pathlib import Path
import pandas as pd
import numpy as np
import pytest
import anndata as ad

from czbenchmarks.datasets.single_cell_perturbation import SingleCellPerturbationDataset
from czbenchmarks.datasets.types import Organism
from tests.datasets.test_single_cell_dataset import SingleCellDatasetTests
from tests.utils import create_dummy_anndata


class TestSingleCellPerturbationDataset(SingleCellDatasetTests):
    """Tests for the SingleCellPerturbationDataset class."""

    @pytest.fixture
    def valid_dataset(
        self,
        tmp_path,
        condition_key: str = "condition",
        control_name: str = "ctrl",
        condition_control_sep: str = "_",
        de_gene_col: str = "gene",
    ) -> SingleCellPerturbationDataset:
        """Fixture to provide a valid SingleCellPerturbationDataset H5AD file."""

        return SingleCellPerturbationDataset(
            path=self.valid_dataset_file(
                tmp_path,
                condition_key=condition_key,
                control_name=control_name,
                condition_control_sep=condition_control_sep,
                de_gene_col=de_gene_col,
            ),
            organism=Organism.HUMAN,
            condition_key=condition_key,
            control_name=control_name,
            condition_control_sep=condition_control_sep,
            de_gene_col=de_gene_col,
            percent_genes_to_mask=0.5,
            min_de_genes_to_mask=5,
            pval_threshold=1e-4,
            min_logfoldchange=1.0,
        )

    def valid_dataset_file(
        self,
        tmp_path,
        condition_key: str = "condition",
        control_name: str = "ctrl",
        condition_control_sep: str = "_",
        de_gene_col: str = "gene",
    ) -> Path:
        """Creates a valid SingleCellPerturbationDataset H5AD file."""
        file_path = tmp_path / "dummy_perturbation.h5ad"
        adata = create_dummy_anndata(
            n_cells=6,
            n_genes=3,
            obs_columns=[condition_key],
            organism=Organism.HUMAN,
        )
        # Use granular barcodes for obs_names, but high-level conditions in obs
        obs_names = [
            condition_control_sep.join([control_name, "test1a"]),  # control cell 1
            condition_control_sep.join([control_name, "test2b"]),  # control cell 2
            "test1a",
            "test1b",
            "test2a",
            "test2b",
        ]
        adata.obs_names = obs_names
        adata.obs[condition_key] = [
            condition_control_sep.join([control_name, "test1"]),
            condition_control_sep.join([control_name, "test2"]),
            "test1",
            "test1",
            "test2",
            "test2",
        ]

        # Provide matched control cell IDs per condition using the two control cells above
        adata.uns["control_cells_ids"] = {
            "test1": {
                "test1a": condition_control_sep.join([control_name, "test1a"]),
                "test2a": condition_control_sep.join([control_name, "test2a"]),
            },
            "test2": {
                "test1b": condition_control_sep.join([control_name, "test1b"]),
                "test2b": condition_control_sep.join([control_name, "test2b"]),
            },
        }
        # Provide sufficient DE results to pass internal filtering and sampling
        de_conditions = ["test1"] * 10 + ["test2"] * 10
        de_genes = [f"ENSG000000000{str(i).zfill(2)}" for i in range(20)]
        adata.uns["de_results_wilcoxon"] = pd.DataFrame(
            {
                condition_key: de_conditions,
                de_gene_col: de_genes,
                "pval_adj": [1e-6] * 20,
                "logfoldchange": [2.0] * 20,
            }
        )
        adata.write_h5ad(file_path)

        return file_path

    @pytest.fixture
    def perturbation_missing_condition_key_h5ad(self, tmp_path) -> Path:
        """Creates a PerturbationSingleCellDataset with invalid condition format."""
        file_path = tmp_path / "perturbation_invalid_condition.h5ad"
        adata = create_dummy_anndata(
            n_cells=6,
            n_genes=3,
            obs_columns=[],
            organism=Organism.HUMAN,
        )
        # Provide required uns keys so load_data() proceeds far enough to access obs
        adata.uns["control_cells_ids"] = {
            "test1": {"cell0": "ctrl_test1"},
            "test2": {"cell1": "ctrl_test2"},
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

    @pytest.fixture
    def perturbation_invalid_condition_h5ad(
        self, tmp_path, condition_key: str = "condition", de_gene_col: str = "gene"
    ) -> Path:
        """Creates a PerturbationSingleCellDataset with invalid condition format."""
        file_path = tmp_path / "perturbation_invalid_condition.h5ad"
        adata = create_dummy_anndata(
            n_cells=6,
            n_genes=3,
            obs_columns=[condition_key],
            organism=Organism.HUMAN,
        )
        adata.obs[condition_key] = [
            "BAD0",
            "BAD1",
            "test1",
            "test1",
            "test2",
            "test2",
        ]
        # Ensure required uns keys exist so load_data() succeeds, and failure occurs at validate()
        adata.uns["control_cells_ids"] = {
            "test1": {"cell0": "BADctrl0", 
                      "cell1": "BADctrl1"},
            "test2": {"cell0": "BADctrl0", 
                      "cell1": "BADctrl1"},
        }
        de_conditions = ["test1"] * 10 + ["test2"] * 10
        de_genes = [f"ENSG000000000{str(i).zfill(2)}" for i in range(20)]
        adata.uns["de_results_wilcoxon"] = pd.DataFrame(
            {
                condition_key: de_conditions,
                de_gene_col: de_genes,
                "pval_adj": [1e-6] * 20,
                "logfoldchange": [2.0] * 20,
            }
        )
        adata.write_h5ad(file_path)

        return file_path

    # TODO split the naming and parameter value tests into separate tests
    @pytest.mark.parametrize("condition_key", ["condition", "perturbation"])
    @pytest.mark.parametrize("control_name", ["ctrl", "control"])
    @pytest.mark.parametrize("condition_control_sep", ["_", "+"])
    @pytest.mark.parametrize("de_gene_col", ["gene_id", "gene_name"])
    @pytest.mark.parametrize("percent_genes_to_mask", [0.5, 1.0])
    @pytest.mark.parametrize("min_de_genes_to_mask", [1, 5])
    @pytest.mark.parametrize("pval_threshold", [1e-4, 1e-2])
    def test_perturbation_dataset_load_data(
        self,
        tmp_path,
        percent_genes_to_mask: float,
        min_de_genes_to_mask: int,
        pval_threshold: float,
        condition_key: str,
        control_name: str,
        condition_control_sep: str,
        de_gene_col: str,
    ):
        """Tests the loading of perturbation dataset data across parameter combinations."""

        dataset = SingleCellPerturbationDataset(
            path=self.valid_dataset_file(
                tmp_path,
                condition_key=condition_key,
                control_name=control_name,
                condition_control_sep=condition_control_sep,
                de_gene_col=de_gene_col,
            ),
            organism=Organism.HUMAN,
            condition_key=condition_key,
            control_name=control_name,
            condition_control_sep=condition_control_sep,
            de_gene_col=de_gene_col,
            percent_genes_to_mask=percent_genes_to_mask,
            min_de_genes_to_mask=min_de_genes_to_mask,
            pval_threshold=pval_threshold,
            min_logfoldchange=1.0,
        )

        dataset.load_data()

        assert hasattr(dataset, "control_matched_adata")
        assert isinstance(dataset.control_matched_adata, ad.AnnData)
        # After loading, data should be created for each perturbation with matched controls
        # Expect 2 conditions (test1, test2), each with 2 perturbed + 2 control cells -> 8 total
        assert dataset.control_matched_adata.shape == (8, 3)
        assert condition_key in dataset.control_matched_adata.obs.columns
        # assert de_gene_col in dataset.control_matched_adata.var.columns
        assert dataset.control_matched_adata.obs[condition_key].str.contains(control_name).any()
        # Control labels are formatted as "{control_name}{condition_control_sep}{condition}"
        assert (
            dataset.control_matched_adata.obs[condition_key]
            .astype(str)
            .str.contains(
                f"{control_name}{condition_control_sep}test1", regex=False
            )
            .any()
        )
        assert (
            dataset.control_matched_adata.obs[condition_key]
            .astype(str)
            .str.contains(
                f"{control_name}{condition_control_sep}test2", regex=False
            )
            .any()
        )

        # Target genes should be stored per cell (for each unique cell index)
        assert hasattr(dataset, "target_conditions_dict")
        unique_condition_count = len(
            np.unique(
                dataset.control_matched_adata.obs[condition_key][
                    ~dataset.control_matched_adata.obs[condition_key].str.startswith(
                        control_name
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
        perturbation_missing_condition_key_h5ad,
        condition_key: str = "condition",
        control_name: str = "ctrl",
        condition_control_sep: str = "_",
        de_gene_col: str = "gene",
    ):
        """Tests that loading data fails when the condition column is missing."""
        invalid_dataset = SingleCellPerturbationDataset(
            perturbation_missing_condition_key_h5ad,
            organism=Organism.HUMAN,
            condition_key=condition_key,
            control_name=control_name,
            condition_control_sep=condition_control_sep,
            de_gene_col=de_gene_col,
            percent_genes_to_mask=0.5,
            min_de_genes_to_mask=5,
            pval_threshold=1e-4,
            min_logfoldchange=1.0,
        )

        # Framework currently loads first and accesses uns before validating obs columns
        with pytest.raises(KeyError):
            invalid_dataset.load_data()

    def test_perturbation_dataset_validate_invalid_condition(
        self,
        perturbation_invalid_condition_h5ad,
        condition_key: str = "condition",
        control_name: str = "ctrl",
        condition_control_sep: str = "_",
        de_gene_col: str = "gene",
    ):
        """Test that validation fails with invalid condition format."""
        dataset = SingleCellPerturbationDataset(
            perturbation_invalid_condition_h5ad,
            organism=Organism.HUMAN,
            condition_key=condition_key,
            control_name=control_name,
            condition_control_sep=condition_control_sep,
            de_gene_col=de_gene_col,
            percent_genes_to_mask=0.5,
            min_de_genes_to_mask=5,
            pval_threshold=1e-4,
            min_logfoldchange=1.0,
        )
        dataset.load_data()
        with pytest.raises(ValueError, match=""):
            dataset.validate()


    def test_custom_input_tasks_dir_is_used(self, tmp_path, valid_dataset):
        """Test that custom task inputs directory is used if provided"""
        custom_task_inputs_dir = tmp_path / "custom_task_inputs"
        valid_dataset.task_inputs_dir = custom_task_inputs_dir

        valid_dataset.load_data()
        # Avoid AnnData write conflict where index name equals column name
        valid_dataset.control_matched_adata.obs.index.name = None

        valid_dataset.store_task_inputs()

        assert custom_task_inputs_dir.exists()
        assert len(list(custom_task_inputs_dir.iterdir())) > 0


    def test_perturbation_dataset_store_task_inputs(
        self,
        tmp_path,
        condition_key: str = "condition",
        control_name: str = "ctrl",
        condition_control_sep: str = "_",
        de_gene_col: str = "gene",
    ):
        """Tests that the store_task_inputs method writes expected files."""
        dataset = SingleCellPerturbationDataset(
            path=self.valid_dataset_file(
                tmp_path,
                condition_control_sep=condition_control_sep,
                de_gene_col=de_gene_col,
            ),
            organism=Organism.HUMAN,
            condition_key=condition_key,
            control_name=control_name,
            condition_control_sep=condition_control_sep,
            de_gene_col=de_gene_col,
            percent_genes_to_mask=0.5,
            min_de_genes_to_mask=5,
            pval_threshold=1e-4,
            min_logfoldchange=1.0,
        )
        dataset.load_data()

        out_dir = dataset.store_task_inputs()
        adata_file = out_dir / "control_matched_adata.h5ad"
        target_conditions_file = out_dir / "target_conditions_dict.json"
        de_results_file = out_dir / "de_results.parquet"

        assert adata_file.exists()
        assert target_conditions_file.exists()
        assert de_results_file.exists()

        # Validate that DE results parquet is readable and has expected columns
        de_df = pd.read_parquet(de_results_file)
        assert not de_df.empty
        base_cols = {condition_key, de_gene_col, "pval_adj"}
        assert base_cols.issubset(set(de_df.columns))
        assert "logfoldchange" in de_df.columns
