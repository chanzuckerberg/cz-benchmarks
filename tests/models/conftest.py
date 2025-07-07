import pytest
import numpy as np
import pandas as pd
from czbenchmarks.datasets.single_cell import SingleCellLabeledDataset
from czbenchmarks.datasets.perturbation_single_cell import PerturbationSingleCellDataset
from czbenchmarks.datasets.types import Organism, DataType
from tests.utils import create_dummy_anndata


@pytest.fixture
def dummy_single_cell_dataset(tmp_path):
    """Creates a SingleCellDataset with all required fields for model validation."""
    file_path = tmp_path / "dummy.h5ad"
    # Create anndata with all possible required obs/var columns that
    # any validator might need
    obs_columns = [
        "dataset_id",
        "assay",
        "suspension_type",
        "donor_id",
        "cell_type",
        "batch",
    ]
    var_columns = ["feature_id", "ensembl_id", "feature_name"]

    adata = create_dummy_anndata(
        n_cells=10,
        n_genes=20,
        obs_columns=obs_columns,
        var_columns=var_columns,
        organism=Organism.HUMAN,
    )
    adata.write_h5ad(file_path)

    dataset = SingleCellLabeledDataset(str(file_path), organism=Organism.HUMAN)
    dataset.load_data()
    dataset.set_input(DataType.METADATA, adata.obs)
    return dataset


@pytest.fixture
def dummy_perturbation_dataset(tmp_path):
    """Creates a PerturbationSingleCellDataset with required fields for model
    validation."""
    file_path = tmp_path / "dummy_perturbation.h5ad"

    # Create anndata with all possible required obs/var columns
    obs_columns = [
        "dataset_id",
        "assay",
        "suspension_type",
        "donor_id",
        "condition",
        "split",
    ]
    var_columns = ["feature_id", "ensembl_id", "feature_name"]

    adata = create_dummy_anndata(
        n_cells=10,
        n_genes=20,
        obs_columns=obs_columns,
        var_columns=var_columns,
        organism=Organism.HUMAN,
    )

    # Set up valid perturbation conditions and splits
    adata.obs["condition"] = (
        ["ctrl"] * 4
        + ["ENSG00000123456+ctrl"] * 3
        + ["ENSG00000123456+ENSG00000789012"] * 3
    )
    adata.obs["split"] = ["train"] * 4 + ["test"] * 6

    adata.write_h5ad(file_path)

    dataset = PerturbationSingleCellDataset(
        str(file_path),
        organism=Organism.HUMAN,
        condition_key="condition",
        split_key="split",
    )
    dataset.load_data()

    # Set required perturbation truth input
    dummy_truth = {
        "ENSG00000123456+ctrl": pd.DataFrame(
            np.ones((4, adata.n_vars)), columns=adata.var_names
        ),
        "ENSG00000123456+ENSG00000789012": pd.DataFrame(
            np.ones((4, adata.n_vars)), columns=adata.var_names
        ),
    }
    dataset.set_input(DataType.PERTURBATION_TRUTH, dummy_truth)
    dataset.set_input(DataType.METADATA, adata.obs)

    return dataset


@pytest.fixture
def dummy_mouse_dataset(tmp_path):
    """Creates a SingleCellDataset with mouse data for cross-species testing."""
    file_path = tmp_path / "dummy_mouse.h5ad"
    obs_columns = [
        "dataset_id",
        "assay",
        "suspension_type",
        "donor_id",
        "cell_type",
    ]
    var_columns = ["feature_id", "ensembl_id", "feature_name"]

    adata = create_dummy_anndata(
        n_cells=10,
        n_genes=20,
        obs_columns=obs_columns,
        var_columns=var_columns,
        organism=Organism.MOUSE,
    )
    adata.write_h5ad(file_path)

    dataset = SingleCellLabeledDataset(str(file_path), organism=Organism.MOUSE)
    dataset.load_data()
    dataset.set_input(DataType.METADATA, adata.obs)
    return dataset
