import pytest
import numpy as np
import pandas as pd
from czbenchmarks.datasets.single_cell import (
    SingleCellDataset,
    PerturbationSingleCellDataset,
)
from czbenchmarks.datasets.types import Organism, DataType
from tests.utils import create_dummy_anndata
import os


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

    dataset = SingleCellDataset(str(file_path), organism=Organism.HUMAN)
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

    dataset = SingleCellDataset(str(file_path), organism=Organism.MOUSE)
    dataset.load_data()
    dataset.set_input(DataType.METADATA, adata.obs)
    return dataset


# region model regression tests


def pytest_addoption(parser):
    parser.addoption(
        "--run-model-tests",
        action="store_true",
        default=False,
        help="Run model regression tests",
    )


def pytest_configure(config):
    pytest.run_model_tests = config.getoption("--run-model-tests")


@pytest.fixture
def run_model_tests(request):
    return request.config.getoption("--run-model-tests", default=False)


def parse_cases_from_env():
    cases = os.environ.get("MODEL_CASES")
    if not cases:
        return None
    parsed = []
    for case in cases.split(";"):
        parts = case.split(",")
        if len(parts) == 4:
            parsed.append(tuple(parts))
    return parsed


def pytest_generate_tests(metafunc):
    if {
        name in metafunc.fixturenames
        for name in ["model_name", "variant", "dataset_name", "task_name"]
    }:
        cases = parse_cases_from_env() or [
            ("SCGPT", "human", "human_spermatogenesis", "clustering"),
            ("SCVI", "homo_sapiens", "human_spermatogenesis", "clustering"),
            ("GENEFORMER", "gf_6L_30M", "human_spermatogenesis", "clustering"),
            ("SCGENEPT", "scgpt", "adamson_perturb", "perturbation"),
            ("UCE", "4l", "human_spermatogenesis", "clustering"),
            ("TRANSCRIPTFORMER", "tf-sapiens", "tsv2_bladder", "clustering"),
            ("AIDO", "aido_cell_3m", "human_spermatogenesis", "clustering"),
        ]
        metafunc.parametrize(
            ("model_name", "variant", "dataset_name", "task_name"), cases
        )


# endregion model regression tests
