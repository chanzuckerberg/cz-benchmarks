import pytest
import numpy as np
import pandas as pd
import anndata as ad

from tests.utils import create_dummy_anndata, DummyDataset
from czbenchmarks.datasets.types import DataType

# Define fixtures and other configuration options
# that can be shared across multiple test files


# Creates a dummy AnnData object with default characteristics.
@pytest.fixture
def dummy_anndata_fixture():
    return create_dummy_anndata(
        n_cells=5,
        n_genes=3,
        obs_columns=None,
        var_columns=None,
    )


# Creates a dummy dataset with AnnData and metadata inputs.
@pytest.fixture
def dummy_dataset(dummy_anndata_fixture):
    ds = DummyDataset("dummy_path")
    ds.set_input(DataType.ANNDATA, dummy_anndata_fixture)
    ds.set_input(DataType.METADATA, pd.DataFrame({"col": [1, 2, 3, 4, 5]}))
    return ds


# Creates a dummy AnnData object with valid gene names and saves it to a temporary file.
@pytest.fixture
def dummy_human_anndata(tmp_path):
    file_path = tmp_path / "dummy.h5ad"
    gene_names = [f"ENSG{i}" for i in range(1, 4)]
    X = np.ones((5, 3))
    obs = pd.DataFrame(
        {
            "dataset_id": [f"ds_{i}" for i in range(5)],
            "assay": [f"assay_{i}" for i in range(5)],
            "suspension_type": [f"type_{i}" for i in range(5)],
            "donor_id": [f"donor_{i}" for i in range(5)],
        },
        index=[f"cell_{i}" for i in range(5)],
    )
    var = pd.DataFrame(
        {"ensembl_id": gene_names, "other": gene_names}, index=gene_names
    )
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.write_h5ad(file_path)
    return str(file_path)


# Creates a dummy AnnData object with invalid gene name prefixes
# and saves it to a temporary file.
@pytest.fixture
def dummy_human_anndata_wrong_prefix(tmp_path):
    file_path = tmp_path / "dummy_wrong.h5ad"
    gene_names = [f"BAD{i}" for i in range(1, 4)]  # Not starting with 'ENSG'
    fallback_ensembl = [f"ENSG{i}" for i in range(1, 4)]
    X = np.ones((5, 3))
    obs = pd.DataFrame(
        {
            "dataset_id": [f"ds_{i}" for i in range(5)],
            "assay": [f"assay_{i}" for i in range(5)],
            "suspension_type": [f"type_{i}" for i in range(5)],
            "donor_id": [f"donor_{i}" for i in range(5)],
        },
        index=[f"cell_{i}" for i in range(5)],
    )
    var = pd.DataFrame({"ensembl_id": fallback_ensembl}, index=gene_names)
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.write_h5ad(file_path)
    return str(file_path)
