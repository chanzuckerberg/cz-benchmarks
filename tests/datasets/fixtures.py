import pytest
import pandas as pd
from tests.utils import create_dummy_anndata, DummyDataset
from czbenchmarks.datasets.types import DataType, Organism


@pytest.fixture
def dummy_anndata():
    """Creates a dummy AnnData object with default characteristics."""
    return create_dummy_anndata()


@pytest.fixture
def dummy_dataset(dummy_anndata):
    """Creates a dummy dataset with AnnData and metadata inputs."""
    ds = DummyDataset("dummy_path")
    ds.set_input(DataType.ANNDATA, dummy_anndata)
    ds.set_input(DataType.METADATA, pd.DataFrame({"col": [1, 2, 3, 4, 5]}))
    return ds


@pytest.fixture
def dummy_human_anndata(tmp_path):
    """Creates a dummy AnnData object with valid human gene names."""
    file_path = tmp_path / "dummy.h5ad"
    adata = create_dummy_anndata(
        n_cells=5,
        n_genes=3,
        obs_columns=["dataset_id", "assay", "suspension_type", "donor_id"],
        organism=Organism.HUMAN,
    )
    adata.write_h5ad(file_path)
    return str(file_path)


@pytest.fixture
def dummy_human_anndata_wrong_prefix(tmp_path):
    """Creates a dummy AnnData object with invalid gene name prefixes."""
    file_path = tmp_path / "dummy_wrong.h5ad"
    # Create with wrong gene names but valid ensembl IDs in var
    gene_names = [f"BAD{i}" for i in range(1, 4)]
    fallback_ensembl = [f"ENSG{i}" for i in range(1, 4)]

    # Use create_dummy_anndata but override the var names
    adata = create_dummy_anndata(
        n_cells=5,
        n_genes=3,
        obs_columns=["dataset_id", "assay", "suspension_type", "donor_id"],
        organism=Organism.HUMAN,
    )
    adata.var_names = pd.Index(gene_names)
    adata.var["ensembl_id"] = fallback_ensembl

    adata.write_h5ad(file_path)
    return str(file_path)


@pytest.fixture
def dummy_perturbation_anndata(tmp_path):
    """Creates a dummy AnnData object for perturbation testing."""
    file_path = tmp_path / "dummy_perturbation.h5ad"
    adata = create_dummy_anndata(
        n_cells=6,
        n_genes=3,
        obs_columns=["condition", "split"],
        organism=Organism.HUMAN,
    )
    # Override the default obs values for perturbation testing
    adata.obs["condition"] = [
        "ctrl",
        "ctrl",
        "test1+ctrl",
        "test1+ctrl",
        "test2+ctrl",
        "test2+ctrl",
    ]
    adata.obs["split"] = ["train", "train", "test", "test", "test", "test"]

    adata.write_h5ad(file_path)
    return str(file_path)
