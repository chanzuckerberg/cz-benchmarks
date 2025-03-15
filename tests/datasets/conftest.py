import pytest
import pandas as pd
import numpy as np
import anndata as ad
from tests.utils import create_dummy_anndata, DummyDataset
from czbenchmarks.datasets.types import DataType, Organism
from czbenchmarks.datasets.single_cell import (
    SingleCellDataset,
    PerturbationSingleCellDataset,
)


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
    """Creates a SingleCellDataset with valid human gene names."""
    file_path = tmp_path / "dummy.h5ad"
    adata = create_dummy_anndata(
        n_cells=5,
        n_genes=3,
        obs_columns=["dataset_id", "assay", "suspension_type", "donor_id"],
        organism=Organism.HUMAN,
    )
    adata.write_h5ad(file_path)

    dataset = SingleCellDataset(str(file_path), organism=Organism.HUMAN)
    dataset.load_data()
    return dataset


@pytest.fixture
def dummy_human_anndata_wrong_prefix(tmp_path):
    """Creates a SingleCellDataset with invalid gene name prefixes."""
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

    dataset = SingleCellDataset(str(file_path), organism=Organism.HUMAN)
    dataset.load_data()
    return dataset


@pytest.fixture
def empty_anndata(tmp_path):
    """Creates a SingleCellDataset with an empty AnnData object."""
    file_path = tmp_path / "empty.h5ad"
    adata = ad.AnnData(X=np.empty((0, 0)))
    adata.write_h5ad(file_path)

    dataset = SingleCellDataset(str(file_path), organism=Organism.HUMAN)
    dataset.load_data()
    return dataset


@pytest.fixture
def float_counts_anndata(tmp_path):
    """Creates a SingleCellDataset with float counts instead of integers."""
    file_path = tmp_path / "float_counts.h5ad"
    adata = create_dummy_anndata(n_cells=5, n_genes=3, organism=Organism.HUMAN)
    adata.X = np.ones((5, 3), dtype=np.float32)  # Override X to be float
    adata.write_h5ad(file_path)

    dataset = SingleCellDataset(str(file_path), organism=Organism.HUMAN)
    dataset.load_data()
    return dataset


@pytest.fixture
def dummy_perturbation_anndata(tmp_path):
    """Creates a PerturbationSingleCellDataset with valid data."""
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

    dataset = PerturbationSingleCellDataset(
        str(file_path),
        organism=Organism.HUMAN,
        condition_key="condition",
        split_key="split",
    )
    dataset.load_data()
    return dataset


@pytest.fixture
def perturbation_missing_condition(tmp_path):
    """Creates a PerturbationSingleCellDataset missing the condition column."""
    file_path = tmp_path / "perturbation_missing_condition.h5ad"
    adata = create_dummy_anndata(
        n_cells=6, n_genes=3, obs_columns=["split"], organism=Organism.HUMAN
    )
    adata.obs["split"] = ["train", "train", "test", "test", "test", "test"]
    adata.write_h5ad(file_path)

    dataset = PerturbationSingleCellDataset(
        str(file_path),
        organism=Organism.HUMAN,
        condition_key="condition",
        split_key="split",
    )
    return dataset


@pytest.fixture
def perturbation_missing_split(tmp_path):
    """Creates a PerturbationSingleCellDataset missing the split column."""
    file_path = tmp_path / "perturbation_missing_split.h5ad"
    adata = create_dummy_anndata(
        n_cells=6, n_genes=3, obs_columns=["condition"], organism=Organism.HUMAN
    )
    adata.obs["condition"] = ["ctrl", "ctrl", "test1", "test1", "test2", "test2"]
    adata.write_h5ad(file_path)

    dataset = PerturbationSingleCellDataset(
        str(file_path),
        organism=Organism.HUMAN,
        condition_key="condition",
        split_key="split",
    )
    return dataset


@pytest.fixture
def perturbation_invalid_split(tmp_path):
    """Creates a PerturbationSingleCellDataset with invalid split values."""
    file_path = tmp_path / "perturbation_invalid_split.h5ad"
    adata = create_dummy_anndata(
        n_cells=6,
        n_genes=3,
        obs_columns=["condition", "split"],
        organism=Organism.HUMAN,
    )
    adata.obs["condition"] = ["ctrl", "ctrl", "test1", "test1", "test2", "test2"]
    adata.obs["split"] = ["invalid", "train", "test", "test", "test", "test"]
    adata.write_h5ad(file_path)

    dataset = PerturbationSingleCellDataset(
        str(file_path),
        organism=Organism.HUMAN,
        condition_key="condition",
        split_key="split",
    )
    dataset.load_data()
    return dataset


@pytest.fixture
def perturbation_invalid_condition(tmp_path):
    """Creates a PerturbationSingleCellDataset with invalid condition format."""
    file_path = tmp_path / "perturbation_invalid_condition.h5ad"
    adata = create_dummy_anndata(
        n_cells=6,
        n_genes=3,
        obs_columns=["condition", "split"],
        organism=Organism.HUMAN,
    )
    adata.obs["condition"] = [
        "control",  # "control" instead of "ctrl"
        "control",
        "test1",
        "test1",
        "test2",
        "test2",
    ]
    adata.obs["split"] = ["train", "train", "test", "test", "test", "test"]
    adata.write_h5ad(file_path)

    dataset = PerturbationSingleCellDataset(
        str(file_path),
        organism=Organism.HUMAN,
        condition_key="condition",
        split_key="split",
    )
    dataset.load_data()
    return dataset


@pytest.fixture
def perturbation_invalid_combo(tmp_path):
    """Creates a PerturbationSingleCellDataset with invalid combo perturbation."""
    file_path = tmp_path / "perturbation_invalid_combo.h5ad"
    adata = create_dummy_anndata(
        n_cells=6,
        n_genes=3,
        obs_columns=["condition", "split"],
        organism=Organism.HUMAN,
    )
    adata.obs["condition"] = [
        "ctrl",
        "ctrl",
        "ENSG00000123456+INVALID123",  # Second gene has wrong prefix
        "ENSG00000123456+INVALID123",
        "test2+ctrl",
        "test2+ctrl",
    ]
    adata.obs["split"] = ["train", "train", "test", "test", "test", "test"]
    adata.write_h5ad(file_path)

    dataset = PerturbationSingleCellDataset(
        str(file_path),
        organism=Organism.HUMAN,
        condition_key="condition",
        split_key="split",
    )
    dataset.load_data()
    return dataset


@pytest.fixture
def perturbation_valid_conditions(tmp_path):
    """Creates a PerturbationSingleCellDataset with all valid condition formats."""
    file_path = tmp_path / "perturbation_valid.h5ad"
    adata = create_dummy_anndata(
        n_cells=9,
        n_genes=3,
        obs_columns=["condition", "split"],
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
    adata.obs["split"] = ["train"] * 3 + ["test"] * 6
    adata.write_h5ad(file_path)

    dataset = PerturbationSingleCellDataset(
        str(file_path),
        organism=Organism.HUMAN,
        condition_key="condition",
        split_key="split",
    )
    dataset.load_data()
    return dataset
