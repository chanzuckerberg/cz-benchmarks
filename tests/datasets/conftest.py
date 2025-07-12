from pathlib import Path
import pytest
import pandas as pd
from tests.utils import create_dummy_anndata
from czbenchmarks.datasets.types import Organism


@pytest.fixture
def invalid_labeled_human_h5ad_wrong_prefix(tmp_path) -> Path:
    """Creates a SingleCellLabeledDataset with invalid gene name prefixes."""
    file_path = tmp_path / "dummy_wrong.h5ad"
    # Create with wrong gene names but valid ensembl IDs in var
    gene_names = [f"BAD{i}" for i in range(1, 4)]

    # Use create_dummy_anndata but override the var names
    adata = create_dummy_anndata(
        n_cells=5,
        n_genes=3,
        obs_columns=[],
        organism=Organism.HUMAN,
    )
    adata.var_names = pd.Index(gene_names)

    adata.write_h5ad(file_path)

    return file_path
