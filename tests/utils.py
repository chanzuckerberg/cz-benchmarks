import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

from czbenchmarks.datasets.base import BaseDataset
from czbenchmarks.datasets.types import Organism


def create_dummy_anndata(
    n_cells=5,
    n_genes=3,
    obs_columns=None,
    var_columns=None,
    organism: Organism = Organism.HUMAN,
):
    obs_columns = obs_columns or []
    var_columns = var_columns or []
    # Create a dummy data matrix with random integer values (counts)
    X = sp.csr_matrix(
        np.random.poisson(lam=5, size=(n_cells, n_genes)).astype(np.int32)
    )

    # Create obs dataframe with specified columns
    obs_data = {}
    for col in obs_columns:
        if col == "cell_type":
            # Create balanced cell type labels for testing
            n_types = min(3, n_cells)  # Use at most 3 cell types
            obs_data[col] = [f"type_{i}" for i in range(n_types)] * (
                n_cells // n_types
            ) + ["type_0"] * (n_cells % n_types)
        elif col == "batch":
            # Create balanced batch labels for testing
            n_batches = min(2, n_cells)  # Use at most 2 batches
            obs_data[col] = [f"batch_{i}" for i in range(n_batches)] * (
                n_cells // n_batches
            ) + ["batch_0"] * (n_cells % n_batches)
        else:
            obs_data[col] = [f"{col}_{i}" for i in range(n_cells)]
    obs_df = pd.DataFrame(obs_data, index=[f"cell_{i}" for i in range(n_cells)])

    # Create var dataframe with specified columns, using Ensembl IDs as gene names
    genes = [
        f"{organism.prefix}{str(j).zfill(11)}" for j in range(n_genes)
    ]  # Ensembl IDs are 11 digits
    var_data = {}
    for col in var_columns:
        var_data[col] = genes
    var_df = pd.DataFrame(var_data, index=genes)
    return ad.AnnData(X=X, obs=obs_df, var=var_df)


class DummyDataset(BaseDataset):
    """A dummy dataset implementation for testing that skips file validation."""

    def _validate(self):
        # Skip validation for testing
        pass

    def validate(self):
        # Only validate inputs/outputs, skip file validation
        self._validate()

    def load_data(self):
        pass

    def unload_data(self):
        pass
