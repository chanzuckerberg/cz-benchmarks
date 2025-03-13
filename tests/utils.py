import numpy as np
import pandas as pd
import anndata as ad

from czbenchmarks.datasets.base import BaseDataset


def create_dummy_anndata(n_cells=5, n_genes=3, obs_columns=None, var_columns=None):
    obs_columns = obs_columns or []
    var_columns = var_columns or []
    # Create a dummy data matrix
    X = np.ones((n_cells, n_genes))

    # Create obs dataframe with specified columns
    obs_data = {}
    for col in obs_columns:
        obs_data[col] = [f"{col}_{i}" for i in range(n_cells)]
    obs_df = pd.DataFrame(obs_data, index=[f"cell_{i}" for i in range(n_cells)])

    # Create var dataframe with specified columns, using gene names as data
    genes = [f"gene_{j}" for j in range(n_genes)]
    var_data = {}
    for col in var_columns:
        var_data[col] = genes
    var_df = pd.DataFrame(var_data, index=genes)
    return ad.AnnData(X=X, obs=obs_df, var=var_df)


class DummyDataset(BaseDataset):
    def _validate(self):
        pass

    def load_data(self):
        pass

    def unload_data(self):
        pass
