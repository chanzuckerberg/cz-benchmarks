import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

from czbenchmarks.datasets.types import Organism, CellRepresentation
from czbenchmarks.constants import RANDOM_SEED
from typing import List
from czbenchmarks.tasks.base import BaseTask
from czbenchmarks.tasks.types import TaskInput, MetricInput
from czbenchmarks.datasets import (
    BaseDataset,
)
from czbenchmarks.metrics.types import MetricResult, MetricType


class DummyTaskInput(TaskInput):
    pass


class DummyMetricInput(MetricInput):
    pass


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


def create_dummy_perturbation_anndata(
    n_cells=500,
    n_genes=200,
    organism: Organism = Organism.HUMAN,
    condition_column="condition",
    split_column="split",
):
    # Create dummy data with control and perturbed cells
    n_genes = 200
    n_ctrl = n_cells // 2
    n_pert = n_cells - n_ctrl

    # Create base anndata with all cells
    adata = create_dummy_anndata(
        n_cells=n_cells,
        n_genes=n_genes,
        obs_columns=None,
        var_columns=None,
        organism=organism,
    )

    # Create perturbed cells
    gene_pert = f"{str(organism.prefix)}{str(123456).zfill(11)}+ctrl"
    adata.obs[condition_column] = ["ctrl"] * n_ctrl + [gene_pert] * n_pert
    adata.obs[split_column] = ["train"] * n_ctrl + ["test"] * n_pert

    # Create and set predicted perturbation effect
    pert_pred = pd.DataFrame(
        data=np.random.normal(size=(n_ctrl, n_genes)),
        columns=adata.var_names,
        index=adata[adata.obs[condition_column] == "ctrl"].obs_names,
    )

    conditions = np.array(list(adata.obs[condition_column]))
    test_conditions = set(
        adata.obs[condition_column][adata.obs[split_column] == "test"]
    )

    pert_truth = {
        str(condition): pd.DataFrame(
            data=adata[conditions == condition].X.toarray(),
            index=adata[conditions == condition].obs_names,
            columns=adata[conditions == condition].var_names,
        )
        for condition in test_conditions
    }

    return {
        "adata": adata,
        "gene_pert": gene_pert,
        "pert_pred": pert_pred,
        "pert_truth": pert_truth,
    }


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

    def cache_data(self):
        pass


class DummyTask(BaseTask):
    """A dummy task implementation for testing."""

    def __init__(
        self,
        requires_multiple_datasets: bool = False,
        *,
        random_seed: int = RANDOM_SEED,
    ):
        super().__init__(random_seed=random_seed)
        self.name = "dummy task"
        self.requires_multiple_datasets = requires_multiple_datasets

    def _run_task(self, embedding: CellRepresentation, task_input: TaskInput):
        # Dummy implementation that does nothing
        return {}

    def _compute_metrics(
        self, task_output: dict, metric_input: MetricInput
    ) -> List[MetricResult]:
        # Return a dummy metric result
        return [
            MetricResult(
                name="dummy", value=1.0, metric_type=MetricType.ADJUSTED_RAND_INDEX
            )
        ]
