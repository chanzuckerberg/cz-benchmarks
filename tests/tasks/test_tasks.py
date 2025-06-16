import pytest

from typing import Dict
from czbenchmarks.tasks.clustering import ClusteringTask
from czbenchmarks.tasks.embedding import EmbeddingTask
from czbenchmarks.tasks.integration import BatchIntegrationTask
from czbenchmarks.tasks.label_prediction import MetadataLabelPredictionTask
from czbenchmarks.tasks.single_cell.cross_species import (
    CrossSpeciesIntegrationTask,
)
from czbenchmarks.tasks.single_cell.perturbation import PerturbationTask
from czbenchmarks.datasets.types import (
    Organism,
    Embedding,
    ListLike,
)
from czbenchmarks.metrics.types import MetricResult

# from tests.utils import DummyTask, create_dummy_anndata
from czbenchmarks.tasks.base import BaseTask
from czbenchmarks.metrics.types import MetricType
from czbenchmarks.tasks.utils import RANDOM_SEED
import numpy as np
import scipy.sparse as sp
import pandas as pd
from typing import List
import anndata as ad

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


class DummyTask(BaseTask):
    """A dummy task implementation for testing."""

    def __init__(
        self, requires_multiple_datasets: bool = False, *, random_seed: int = RANDOM_SEED
    ):
        super().__init__(random_seed=random_seed)
        self.name = "dummy task"
        self.requires_multiple_datasets = requires_multiple_datasets

    def _run_task(self, embedding: Embedding):
        # Dummy implementation that does nothing
        return {}

    def _compute_metrics(self, **kwargs) -> List[MetricResult]:
        # Return a dummy metric result
        return [
            MetricResult(
                name="dummy", value=1.0, metric_type=MetricType.ADJUSTED_RAND_INDEX
            )
        ]


# FIXME MICHELLE
# Move this data prep code to tasks/conftest.py ?
n_cells: int = 500
n_genes: int = 128
n_emb_dim: int = 32
organism: Organism = Organism.HUMAN
obs_columns = ["cell_type", "batch"]
var_columns = ["feature_name"]
adata = create_dummy_anndata(
    n_cells=n_cells,
    n_genes=n_genes,
    organism=organism,
    obs_columns=obs_columns,
    var_columns=var_columns,
)

n_ctrl = n_cells // 2
n_pert = n_cells - n_ctrl
gene_pert = "ENSG00000123456+ctrl"
adata.obs["condition"] = ["ctrl"] * n_ctrl + [gene_pert] * n_pert
adata.obs["split"] = ["train"] * n_ctrl + ["test"] * n_pert
PERT_PRED: Embedding = pd.DataFrame(
    data=np.random.normal(size=(n_ctrl, n_genes)),
    columns=adata.var_names,
    index=adata[adata.obs["condition"] == "ctrl"].obs_names,
)
conditions = np.array(list(adata.obs["condition"]))
test_conditions = set(
    adata.obs["condition"][adata.obs["split"] == "test"]
)
PERT_TRUTH: Dict[str, pd.DataFrame] = {
    str(condition): pd.DataFrame(
        data=adata[conditions == condition].X.toarray(),
        index=adata[conditions == condition].obs_names,
        columns=adata[conditions == condition].var_names,
    )
    for condition in test_conditions
}

EXPRESSION_MATRIX: Embedding = adata.X.copy()
OBS: ListLike = adata.obs.copy()
VAR_EXP: ListLike = adata.var.copy()

# FIXME MICHELLE
# sc.pp.pca(adata, n_comps=n_emb_dim, svd_solver="arpack",
# random_state=RANDOM_SEED, use_highly_variable=True, key_added=OBSM_KEY)
EMBEDDING_MATRIX: Embedding = (
    adata.X.copy().toarray()
)  # adata.obsm[OBSM_KEY].copy()
VAR_EMB: ListLike = VAR_EXP


@pytest.mark.parametrize(
    "requires_multiple_datasets, embedding",
    [
        (False, EMBEDDING_MATRIX),
        (True, [EMBEDDING_MATRIX, EMBEDDING_MATRIX]),
    ],
)
def test_embedding_valid_input_output(requires_multiple_datasets, embedding):
    """Test that embedding is accepted and List[MetricResult] is returned."""
    task = DummyTask(requires_multiple_datasets=requires_multiple_datasets)
    results = task.run(cell_representation=embedding)

    assert isinstance(results, list)
    assert all(isinstance(r, MetricResult) for r in results)


@pytest.mark.parametrize(
    "requires_multiple_datasets, embedding_list, error_message",
    [
        (False, "abcd", "This task requires a single cell representation for input"),
        (
            False,
            [EMBEDDING_MATRIX],
            "This task requires a single cell representation for input",
        ),
        (
            False,
            [EMBEDDING_MATRIX, EMBEDDING_MATRIX],
            "This task requires a single cell representation for input",
        ),
        (True, EMBEDDING_MATRIX, "This task requires a list of cell representations"),
        (
            True,
            ["abcd", EMBEDDING_MATRIX],
            "This task requires a list of cell representations",
        ),
        (
            True,
            [EMBEDDING_MATRIX],
            "This task requires a list of cell representations but only one was provided",
        ),
    ],
)
def test_embedding_invalid_input(
    requires_multiple_datasets, embedding_list, error_message
):
    """Test ValueError for mismatch with requires_multiple_datasets."""
    task = DummyTask(requires_multiple_datasets=requires_multiple_datasets)
    with pytest.raises(ValueError, match=error_message):
        task.run(cell_representation=embedding_list)


@pytest.mark.parametrize(
    "task_class, embedding, task_kwargs, metric_kwargs",
    [
        (
            ClusteringTask,
            EMBEDDING_MATRIX,
            {"obs": OBS, "var": VAR_EMB},
            {"input_labels": OBS["cell_type"]},
        ),
        (
            EmbeddingTask,
            EMBEDDING_MATRIX,
            {},
            {"input_labels": OBS["cell_type"]},
        ),
        (
            BatchIntegrationTask,
            EMBEDDING_MATRIX,
            {},
            {"labels": OBS["cell_type"], "batch_labels": OBS["batch"]},
        ),
        (
            MetadataLabelPredictionTask,
            EMBEDDING_MATRIX,
            {"labels": OBS["cell_type"]},
            {},
        ),

    ],
)
def test_task_execution(
    task_class,
    embedding,
    task_kwargs,
    metric_kwargs,
    expression_data=EXPRESSION_MATRIX,
):
    """Test that each task executes without errors on compatible data."""
    task = task_class()

    try:
        # Test regular task execution
        results = task.run(
            cell_representation=embedding,
            task_kwargs=task_kwargs,
            metric_kwargs=metric_kwargs,
        )
        assert isinstance(results, list)
        assert all(isinstance(r, MetricResult) for r in results)

        # Test baseline execution if implemented
        try:
            n_pcs = min(50, expression_data.shape[1] - 1)
            baseline_embedding = task.set_baseline(
                expression_data, n_pcs=n_pcs
            )
            if "var" in task_kwargs:
                task_kwargs["var"] = task_kwargs["var"].iloc[:n_pcs]

            baseline_results = task.run(
                cell_representation=baseline_embedding,
                task_kwargs=task_kwargs,
                metric_kwargs=metric_kwargs,
            )
            assert isinstance(baseline_results, list)
            assert all(isinstance(r, MetricResult) for r in baseline_results)
        except NotImplementedError:
            # Some tasks may not implement set_baseline
            pass

    except Exception as e:
        pytest.fail(f"Task {task_class.__name__} failed unexpectedly: {e}")


def test_cross_species_task(
    embedding=EMBEDDING_MATRIX, labels=OBS["cell_type"]
):
    """Test that CrossSpeciesIntegrationTask executes without errors."""
    task = CrossSpeciesIntegrationTask()
    embedding_list = [embedding, embedding]
    labels_list = [labels, labels]
    organism_list = [Organism.HUMAN, Organism.MOUSE]
    task_kwargs = {
        "labels": labels_list,
        "organism_list": organism_list,
    }

    try:
        # Test regular task execution
        results = task.run(
            cell_representation=embedding_list,
            task_kwargs=task_kwargs,
        )

        # Verify results structure
        assert isinstance(results, list)
        assert all(isinstance(r, MetricResult) for r in results)

        # Test that baseline raises NotImplementedError
        with pytest.raises(NotImplementedError):
            task.set_baseline()

    except Exception as e:
        pytest.fail(f"CrossSpeciesIntegrationTask failed unexpectedly: {e}")


# FIXME MICHELLE
def test_perturbation_task(
    cell_representation = EXPRESSION_MATRIX,
    var_names = VAR_EXP["feature_name"],
    obs_names = OBS["cell_type"],
    gene_pert = gene_pert,
    pert_pred = PERT_PRED,
    pert_truth = PERT_TRUTH,
):
    """Test that PerturbationTask executes without errors."""
    task = PerturbationTask()
    task_kwargs = {
        "var_names": var_names,
    }
    metric_kwargs = {
        "gene_pert": gene_pert,
        "perturbation_pred": pert_pred,
        "perturbation_truth": pert_truth,
    }

    try:
        # Test regular task execution
        results = task.run(
            cell_representation, 
            task_kwargs,
            metric_kwargs,
        )

        # Verify results structure
        assert isinstance(results, list)
        assert all(isinstance(r, MetricResult) for r in results)
        # Eight metrics: MSE and R2 for all/top20/top100 genes, Jaccard
        # for top20/100
        num_metrics = 8
        assert len(results) == num_metrics

        # Test baseline with both mean and median
        for baseline_type in ["mean", "median"]:
            baseline_embedding = task.set_baseline(
                cell_representation=cell_representation,
                var_names=var_names,
                obs_names=obs_names,
                baseline_type=baseline_type,
            )
            baseline_results = task.run(
                baseline_embedding, task_kwargs, metric_kwargs
            )
            assert isinstance(baseline_results, list)
            assert all(isinstance(r, MetricResult) for r in baseline_results)
            assert len(baseline_results) == num_metrics

    except Exception as e:
        pytest.fail(f"PerturbationTask failed unexpectedly: {e}")


if __name__ == "__main__":
    pytest.main(["-v", __file__, "-k", "test_perturbation_task"])
