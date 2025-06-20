import pytest

import anndata as ad
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
    CellRepresentation,
    ListLike,
)
from czbenchmarks.metrics.types import MetricResult

from tests.utils import (
    create_dummy_anndata,
    DummyTask,
    create_dummy_perturbation_anndata,
)


n_cells: int = 500
n_genes: int = 200
organism: Organism = Organism.HUMAN
obs_columns: list[str] = ["cell_type", "batch"]
var_columns: list[str] = ["feature_name"]
anndata: ad.AnnData = create_dummy_anndata(
    n_cells=n_cells,
    n_genes=n_genes,
    organism=organism,
    obs_columns=obs_columns,
    var_columns=var_columns,
)

EXPRESSION_MATRIX: CellRepresentation = anndata.X.copy()
OBS: ListLike = anndata.obs.copy()
VAR_EXP: ListLike = anndata.var.copy()

EMBEDDING_MATRIX: CellRepresentation = EXPRESSION_MATRIX.toarray()
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
            baseline_embedding = task.set_baseline(expression_data, n_pcs=n_pcs)
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


def test_cross_species_task(embedding=EMBEDDING_MATRIX, labels=OBS["cell_type"]):
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


# # FIXME MICHELLE
def test_perturbation_task():
    """Test that PerturbationTask executes without errors."""
    perturbation_anndata = create_dummy_perturbation_anndata(
        n_cells=500,
        n_genes=200,
        organism=Organism.HUMAN,
        condition_column="condition",
        split_column="split",
    )
    pert_pred = perturbation_anndata["pert_pred"]
    pert_truth = perturbation_anndata["pert_truth"]
    gene_pert = perturbation_anndata["gene_pert"]
    cell_representation = perturbation_anndata["adata"].X
    var_names = perturbation_anndata["adata"].var_names
    obs_names = perturbation_anndata["adata"].obs_names

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
            baseline_results = task.run(baseline_embedding, task_kwargs, metric_kwargs)
            assert isinstance(baseline_results, list)
            assert all(isinstance(r, MetricResult) for r in baseline_results)
            assert len(baseline_results) == num_metrics

    except Exception as e:
        pytest.fail(f"PerturbationTask failed unexpectedly: {e}")


if __name__ == "__main__":
    pytest.main(["-v", __file__, "-k", "test_perturbation_task"])
