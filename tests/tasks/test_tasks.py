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


# TODO these tests could be split into multiple files and fixtures moved to conftest.py


@pytest.fixture
def dummy_anndata():
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

    expression_matrix: CellRepresentation = anndata.X.copy()
    obs: ListLike = anndata.obs.copy()
    var: ListLike = anndata.var.copy()

    # TODO perform PCA on expression matrix to get true embedding
    embedding_matrix: CellRepresentation = expression_matrix.toarray()

    return {
        "anndata": anndata,
        "expression_matrix": expression_matrix,
        "obs": obs,
        "var": var,
        "embedding_matrix": embedding_matrix,
    }


@pytest.fixture
def expression_matrix(dummy_anndata):
    return dummy_anndata["expression_matrix"]


@pytest.fixture
def embedding_matrix(dummy_anndata):
    return dummy_anndata["embedding_matrix"]


@pytest.fixture
def obs(dummy_anndata):
    return dummy_anndata["obs"]


@pytest.fixture
def var(dummy_anndata):
    return dummy_anndata["var"]


@pytest.fixture
def fixture_data(request):
    # Enables lazy generation of fixture data so fixtures can be used as parameters
    valid_fixture_names = ["expression_matrix", "embedding_matrix", "obs", "var"]
    fixture_name, other_data = request.param
    if isinstance(fixture_name, str):
        fixture_data = (
            request.getfixturevalue(fixture_name)
            if fixture_name in valid_fixture_names
            else fixture_name
        )
    else:
        fixture_data = [
            request.getfixturevalue(f) if f in valid_fixture_names else f
            for f in fixture_name
        ]
    return fixture_data, other_data


@pytest.mark.parametrize(
    "fixture_data",
    [
        ("expression_matrix", False),
        (["expression_matrix", "expression_matrix"], True),
    ],
    indirect=True,
)
def test_embedding_valid_input_output(fixture_data):
    """Test that embedding is accepted and List[MetricResult] is returned."""
    embedding, requires_multiple_datasets = fixture_data
    task = DummyTask(requires_multiple_datasets=requires_multiple_datasets)
    results = task.run(cell_representation=embedding)

    assert isinstance(results, list)
    assert all(isinstance(r, MetricResult) for r in results)


@pytest.mark.parametrize(
    "fixture_data",
    [
        ("abcd", [False, "This task requires a single cell representation for input"]),
        (
            ["embedding_matrix"],
            [False, "This task requires a single cell representation for input"],
        ),
        (
            ["embedding_matrix", "embedding_matrix"],
            [False, "This task requires a single cell representation for input"],
        ),
        (
            "embedding_matrix",
            [True, "This task requires a list of cell representations"],
        ),
        (
            ["abcd", "embedding_matrix"],
            [True, "This task requires a list of cell representations"],
        ),
        (
            ["embedding_matrix"],
            [
                True,
                "This task requires a list of cell representations but only one was provided",
            ],
        ),
    ],
    indirect=True,
)
def test_embedding_invalid_input(fixture_data):
    """Test ValueError for mismatch with requires_multiple_datasets."""
    embedding_list, (requires_multiple_datasets, error_message) = fixture_data
    task = DummyTask(requires_multiple_datasets=requires_multiple_datasets)
    with pytest.raises(ValueError, match=error_message):
        task.run(cell_representation=embedding_list)


@pytest.mark.parametrize(
    "task_class, task_list, metric_list",
    [
        (
            ClusteringTask,
            ["obs", "var"],
            ["input_labels"],
        ),
        (
            EmbeddingTask,
            [],
            ["input_labels"],
        ),
        (
            BatchIntegrationTask,
            [],
            ["labels", "batch_labels"],
        ),
        (
            MetadataLabelPredictionTask,
            ["labels"],
            [],
        ),
    ],
)
def test_task_execution(
    task_class, task_list, metric_list, embedding_matrix, expression_matrix, obs, var
):
    """Test that each task executes without errors on compatible data."""

    kwargs_dict = {
        "obs": obs,
        "var": var,
        "labels": obs["cell_type"],
        "input_labels": obs["cell_type"],
        "batch_labels": obs["batch"],
    }
    task_kwargs = {key: kwargs_dict[key] for key in task_list}
    metric_kwargs = {key: kwargs_dict[key] for key in metric_list}

    task = task_class()

    try:
        # Test regular task execution
        results = task.run(
            cell_representation=embedding_matrix,
            task_kwargs=task_kwargs,
            metric_kwargs=metric_kwargs,
        )
        assert isinstance(results, list)
        assert all(isinstance(r, MetricResult) for r in results)

        # Test baseline execution if implemented
        try:
            n_pcs = min(50, expression_matrix.shape[1] - 1)
            baseline_embedding = task.set_baseline(expression_matrix, n_pcs=n_pcs)
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


def test_cross_species_task(embedding_matrix, obs):
    """Test that CrossSpeciesIntegrationTask executes without errors."""
    task = CrossSpeciesIntegrationTask()
    embedding_list = [embedding_matrix, embedding_matrix]
    labels = obs["cell_type"]
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


def test_perturbation_task():
    """Test that PerturbationTask executes without errors."""
    # Create dummy perturbation data
    perturbation_data: dict = create_dummy_perturbation_anndata(
        n_cells=500,
        n_genes=200,
        organism=Organism.HUMAN,
        condition_column="condition",
        split_column="split",
    )
    gene_pert = perturbation_data["gene_pert"]
    pert_pred = perturbation_data["pert_pred"]
    pert_truth = perturbation_data["pert_truth"]
    cell_representation = perturbation_data["adata"].X
    var_names = perturbation_data["adata"].var_names
    obs_names = perturbation_data["adata"].obs_names

    # Task and argument setup
    task = PerturbationTask()
    task_kwargs = {
        "var_names": var_names,
    }
    metric_kwargs = {
        "gene_pert": gene_pert,
        "perturbation_pred": pert_pred,
        "perturbation_truth": pert_truth,
    }

    # Eight metrics: MSE and R2 for all/top20/top100 genes, Jaccard top20/100
    num_metrics = 8

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
