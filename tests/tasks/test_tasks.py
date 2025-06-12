import pytest
import numpy as np
import pandas as pd
from czbenchmarks.tasks.clustering import ClusteringTask
from czbenchmarks.tasks.embedding import EmbeddingTask
from czbenchmarks.tasks.integration import BatchIntegrationTask
from czbenchmarks.tasks.label_prediction import MetadataLabelPredictionTask
from czbenchmarks.tasks.single_cell.cross_species import CrossSpeciesIntegrationTask
from czbenchmarks.tasks.single_cell.perturbation import PerturbationTask
from czbenchmarks.datasets.types import Organism, Embedding, ListLike
from czbenchmarks.metrics.types import MetricResult

from tests.utils import DummyTask


# Move this to tests/utils.py ?
NUM_CELLS: int = 30
NUM_GENES: int = 20
NUM_EMB_DIM: int = 15

OBS: ListLike = pd.DataFrame(
    {
        "cell_type": np.random.choice(a=["A", "B", "C"], size=NUM_CELLS),
        "batch": np.random.choice(a=["1", "2", "3"], size=NUM_CELLS),
    }
)
VAR_EXP: ListLike = pd.DataFrame(
    {"feature_name": np.random.choice(a=["X", "Y", "Z"], size=NUM_GENES)}
)
EXPRESSION_MATRIX: Embedding = np.random.normal(size=(NUM_CELLS, NUM_GENES))

VAR_EMB: ListLike = pd.DataFrame(
    {"feature_name": np.random.choice(a=["X", "Y", "Z"], size=NUM_EMB_DIM)}
)
EMBEDDING_MATRIX: Embedding = np.random.normal(size=(NUM_CELLS, NUM_EMB_DIM))


@pytest.mark.parametrize(
    "requires_multiple_datasets, embedding",
    [
        (False, EMBEDDING_MATRIX),
        (True, [EMBEDDING_MATRIX, EMBEDDING_MATRIX]),
    ],
)
def test_embedding_valid_input_output(requires_multiple_datasets, embedding):
    """Test that and embedding is accepted and List[MetricResult] is returned."""
    task = DummyTask(requires_multiple_datasets=requires_multiple_datasets)
    results = task.run(embedding)

    assert isinstance(results, list)
    assert all(isinstance(r, MetricResult) for r in results)


@pytest.mark.parametrize(
    "requires_multiple_datasets, embedding_list, error_message",
    [
        (False, "abcd", "This task requires a single embedding for input"),
        (False, [EMBEDDING_MATRIX], "This task requires a single embedding for input"),
        (
            False,
            [EMBEDDING_MATRIX, EMBEDDING_MATRIX],
            "This task requires a single embedding for input",
        ),
        (True, EMBEDDING_MATRIX, "This task requires a list of embeddings"),
        (True, ["abcd", EMBEDDING_MATRIX], "This task requires a list of embeddings"),
        (
            True,
            [EMBEDDING_MATRIX],
            "This task requires a list of embeddings but only one embedding provided",
        ),
    ],
)
def test_embedding_invalid_input(
    requires_multiple_datasets, embedding_list, error_message
):
    """Test that ValueError is raised appropriately when requires_multiple_datasets is True/False"""
    task = DummyTask(requires_multiple_datasets=requires_multiple_datasets)
    with pytest.raises(ValueError, match=error_message):
        task.run(embedding_list)


@pytest.mark.parametrize(
    "task_class, embedding, task_kwargs, metric_kwargs",
    [
        (
            ClusteringTask,
            EMBEDDING_MATRIX,
            {"obs": OBS, "var": VAR_EMB},
            {"input_labels": OBS["cell_type"]},
        ),
        (EmbeddingTask, EMBEDDING_MATRIX, {}, {"input_labels": OBS["cell_type"]}),
        (
            BatchIntegrationTask,
            EMBEDDING_MATRIX,
            {},
            {"labels": OBS["cell_type"], "batch_labels": OBS["batch"]},
        ),  # FIXME MICHELLE
        (
            MetadataLabelPredictionTask,
            EMBEDDING_MATRIX,
            {"labels": OBS["cell_type"]},
            {},
        ),
    ],
)
def test_task_execution(
    task_class, embedding, task_kwargs, metric_kwargs, expression_data=EXPRESSION_MATRIX
):
    """Test that each task executes without errors on compatible data."""
    task = task_class()

    try:
        # Test regular task execution
        results = task.run(
            embedding=embedding, task_kwargs=task_kwargs, metric_kwargs=metric_kwargs
        )
        assert isinstance(results, list)
        assert all(isinstance(r, MetricResult) for r in results)

        # Test baseline execution if implemented
        # try:  # FIXME MICHELLE
        #     baseline_embedding = task.set_baseline(expression_data, n_pcs=expression_data.shape[1])
        #     baseline_results = task.run(
        #         embedding=baseline_embedding, task_kwargs=task_kwargs, metric_kwargs=metric_kwargs
        #     )
        #     assert isinstance(baseline_results, list)
        #     assert all(isinstance(r, MetricResult) for r in baseline_results)
        # except NotImplementedError:
        #     # Some tasks may not implement set_baseline
        #     pass

    except Exception as e:
        pytest.fail(f"Task {task_class.__name__} failed unexpectedly: {e}")


# FIXME MICHELLE
def test_cross_species_task(embedding=EMBEDDING_MATRIX, labels=OBS["cell_type"]):
    """Test that CrossSpeciesIntegrationTask executes without errors."""
    task = CrossSpeciesIntegrationTask()
    embedding_list = [embedding, embedding]
    labels_list = [labels, labels]
    organism_list = [Organism.HUMAN, Organism.MOUSE]

    try:
        # Test regular task execution
        results = task.run(
            embedding=embedding_list, labels=labels_list, organism_list=organism_list
        )

        # Verify results structure
        assert isinstance(results, list)
        assert all(isinstance(r, MetricResult) for r in results)

        # Test that baseline raises NotImplementedError
        with pytest.raises(NotImplementedError):
            task.set_baseline()

    except Exception as e:
        pytest.fail(f"CrossSpeciesIntegrationTask failed unexpectedly: {e}")


def test_perturbation_task(
    expression_data=EXPRESSION_MATRIX,
    var_names=VAR_EXP["feature_name"],
    obs_names=OBS["cell_id"],
):
    """Test that PerturbationTask executes without errors."""
    task = PerturbationTask()

    try:
        # Test regular task execution
        results = task.run(expression_data=expression_data, var_names=var_names)

        # Verify results structure
        assert isinstance(results, list)
        assert all(isinstance(r, MetricResult) for r in results)
        assert (
            len(results) == 8
        )  # Eight metrics: MSE and R2 for all/top20/top100 genes, Jaccard for top20/100

        # Test baseline with both mean and median
        for baseline_type in ["mean", "median"]:
            baseline_embedding = task.set_baseline(
                expression_data=expression_data,
                var_names=var_names,
                obs_names=obs_names,
                gene_pert="ENSG00000123456+ctrl",
                baseline_type=baseline_type,
            )
            baseline_results = task.run(
                expression_data=baseline_embedding, var_names=var_names
            )
            assert isinstance(baseline_results, list)
            assert all(isinstance(r, MetricResult) for r in baseline_results)
            assert len(baseline_results) == 8

    except Exception as e:
        pytest.fail(f"PerturbationTask failed unexpectedly: {e}")


if __name__ == "__main__":
    pytest.main(["-v", __file__, "-k", "test_task_execution"])
