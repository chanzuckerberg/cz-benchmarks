import pytest
from typing import List
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
from czbenchmarks.constants import RANDOM_SEED


# from tests.utils import DummyDataset, create_dummy_anndata, DummyTask
from czbenchmarks.tasks.base import BaseTask
from czbenchmarks.metrics.types import MetricType


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

NUM_CELLS: int = 30
NUM_GENES: int = 20
NUM_EMB_DIM: int = 15

OBS: ListLike = pd.DataFrame({"cell_type": np.random.choice(a=["A", "B", "C"], size=NUM_CELLS)})
VAR_EXP: ListLike = pd.DataFrame({"feature_name": np.random.choice(a=["X", "Y", "Z"], size=NUM_GENES)})
EXPRESSION_MATRIX: Embedding = np.random.normal(size=(NUM_CELLS, NUM_GENES))

VAR_EMB: ListLike = pd.DataFrame({"feature_name": np.random.choice(a=["X", "Y", "Z"], size=NUM_EMB_DIM)})
EMBEDDING_MATRIX: Embedding = np.random.normal(size=(NUM_CELLS, NUM_EMB_DIM))


@pytest.mark.parametrize("requires_multiple_datasets, embedding", [
    (False, EMBEDDING_MATRIX),
    (True, [EMBEDDING_MATRIX, EMBEDDING_MATRIX]),
])
def test_embedding_valid_input_output(requires_multiple_datasets, embedding):
    """Test that and embedding is accepted and List[MetricResult] is returned."""
    task = DummyTask(requires_multiple_datasets=requires_multiple_datasets)
    results = task.run(embedding)

    assert isinstance(results, list)
    assert all(isinstance(r, MetricResult) for r in results)


@pytest.mark.parametrize("requires_multiple_datasets, embedding_list, error_message", [
    (False, "abcd", "This task requires a single embedding for input"),
    (False, [EMBEDDING_MATRIX], "This task requires a single embedding for input"),
    (False, [EMBEDDING_MATRIX, EMBEDDING_MATRIX], "This task requires a single embedding for input"),
    (True, EMBEDDING_MATRIX, "This task requires a list of embeddings"),
    (True, ["abcd", EMBEDDING_MATRIX], "This task requires a list of embeddings"),
    (True, [EMBEDDING_MATRIX], "This task requires a list of embeddings but only one embedding provided"),
])
def test_embedding_invalid_input(requires_multiple_datasets, embedding_list, error_message):
    """Test that ValueError is raised appropriately when requires_multiple_datasets is True/False"""
    task = DummyTask(requires_multiple_datasets=requires_multiple_datasets)
    with pytest.raises(ValueError, match=error_message):
        task.run(embedding_list)


@pytest.mark.parametrize(
    "task_class,task_kwargs",
    [
        (ClusteringTask, {"label_key": "cell_type"}),
        (EmbeddingTask, {"label_key": "cell_type"}),
        (BatchIntegrationTask, {"label_key": "cell_type", "batch_key": "batch"}),
        (MetadataLabelPredictionTask, {"label_key": "cell_type", "n_folds": 3}),
    ],
)
def test_task_execution(task_class, task_kwargs, dummy_single_cell_dataset):
    """Test that each task executes without errors on compatible data."""
    task = task_class(**task_kwargs)

    try:
        # Test regular task execution
        results = task.run(dummy_single_cell_dataset)
        assert isinstance(results, dict)
        assert ModelType.SCVI in results
        assert isinstance(results[ModelType.SCVI], list)
        assert all(isinstance(m, MetricResult) for m in results[ModelType.SCVI])

        # Test baseline execution if implemented
        try:
            task.set_baseline(dummy_single_cell_dataset)
            baseline_results = task.run(
                dummy_single_cell_dataset, model_types=[ModelType.BASELINE]
            )
            assert ModelType.BASELINE in baseline_results
            assert isinstance(baseline_results[ModelType.BASELINE], list)
            assert all(
                isinstance(m, MetricResult)
                for m in baseline_results[ModelType.BASELINE]
            )
        except NotImplementedError:
            # Some tasks may not implement set_baseline
            pass

    except Exception as e:
        pytest.fail(f"Task {task_class.__name__} failed unexpectedly: {e}")


def test_cross_species_task(dummy_cross_species_datasets):
    """Test that CrossSpeciesIntegrationTask executes without errors."""
    task = CrossSpeciesIntegrationTask(label_key="cell_type")

    try:
        # Test regular task execution
        results = task.run(dummy_cross_species_datasets)

        # Verify results structure
        assert isinstance(results, dict)
        assert ModelType.UCE in results
        assert isinstance(results[ModelType.UCE], list)
        assert all(isinstance(m, MetricResult) for m in results[ModelType.UCE])
        assert (
            len(results[ModelType.UCE]) == 2
        )  # Should have entropy and silhouette metrics

        # Test that baseline raises NotImplementedError
        with pytest.raises(NotImplementedError):
            task.set_baseline(dummy_cross_species_datasets)

    except Exception as e:
        pytest.fail(f"CrossSpeciesIntegrationTask failed unexpectedly: {e}")


def test_perturbation_task(dummy_perturbation_dataset):
    """Test that PerturbationTask executes without errors."""
    task = PerturbationTask()

    try:
        # Test regular task execution
        results = task.run(dummy_perturbation_dataset)

        # Verify results structure
        assert isinstance(results, dict)
        assert ModelType.SCGENEPT in results
        assert isinstance(results[ModelType.SCGENEPT], list)
        assert all(isinstance(m, MetricResult) for m in results[ModelType.SCGENEPT])

        # Should have 8 metrics: MSE and R2 for all/top20/top100 genes,
        # plus Jaccard for top20/100
        assert len(results[ModelType.SCGENEPT]) == 8

        # Test baseline with both mean and median
        for baseline_type in ["mean", "median"]:
            task.set_baseline(
                dummy_perturbation_dataset,
                gene_pert="ENSG00000123456+ctrl",
                baseline_type=baseline_type,
            )
            baseline_results = task.run(
                dummy_perturbation_dataset, model_types=[ModelType.BASELINE]
            )
            assert ModelType.BASELINE in baseline_results
            assert isinstance(baseline_results[ModelType.BASELINE], list)
            assert len(baseline_results[ModelType.BASELINE]) == 8

    except Exception as e:
        pytest.fail(f"PerturbationTask failed unexpectedly: {e}")
