import pytest
from czbenchmarks.tasks import (
    ClusteringTask,
    ClusteringTaskInput,
    EmbeddingTask,
    EmbeddingTaskInput,
    BatchIntegrationTask,
    BatchIntegrationTaskInput,
    MetadataLabelPredictionTask,
    MetadataLabelPredictionTaskInput,
)
from czbenchmarks.tasks.single_cell import (
    CrossSpeciesIntegrationTask,
    CrossSpeciesIntegrationTaskInput,
)
from czbenchmarks.datasets.types import Organism
from czbenchmarks.metrics.types import MetricResult

from tests.utils import (
    DummyTask,
    DummyTaskInput,
)


def test_embedding_valid_input_output(dummy_anndata):
    """Test that embedding is accepted and List[MetricResult] is returned."""
    expression_matrix = dummy_anndata["expression_matrix"]

    # Test single dataset
    task = DummyTask(requires_multiple_datasets=False)
    results = task.run(
        cell_representation=expression_matrix,
        task_input=DummyTaskInput(),
    )
    assert isinstance(results, list)
    assert all(isinstance(r, MetricResult) for r in results)

    # Test multiple datasets
    task = DummyTask(requires_multiple_datasets=True)
    results = task.run(
        cell_representation=[expression_matrix, expression_matrix],
        task_input=DummyTaskInput(),
    )
    assert isinstance(results, list)
    assert all(isinstance(r, MetricResult) for r in results)


def test_embedding_invalid_input(dummy_anndata):
    """Test ValueError for mismatch with requires_multiple_datasets."""
    embedding_matrix = dummy_anndata["embedding_matrix"]

    # Test invalid string input for single dataset task
    task = DummyTask(requires_multiple_datasets=False)
    with pytest.raises(
        ValueError, match="This task requires a single cell representation for input"
    ):
        task.run(cell_representation="abcd", task_input=DummyTaskInput())

    # Test list input for single dataset task
    with pytest.raises(
        ValueError, match="This task requires a single cell representation for input"
    ):
        task.run(cell_representation=[embedding_matrix], task_input=DummyTaskInput())

    # Test multiple datasets for single dataset task
    with pytest.raises(
        ValueError, match="This task requires a single cell representation for input"
    ):
        task.run(
            cell_representation=[embedding_matrix, embedding_matrix],
            task_input=DummyTaskInput(),
        )

    # Test single dataset for multiple dataset task
    task = DummyTask(requires_multiple_datasets=True)
    with pytest.raises(
        ValueError, match="This task requires a list of cell representations"
    ):
        task.run(cell_representation=embedding_matrix, task_input=DummyTaskInput())

    # Test invalid input in list for multiple dataset task
    with pytest.raises(
        ValueError, match="This task requires a list of cell representations"
    ):
        task.run(
            cell_representation=["abcd", embedding_matrix], task_input=DummyTaskInput()
        )

    # Test single item list for multiple dataset task
    with pytest.raises(
        ValueError,
        match="This task requires a list of cell representations but only one was provided",
    ):
        task.run(cell_representation=[embedding_matrix], task_input=DummyTaskInput())


@pytest.mark.parametrize(
    "task_class,task_input_builder",
    [
        (
            ClusteringTask,
            lambda obs: ClusteringTaskInput(obs=obs, input_labels=obs["cell_type"]),
        ),
        (
            EmbeddingTask,
            lambda obs: EmbeddingTaskInput(input_labels=obs["cell_type"]),
        ),
        (
            BatchIntegrationTask,
            lambda obs: BatchIntegrationTaskInput(
                labels=obs["cell_type"], batch_labels=obs["batch"]
            ),
        ),
        (
            MetadataLabelPredictionTask,
            lambda obs: MetadataLabelPredictionTaskInput(labels=obs["cell_type"]),
        ),
    ],
)
def test_task_execution(
    task_class,
    task_input_builder,
    dummy_anndata,
):
    """Test that each task executes without errors on compatible data."""

    embedding_matrix = dummy_anndata["embedding_matrix"]
    expression_matrix = dummy_anndata["expression_matrix"]
    obs = dummy_anndata["obs"]

    task_input = task_input_builder(obs)

    task = task_class()

    try:
        # Test regular task execution
        results = task.run(
            cell_representation=embedding_matrix,
            task_input=task_input,
        )
        assert isinstance(results, list)
        assert all(isinstance(r, MetricResult) for r in results)

        # Test baseline execution if implemented
        try:
            n_pcs = min(50, expression_matrix.shape[1] - 1)
            baseline_embedding = task.compute_baseline(expression_matrix, n_pcs=n_pcs)
            if hasattr(task_input, "var"):
                task_input.var = task_input.var.iloc[:n_pcs]

            baseline_results = task.run(
                cell_representation=baseline_embedding,
                task_input=task_input,
            )
            assert isinstance(baseline_results, list)
            assert all(isinstance(r, MetricResult) for r in baseline_results)
        except NotImplementedError:
            # Some tasks may not implement compute_baseline
            pass

    except Exception as e:
        pytest.fail(f"Task {task_class.__name__} failed unexpectedly: {e}")


def test_cross_species_task(dummy_anndata):
    """Test that CrossSpeciesIntegrationTask executes without errors."""
    task = CrossSpeciesIntegrationTask()
    embedding_matrix = dummy_anndata["embedding_matrix"]
    obs = dummy_anndata["obs"]
    embedding_list = [embedding_matrix, embedding_matrix]
    labels = obs["cell_type"]
    labels_list = [labels, labels]
    organism_list = [Organism.HUMAN, Organism.MOUSE]
    task_input = CrossSpeciesIntegrationTaskInput(
        labels=labels_list, organism_list=organism_list
    )

    try:
        # Test regular task execution
        results = task.run(
            cell_representation=embedding_list,
            task_input=task_input,
        )

        # Verify results structure
        assert isinstance(results, list)
        assert all(isinstance(r, MetricResult) for r in results)

        # Test that baseline raises NotImplementedError
        with pytest.raises(NotImplementedError):
            task.compute_baseline()

    except Exception as e:
        pytest.fail(f"CrossSpeciesIntegrationTask failed unexpectedly: {e}")
