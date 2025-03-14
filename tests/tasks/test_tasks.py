import pytest
import numpy as np
from typing import List, Set

from czbenchmarks.tasks.base import BaseTask
from czbenchmarks.tasks.clustering import ClusteringTask
from czbenchmarks.tasks.embedding import EmbeddingTask
from czbenchmarks.tasks.integration import BatchIntegrationTask
from czbenchmarks.tasks.label_prediction import MetadataLabelPredictionTask
from czbenchmarks.datasets import BaseDataset, DataType, SingleCellDataset
from czbenchmarks.models.types import ModelType
from czbenchmarks.metrics.types import MetricResult
from tests.utils import DummyDataset, create_dummy_anndata


class DummyTask(BaseTask):
    """A dummy task implementation for testing."""

    def __init__(self, requires_multiple: bool = False):
        self._requires_multiple = requires_multiple

    @property
    def required_inputs(self) -> Set[DataType]:
        return {DataType.ANNDATA, DataType.METADATA}

    @property
    def required_outputs(self) -> Set[DataType]:
        return {DataType.EMBEDDING}

    @property
    def requires_multiple_datasets(self) -> bool:
        return self._requires_multiple

    def _run_task(self, data: BaseDataset, model_type: ModelType):
        # Dummy implementation that does nothing
        pass

    def _compute_metrics(self) -> List[MetricResult]:
        # Return a dummy metric result
        return [MetricResult(name="dummy", value=1.0)]


def test_missing_required_inputs_outputs():
    """Test that validation fails when dataset is missing required inputs/outputs."""
    task = DummyTask()
    dataset = DummyDataset("dummy_path")

    # Don't set any inputs/outputs
    with pytest.raises(ValueError, match="Missing required inputs"):
        task.validate(dataset)

    # Set only inputs but no outputs
    dataset.set_input(DataType.ANNDATA, np.array([]))
    dataset.set_input(DataType.METADATA, {})
    with pytest.raises(ValueError, match="Missing required outputs"):
        task.validate(dataset)


def test_requires_multiple_datasets_validation():
    """Test that ValueError is raised when requires_multiple_datasets is True
    but input is not a list."""
    task = DummyTask(requires_multiple=True)
    dataset = DummyDataset("dummy_path")

    # Set required inputs/outputs
    dataset.set_input(DataType.ANNDATA, np.array([]))
    dataset.set_input(DataType.METADATA, {})
    dataset.set_output(ModelType.BASELINE, DataType.EMBEDDING, np.array([]))

    with pytest.raises(ValueError, match="This task requires a list of datasets"):
        task.run(dataset)


def test_invalid_input_type():
    """Test that ValueError is raised when input is not BaseDataset or list of
    BaseDatasets."""
    task = DummyTask()

    with pytest.raises(ValueError, match="Invalid data type"):
        task.run("not a dataset")

    with pytest.raises(ValueError, match="Invalid data type"):
        task.run([1, 2, 3])


def test_list_input_single_task():
    """Test that List[Dict[ModelType, List[MetricResult]]] is returned for list
    input on single task."""
    task = DummyTask(requires_multiple=False)
    datasets = [DummyDataset("dummy1"), DummyDataset("dummy2")]

    # Set required inputs/outputs for both datasets
    for dataset in datasets:
        dataset.set_input(DataType.ANNDATA, np.array([]))
        dataset.set_input(DataType.METADATA, {})
        dataset.set_output(ModelType.BASELINE, DataType.EMBEDDING, np.array([]))

    results = task.run(datasets)

    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(r, dict) for r in results)
    assert all(ModelType.BASELINE in r for r in results)
    assert all(isinstance(r[ModelType.BASELINE], list) for r in results)


def test_list_input_multiple_task():
    """Test that Dict[ModelType, List[MetricResult]] is returned for list input
    on multiple task."""
    task = DummyTask(requires_multiple=True)
    datasets = [DummyDataset("dummy1"), DummyDataset("dummy2")]

    # Set required inputs/outputs for both datasets
    for dataset in datasets:
        dataset.set_input(DataType.ANNDATA, np.array([]))
        dataset.set_input(DataType.METADATA, {})
        dataset.set_output(ModelType.BASELINE, DataType.EMBEDDING, np.array([]))

    results = task.run(datasets)

    assert isinstance(results, dict)
    assert ModelType.BASELINE in results
    assert isinstance(results[ModelType.BASELINE], list)


def test_single_dataset_result():
    """Test that Dict[ModelType, List[MetricResult]] is returned for single dataset."""
    task = DummyTask()
    dataset = DummyDataset("dummy")

    # Set required inputs/outputs
    dataset.set_input(DataType.ANNDATA, np.array([]))
    dataset.set_input(DataType.METADATA, {})
    dataset.set_output(ModelType.BASELINE, DataType.EMBEDDING, np.array([]))

    results = task.run(dataset)

    assert isinstance(results, dict)
    assert ModelType.BASELINE in results
    assert isinstance(results[ModelType.BASELINE], list)
    assert len(results[ModelType.BASELINE]) == 1
    assert isinstance(results[ModelType.BASELINE][0], MetricResult)


@pytest.fixture
def dummy_dataset_100x500():
    """Create a dummy dataset with 100 cells and 500 genes."""
    dataset = SingleCellDataset("dummy_path")
    adata = create_dummy_anndata(
        n_cells=100,
        n_genes=500,
        obs_columns=["cell_type", "batch"],
        var_columns=["feature_id"],
    )
    dataset.set_input(DataType.ANNDATA, adata)
    dataset.set_input(DataType.METADATA, adata.obs)

    # Create a random embedding
    embedding = np.random.normal(size=(100, 32))
    dataset.set_output(ModelType.BASELINE, DataType.EMBEDDING, embedding)

    return dataset


@pytest.mark.parametrize(
    "task_class,task_kwargs",
    [
        (ClusteringTask, {"label_key": "cell_type"}),
        (EmbeddingTask, {"label_key": "cell_type"}),
        (BatchIntegrationTask, {"label_key": "cell_type", "batch_key": "batch"}),
        (MetadataLabelPredictionTask, {"label_key": "cell_type", "n_folds": 3}),
    ],
)
def test_task_execution(task_class, task_kwargs, dummy_dataset_100x500):
    """Test that each task executes without errors on compatible data."""
    task = task_class(**task_kwargs)

    try:
        # Test regular task execution
        results = task.run(dummy_dataset_100x500)
        assert isinstance(results, dict)
        assert ModelType.BASELINE in results
        assert isinstance(results[ModelType.BASELINE], list)
        assert all(isinstance(m, MetricResult) for m in results[ModelType.BASELINE])

        # Test baseline execution if implemented
        try:
            baseline_results = task.run_baseline(dummy_dataset_100x500)
            assert isinstance(baseline_results, list)
            assert all(isinstance(m, MetricResult) for m in baseline_results)
        except NotImplementedError:
            # Some tasks may not implement run_baseline
            pass

    except Exception as e:
        pytest.fail(f"Task {task_class.__name__} failed unexpectedly: {e}")
