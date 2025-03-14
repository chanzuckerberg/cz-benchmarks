import pytest
import numpy as np
import tempfile
import os
from typing import List, Set

from czbenchmarks.tasks.base import BaseTask
from czbenchmarks.tasks.clustering import ClusteringTask
from czbenchmarks.tasks.embedding import EmbeddingTask
from czbenchmarks.tasks.integration import BatchIntegrationTask
from czbenchmarks.tasks.label_prediction import MetadataLabelPredictionTask
from czbenchmarks.datasets import BaseDataset, DataType, SingleCellDataset
from czbenchmarks.datasets.types import Organism
from czbenchmarks.models.types import ModelType
from czbenchmarks.metrics.types import MetricResult, MetricType
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
        return [
            MetricResult(
                name="dummy", value=1.0, metric_type=MetricType.ADJUSTED_RAND_INDEX
            )
        ]


def test_missing_required_inputs_outputs():
    """Test that validation fails when dataset is missing required inputs/outputs."""
    task = DummyTask()
    dataset = DummyDataset("dummy_path")

    # Don't set any inputs/outputs
    with pytest.raises(
        ValueError,
        match=".*Missing required inputs.*",
    ):
        task.validate(dataset)

    # Set inputs but no model outputs at all
    adata = create_dummy_anndata(n_cells=10, n_genes=20, organism=Organism.HUMAN)
    dataset.set_input(DataType.ANNDATA, adata)
    dataset.set_input(DataType.METADATA, adata.obs)
    with pytest.raises(
        ValueError,
        match=".*No model outputs available.*",
    ):
        task.validate(dataset)

    # Set inputs and model type but missing required outputs
    dataset.set_output(ModelType.BASELINE, DataType.ANNDATA, adata)
    with pytest.raises(
        ValueError,
        match=".*Missing required outputs for model type BASELINE.*",
    ):
        task.validate(dataset)


def test_requires_multiple_datasets_validation():
    """Test that ValueError is raised when requires_multiple_datasets is True
    but input is not a list."""
    task = DummyTask(requires_multiple=True)
    dataset = DummyDataset("dummy_path")

    # Set required inputs/outputs
    adata = create_dummy_anndata(n_cells=10, n_genes=20, organism=Organism.HUMAN)
    dataset.set_input(DataType.ANNDATA, adata)
    dataset.set_input(DataType.METADATA, adata.obs)
    dataset.set_output(ModelType.BASELINE, DataType.EMBEDDING, adata.X.toarray())

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
    input on a task requiring a single dataset."""
    task = DummyTask(requires_multiple=False)
    datasets = [DummyDataset("dummy1"), DummyDataset("dummy2")]

    # Set required inputs/outputs for both datasets
    for dataset in datasets:
        adata = create_dummy_anndata(n_cells=10, n_genes=20, organism=Organism.HUMAN)
        dataset.set_input(DataType.ANNDATA, adata)
        dataset.set_input(DataType.METADATA, adata.obs)
        dataset.set_output(ModelType.SCVI, DataType.EMBEDDING, adata.X.toarray())

    results = task.run(datasets)

    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(r, dict) for r in results)
    assert all(ModelType.SCVI in r for r in results)
    assert all(isinstance(r[ModelType.SCVI], list) for r in results)


def test_list_input_multiple_task():
    """Test that Dict[ModelType, List[MetricResult]] is returned for list input
    on a task requiring multiple datasets."""
    task = DummyTask(requires_multiple=True)
    datasets = [DummyDataset("dummy1"), DummyDataset("dummy2")]

    # Set required inputs/outputs for both datasets
    for dataset in datasets:
        adata = create_dummy_anndata(n_cells=10, n_genes=20, organism=Organism.HUMAN)
        dataset.set_input(DataType.ANNDATA, adata)
        dataset.set_input(DataType.METADATA, adata.obs)
        dataset.set_output(ModelType.SCVI, DataType.EMBEDDING, adata.X.toarray())
    results = task.run(datasets)

    assert isinstance(results, dict)
    assert ModelType.SCVI in results
    assert isinstance(results[ModelType.SCVI], list)


def test_single_dataset_result():
    """Test that Dict[ModelType, List[MetricResult]] is returned for single dataset."""
    task = DummyTask()
    dataset = DummyDataset("dummy")

    # Set required inputs/outputs
    adata = create_dummy_anndata(n_cells=10, n_genes=20, organism=Organism.HUMAN)
    dataset.set_input(DataType.ANNDATA, adata)
    dataset.set_input(DataType.METADATA, adata.obs)
    dataset.set_output(ModelType.BASELINE, DataType.EMBEDDING, adata.X.toarray())

    results = task.run(dataset)

    assert isinstance(results, dict)
    assert ModelType.BASELINE in results
    assert isinstance(results[ModelType.BASELINE], list)
    assert len(results[ModelType.BASELINE]) == 1
    assert isinstance(results[ModelType.BASELINE][0], MetricResult)


@pytest.fixture
def dummy_single_cell_dataset(n_cells: int = 1000, n_genes: int = 500):
    """Create a dummy dataset with 100 cells and 500 genes."""
    # Create a temporary directory and file
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = os.path.join(tmp_dir, "dummy.h5ad")

        # Create the dataset with the temporary file path
        dataset = SingleCellDataset(tmp_file, organism=Organism.HUMAN)
        adata = create_dummy_anndata(
            n_cells=n_cells,
            n_genes=n_genes,
            obs_columns=["cell_type", "batch"],
            var_columns=["feature_id"],
            organism=Organism.HUMAN,
        )

        # Write the AnnData object to the temporary file
        adata.write_h5ad(tmp_file)

        dataset.set_input(DataType.ANNDATA, adata)
        dataset.set_input(DataType.METADATA, adata.obs)

        # Create a random embedding
        embedding = np.random.normal(size=(n_cells, 32))
        dataset.set_output(ModelType.SCVI, DataType.EMBEDDING, embedding)
        yield dataset


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
