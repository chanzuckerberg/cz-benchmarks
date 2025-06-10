import pytest
import pandas as pd
from czbenchmarks.tasks.clustering import ClusteringTask
from czbenchmarks.tasks.embedding import EmbeddingTask
from czbenchmarks.tasks.integration import BatchIntegrationTask
from czbenchmarks.tasks.label_prediction import MetadataLabelPredictionTask
from czbenchmarks.tasks.single_cell.cross_species import CrossSpeciesIntegrationTask
from czbenchmarks.tasks.single_cell.perturbation import PerturbationTask
from czbenchmarks.datasets.types import Organism
from czbenchmarks.metrics.types import MetricResult
from tests.utils import DummyDataset, create_dummy_anndata, DummyTask


def test_missing_required_inputs_outputs():
    """Test that validation fails when dataset is missing required inputs/outputs."""
    task = DummyTask()
    dataset = DummyDataset("dummy_path", Organism.HUMAN)

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
    dataset.set_output(
        ModelType.BASELINE, DataType.PERTURBATION_PRED, ("", pd.DataFrame())
    )
    with pytest.raises(
        ValueError,
        match=".*Missing required outputs for model type BASELINE.*",
    ):
        task.validate(dataset)


def test_requires_multiple_datasets_validation():
    """Test that ValueError is raised when requires_multiple_datasets is True
    but input is not a list."""
    task = DummyTask(requires_multiple=True)
    dataset = DummyDataset("dummy_path", Organism.HUMAN)

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
    datasets = [
        DummyDataset("dummy1", Organism.HUMAN),
        DummyDataset("dummy2", Organism.HUMAN),
    ]

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
    datasets = [
        DummyDataset("dummy1", Organism.HUMAN),
        DummyDataset("dummy2", Organism.HUMAN),
    ]

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
    dataset = DummyDataset("dummy", Organism.HUMAN)

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
