import pytest
import pandas as pd
import numpy as np

import anndata as ad
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
    PerturbationTask,
    PerturbationTaskInput,
    PerturbationExpressionPredictionTask,
    PerturbationExpressionPredictionTaskInput,
)
from czbenchmarks.tasks.types import CellRepresentation
from czbenchmarks.datasets.types import Organism
from czbenchmarks.metrics.types import MetricResult, MetricType

from tests.utils import (
    create_dummy_anndata,
    DummyTask,
    create_dummy_perturbation_anndata,
    DummyTaskInput,
)


# FIXME these tests could be split into multiple files and fixtures moved to
# conftest.py


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
    obs: pd.DataFrame = anndata.obs.copy()
    var: pd.DataFrame = anndata.var.copy()

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
    # Enables lazy generation of fixture data so fixtures can be used as
    # parameters
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
    results = task.run(
        cell_representation=embedding,
        task_input=DummyTaskInput(),
    )

    assert isinstance(results, list)
    assert all(isinstance(r, MetricResult) for r in results)


@pytest.mark.parametrize(
    "fixture_data",
    [
        (
            "abcd",
            [False, "This task requires a single cell representation for input"],
        ),
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
                "This task requires a list of cell representations but only one "
                "was provided",
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
        task.run(
            cell_representation=embedding_list,
            task_input=DummyTaskInput(),
        )


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
    embedding_matrix,
    expression_matrix,
    obs,
):
    """Test that each task executes without errors on compatible data."""

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


def test_cross_species_task(embedding_matrix, obs):
    """Test that CrossSpeciesIntegrationTask executes without errors."""
    task = CrossSpeciesIntegrationTask()
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
    task_input = PerturbationTaskInput(
        var_names=var_names,
        gene_pert=gene_pert,
        perturbation_pred=pert_pred,
        perturbation_truth=pert_truth,
    )

    # Eight metrics: MSE and R2 for all/top20/top100 genes, Jaccard
    # top20/100
    num_metrics = 8

    try:
        # Test regular task execution
        results = task.run(
            cell_representation,
            task_input,
        )

        # Verify results structure
        assert isinstance(results, list)
        assert all(isinstance(r, MetricResult) for r in results)
        assert len(results) == num_metrics

        # Test baseline with both mean and median
        for baseline_type in ["mean", "median"]:
            baseline_embedding = task.compute_baseline(
                cell_representation=cell_representation,
                var_names=var_names,
                obs_names=obs_names,
                baseline_type=baseline_type,
            )
            task_input.perturbation_pred = baseline_embedding
            baseline_results = task.run(cell_representation, task_input)
            assert isinstance(baseline_results, list)
            assert all(isinstance(r, MetricResult) for r in baseline_results)
            assert len(baseline_results) == num_metrics

    except Exception as e:
        pytest.fail(f"PerturbationTask failed unexpectedly: {e}")


def test_perturbation_expression_prediction_task_wilcoxon():
    """Test Wilcoxon path computes correct vectors and metrics."""
    # Deterministic gene set and per-condition true/predicted effects
    gene_names = ["G0", "G1", "G2", "G3"]
    true_lfc_gene_A = np.array([1.0, 0.5, -0.5, -1.0])
    true_lfc_gene_B = np.array([2.0, 1.0, -1.0, -2.0])

    # Build DE results matching the designed true effects for both conditions
    de_res_wilcoxon_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "logfoldchange": true_lfc_gene_A,
                    "target_gene": ["gene_A"] * len(gene_names),
                    "names": gene_names,
                    "pval": [0.001] * len(gene_names),
                    "condition": ["gene_A"] * len(gene_names),
                    "condition_ensembl_id": ["ENSG_A"] * len(gene_names),
                }
            ),
            pd.DataFrame(
                {
                    "logfoldchange": true_lfc_gene_B,
                    "target_gene": ["gene_B"] * len(gene_names),
                    "names": gene_names,
                    "pval": [0.001] * len(gene_names),
                    "condition": ["gene_B"] * len(gene_names),
                    "condition_ensembl_id": ["ENSG_B"] * len(gene_names),
                }
            ),
        ],
        ignore_index=True,
    )

    # Build AnnData with 4 groups: condition/control for A and B
    n_per_group = 4
    conditions = (
        ["gene_A"] * n_per_group
        + ["ctrl_gene_A"] * n_per_group
        + ["gene_B"] * n_per_group
        + ["ctrl_gene_B"] * n_per_group
    )
    obs_names = [f"cell_{i}_{cond}" for i, cond in enumerate(conditions)]

    X = np.zeros((len(conditions), len(gene_names)), dtype=float)
    # Set group means so that pred_log_fc equals the designed true_lfc
    X[0:n_per_group, :] = true_lfc_gene_A  # gene_A group
    X[n_per_group : 2 * n_per_group, :] = 0.0  # ctrl_gene_A group
    X[2 * n_per_group : 3 * n_per_group, :] = true_lfc_gene_B  # gene_B group
    X[3 * n_per_group : 4 * n_per_group, :] = 0.0  # ctrl_gene_B group

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame({"condition": conditions}, index=obs_names),
        var=pd.DataFrame(index=gene_names),
    )
    target_genes_to_save = {obs_name: list(gene_names) for obs_name in obs_names}
    cell_representation = X

    # Ensure de_results has the expected gene identifier column
    de_res_wilcoxon_df["gene_id"] = de_res_wilcoxon_df["names"]

    task = PerturbationExpressionPredictionTask(
        min_de_genes=1,
        metric_type="wilcoxon",
        metric_column="logfoldchange",
        min_logfoldchange=0.01,
        pval_threshold=0.1,
        control_gene="ctrl",
    )
    task_input = PerturbationExpressionPredictionTaskInput(
        de_results=de_res_wilcoxon_df,
        dataset_adata=adata,
        target_genes_to_save=target_genes_to_save,
    )

    # First, check that true/pred vectors produced by _run_task match expectations
    task_output = task._run_task(cell_representation, task_input)
    assert set(task_output.pred_log_fc_dict.keys()) == {"gene_A", "gene_B"}
    assert set(task_output.true_log_fc_dict.keys()) == {"gene_A", "gene_B"}
    assert np.allclose(task_output.pred_log_fc_dict["gene_A"], true_lfc_gene_A)
    assert np.allclose(task_output.pred_log_fc_dict["gene_B"], true_lfc_gene_B)
    assert np.allclose(task_output.true_log_fc_dict["gene_A"], true_lfc_gene_A)
    assert np.allclose(task_output.true_log_fc_dict["gene_B"], true_lfc_gene_B)

    # Then, run the full task and validate metrics are perfect
    num_metric_types = 5  # accuracy, precision, recall, f1, spearman
    try:
        results = task.run(cell_representation, task_input)

        assert isinstance(results, dict)
        assert len(results) == num_metric_types

        # Each metric list should contain perfect scores for both conditions
        for name, metric_list in results.items():
            assert isinstance(metric_list, list)
            assert all(isinstance(r, MetricResult) for r in metric_list)
            # Expect results for both conditions
            assert len(metric_list) == 2
            for r in metric_list:
                assert np.isclose(r.value, 1.0)

        # Verify metric types present
        all_results = [
            result for metric_list in results.values() for result in metric_list
        ]
        metric_types = {result.metric_type for result in all_results}
        expected_types = {
            MetricType.ACCURACY,
            MetricType.PRECISION,
            MetricType.RECALL,
            MetricType.F1,
            MetricType.SPEARMAN_CORRELATION,
        }
        assert expected_types.issubset(metric_types)
    except Exception as e:
        pytest.fail(
            f"PerturbationExpressionPredictionTask (Wilcoxon) failed unexpectedly: {e}"
        )


def test_perturbation_expression_prediction_task_ttest():
    """Test t-test path computes correct vectors and metrics."""
    # Deterministic gene set and per-condition true/predicted effects
    gene_names = ["G0", "G1", "G2", "G3"]
    true_smd_gene_A = np.array([0.2, 0.5, -0.5, -0.2])
    true_smd_gene_B = np.array([1.0, 0.7, -0.7, -1.0])

    # Build DE results matching the designed true effects for both conditions
    de_res_ttest_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "standardized_mean_diff": true_smd_gene_A,
                    "target_gene": ["gene_A"] * len(gene_names),
                    "names": gene_names,
                    "pval": [0.001] * len(gene_names),
                    "condition": ["gene_A"] * len(gene_names),
                    "condition_ensembl_id": ["ENSG_A"] * len(gene_names),
                }
            ),
            pd.DataFrame(
                {
                    "standardized_mean_diff": true_smd_gene_B,
                    "target_gene": ["gene_B"] * len(gene_names),
                    "names": gene_names,
                    "pval": [0.001] * len(gene_names),
                    "condition": ["gene_B"] * len(gene_names),
                    "condition_ensembl_id": ["ENSG_B"] * len(gene_names),
                }
            ),
        ],
        ignore_index=True,
    )

    # Build AnnData with 4 groups: condition/control for A and B
    n_per_group = 4
    conditions = (
        ["gene_A"] * n_per_group
        + ["ctrl_gene_A"] * n_per_group
        + ["gene_B"] * n_per_group
        + ["ctrl_gene_B"] * n_per_group
    )
    obs_names = [f"cell_{i}_{cond}" for i, cond in enumerate(conditions)]

    X = np.zeros((len(conditions), len(gene_names)), dtype=float)
    # Set group means so that pred_log_fc equals the designed "true" standardized_mean_diff
    X[0:n_per_group, :] = true_smd_gene_A  # gene_A group
    X[n_per_group : 2 * n_per_group, :] = 0.0  # ctrl_gene_A group
    X[2 * n_per_group : 3 * n_per_group, :] = true_smd_gene_B  # gene_B group
    X[3 * n_per_group : 4 * n_per_group, :] = 0.0  # ctrl_gene_B group

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame({"condition": conditions}, index=obs_names),
        var=pd.DataFrame(index=gene_names),
    )
    target_genes_to_save = {obs_name: list(gene_names) for obs_name in obs_names}
    cell_representation = X

    # Ensure de_results has the expected gene identifier column
    de_res_ttest_df["gene_id"] = de_res_ttest_df["names"]

    task = PerturbationExpressionPredictionTask(
        min_de_genes=1,
        metric_type="t-test",
        metric_column="standardized_mean_diff",
        standardized_mean_diff=0.01,
        pval_threshold=0.1,
        control_gene="ctrl",
    )
    task_input = PerturbationExpressionPredictionTaskInput(
        de_results=de_res_ttest_df,
        dataset_adata=adata,
        target_genes_to_save=target_genes_to_save,
    )

    # First, check that true/pred vectors produced by _run_task match expectations
    task_output = task._run_task(cell_representation, task_input)
    assert set(task_output.pred_log_fc_dict.keys()) == {"gene_A", "gene_B"}
    assert set(task_output.true_log_fc_dict.keys()) == {"gene_A", "gene_B"}
    assert np.allclose(task_output.pred_log_fc_dict["gene_A"], true_smd_gene_A)
    assert np.allclose(task_output.pred_log_fc_dict["gene_B"], true_smd_gene_B)
    assert np.allclose(task_output.true_log_fc_dict["gene_A"], true_smd_gene_A)
    assert np.allclose(task_output.true_log_fc_dict["gene_B"], true_smd_gene_B)

    # Then, run the full task and validate metrics are perfect
    num_metric_types = 5  # accuracy, precision, recall, f1, spearman
    try:
        results = task.run(cell_representation, task_input)

        assert isinstance(results, dict)
        assert len(results) == num_metric_types

        # Each metric list should contain perfect scores for both conditions
        for name, metric_list in results.items():
            assert isinstance(metric_list, list)
            assert all(isinstance(r, MetricResult) for r in metric_list)
            assert len(metric_list) == 2
            for r in metric_list:
                assert np.isclose(r.value, 1.0)

        all_results = [
            result for metric_list in results.values() for result in metric_list
        ]
        metric_types = {result.metric_type for result in all_results}
        expected_types = {
            MetricType.ACCURACY,
            MetricType.PRECISION,
            MetricType.RECALL,
            MetricType.F1,
            MetricType.SPEARMAN_CORRELATION,
        }
        assert expected_types.issubset(metric_types)
    except Exception as e:
        pytest.fail(
            f"PerturbationExpressionPredictionTask (t-test) failed unexpectedly: {e}"
        )
