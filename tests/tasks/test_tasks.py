import pytest
import pandas as pd
import numpy as np
from pathlib import Path
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
    """Test that PerturbationExpressionPredictionTask executes without errors."""
    # Create dummy perturbation data
    perturbation_data: dict = create_dummy_perturbation_anndata(
        n_cells=500,
        n_genes=200,
        organism=Organism.HUMAN,
        condition_column="condition",
        split_column="split",
    )
    gene_pert = perturbation_data["gene_pert"]
    # Convert sparse matrix to dense array to avoid matrix object issues
    cell_representation = perturbation_data["adata"].X.toarray()
    var_names = perturbation_data["adata"].var_names

    # Task and argument setup
    task = PerturbationExpressionPredictionTask()

    # Create DE results DataFrame matching expected structure
    de_results = pd.DataFrame(
        {
            "condition": [gene_pert] * len(var_names),
            "gene_id": var_names,
            "logfoldchange": np.random.randn(len(var_names)),
            "pval_adj": np.random.uniform(0, 0.01, len(var_names)),
        }
    )

    # Create masked_adata_obs DataFrame and fix condition naming
    adata = perturbation_data["adata"]
    masked_adata_obs = adata.obs.copy()

    # Fix condition naming to match task expectations
    # Task expects control cells to be named: {control_prefix}_{condition}
    control_condition_name = f"non-targeting_{gene_pert}"
    masked_adata_obs.loc[masked_adata_obs["condition"] == "ctrl", "condition"] = (
        control_condition_name
    )

    # Create target_conditions_dict dict - map condition names to lists of genes to mask
    target_conditions_dict = {}
    # Map each perturbation condition to genes to mask
    unique_conditions = np.unique(
        masked_adata_obs["condition"][
            ~masked_adata_obs["condition"].str.startswith("non-targeting")
        ]
    )
    for condition in unique_conditions:
        # Sample some genes to mask for each condition
        n_genes_to_mask = min(10, len(var_names) // 2)
        target_conditions_dict[condition] = list(
            np.random.choice(var_names, n_genes_to_mask, replace=False)
        )

    # Create AnnData with required data
    test_adata = ad.AnnData(
        X=adata.X, obs=masked_adata_obs, var=pd.DataFrame(index=var_names)
    )
    test_adata.uns["cell_barcode_index"] = adata.obs.index.astype(str).values

    task_input = PerturbationExpressionPredictionTaskInput(
        adata=test_adata,
        target_conditions_dict=target_conditions_dict,
        de_results=de_results,
    )

    # Five metrics per condition: accuracy, precision, recall, f1, correlation
    # We have one perturbed condition, so 5 metrics total
    num_metrics = 5

    try:
        # Test regular task execution
        results = task.run(
            cell_representation,
            task_input,
        )

        # Verify results structure - the method returns a list of MetricResult
        assert isinstance(results, list)
        assert all(isinstance(r, MetricResult) for r in results)
        assert len(results) == num_metrics

        # Test baseline with both mean and median
        for baseline_type in ["mean", "median"]:
            baseline_embedding = task.compute_baseline(
                cell_representation=cell_representation,
                baseline_type=baseline_type,
            )
            # Create a new task input with the baseline embedding
            baseline_results = task.run(baseline_embedding, task_input)
            assert isinstance(baseline_results, list)
            assert all(isinstance(r, MetricResult) for r in baseline_results)
            assert len(baseline_results) == num_metrics
    except Exception as e:
        pytest.fail(f"Test failed with exception: {e}")


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
                    "pval_adj": [0.001] * len(gene_names),
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
                    "pval_adj": [0.001] * len(gene_names),
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
    # Create base cell names without underscores (so str.split("_").str[0] works correctly)
    base_cell_names = [f"cellbarcode{i}" for i in range(len(conditions))]
    # Create extended obs names like real dataset: base_name + "_" + condition
    obs_names = [
        f"{base_name}_{cond}" for base_name, cond in zip(base_cell_names, conditions)
    ]

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
    target_conditions_dict = {"gene_A": list(gene_names), "gene_B": list(gene_names)}
    cell_representation = X

    # Ensure de_results has the expected gene identifier column
    de_res_wilcoxon_df["gene_id"] = de_res_wilcoxon_df["names"]

    task = PerturbationExpressionPredictionTask(
        control_prefix="ctrl",
    )
    # Create AnnData with required data
    test_adata = adata.copy()
    test_adata.uns["cell_barcode_index"] = pd.Index(base_cell_names).astype(str).values

    task_input = PerturbationExpressionPredictionTaskInput(
        adata=test_adata,
        target_conditions_dict=target_conditions_dict,
        de_results=de_res_wilcoxon_df,
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
    results = task.run(cell_representation, task_input)

    assert isinstance(results, list)
    assert all(isinstance(r, MetricResult) for r in results)
    # Expect results for both conditions x 5 metric types = 10 total results
    assert len(results) == 10

    # Each result should have perfect scores
    for r in results:
        assert np.isclose(r.value, 1.0)

    # Verify metric types present
    metric_types = {result.metric_type for result in results}
    expected_types = {
        MetricType.ACCURACY_CALCULATION,
        MetricType.PRECISION_CALCULATION,
        MetricType.RECALL_CALCULATION,
        MetricType.F1_CALCULATION,
        MetricType.SPEARMAN_CORRELATION_CALCULATION,
    }
    assert expected_types.issubset(metric_types)


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
                    "logfoldchange": true_smd_gene_A,  # Add logfoldchange column for compatibility
                    "target_gene": ["gene_A"] * len(gene_names),
                    "names": gene_names,
                    "pval": [0.001] * len(gene_names),
                    "pval_adj": [0.001] * len(gene_names),
                    "condition": ["gene_A"] * len(gene_names),
                    "condition_ensembl_id": ["ENSG_A"] * len(gene_names),
                }
            ),
            pd.DataFrame(
                {
                    "standardized_mean_diff": true_smd_gene_B,
                    "logfoldchange": true_smd_gene_B,  # Add logfoldchange column for compatibility
                    "target_gene": ["gene_B"] * len(gene_names),
                    "names": gene_names,
                    "pval": [0.001] * len(gene_names),
                    "pval_adj": [0.001] * len(gene_names),
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
    # Create base cell names without underscores (so str.split("_").str[0] works correctly)
    base_cell_names = [f"cellbarcode{i}" for i in range(len(conditions))]
    # Create extended obs names like real dataset: base_name + "_" + condition
    obs_names = [
        f"{base_name}_{cond}" for base_name, cond in zip(base_cell_names, conditions)
    ]

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
    target_conditions_dict = {"gene_A": list(gene_names), "gene_B": list(gene_names)}
    cell_representation = X

    # Ensure de_results has the expected gene identifier column
    de_res_ttest_df["gene_id"] = de_res_ttest_df["names"]

    task = PerturbationExpressionPredictionTask(
        control_prefix="ctrl",
    )
    # Create AnnData with required data
    test_adata = adata.copy()
    test_adata.uns["cell_barcode_index"] = pd.Index(base_cell_names).astype(str).values

    task_input = PerturbationExpressionPredictionTaskInput(
        adata=test_adata,
        target_conditions_dict=target_conditions_dict,
        de_results=de_res_ttest_df,
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
    try:
        results = task.run(cell_representation, task_input)

        assert isinstance(results, list)
        assert all(isinstance(r, MetricResult) for r in results)
        # Expect results for both conditions x 5 metric types = 10 total results
        assert len(results) == 10

        # Each result should have perfect scores
        for r in results:
            assert np.isclose(r.value, 1.0)

        metric_types = {result.metric_type for result in results}
        expected_types = {
            MetricType.ACCURACY_CALCULATION,
            MetricType.PRECISION_CALCULATION,
            MetricType.RECALL_CALCULATION,
            MetricType.F1_CALCULATION,
            MetricType.SPEARMAN_CORRELATION_CALCULATION,
        }
        assert expected_types.issubset(metric_types)
    except Exception as e:
        pytest.fail(
            f"PerturbationExpressionPredictionTask (t-test) failed unexpectedly: {e}"
        )


def test_perturbation_expression_prediction_task_load_from_task_inputs(tmp_path):
    """Test that the task can load inputs from stored task files."""
    from czbenchmarks.datasets.single_cell_perturbation import (
        SingleCellPerturbationDataset,
    )
    from czbenchmarks.datasets.types import Organism
    from tests.utils import create_dummy_anndata

    # Create a dummy dataset and store its task inputs
    file_path = tmp_path / "dummy_perturbation.h5ad"
    adata = create_dummy_anndata(
        n_cells=6,
        n_genes=3,
        obs_columns=["condition"],
        organism=Organism.HUMAN,
    )
    adata.obs["condition"] = ["ctrl", "ctrl", "test1", "test1", "test2", "test2"]
    adata.obs_names = [
        "ctrl_test1_a",
        "ctrl_test2_b",
        "cond_test1_a",
        "cond_test1_b",
        "cond_test2_a",
        "cond_test2_b",
    ]
    # Provide matched control cell IDs and DE results
    adata.uns["control_cells_ids"] = {
        "test1": ["ctrl_test1_a", "ctrl_test2_b"],
        "test2": ["ctrl_test1_a", "ctrl_test2_b"],
    }
    de_conditions = ["test1"] * 10 + ["test2"] * 10
    de_genes = [f"ENSG000000000{str(i).zfill(2)}" for i in range(20)]
    adata.uns["de_results_wilcoxon"] = pd.DataFrame(
        {
            "condition": de_conditions,
            "gene": de_genes,
            "pval_adj": [1e-6] * 20,
            "logfoldchange": [2.0] * 20,
        }
    )
    adata.write_h5ad(file_path)

    # Create dataset and store task inputs
    dataset = SingleCellPerturbationDataset(
        path=file_path,
        organism=Organism.HUMAN,
        condition_key="condition",
        control_name="ctrl",
    )
    dataset.load_data()
    task_inputs_dir = dataset.store_task_inputs()

    # Test loading task inputs using the standalone function
    from czbenchmarks.tasks.single_cell.perturbation_expression_prediction import (
        load_perturbation_task_input_from_saved_files,
    )

    task_input = load_perturbation_task_input_from_saved_files(task_inputs_dir)

    # Verify the loaded task input has the expected structure
    assert isinstance(task_input, PerturbationExpressionPredictionTaskInput)
    assert hasattr(task_input, "adata")
    assert hasattr(task_input, "target_conditions_dict")
    assert hasattr(task_input, "de_results")
    assert isinstance(task_input.adata, ad.AnnData)
    assert isinstance(task_input.target_conditions_dict, dict)
    assert isinstance(task_input.de_results, pd.DataFrame)

    # Verify AnnData contains required data in uns
    assert "cell_barcode_index" in task_input.adata.uns
    assert "control_cells_ids" in task_input.adata.uns

    # Verify data integrity
    assert task_input.de_results.shape[0] > 0
    assert task_input.adata.obs.shape[0] > 0
    assert len(task_input.adata.var.index) > 0
    assert len(task_input.target_conditions_dict) > 0

    # Verify cell barcode index matches adata size
    assert len(task_input.adata.uns["cell_barcode_index"]) == dataset.adata.shape[0]


def test_perturbation_expression_prediction_task_with_shuffled_input():
    """Test that the perturbation task works with shuffled AnnData input."""
    from czbenchmarks.datasets.single_cell_perturbation import (
        SingleCellPerturbationDataset,
    )
    from czbenchmarks.datasets.types import Organism
    from tests.utils import create_dummy_anndata
    import tempfile

    # Create a dummy dataset
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = Path(tmp_dir) / "dummy_perturbation.h5ad"
        adata = create_dummy_anndata(
            n_cells=6,
            n_genes=4,  # Use 4 genes for easier testing
            obs_columns=["condition"],
            organism=Organism.HUMAN,
        )
        adata.obs["condition"] = ["ctrl", "ctrl", "test1", "test1", "test2", "test2"]
        adata.obs_names = [
            "ctrl_test1_a",
            "ctrl_test2_b",
            "cond_test1_a",
            "cond_test1_b",
            "cond_test2_a",
            "cond_test2_b",
        ]
        # Provide matched control cell IDs and DE results
        adata.uns["control_cells_ids"] = {
            "test1": ["ctrl_test1_a", "ctrl_test2_b"],
            "test2": ["ctrl_test1_a", "ctrl_test2_b"],
        }

        # Create DE results that match the actual gene names in the dataset
        # The create_dummy_anndata creates genes with names like "ENSG000000000{i:02d}"
        gene_names = (
            adata.var.index.tolist()
        )  # Get actual gene names from the created dataset

        # Create sufficient DE results for both conditions using actual gene names
        de_data = []
        for condition in ["test1", "test2"]:
            for gene in gene_names:
                de_data.append(
                    {
                        "condition": condition,
                        "gene_id": gene,  # Use gene_id instead of gene
                        "pval_adj": 1e-6,
                        "logfoldchange": 2.0,
                    }
                )

        adata.uns["de_results_wilcoxon"] = pd.DataFrame(de_data)
        adata.write_h5ad(file_path)

        # Create dataset and load data with relaxed parameters
        dataset = SingleCellPerturbationDataset(
            path=file_path,
            organism=Organism.HUMAN,
            condition_key="condition",
            control_name="ctrl",
            deg_test_name="wilcoxon",
            percent_genes_to_mask=1.0,  # Use all genes to avoid filtering
            min_de_genes_to_mask=1,  # Minimum threshold
            pval_threshold=1.0,  # Accept all p-values
            min_logfoldchange=0.0,  # Accept all log fold changes
        )
        dataset.load_data()

        # Create task input with original ordering
        original_task_input = PerturbationExpressionPredictionTaskInput(
            adata=dataset.control_matched_adata,
            target_conditions_dict=dataset.target_conditions_dict,
            de_results=dataset.de_results,
        )

        # Create shuffled version of the AnnData
        shuffled_adata = dataset.control_matched_adata.copy()

        # Shuffle obs and var with fixed random seed for reproducibility
        np.random.seed(42)
        obs_shuffled = shuffled_adata.obs.sample(frac=1, random_state=42)
        var_shuffled = shuffled_adata.var.sample(frac=1, random_state=42)

        # Get the indices for shuffling
        obs_order = [shuffled_adata.obs.index.get_loc(i) for i in obs_shuffled.index]
        var_order = [shuffled_adata.var.index.get_loc(i) for i in var_shuffled.index]

        # Shuffle the data matrix accordingly
        X_shuffled = shuffled_adata.X[np.ix_(obs_order, var_order)]

        # Create new AnnData with shuffled data
        shuffled_adata = ad.AnnData(X=X_shuffled, obs=obs_shuffled, var=var_shuffled)

        # Copy over the uns data
        shuffled_adata.uns = dataset.control_matched_adata.uns.copy()

        # Create task input with shuffled AnnData
        shuffled_task_input = PerturbationExpressionPredictionTaskInput(
            adata=shuffled_adata,
            target_conditions_dict=dataset.target_conditions_dict,
            de_results=dataset.de_results,
        )

        # Create identical model output for both (using original dimensions)
        model_output = np.random.rand(
            dataset.control_matched_adata.shape[0],
            dataset.control_matched_adata.shape[1],
        )

        # For shuffled input, we need to reorder the model output to match
        model_output_shuffled = model_output[np.ix_(obs_order, var_order)]

        # Initialize task
        task = PerturbationExpressionPredictionTask()

        # Run task with both inputs
        original_result = task._run_task(model_output, original_task_input)
        shuffled_result = task._run_task(model_output_shuffled, shuffled_task_input)

        # Results should be identical (same conditions, same relative data)
        assert set(original_result.pred_log_fc_dict.keys()) == set(
            shuffled_result.pred_log_fc_dict.keys()
        )
        assert set(original_result.true_log_fc_dict.keys()) == set(
            shuffled_result.true_log_fc_dict.keys()
        )

        # The predicted values should be the same since we used correspondingly shuffled model output
        for condition in original_result.pred_log_fc_dict.keys():
            np.testing.assert_allclose(
                original_result.pred_log_fc_dict[condition],
                shuffled_result.pred_log_fc_dict[condition],
                rtol=1e-10,
                err_msg=f"Predicted log fold changes differ for condition {condition}",
            )
            np.testing.assert_allclose(
                original_result.true_log_fc_dict[condition],
                shuffled_result.true_log_fc_dict[condition],
                rtol=1e-10,
                err_msg=f"True log fold changes differ for condition {condition}",
            )


def test_perturbation_task_apply_model_ordering():
    """Test the apply_model_ordering method for PerturbationExpressionPredictionTaskInput."""
    from czbenchmarks.datasets.single_cell_perturbation import (
        SingleCellPerturbationDataset,
    )
    from czbenchmarks.datasets.types import Organism
    from tests.utils import create_dummy_anndata
    import tempfile

    # Create a dummy dataset
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = Path(tmp_dir) / "dummy_perturbation.h5ad"
        adata = create_dummy_anndata(
            n_cells=6,
            n_genes=4,
            obs_columns=["condition"],
            organism=Organism.HUMAN,
        )
        adata.obs["condition"] = ["ctrl", "ctrl", "test1", "test1", "test2", "test2"]
        adata.obs_names = [
            "ctrl_test1_a",
            "ctrl_test2_b",
            "cond_test1_a",
            "cond_test1_b",
            "cond_test2_a",
            "cond_test2_b",
        ]

        # Provide matched control cell IDs and DE results
        adata.uns["control_cells_ids"] = {
            "test1": ["ctrl_test1_a", "ctrl_test2_b"],
            "test2": ["ctrl_test1_a", "ctrl_test2_b"],
        }

        # Create DE results
        gene_names = adata.var.index.tolist()
        de_data = []
        for condition in ["test1", "test2"]:
            for gene in gene_names:
                de_data.append(
                    {
                        "condition": condition,
                        "gene_id": gene,
                        "pval_adj": 1e-6,
                        "logfoldchange": 2.0,
                    }
                )

        adata.uns["de_results_wilcoxon"] = pd.DataFrame(de_data)
        adata.write_h5ad(file_path)

        # Create dataset and get task input
        dataset = SingleCellPerturbationDataset(
            path=file_path,
            organism=Organism.HUMAN,
            condition_key="condition",
            control_name="ctrl",
            min_de_genes_to_mask=1,  # Lower threshold so genes get sampled
            percent_genes_to_mask=1.0,  # Use all genes
        )
        dataset.load_data()

        # Create task input
        task_input = PerturbationExpressionPredictionTaskInput(
            adata=dataset.control_matched_adata,
            target_conditions_dict=dataset.target_conditions_dict,
            de_results=dataset.de_results,
        )

        # Create model data with shuffled ordering (same content, different order)
        model_adata = dataset.control_matched_adata.copy()

        # Shuffle gene order
        import numpy as np

        np.random.seed(42)  # For reproducible test
        gene_order = np.random.permutation(model_adata.var.index)
        model_adata = model_adata[:, gene_order]

        # Shuffle cell order
        cell_order = np.random.permutation(model_adata.obs.index)
        model_adata = model_adata[cell_order, :]

        # Store original orderings
        original_gene_order = task_input.adata.var.index.copy()
        original_cell_barcode_index = task_input.adata.uns["cell_barcode_index"].copy()

        # Apply model ordering
        task_input.apply_model_ordering(model_adata)

        # Verify that orderings have changed to match model data
        pd.testing.assert_index_equal(task_input.adata.var.index, model_adata.var.index)
        np.testing.assert_array_equal(
            task_input.adata.uns["cell_barcode_index"],
            model_adata.obs.index.astype(str).values,
        )

        # Verify orderings are different from original (unless by chance they're the same)
        assert not task_input.adata.var.index.equals(original_gene_order)
        assert not np.array_equal(
            task_input.adata.uns["cell_barcode_index"], original_cell_barcode_index
        )


def test_perturbation_task_apply_model_ordering_validation():
    """Test that apply_model_ordering validates matching gene and cell sets."""
    from czbenchmarks.datasets.single_cell_perturbation import (
        SingleCellPerturbationDataset,
    )
    from czbenchmarks.datasets.types import Organism
    from tests.utils import create_dummy_anndata
    import tempfile

    # Create a dummy dataset
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = Path(tmp_dir) / "dummy_perturbation.h5ad"
        adata = create_dummy_anndata(
            n_cells=4,
            n_genes=3,
            obs_columns=["condition"],
            organism=Organism.HUMAN,
        )
        adata.obs["condition"] = ["ctrl", "ctrl", "test1", "test1"]
        adata.obs_names = ["ctrl_a", "ctrl_b", "cond_a", "cond_b"]

        # Provide matched control cell IDs and DE results
        adata.uns["control_cells_ids"] = {"test1": ["ctrl_a", "ctrl_b"]}

        # Create DE results
        gene_names = adata.var.index.tolist()
        de_data = []
        for gene in gene_names:
            de_data.append(
                {
                    "condition": "test1",
                    "gene_id": gene,
                    "pval_adj": 1e-6,
                    "logfoldchange": 2.0,
                }
            )

        adata.uns["de_results_wilcoxon"] = pd.DataFrame(de_data)
        adata.write_h5ad(file_path)

        # Create dataset and get task input
        dataset = SingleCellPerturbationDataset(
            path=file_path,
            organism=Organism.HUMAN,
            condition_key="condition",
            control_name="ctrl",
            min_de_genes_to_mask=1,  # Lower threshold so genes get sampled
            percent_genes_to_mask=1.0,  # Use all genes
        )
        dataset.load_data()

        # Create task input
        task_input = PerturbationExpressionPredictionTaskInput(
            adata=dataset.control_matched_adata,
            target_conditions_dict=dataset.target_conditions_dict,
            de_results=dataset.de_results,
        )

        # Test with mismatched genes
        model_adata_bad_genes = create_dummy_anndata(
            n_cells=4,
            n_genes=3,
            obs_columns=["condition"],
            organism=Organism.HUMAN,
        )
        model_adata_bad_genes.obs_names = task_input.adata.obs.index  # Same cells
        model_adata_bad_genes.var_names = [
            "different_gene1",
            "different_gene2",
            "different_gene3",
        ]  # Different genes

        with pytest.raises(
            ValueError, match="Gene indices in task input and model data do not match"
        ):
            task_input.apply_model_ordering(model_adata_bad_genes)

        # Test with mismatched cells
        model_adata_bad_cells = task_input.adata.copy()
        model_adata_bad_cells.obs_names = [
            "different_cell1",
            "different_cell2",
            "different_cell3",
            "different_cell4",
        ]  # Different cells

        with pytest.raises(
            ValueError, match="Cell indices in task input and model data do not match"
        ):
            task_input.apply_model_ordering(model_adata_bad_cells)
