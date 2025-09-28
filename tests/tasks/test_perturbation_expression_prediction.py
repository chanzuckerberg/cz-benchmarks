import pytest
import pandas as pd
import numpy as np

import anndata as ad
from czbenchmarks.tasks.single_cell import (
    PerturbationExpressionPredictionTask,
    PerturbationExpressionPredictionTaskInput,
)
from czbenchmarks.metrics.types import MetricResult, MetricType
from czbenchmarks.datasets.single_cell_perturbation import (
    SingleCellPerturbationDataset,
)
from czbenchmarks.tasks.single_cell.perturbation_expression_prediction import (
    load_perturbation_task_input_from_saved_files,
)

from czbenchmarks.datasets.types import Organism
from tests.utils import (
    create_dummy_anndata,
    create_dummy_perturbation_anndata,
)


@pytest.mark.parametrize("condition_column", ["condition", "perturbation"])
@pytest.mark.parametrize("control_name", ["ctrl", "control"])
@pytest.mark.parametrize("de_gene_col", ["gene_id", "gene_name"])
def test_perturbation_task(
    condition_column: str, control_name: str, de_gene_col: str, condition_control_sep: str = "_"
):
    """Test that PerturbationExpressionPredictionTask executes without errors."""
    # Create dummy perturbation data
    perturbation_data: dict = create_dummy_perturbation_anndata(
        n_cells=500,
        n_genes=200,
        organism=Organism.HUMAN,
        condition_column=condition_column,
        control_name=control_name,
        split_column="split",
    )
    gene_pert = perturbation_data["gene_pert"]
    # Convert sparse matrix to dense array to avoid matrix object issues
    cell_representation = perturbation_data["adata"].X.toarray()
    var_names = perturbation_data["adata"].var_names

    # Task and argument setup
    task = PerturbationExpressionPredictionTask(
        control_prefix=control_name,
        condition_column=condition_column,
        de_gene_col=de_gene_col,
        condition_control_sep=condition_control_sep,
    )

    # Create DE results DataFrame matching expected structure
    de_results = pd.DataFrame(
        {
            condition_column: [gene_pert] * len(var_names),
            de_gene_col: var_names,
            "logfoldchange": np.random.randn(len(var_names)),
            "pval_adj": np.random.uniform(0, 0.01, len(var_names)),
        }
    )

    # Create masked_adata_obs DataFrame and fix condition naming
    adata = perturbation_data["adata"]
    masked_adata_obs = adata.obs.copy()

    # Fix condition naming to match task expectations
    # Task expects control cells to be named: {control_prefix}_{condition}
    control_condition_name = f"{control_name}_{gene_pert}"
    masked_adata_obs.loc[
        masked_adata_obs[condition_column] == control_name, condition_column
    ] = control_condition_name

    # Create target_conditions_to_save dict - map cell IDs to lists of genes to mask
    target_conditions_to_save = {}
    for cell_id in adata.obs_names:
        # Sample some genes to mask for each cell
        n_genes_to_mask = min(10, len(var_names) // 2)
        target_conditions_to_save[cell_id] = list(
            np.random.choice(var_names, n_genes_to_mask, replace=False)
        )

    task_input = PerturbationExpressionPredictionTaskInput(
        de_results=de_results,
        masked_adata_obs=masked_adata_obs,
        var_index=var_names,
        target_conditions_to_save=target_conditions_to_save,
        row_index=adata.obs.index,
    )

    # Five metrics per condition: accuracy, precision, recall, f1, correlation
    # Determine number of non-control conditions dynamically
    cond_series = masked_adata_obs[condition_column].astype(str)
    non_control_conditions = [
        c for c in cond_series.unique() if not c.startswith(control_name + condition_control_sep)
    ]
    num_metrics = 5 * len(non_control_conditions)

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


@pytest.mark.parametrize("condition_column", ["condition", "perturbation"])
@pytest.mark.parametrize("control_name", ["ctrl", "control"])
@pytest.mark.parametrize("de_gene_col", ["gene_id", "gene_name"])
def test_perturbation_expression_prediction_task_wilcoxon(
    condition_column: str, control_name: str, de_gene_col: str, condition_control_sep: str = "_"
):
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
                    de_gene_col: gene_names,
                    "pval": [0.001] * len(gene_names),
                    "pval_adj": [0.001] * len(gene_names),
                    condition_column: ["gene_A"] * len(gene_names),
                }
            ),
            pd.DataFrame(
                {
                    "logfoldchange": true_lfc_gene_B,
                    de_gene_col: gene_names,
                    "pval": [0.001] * len(gene_names),
                    "pval_adj": [0.001] * len(gene_names),
                    condition_column: ["gene_B"] * len(gene_names),
                }
            ),
        ],
        ignore_index=True,
    )

    # Build AnnData with 4 groups: condition/control for A and B
    n_per_group = 4
    conditions = (
        ["gene_A"] * n_per_group
        + [condition_control_sep.join([control_name, "gene_A"])] * n_per_group
        + ["gene_B"] * n_per_group
        + [condition_control_sep.join([control_name, "gene_B"])] * n_per_group
    )
    # Create base cell names without underscores
    base_cell_names = [f"cellbarcode{i}" for i in range(len(conditions))]
    # Create extended obs names like real dataset: base + condition_control_sep + condition
    obs_names = [condition_control_sep.join([base_name, cond]) for base_name, cond in zip(base_cell_names, conditions)]

    X = np.zeros((len(conditions), len(gene_names)), dtype=float)
    # Set group means so that pred_log_fc equals the designed true_lfc
    X[0:n_per_group, :] = true_lfc_gene_A  # gene_A group
    X[n_per_group:2 * n_per_group, :] = 0.0  # ctrl_gene_A group
    X[2 * n_per_group:3 * n_per_group, :] = true_lfc_gene_B  # gene_B group
    X[3 * n_per_group:4 * n_per_group, :] = 0.0  # ctrl_gene_B group

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame({condition_column: conditions}, index=obs_names),
        var=pd.DataFrame(index=gene_names),
    )
    target_conditions_to_save = {obs_name: list(gene_names) for obs_name in obs_names}
    cell_representation = X

    task = PerturbationExpressionPredictionTask(
        control_prefix=control_name,
        condition_column=condition_column,
        de_gene_col=de_gene_col,
        condition_control_sep=condition_control_sep,
    )
    task_input = PerturbationExpressionPredictionTaskInput(
        de_results=de_res_wilcoxon_df,
        masked_adata_obs=adata.obs,
        var_index=adata.var_names,
        target_conditions_to_save=target_conditions_to_save,
        row_index=pd.Index(base_cell_names),  # Full dataset uses base names
    )

    # First, check that true/pred vectors produced by _run_task match expectations
    task_output = task._run_task(cell_representation, task_input)
    expected_conditions = {c for c in conditions if not c.startswith(control_name + condition_control_sep)}
    assert set(task_output.pred_log_fc_dict.keys()) == expected_conditions
    assert set(task_output.true_log_fc_dict.keys()) == expected_conditions
    assert np.allclose(
        task_output.pred_log_fc_dict["gene_A"],
        true_lfc_gene_A,
    )
    assert np.allclose(
        task_output.pred_log_fc_dict["gene_B"],
        true_lfc_gene_B,
    )
    assert np.allclose(
        task_output.true_log_fc_dict["gene_A"],
        true_lfc_gene_A,
    )
    assert np.allclose(
        task_output.true_log_fc_dict["gene_B"],
        true_lfc_gene_B,
    )

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


@pytest.mark.parametrize("condition_column", ["condition", "perturbation"])
@pytest.mark.parametrize("control_name", ["ctrl", "control"])
@pytest.mark.parametrize("de_gene_col", ["gene_id", "gene_name"])
def test_perturbation_expression_prediction_task_load_from_task_inputs(
    tmp_path, condition_column: str, control_name: str, de_gene_col: str, condition_control_sep: str = "_"
):
    """Test that the task can load inputs from stored task files."""
    # Create a dummy dataset and store its task inputs
    file_path = tmp_path / "dummy_perturbation.h5ad"
    adata = create_dummy_anndata(
        n_cells=6,
        n_genes=3,
        obs_columns=[condition_column],
        organism=Organism.HUMAN,
    )
    adata.obs[condition_column] = [
        control_name,
        control_name,
        "test1",
        "test1",
        "test2",
        "test2",
    ]
    adata.obs_names = [
        condition_control_sep.join([control_name, "test1_a"]),
        condition_control_sep.join([control_name, "test2_b"]),
        "cond_test1_a",
        "cond_test1_b",
        "cond_test2_a",
        "cond_test2_b",
    ]
    # Provide matched control cell IDs and DE results
    adata.uns["control_cells_ids"] = {
        "test1": [condition_control_sep.join([control_name, "test1_a"]), condition_control_sep.join([control_name, "test2_b"])],
        "test2": [condition_control_sep.join([control_name, "test1_a"]), condition_control_sep.join([control_name, "test2_b"])],
    }
    de_conditions = ["test1"] * 10 + ["test2"] * 10
    de_genes = [f"ENSG000000000{str(i).zfill(2)}" for i in range(20)]
    adata.uns["de_results_wilcoxon"] = pd.DataFrame(
        {
            condition_column: de_conditions,
            de_gene_col: de_genes,
            "pval_adj": [1e-6] * 20,
            "logfoldchange": [2.0] * 20,
        }
    )
    adata.write_h5ad(file_path)

    # Create dataset and store task inputs
    dataset = SingleCellPerturbationDataset(
        path=file_path,
        organism=Organism.HUMAN,
        condition_key=condition_column,
        control_name=control_name,
        de_gene_col=de_gene_col,
    )
    dataset.load_data()
    stored_dir = dataset.store_task_inputs()

    # Test loading task inputs using the standalone function
    task_input = load_perturbation_task_input_from_saved_files(stored_dir)

    # Verify the loaded task input has the expected structure
    assert isinstance(task_input, PerturbationExpressionPredictionTaskInput)
    assert isinstance(task_input.de_results, pd.DataFrame)
    assert isinstance(task_input.masked_adata_obs, pd.DataFrame)
    assert len(task_input.target_conditions_to_save) > 0
    assert all(
        isinstance(v, list) for v in task_input.target_conditions_to_save.values()
    )

    # Verify data integrity
    assert task_input.de_results.shape[0] > 0
    assert task_input.masked_adata_obs.shape[0] > 0
    assert len(task_input.var_index) > 0
