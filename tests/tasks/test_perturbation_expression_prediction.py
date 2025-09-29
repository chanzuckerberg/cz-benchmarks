import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
import anndata as ad
from czbenchmarks.tasks.single_cell import (
    PerturbationExpressionPredictionTask,
    PerturbationExpressionPredictionTaskInput,
)
from czbenchmarks.tasks.single_cell.perturbation_expression_prediction import (
    PerturbationExpressionPredictionTaskInput,
    build_task_input_from_predictions,
)
from czbenchmarks.datasets.types import Organism
from czbenchmarks.datasets.single_cell_perturbation import (
    SingleCellPerturbationDataset,
)
from czbenchmarks.metrics.types import MetricResult, MetricType
from tests.utils import create_dummy_anndata, create_dummy_perturbation_anndata


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
    # Log-normalize the data to pass validation (add small constant to avoid log(0))
    cell_representation = np.log1p(cell_representation)
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
    # Populate UNS like dataset would
    test_adata.uns["control_cells_ids"] = {}
    test_adata.uns["de_results"] = de_results[["condition", "gene_id", "logfoldchange"]]
    test_adata.uns["metric_column"] = "logfoldchange"
    # Provide explicit target gene lists per condition
    test_adata.uns["target_conditions_dict"] = target_conditions_dict

    # Provide strict 1-1 control mapping for the single condition
    treated_idx = masked_adata_obs.index[masked_adata_obs["condition"] == gene_pert]
    control_idx = masked_adata_obs.index[
        masked_adata_obs["condition"] == control_condition_name
    ]
    mapping = {}
    if len(control_idx) > 0:
        for i, tb in enumerate(treated_idx):
            mapping[str(tb)] = str(control_idx[i % len(control_idx)])
    test_adata.uns["control_cells_map"] = {gene_pert: mapping}

    task_input = PerturbationExpressionPredictionTaskInput(
        adata=test_adata,
        gene_index=test_adata.var.index,
        cell_index=test_adata.obs.index,
    )

    # Only one metric per condition (Spearman correlation)
    num_metrics = 1

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

    # Create log-normalized data that will pass validation and gives expected differences
    # Since the task computes: mean(treatment) - mean(control), we create data where this equals true_lfc
    base_value = 1.0  # Base value that will become 0 after log transformation
    cell_representation = np.zeros_like(X, dtype=float)

    # For treatment groups: set to base_value + true_lfc so log difference will be true_lfc
    cell_representation[0:n_per_group, :] = base_value + true_lfc_gene_A  # gene_A group
    cell_representation[n_per_group : 2 * n_per_group, :] = (
        base_value  # ctrl_gene_A group
    )
    cell_representation[2 * n_per_group : 3 * n_per_group, :] = (
        base_value + true_lfc_gene_B
    )  # gene_B group
    cell_representation[3 * n_per_group : 4 * n_per_group, :] = (
        base_value  # ctrl_gene_B group
    )

    # Add small constant to ensure fractional cell sums for validation (must exceed epsilon=1e-2)
    cell_representation += 0.003

    # Ensure de_results has the expected gene identifier column
    de_res_wilcoxon_df["gene_id"] = de_res_wilcoxon_df["names"]

    task = PerturbationExpressionPredictionTask(
        control_name="ctrl",
    )
    # Create AnnData with required data
    test_adata = adata.copy()
    # UNS required by task
    dr = de_res_wilcoxon_df.copy()
    dr["gene_id"] = dr["names"]
    test_adata.uns["de_results"] = dr[["condition", "gene_id", "logfoldchange"]]
    test_adata.uns["metric_column"] = "logfoldchange"
    test_adata.uns["control_cells_ids"] = {}
    test_adata.uns["target_conditions_dict"] = target_conditions_dict
    # Build 1-1 mapping per condition using block structure
    treated_A = obs_names[0:n_per_group]
    ctrl_A = obs_names[n_per_group : 2 * n_per_group]
    treated_B = obs_names[2 * n_per_group : 3 * n_per_group]
    ctrl_B = obs_names[3 * n_per_group : 4 * n_per_group]
    map_A = {t: c for t, c in zip(treated_A, ctrl_A)}
    map_B = {t: c for t, c in zip(treated_B, ctrl_B)}
    test_adata.uns["control_cells_map"] = {"gene_A": map_A, "gene_B": map_B}

    task_input = PerturbationExpressionPredictionTaskInput(
        adata=test_adata,
        gene_index=test_adata.var.index,
        cell_index=test_adata.obs.index,
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
    # Task returns an aggregate metric across conditions
    assert len(results) == 1

    # Each result should have perfect correlation
    for r in results:
        assert np.isclose(r.value, 1.0)

    # Verify metric types present
    metric_types = {result.metric_type for result in results}
    expected_types = {MetricType.SPEARMAN_CORRELATION_CALCULATION}
    assert expected_types.issubset(metric_types)


def test_task_uses_strict_pair_mapping_for_pred_lfc():
    # Build toy dataset: 2 genes, 4 cells (2 control, 2 treated for pertA)
    X = np.array(
        [
            [1.0, 2.0],  # NT1
            [2.0, 1.0],  # NT2
            [3.0, 4.0],  # T1
            [5.0, 3.0],  # T2
        ]
    )
    obs = pd.DataFrame(
        {
            "condition": ["non-targeting", "non-targeting", "pertA", "pertA"],
        },
        index=["NT1", "NT2", "T1", "T2"],
    )
    var = pd.DataFrame(index=["g1", "g2"])
    adata = ad.AnnData(X=X, obs=obs, var=var)

    # Provide required UNS for the task
    de = pd.DataFrame(
        {
            "condition": ["pertA", "pertA"],
            "gene_id": ["g1", "g2"],
            "logfoldchange": [1.0, 2.0],
            "pval_adj": [1e-6, 1e-4],
        }
    )
    adata.uns["de_results"] = de
    adata.uns["metric_column"] = "logfoldchange"
    adata.uns["control_cells_map"] = {"pertA": {"T1": "NT1", "T2": "NT2"}}
    adata.uns["target_conditions_dict"] = {"pertA": ["g1", "g2"]}

    # Use log1p as prediction matrix to satisfy log-normalization check
    preds = np.log1p(X)
    task = PerturbationExpressionPredictionTask(
        condition_key="condition",
        control_name="non-targeting",
    )
    task_input = build_task_input_from_predictions(
        predictions_adata=adata, dataset_adata=adata
    )

    output = task._run_task(preds, task_input)
    assert "pertA" in output.pred_log_fc_dict
    pred_lfc = output.pred_log_fc_dict["pertA"]
    assert pred_lfc.shape == (2,)
    assert np.all(np.isfinite(pred_lfc))
    assert not np.allclose(pred_lfc, 0.0)


def test_requires_control_map_error_when_missing():
    X = np.array([[1.0, 2.0], [2.0, 1.0]])
    obs = pd.DataFrame({"condition": ["pertA", "pertA"]}, index=["T1", "T2"]) 
    var = pd.DataFrame(index=["g1", "g2"]) 
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.uns["de_results"] = pd.DataFrame(
        {"condition": ["pertA"], "gene_id": ["g1"], "logfoldchange": [1.0], "pval_adj": [1e-6]}
    )
    adata.uns["target_conditions_dict"] = {"pertA": ["g1"]}
    preds = np.log1p(X)
    task = PerturbationExpressionPredictionTask(condition_key="condition", control_name="non-targeting")
    ti = build_task_input_from_predictions(adata, adata)
    with pytest.raises(ValueError):
        task._run_task(preds, ti)


def test_missing_condition_mapping_raises():
    X = np.array([[1.0, 2.0], [2.0, 1.0]])
    obs = pd.DataFrame({"condition": ["pertA", "pertA"]}, index=["T1", "T2"]) 
    var = pd.DataFrame(index=["g1", "g2"]) 
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.uns["de_results"] = pd.DataFrame(
        {"condition": ["pertA"], "gene_id": ["g1"], "logfoldchange": [1.0], "pval_adj": [1e-6]}
    )
    adata.uns["control_cells_map"] = {"pertB": {"T1": "NT1"}}
    adata.uns["target_conditions_dict"] = {"pertA": ["g1"]}
    preds = np.log1p(X)
    task = PerturbationExpressionPredictionTask(condition_key="condition", control_name="non-targeting")
    ti = build_task_input_from_predictions(adata, adata)
    with pytest.raises(ValueError):
        task._run_task(preds, ti)


def test_invalid_mapping_type_raises():
    X = np.array([[1.0]])
    obs = pd.DataFrame({"condition": ["pertA"]}, index=["T1"]) 
    var = pd.DataFrame(index=["g1"]) 
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.uns["de_results"] = pd.DataFrame(
        {"condition": ["pertA"], "gene_id": ["g1"], "logfoldchange": [0.0], "pval_adj": [1e-6]}
    )
    adata.uns["control_cells_map"] = {"pertA": ["NT1"]}
    adata.uns["target_conditions_dict"] = {"pertA": ["g1"]}
    preds = np.log1p(X)
    task = PerturbationExpressionPredictionTask(condition_key="condition", control_name="non-targeting")
    ti = build_task_input_from_predictions(adata, adata)
    with pytest.raises(ValueError):
        task._run_task(preds, ti)


def test_custom_indices_subset_and_order():
    X = np.array(
        [
            [1.0, 2.0],  # NT1
            [2.0, 1.0],  # NT2
            [3.0, 4.0],  # T1
            [5.0, 3.0],  # T2
        ]
    )
    obs = pd.DataFrame({"condition": ["non-targeting", "non-targeting", "pertA", "pertA"]}, index=["NT1", "NT2", "T1", "T2"]) 
    var = pd.DataFrame(index=["g1", "g2"]) 
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.uns["de_results"] = pd.DataFrame(
        {"condition": ["pertA"], "gene_id": ["g1"], "logfoldchange": [1.0], "pval_adj": [1e-6]}
    )
    adata.uns["control_cells_map"] = {"pertA": {"T1": "NT1", "T2": "NT2"}}
    adata.uns["target_conditions_dict"] = {"pertA": ["g1"]}
    preds_adata = ad.AnnData(X=np.log1p(X)[:, [0]], obs=obs.iloc[::-1].copy(), var=pd.DataFrame(index=["g1"]))
    task = PerturbationExpressionPredictionTask(condition_key="condition", control_name="non-targeting")
    ti = build_task_input_from_predictions(preds_adata, adata)
    output = task._run_task(preds_adata.X, ti)
    assert "pertA" in output.pred_log_fc_dict
    assert output.pred_log_fc_dict["pertA"].shape == (1,)


def test_ignores_pairs_not_in_index():
    X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]])
    obs = pd.DataFrame({"condition": ["non-targeting", "pertA", "pertA"]}, index=["NT1", "T1", "T2"]) 
    var = pd.DataFrame(index=["g1", "g2"]) 
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.uns["de_results"] = pd.DataFrame(
        {"condition": ["pertA"], "gene_id": ["g1"], "logfoldchange": [1.0], "pval_adj": [1e-6]}
    )
    adata.uns["control_cells_map"] = {"pertA": {"T1": "NT_MISSING", "T2": "NT1"}}
    adata.uns["target_conditions_dict"] = {"pertA": ["g1"]}
    preds = np.log1p(X)
    task = PerturbationExpressionPredictionTask(condition_key="condition", control_name="non-targeting")
    ti = build_task_input_from_predictions(adata, adata)
    out = task._run_task(preds, ti)
    assert "pertA" in out.pred_log_fc_dict


def test_non_lognorm_predictions_are_supported():
    X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [5.0, 3.0]])
    obs = pd.DataFrame({"condition": ["non-targeting", "non-targeting", "pertA", "pertA"]}, index=["NT1", "NT2", "T1", "T2"]) 
    var = pd.DataFrame(index=["g1", "g2"]) 
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.uns["de_results"] = pd.DataFrame(
        {"condition": ["pertA", "pertA"], "gene_id": ["g1", "g2"], "logfoldchange": [1.0, 2.0], "pval_adj": [1e-6, 1e-5]}
    )
    adata.uns["control_cells_map"] = {"pertA": {"T1": "NT1", "T2": "NT2"}}
    adata.uns["target_conditions_dict"] = {"pertA": ["g1", "g2"]}
    preds = X  # raw/non-log-normalized predictions supported
    task = PerturbationExpressionPredictionTask(condition_key="condition", control_name="non-targeting")
    ti = build_task_input_from_predictions(adata, adata)
    out = task._run_task(preds, ti)
    assert "pertA" in out.pred_log_fc_dict and "pertA" in out.true_log_fc_dict
    assert out.pred_log_fc_dict["pertA"].shape == out.true_log_fc_dict["pertA"].shape
    assert np.all(np.isfinite(out.pred_log_fc_dict["pertA"]))


def test_de_results_missing_columns_raises():
    X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]])
    obs = pd.DataFrame({"condition": ["non-targeting", "pertA", "pertA"]}, index=["NT1", "T1", "T2"]) 
    var = pd.DataFrame(index=["g1", "g2"]) 
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.uns["de_results"] = pd.DataFrame({"condition": ["pertA"], "gene_id": ["g1"]})
    adata.uns["metric_column"] = "logfoldchange"
    adata.uns["control_cells_map"] = {"pertA": {"T1": "NT1"}}
    adata.uns["target_conditions_dict"] = {"pertA": ["g1"]}
    preds = np.log1p(X)
    task = PerturbationExpressionPredictionTask(condition_key="condition", control_name="non-targeting")
    ti = build_task_input_from_predictions(adata, adata)
    with pytest.raises(ValueError):
        task._run_task(preds, ti)


def test_nan_metric_skips_condition():
    X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [5.0, 3.0]])
    obs = pd.DataFrame({"condition": ["non-targeting", "non-targeting", "pertA", "pertB"]}, index=["NT1", "NT2", "T1", "T2"]) 
    var = pd.DataFrame(index=["g1", "g2"]) 
    adata = ad.AnnData(X=X, obs=obs, var=var)
    de = pd.DataFrame(
        {
            "condition": ["pertA", "pertB"],
            "gene_id": ["g1", "g2"],
            "logfoldchange": [np.nan, 1.0],
            "pval_adj": [1e-6, 1e-6],
        }
    )
    adata.uns["de_results"] = de
    adata.uns["control_cells_map"] = {"pertA": {"T1": "NT1"}, "pertB": {"T2": "NT2"}}
    adata.uns["target_conditions_dict"] = {"pertA": ["g1"], "pertB": ["g2"]}
    preds = np.log1p(X)
    task = PerturbationExpressionPredictionTask(condition_key="condition", control_name="non-targeting")
    ti = build_task_input_from_predictions(adata, adata)
    out = task._run_task(preds, ti)
    assert "pertA" not in out.pred_log_fc_dict
    assert "pertB" in out.pred_log_fc_dict


def test_multiple_conditions_pair_mapping():
    X = np.array(
        [
            [1.0, 2.0],  # NT1
            [2.0, 1.0],  # NT2
            [1.1, 1.9],  # NT3
            [3.0, 4.0],  # T1 (A)
            [5.0, 3.0],  # T2 (A)
            [2.5, 2.5],  # T3 (B)
        ]
    )
    obs = pd.DataFrame({"condition": ["non-targeting", "non-targeting", "non-targeting", "pertA", "pertA", "pertB"]}, index=["NT1", "NT2", "NT3", "T1", "T2", "T3"]) 
    var = pd.DataFrame(index=["g1", "g2"]) 
    adata = ad.AnnData(X=X, obs=obs, var=var)
    de = pd.DataFrame(
        {
            "condition": ["pertA", "pertA", "pertB"],
            "gene_id": ["g1", "g2", "g1"],
            "logfoldchange": [1.0, 2.0, 0.5],
            "pval_adj": [1e-6, 1e-6, 1e-6],
        }
    )
    adata.uns["de_results"] = de
    adata.uns["control_cells_map"] = {"pertA": {"T1": "NT1", "T2": "NT2"}, "pertB": {"T3": "NT3"}}
    adata.uns["target_conditions_dict"] = {"pertA": ["g1", "g2"], "pertB": ["g1"]}
    preds = np.log1p(X)
    task = PerturbationExpressionPredictionTask(condition_key="condition", control_name="non-targeting")
    ti = build_task_input_from_predictions(adata, adata)
    out = task._run_task(preds, ti)
    assert set(out.pred_log_fc_dict.keys()) == {"pertA", "pertB"}


def test_perturbation_expression_prediction_task_load_from_task_inputs(tmp_path):
    """Test that the task can load inputs from stored task files."""

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
    # Provide strict control map and DE results
    adata.uns["control_cells_map"] = {
        "test1": {
            "cond_test1_a": "ctrl_test1_a",
            "cond_test1_b": "ctrl_test2_b",
        },
        "test2": {
            "cond_test2_a": "ctrl_test1_a",
            "cond_test2_b": "ctrl_test2_b",
        },
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

    # Read h5ad directly and construct task input
    loaded_adata = ad.read_h5ad(task_inputs_dir / "perturbation_task_inputs.h5ad")
    assert isinstance(loaded_adata, ad.AnnData)
    assert "de_results" in loaded_adata.uns
    task_input = PerturbationExpressionPredictionTaskInput(adata=loaded_adata)


def test_perturbation_expression_prediction_task_with_shuffled_input(monkeypatch):
    """Test that the perturbation task works with shuffled AnnData input."""

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
            "test1": {
                "cond_test1_a": "non-targeting_test1_a",
                "cond_test1_b": "non-targeting_test2_b",
            },
            "test2": {
                "cond_test2_a": "non-targeting_test1_a",
                "cond_test2_b": "non-targeting_test2_b",
            },
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
        # Build a strict control map on the base AnnData prior to dataset.load_data()
        obs = adata.obs
        ctrl_barcodes = obs.index[obs["condition"] == "ctrl"].tolist()
        cm = {}
        for c in ["test1", "test2"]:
            treated = obs.index[obs["condition"] == c].tolist()
            cm[c] = {str(t): str(ctrl_barcodes[i % len(ctrl_barcodes)]) for i, t in enumerate(treated)}
        adata.uns["control_cells_map"] = cm
        adata.write_h5ad(file_path)

        # Use the preprocessing pipeline to generate mappings and DE
        from dge_files import preprocess_pipeline as pp

        # Stub matching to produce strict 1-1 mapping using first two ctrl cells
        def _stub_get_matched_controls(**kwargs):
            adata_local = kwargs["adata"]
            cond = kwargs.get("perturbation")
            treated = adata_local.obs.index[adata_local.obs["condition"] == cond].tolist()
            ctrls = adata_local.obs.index[adata_local.obs["condition"] == "ctrl"].tolist()
            if len(treated) == 0 or len(ctrls) == 0:
                return {}
            mapping = {str(t): str(ctrls[i % len(ctrls)]) for i, t in enumerate(treated)}
            return mapping

        monkeypatch.setattr(
            "dge_files.preprocess_pipeline.get_matched_controls",
            _stub_get_matched_controls,
        )

        # Stub DE to return simple significance across all genes per condition
        def _stub_run_multicondition_dge_analysis(**kwargs):
            adata_local = kwargs["adata"]
            gene_names_local = adata_local.var.index.tolist()
            conditions_local = [c for c in adata_local.obs["condition"].unique() if str(c) != "ctrl"]
            rows = []
            for c in conditions_local:
                for g in gene_names_local:
                    rows.append({
                        "condition": c,
                        "gene_id": g,
                        "logfoldchange": 1.0,
                        "pval_adj": 1e-6,
                    })
            return pd.DataFrame(rows), None

        monkeypatch.setattr(
            "dge_files.preprocess_pipeline.run_multicondition_dge_analysis",
            _stub_run_multicondition_dge_analysis,
        )

        processed = pp.run_pipeline(
            str(file_path),
            condition_col="condition",
            control_name="ctrl",
            filter_min_cells=0,
            filter_min_genes=0,
            min_pert_cells=1,
        )

        # Write processed adata and construct dataset from it
        processed_path = Path(tmp_dir) / "dummy_perturbation_processed.h5ad"
        processed.write_h5ad(processed_path)

        dataset = SingleCellPerturbationDataset(
            path=processed_path,
            organism=Organism.HUMAN,
            condition_key="condition",
            control_name="ctrl",
            percent_genes_to_mask=1.0,
            min_de_genes_to_mask=1,
            pval_threshold=1.0,
            min_logfoldchange=0.0,
        )
        dataset.load_data()

        # Create task input with original ordering
        original_task_input = PerturbationExpressionPredictionTaskInput(
            adata=dataset.adata,
            gene_index=dataset.adata.var.index,
            cell_index=dataset.adata.obs.index,
        )

        # Create identical model output for both (using original dimensions)
        model_output = np.random.rand(dataset.adata.shape[0], dataset.adata.shape[1])
        # Shuffle obs and var with fixed random seed for reproducibility
        np.random.seed(42)
        obs_shuffled = dataset.adata.obs.sample(frac=1, random_state=42)
        var_shuffled = dataset.adata.var.sample(frac=1, random_state=42)

        # Get the indices for shuffling
        obs_order = [dataset.adata.obs.index.get_loc(i) for i in obs_shuffled.index]
        var_order = [dataset.adata.var.index.get_loc(i) for i in var_shuffled.index]

        # For shuffled input, we need to reorder the model output to match
        model_output_shuffled = model_output[np.ix_(obs_order, var_order)]

        # Initialize task
        task = PerturbationExpressionPredictionTask()

        # Run task with both inputs (use matching indices for shuffled case)
        original_result = task._run_task(model_output, original_task_input)
        shuffled_task_input = PerturbationExpressionPredictionTaskInput(
            adata=dataset.adata,
            gene_index=dataset.adata.var.index[var_order],
            cell_index=dataset.adata.obs.index[obs_order],
        )
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
    """Ensure results are invariant to shuffled model ordering when indices are provided."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = Path(tmp_dir) / "dummy_perturbation.h5ad"
        adata = create_dummy_anndata(
            n_cells=6,
            n_genes=4,
            obs_columns=["condition"],
            organism=Organism.HUMAN,
        )
        adata.obs["condition"] = ["ctrl", "ctrl", "test1", "test1", "test2", "test2"]
        adata.write_h5ad(file_path)

        dataset = SingleCellPerturbationDataset(
            path=file_path,
            organism=Organism.HUMAN,
            condition_key="condition",
            control_name="ctrl",
            percent_genes_to_mask=1.0,
            min_de_genes_to_mask=1,
            pval_threshold=1.0,
            min_logfoldchange=0.0,
        )
        # Provide minimal DE results to uns before loading
        # Use all genes for both conditions
        gene_names = adata.var.index.tolist()
        de_rows = []
        for condition in ["test1", "test2"]:
            for g in gene_names:
                de_rows.append({"condition": condition, "gene_id": g, "logfoldchange": 1.0, "pval_adj": 1e-6})
        adata.uns["de_results_wilcoxon"] = pd.DataFrame(de_rows)
        # Build and persist a strict control map on the base AnnData prior to dataset.load_data()
        obs_base = adata.obs
        ctrl_barcodes_base = obs_base.index[obs_base["condition"] == "ctrl"].tolist()
        cm_base = {}
        for c in ["test1", "test2"]:
            treated_base = obs_base.index[obs_base["condition"] == c].tolist()
            cm_base[c] = {str(t): str(ctrl_barcodes_base[i % len(ctrl_barcodes_base)]) for i, t in enumerate(treated_base)}
        adata.uns["control_cells_map"] = cm_base
        adata.write_h5ad(file_path)

        dataset.load_data()

        task_input = PerturbationExpressionPredictionTaskInput(
            adata=dataset.adata,
            gene_index=dataset.adata.var.index,
            cell_index=dataset.adata.obs.index,
        )

        model_output = np.random.rand(dataset.adata.shape[0], dataset.adata.shape[1])
        # Shuffle
        np.random.seed(0)
        obs_order = np.random.permutation(dataset.adata.n_obs)
        var_order = np.random.permutation(dataset.adata.n_vars)
        shuffled = model_output[np.ix_(obs_order, var_order)]

        task = PerturbationExpressionPredictionTask(control_name="ctrl")
        out_orig = task._run_task(model_output, task_input)
        # supply indices to align to shuffled
        shuffled_input = PerturbationExpressionPredictionTaskInput(
            adata=dataset.adata,
            gene_index=dataset.adata.var.index[var_order],
            cell_index=dataset.adata.obs.index[obs_order],
        )
        out_shuf = task._run_task(shuffled, shuffled_input)
        assert set(out_orig.pred_log_fc_dict.keys()) == set(out_shuf.pred_log_fc_dict.keys())
        assert set(out_orig.true_log_fc_dict.keys()) == set(out_shuf.true_log_fc_dict.keys())
        for k in out_orig.pred_log_fc_dict.keys():
            np.testing.assert_allclose(out_orig.pred_log_fc_dict[k], out_shuf.pred_log_fc_dict[k])
            np.testing.assert_allclose(out_orig.true_log_fc_dict[k], out_shuf.true_log_fc_dict[k])

