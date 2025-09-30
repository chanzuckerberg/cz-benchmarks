import pytest
import pandas as pd
import numpy as np
import anndata as ad
import tempfile
from pathlib import Path
from czbenchmarks.tasks.single_cell import (
    PerturbationExpressionPredictionTask,
    PerturbationExpressionPredictionTaskInput,
)
from czbenchmarks.tasks.single_cell.perturbation_expression_prediction import (
    build_task_input_from_predictions,
)
from czbenchmarks.datasets.types import Organism
from czbenchmarks.datasets.single_cell_perturbation import (
    SingleCellPerturbationDataset,
)
from czbenchmarks.metrics.types import MetricType
from tests.utils import create_dummy_perturbation_anndata, create_dummy_anndata


# Test-specific fixtures


@pytest.fixture
def wilcoxon_test_data():
    """Create deterministic test data for Wilcoxon testing."""
    gene_names = ["G0", "G1", "G2", "G3"]
    true_lfc_gene_A = np.array([1.0, 0.5, -0.5, -1.0])
    true_lfc_gene_B = np.array([2.0, 1.0, -1.0, -2.0])

    # Create DE results for both conditions
    conditions_data = [
        ("gene_A", true_lfc_gene_A, "ENSG_A"),
        ("gene_B", true_lfc_gene_B, "ENSG_B"),
    ]

    de_results = pd.concat(
        [
            pd.DataFrame(
                {
                    "logfoldchange": lfc_values,
                    "target_gene": [condition] * len(gene_names),
                    "names": gene_names,
                    "pval": [0.001] * len(gene_names),
                    "pval_adj": [0.001] * len(gene_names),
                    "condition": [condition] * len(gene_names),
                    "condition_ensembl_id": [ensembl_id] * len(gene_names),
                }
            )
            for condition, lfc_values, ensembl_id in conditions_data
        ],
        ignore_index=True,
    )

    return {
        "de_results": de_results,
        "true_lfc_gene_A": true_lfc_gene_A,
        "true_lfc_gene_B": true_lfc_gene_B,
        "gene_names": gene_names,
    }


@pytest.fixture
def simple_toy_data():
    """Create simple toy dataset for pair mapping tests."""
    X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [5.0, 3.0]])
    obs = pd.DataFrame(
        {"condition": ["ctrl", "ctrl", "pertA", "pertA"]},
        index=["NT1", "NT2", "T1", "T2"],
    )
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=["g1", "g2"]))

    de_results = pd.DataFrame(
        {
            "condition": ["pertA", "pertA"],
            "gene_id": ["g1", "g2"],
            "logfoldchange": [1.0, 2.0],
            "pval_adj": [1e-6, 1e-4],
        }
    )

    return {"adata": adata, "de_results": de_results, "predictions": np.log1p(X)}


def test_perturbation_task_executes_without_errors(
    adata_uns_setup, assert_metric_results
):
    """Test that PerturbationExpressionPredictionTask executes without errors."""
    perturbation_data = create_dummy_perturbation_anndata(
        n_cells=500, n_genes=200, organism=Organism.HUMAN, condition_column="condition"
    )
    gene_pert = perturbation_data["gene_pert"]
    adata = perturbation_data["adata"]
    var_names = adata.var_names
    cell_representation = np.log1p(adata.X.toarray())

    # Create DE results and setup data
    de_results = pd.DataFrame(
        {
            "condition": [gene_pert] * len(var_names),
            "gene_id": var_names,
            "logfoldchange": np.random.randn(len(var_names)),
            "pval_adj": np.random.uniform(0, 0.01, len(var_names)),
        }
    )

    # Setup AnnData with proper condition naming
    masked_adata_obs = adata.obs.copy()
    control_condition_name = f"ctrl_{gene_pert}"
    masked_adata_obs.loc[masked_adata_obs["condition"] == "ctrl", "condition"] = (
        control_condition_name
    )

    # Create target conditions and control mapping
    unique_conditions = masked_adata_obs["condition"][
        ~masked_adata_obs["condition"].str.startswith("ctrl")
    ].unique()
    target_conditions_dict = {
        cond: list(
            np.random.choice(var_names, min(10, len(var_names) // 2), replace=False)
        )
        for cond in unique_conditions
    }

    treated_idx = masked_adata_obs.index[masked_adata_obs["condition"] == gene_pert]
    control_idx = masked_adata_obs.index[
        masked_adata_obs["condition"] == control_condition_name
    ]
    control_mapping = {
        gene_pert: {
            str(treated_idx[i]): str(control_idx[i % len(control_idx)])
            for i in range(len(treated_idx))
        }
        if len(control_idx) > 0
        else {}
    }

    # Setup test AnnData using helper
    test_adata = ad.AnnData(
        X=adata.X, obs=masked_adata_obs, var=pd.DataFrame(index=var_names)
    )
    adata_uns_setup(test_adata, de_results, target_conditions_dict, control_mapping)

    task_input = PerturbationExpressionPredictionTaskInput(
        adata=test_adata,
        gene_index=test_adata.var.index,
        cell_index=test_adata.obs.index,
    )

    # Test regular execution
    task = PerturbationExpressionPredictionTask()
    results = task.run(cell_representation, task_input)
    assert_metric_results(results, expected_count=1)

    # Test baselines
    for baseline_type in ["mean", "median"]:
        baseline_embedding = task.compute_baseline(cell_representation, baseline_type)
        baseline_results = task.run(baseline_embedding, task_input)
        assert_metric_results(baseline_results, expected_count=1)


def test_perturbation_expression_prediction_task_wilcoxon(
    wilcoxon_test_data, adata_uns_setup, assert_metric_results
):
    """Test Wilcoxon path computes correct vectors and metrics."""
    de_results = wilcoxon_test_data["de_results"]
    true_lfc_gene_A = wilcoxon_test_data["true_lfc_gene_A"]
    true_lfc_gene_B = wilcoxon_test_data["true_lfc_gene_B"]
    gene_names = wilcoxon_test_data["gene_names"]

    # Build AnnData with 4 groups: condition/control for A and B
    n_per_group = 4
    conditions = (
        ["gene_A"] * n_per_group
        + ["ctrl_gene_A"] * n_per_group
        + ["gene_B"] * n_per_group
        + ["ctrl_gene_B"] * n_per_group
    )
    obs_names = [f"cellbarcode{i}_{cond}" for i, cond in enumerate(conditions)]

    # Create cell representation that gives expected log fold changes
    base_value = 1.0
    cell_representation = np.zeros((len(conditions), len(gene_names)), dtype=float)
    cell_representation[0:n_per_group, :] = base_value + true_lfc_gene_A
    cell_representation[n_per_group : 2 * n_per_group, :] = base_value
    cell_representation[2 * n_per_group : 3 * n_per_group, :] = (
        base_value + true_lfc_gene_B
    )
    cell_representation[3 * n_per_group : 4 * n_per_group, :] = base_value
    cell_representation += 0.003  # Ensure validation passes

    # Setup AnnData
    adata = ad.AnnData(
        X=np.zeros_like(cell_representation),
        obs=pd.DataFrame({"condition": conditions}, index=obs_names),
        var=pd.DataFrame(index=gene_names),
    )

    # Add required UNS data using helper
    de_results["gene_id"] = de_results["names"]
    target_conditions_dict = {"gene_A": list(gene_names), "gene_B": list(gene_names)}
    control_cells_map = {
        "gene_A": {
            obs_names[i]: obs_names[i + n_per_group] for i in range(n_per_group)
        },
        "gene_B": {
            obs_names[i + 2 * n_per_group]: obs_names[i + 3 * n_per_group]
            for i in range(n_per_group)
        },
    }
    adata_uns_setup(adata, de_results, target_conditions_dict, control_cells_map)

    task_input = PerturbationExpressionPredictionTaskInput(
        adata=adata, gene_index=adata.var.index, cell_index=adata.obs.index
    )

    # Verify internal task output matches expectations
    task = PerturbationExpressionPredictionTask()
    task_output = task._run_task(cell_representation, task_input)
    assert set(task_output.pred_log_fc_dict.keys()) == {"gene_A", "gene_B"}
    assert set(task_output.true_log_fc_dict.keys()) == {"gene_A", "gene_B"}
    assert np.allclose(task_output.pred_log_fc_dict["gene_A"], true_lfc_gene_A)
    assert np.allclose(task_output.pred_log_fc_dict["gene_B"], true_lfc_gene_B)
    assert np.allclose(task_output.true_log_fc_dict["gene_A"], true_lfc_gene_A)
    assert np.allclose(task_output.true_log_fc_dict["gene_B"], true_lfc_gene_B)

    # Test full task execution
    results = task.run(cell_representation, task_input)
    assert_metric_results(
        results,
        expected_count=2,
        expected_types={MetricType.SPEARMAN_CORRELATION_CALCULATION},
        perfect_correlation=True,
    )


def test_task_uses_strict_pair_mapping_for_pred_lfc(simple_toy_data, adata_uns_setup):
    """Test that task uses strict pair mapping for prediction log fold changes."""
    adata = simple_toy_data["adata"]
    de_results = simple_toy_data["de_results"]
    predictions = simple_toy_data["predictions"]

    # Add required UNS data using helper
    adata_uns_setup(
        adata,
        de_results,
        {"pertA": ["g1", "g2"]},
        {"pertA": {"T1": "NT1", "T2": "NT2"}},
    )

    task_input = build_task_input_from_predictions(
        predictions_adata=adata, dataset_adata=adata
    )
    task = PerturbationExpressionPredictionTask()
    output = task._run_task(predictions, task_input)

    assert "pertA" in output.pred_log_fc_dict
    pred_lfc = output.pred_log_fc_dict["pertA"]
    assert (
        pred_lfc.shape == (2,)
        and np.all(np.isfinite(pred_lfc))
        and not np.allclose(pred_lfc, 0.0)
    )


def test_perturbation_task_apply_model_ordering():
    """Ensure results are invariant to shuffled model ordering when indices are provided."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = Path(tmp_dir) / "dummy_perturbation.h5ad"
        adata = create_dummy_anndata(
            n_cells=6, n_genes=4, obs_columns=["condition"], organism=Organism.HUMAN
        )
        adata.obs["condition"] = ["ctrl", "ctrl", "test1", "test1", "test2", "test2"]

        # Setup DE results and control mapping
        gene_names = adata.var.index.tolist()
        de_rows = [
            {"condition": cond, "gene_id": g, "logfoldchange": 1.0, "pval_adj": 1e-6}
            for cond in ["test1", "test2"]
            for g in gene_names
        ]
        adata.uns["de_results_wilcoxon"] = pd.DataFrame(de_rows)

        ctrl_barcodes = adata.obs.index[adata.obs["condition"] == "ctrl"].tolist()
        control_mapping = {}
        for condition in ["test1", "test2"]:
            treated_cells = adata.obs.index[
                adata.obs["condition"] == condition
            ].tolist()
            control_mapping[condition] = {
                str(t): str(ctrl_barcodes[i % len(ctrl_barcodes)])
                for i, t in enumerate(treated_cells)
            }
        adata.uns["control_cells_map"] = control_mapping

        # Save the AnnData to file before creating the dataset
        adata.write_h5ad(file_path)

        # Load dataset and test with shuffled data
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
        dataset.load_data()
        model_output = np.random.rand(dataset.adata.shape[0], dataset.adata.shape[1])
        task_input = PerturbationExpressionPredictionTaskInput(
            adata=dataset.adata,
            gene_index=dataset.adata.var.index,
            cell_index=dataset.adata.obs.index,
        )
        task = PerturbationExpressionPredictionTask()
        out_orig = task._run_task(model_output, task_input)

        # Test with shuffled data
        obs_order = np.random.permutation(dataset.adata.n_obs)
        var_order = np.random.permutation(dataset.adata.n_vars)
        shuffled = model_output[np.ix_(obs_order, var_order)]

        shuffled_adata = ad.AnnData(
            X=shuffled,
            var=dataset.adata.var.reindex(dataset.adata.var.index[var_order]),
            obs=dataset.adata.obs.reindex(dataset.adata.obs.index[obs_order]),
        )
        shuffled_input = build_task_input_from_predictions(
            shuffled_adata, dataset.adata
        )
        out_shuf = task._run_task(shuffled, shuffled_input)

        # Verify results are identical
        assert set(out_orig.pred_log_fc_dict.keys()) == set(
            out_shuf.pred_log_fc_dict.keys()
        )
        assert out_orig.true_log_fc_dict.keys() == out_shuf.true_log_fc_dict.keys()
        for k in out_orig.pred_log_fc_dict.keys():
            np.testing.assert_allclose(
                out_orig.pred_log_fc_dict[k], out_shuf.pred_log_fc_dict[k]
            )
            np.testing.assert_allclose(
                out_orig.true_log_fc_dict[k], out_shuf.true_log_fc_dict[k]
            )
