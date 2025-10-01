import pytest
import pandas as pd
import numpy as np
import anndata as ad
from czbenchmarks.tasks.single_cell import (
    PerturbationExpressionPredictionTask,
    PerturbationExpressionPredictionTaskInput,
)

from czbenchmarks.metrics.types import MetricType


@pytest.fixture
def wilcoxon_test_data():
    """Create deterministic test data for Wilcoxon testing.

    Builds the AnnData, prediction matrix, DE results, and required UNS in one
    place.
    """
    condition_key = "condition"
    control_name = "ctrl"
    de_gene_col = "gene_id"
    config = {
        "condition_key": condition_key,
        "control_name": control_name,
        "de_gene_col": de_gene_col,
    }
    metric_column = "logfoldchange"

    n_per_group = 4
    gene_names = ["G0", "G1", "G2", "G3"]
    true_mean_change_gene_A = np.array([1.0, 0.5, -0.5, -1.0])
    true_mean_change_gene_B = np.array([2.0, 1.0, -1.0, -2.0])

    de_results = pd.DataFrame(
        {
            condition_key: ["gene_A"] * 4 + ["gene_B"] * 4,
            de_gene_col: gene_names * 2,
            metric_column: np.concatenate(
                [true_mean_change_gene_A, true_mean_change_gene_B]
            ),
        }
    )

    # Build AnnData with 4 groups: condition/control for A and B
    conditions = (
        ["gene_A"] * n_per_group
        + [f"{control_name}"] * n_per_group
        + ["gene_B"] * n_per_group
        + [f"{control_name}"] * n_per_group
    )
    obs_names = [f"cellbarcode{i}_{cond}" for i, cond in enumerate(conditions)]

    # Create cell representation that gives expected log fold changes
    eps = 0.003
    cell_representation = np.ones((len(conditions), len(gene_names)), dtype=float) + eps
    cell_representation[0:n_per_group, :] += true_mean_change_gene_A[None, :]
    cell_representation[2 * n_per_group : 3 * n_per_group, :] += (
        true_mean_change_gene_B[None, :]
    )

    # Setup AnnData
    adata = ad.AnnData(
        X=np.zeros_like(cell_representation),
        obs=pd.DataFrame({condition_key: conditions}, index=obs_names),
        var=pd.DataFrame(index=gene_names),
    )

    # Add required UNS data using helper
    target_conditions_dict = {
        "gene_A": list(gene_names),
        "gene_B": list(gene_names),
    }
    control_cells_map = {
        "gene_A": {
            obs_names[i]: obs_names[i + n_per_group] for i in range(n_per_group)
        },
        "gene_B": {
            obs_names[i + 2 * n_per_group]: obs_names[i + 3 * n_per_group]
            for i in range(n_per_group)
        },
    }
    adata.uns.update(
        {
            "control_cells_ids": {},
            "de_results": de_results,
            "metric_column": metric_column,
            "target_conditions_dict": target_conditions_dict,
            "control_cells_map": control_cells_map,
            "config": config,
        }
    )

    return {
        "adata": adata,
        "cell_representation": cell_representation,
        "de_results": de_results,
        "true_mean_change_gene_A": true_mean_change_gene_A,
        "true_mean_change_gene_B": true_mean_change_gene_B,
        "gene_names": gene_names,
    }


@pytest.mark.parametrize("test_custom_ordering", [True, False])
def test_perturbation_expression_prediction_task_wilcoxon(
    wilcoxon_test_data,
    assert_metric_results,
    test_custom_ordering: bool,
    random_seed: int = 42,
):
    """Test Wilcoxon path computes correct vectors and metrics."""
    adata = wilcoxon_test_data["adata"]
    cell_representation = wilcoxon_test_data["cell_representation"]

    if test_custom_ordering:
        # Shuffle rows (cells) and columns (genes) independently
        rng = np.random.RandomState(random_seed)
        cell_shuffle_idx = rng.permutation(len(adata.obs))
        gene_shuffle_idx = rng.permutation(len(adata.var))

        # Shuffle data
        task_input_adata = adata[np.ix_(cell_shuffle_idx, gene_shuffle_idx)].copy()
        task_input_cell_index = adata.obs.index[cell_shuffle_idx]
        task_input_gene_index = adata.var.index[gene_shuffle_idx]
        matrix = cell_representation[np.ix_(cell_shuffle_idx, gene_shuffle_idx)]
    else:
        task_input_adata = adata
        task_input_cell_index = adata.obs.index
        task_input_gene_index = adata.var.index
        matrix = cell_representation

    task_input = PerturbationExpressionPredictionTaskInput(
        adata=task_input_adata,
        gene_index=task_input_gene_index,
        cell_index=task_input_cell_index,
        pred_effect_operation="difference",
    )

    # Verify internal task output matches expectations
    condition_key = adata.uns["config"].get("condition_key", "condition")
    task = PerturbationExpressionPredictionTask()

    # Validate metrics results
    results = task.run(cell_representation=matrix, task_input=task_input)

    assert_metric_results(
        results,
        expected_count=2,
        expected_types={MetricType.SPEARMAN_CORRELATION_CALCULATION},
        perfect_correlation=True,
        expected_conditions={"gene_A", "gene_B"},
        condition_key=condition_key,
    )

    with pytest.raises(NotImplementedError):
        task.compute_baseline(cell_representation=adata.X)