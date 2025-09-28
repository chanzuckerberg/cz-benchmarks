import json
import logging
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
import pandas as pd
from scipy import sparse as sp_sparse

from ...constants import RANDOM_SEED
from ...metrics import metrics_registry
from ...metrics.types import MetricResult, MetricType
from ...tasks.types import CellRepresentation
from ..task import Task, TaskInput, TaskOutput
from ..utils import binarize_values

logger = logging.getLogger(__name__)


class PerturbationExpressionPredictionTaskInput(TaskInput):
    """Pydantic model for PerturbationTask inputs."""

    de_results: pd.DataFrame
    masked_adata_obs: pd.DataFrame
    var_index: pd.Index
    target_conditions_to_save: Dict[str, List[str]]
    row_index: pd.Index


def load_perturbation_task_input_from_saved_files(
    task_inputs_dir: Path,
) -> PerturbationExpressionPredictionTaskInput:
    """
    Load task input from files saved by dataset's `store_task_inputs`.

    This creates a PerturbationExpressionPredictionTaskInput from stored files,
    allowing the task to be instantiated without going through the full dataset
    loading process.

    Args:
        task_inputs_dir: Directory containing task inputs.

    Returns:
        PerturbationExpressionPredictionTaskInput: Task input ready for use.
    """

    inputs_dir = Path(task_inputs_dir)

    # Load DE results
    de_results_path = inputs_dir / "de_results.json"
    de_results = pd.read_json(de_results_path)

    # Load target conditions to save
    target_genes_path = inputs_dir / "target_conditions_to_save.json"
    with target_genes_path.open("r") as f:
        target_conditions_to_save = json.load(f)

    # Rebuild AnnData obs and var
    adata_dir = inputs_dir / "control_matched_adata"
    obs = pd.read_json(adata_dir / "obs.json", orient="split")
    var = pd.read_json(adata_dir / "var.json", orient="split")
    row_index = pd.Index(
        np.load(inputs_dir / "original_adata/obs/index.npy", allow_pickle=True)
    )

    return PerturbationExpressionPredictionTaskInput(
        de_results=de_results,
        masked_adata_obs=obs,
        var_index=var.index,
        target_conditions_to_save=target_conditions_to_save,
        row_index=row_index,
    )


class PerturbationExpressionPredictionOutput(TaskOutput):
    """Output for perturbation task."""

    pred_log_fc_dict: Dict[str, np.ndarray]
    true_log_fc_dict: Dict[str, np.ndarray]


class PerturbationExpressionPredictionTask(Task):
    display_name = "Perturbation Expression Prediction"
    description = "Evaluate the quality of predicted changes in expression levels for genes that are differentially expressed under perturbation(s) using multiple classification and correlation metrics."
    input_model = PerturbationExpressionPredictionTaskInput

    def __init__(
        self,
        control_prefix: str = "non-targeting", # FIXME MICHELLE
        *,
        random_seed: int = RANDOM_SEED,
    ):
        """
        Args:
            control_prefix (str): Prefix for control conditions.
            random_seed (int): Random seed for reproducibility.
        """
        super().__init__(random_seed=random_seed)
        self.metric_column = "logfoldchange" # TODO: logfoldchange only for now
        self.control_prefix = control_prefix

    def _run_task(
        self,
        cell_representation: CellRepresentation,
        task_input: PerturbationExpressionPredictionTaskInput,
    ) -> PerturbationExpressionPredictionOutput:
        """
        Runs the perturbation evaluation task.

        This method computes predicted and ground truth log fold changes for each perturbation
        condition in the dataset, using the provided cell representations and differential
        expression results. It aligns predictions and ground truth values for masked genes,
        and prepares data for downstream metric computation.

        Args:
            cell_representation (CellRepresentation): A numpy matrix of shape (n_cells, n_genes)
            task_input (PerturbationExpressionPredictionTaskInput): Input object containing:
                - de_results (pd.DataFrame): DataFrame with differential expression results,
                  including log fold changes/standard mean deviation and gene names.
                - control_cells_ids (Dict[str, np.ndarray]): Dictionary of control cell IDs for each perturbation condition.

        Returns:
            PerturbationExpressionPredictionOutput: Output object containing dictionaries of predicted and true log fold changes
            for each perturbation condition.
        """

        pred_log_fc_dict = {}
        true_log_fc_dict = {}
        de_results = task_input.de_results

        condition_series = task_input.masked_adata_obs["condition"].astype(str)
        condition_list = np.unique(
            condition_series[~condition_series.str.startswith(self.control_prefix)]
        )
        row_index = task_input.row_index.str.split("_").str[0]

        for condition in condition_list:
            condition_de_df = de_results[de_results["condition"] == condition]

            masked_genes = np.array(
                task_input.target_conditions_to_save[
                    task_input.masked_adata_obs.index[
                        task_input.masked_adata_obs["condition"] == condition
                    ][0]
                ]
            )
            # Filter masked_genes to only those present in var.index
            masked_genes = np.array(
                [g for g in masked_genes if g in task_input.var_index]
            )

            if len(masked_genes) == 0:
                print("Skipping condition because it has no masked genes.")
                continue
            true_log_fc = (
                condition_de_df.set_index("gene_id")
                .reindex(masked_genes)[self.metric_column]
                .values
            )
            valid = ~np.isnan(true_log_fc)
            masked_genes = masked_genes[valid]
            true_log_fc = true_log_fc[valid]
            col_indices = task_input.var_index.get_indexer(masked_genes)
            condition_adata = task_input.masked_adata_obs[
                task_input.masked_adata_obs["condition"] == condition
            ].index
            condition_col_ids = condition_adata.to_series().str.split("_").str[0]
            condition_idx = np.where(row_index.isin(condition_col_ids))[0]
            control_adata = task_input.masked_adata_obs[
                task_input.masked_adata_obs["condition"]
                == f"{self.control_prefix}_{condition}"
            ].index
            control_col_ids = control_adata.to_series().str.split("_").str[0]

            control_idx = np.where(row_index.isin(control_col_ids))[0]
            condition_vals = cell_representation[np.ix_(condition_idx, col_indices)]
            control_vals = cell_representation[np.ix_(control_idx, col_indices)]
            ctrl_mean = np.mean(control_vals, axis=0)
            cond_mean = np.mean(condition_vals, axis=0)
            pred_log_fc = cond_mean - ctrl_mean
            pred_log_fc_dict[condition] = pred_log_fc
            true_log_fc_dict[condition] = true_log_fc
        return PerturbationExpressionPredictionOutput(
            pred_log_fc_dict=pred_log_fc_dict,
            true_log_fc_dict=true_log_fc_dict,
        )

    def _compute_metrics(
        self,
        task_input: PerturbationExpressionPredictionTaskInput,
        task_output: PerturbationExpressionPredictionOutput,
    ) -> List[MetricResult]:
        """
        Computes perturbation prediction quality metrics for cell line perturbation predictions.

        This method evaluates the quality of gene perturbation predictions by comparing predicted
        and true log fold changes across different perturbation conditions. For each condition,
        it computes multiple classification and correlation metrics.

        For each perturbation condition, computes:
        - **Accuracy**: Classification accuracy between binarized predicted and true log fold changes
        - **Precision**: Precision score for binarized predictions (positive predictions that are correct)
        - **Recall**: Recall score for binarized predictions (true positives that are detected)
        - **F1 Score**: Harmonic mean of precision and recall for binarized predictions
        - **Spearman Correlation**: Rank correlation between raw predicted and true log fold changes

        The binarization process converts continuous log fold change values to binary classifications
        (up-regulated vs. not up-regulated) using the `binarize_values` function.

        Args:
            task_input (PerturbationExpressionPredictionTaskInput): Input object containing differential expression
                results and prediction data from the perturbation experiment.
            task_output (PerturbationExpressionPredictionOutput): Output object containing aligned predicted and
                true log fold changes for each perturbation condition.

        Returns:
            List[MetricResult]: A flat list of MetricResult objects, where each result contains
                the metric type, value, and the corresponding perturbation condition in its params.
                Each metric (accuracy, precision, recall, F1 score, and Spearman correlation) is
                computed for every condition and appended to the list.

        Note:
            Each MetricResult includes the condition name in its params for identification.
        """
        accuracy_metric = MetricType.ACCURACY_CALCULATION
        precision_metric = MetricType.PRECISION_CALCULATION
        recall_metric = MetricType.RECALL_CALCULATION
        f1_metric = MetricType.F1_CALCULATION
        spearman_correlation_metric = MetricType.SPEARMAN_CORRELATION_CALCULATION

        metric_results = []
        for condition in task_output.pred_log_fc_dict.keys():
            pred_log_fc = task_output.pred_log_fc_dict[condition]
            true_log_fc = task_output.true_log_fc_dict[condition]
            true_binary, pred_binary = binarize_values(true_log_fc, pred_log_fc)

            # Compute precision, recall, F1, and Spearman correlation for each condition
            precision_value = metrics_registry.compute(
                precision_metric,
                y_true=true_binary,
                y_pred=pred_binary,
            )
            metric_results.append(
                MetricResult(
                    metric_type=precision_metric,
                    value=precision_value,
                    params={"condition": condition},
                )
            )

            recall_value = metrics_registry.compute(
                recall_metric,
                y_true=true_binary,
                y_pred=pred_binary,
            )
            metric_results.append(
                MetricResult(
                    metric_type=recall_metric,
                    value=recall_value,
                    params={"condition": condition},
                )
            )

            f1_value = metrics_registry.compute(
                f1_metric,
                y_true=true_binary,
                y_pred=pred_binary,
            )
            metric_results.append(
                MetricResult(
                    metric_type=f1_metric,
                    value=f1_value,
                    params={"condition": condition},
                )
            )

            # Compute Spearman correlation and accuracy for each condition
            spearman_corr = metrics_registry.compute(
                spearman_correlation_metric,
                a=true_log_fc,
                b=pred_log_fc,
            )
            # If the result has a 'correlation' attribute (e.g., scipy.stats result), use it; otherwise, use the value directly
            spearman_corr_value = getattr(spearman_corr, "correlation", spearman_corr)
            metric_results.append(
                MetricResult(
                    metric_type=spearman_correlation_metric,
                    value=spearman_corr_value,
                    params={"condition": condition},
                )
            )

            accuracy_value = metrics_registry.compute(
                accuracy_metric,
                y_true=true_binary,
                y_pred=pred_binary,
            )
            metric_results.append(
                MetricResult(
                    metric_type=accuracy_metric,
                    value=accuracy_value,
                    params={"condition": condition},
                )
            )
        return metric_results

    @staticmethod
    def compute_baseline(
        cell_representation: CellRepresentation,
        baseline_type: Literal["median", "mean"] = "median",
    ) -> CellRepresentation:
        """Set a baseline perturbation prediction using mean or median expression.

        This method creates a baseline prediction by either taking the mean or
        the median of the control cells' gene expression. This baseline
        represents a simple no-change prediction.

        Args:
            cell_representation: The gene expression matrix of control cells.
            baseline_type: The type of baseline to use, either "mean" or "median".

        Returns:
            A DataFrame representing the baseline perturbation prediction.
        """
        # Create baseline prediction by replicating the aggregated expression values
        # across all cells in the dataset.
        baseline_func = np.median if baseline_type == "median" else np.mean
        if baseline_type == "median" and sp_sparse.issparse(cell_representation):
            cell_representation = cell_representation.toarray()

        perturb_baseline_pred = np.tile(
            baseline_func(cell_representation, axis=0),
            (cell_representation.shape[0], 1),
        )

        # Store the baseline prediction in the dataset for evaluation
        return perturb_baseline_pred
