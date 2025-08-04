from typing import List, Dict
import pandas as pd
import numpy as np
import logging
from ..task import Task, TaskInput, TaskOutput
from ...tasks.types import CellRepresentation
from ...metrics import metrics_registry
from ...metrics.types import MetricResult, MetricType
from ...constants import RANDOM_SEED

from tqdm import tqdm

logger = logging.getLogger(__name__)


def binarize_values(y_true, y_pred):
    ids = np.where(~np.isnan(y_true) & ~np.isinf(y_true))[0]
    y_true = y_true[ids]
    y_pred = y_pred[ids]
    pred_binary = (y_pred > 0).astype(int)
    true_binary = (y_true > 0).astype(int)
    return true_binary, pred_binary


class K562PerturbationTaskInput(TaskInput):
    """Pydantic model for PerturbationTask inputs."""

    de_res_wilcoxon_df: pd.DataFrame
    pred_df: pd.DataFrame


class K562PerturbationOutput(TaskOutput):
    """Output for perturbation task."""

    pred_log_fc_dict: Dict[str, np.ndarray]
    true_log_fc_dict: Dict[str, np.ndarray]


class K562PerturbationTask(Task):
    display_name = "k562_perturbation"

    def __init__(self, min_de_genes: int = 5, *, random_seed: int = RANDOM_SEED):
        super().__init__(random_seed=random_seed)
        self.min_de_genes = min_de_genes

    def _run_task(
        self,
        cell_representation: CellRepresentation,
        task_input: K562PerturbationTaskInput,
    ) -> K562PerturbationOutput:
        """
        Runs the perturbation evaluation task.

        This method computes predicted and ground truth log fold changes for each perturbation
        condition in the dataset, using the provided cell representations and differential
        expression results. It aligns predictions and ground truth values for masked genes,
        and prepares data for downstream metric computation.

        Args:
            cell_representation (CellRepresentation): AnnData or DataFrame containing cell-level information,
                including gene assignments and cell indices.
            task_input (K562PerturbationTaskInput): Input object containing:
                - de_res_wilcoxon_df (pd.DataFrame): DataFrame with differential expression results,
                  including log fold changes and gene names.
                - pred_df (pd.DataFrame): DataFrame with model predictions, sample IDs, and target genes.

        Returns:
            K562PerturbationOutput: Output object containing dictionaries of predicted and true log fold changes
            for each perturbation condition.
        """
        genes_dict = {}
        predictions_dict = {}
        pred_log_fc_dict = {}
        true_log_fc_dict = {}

        for sample, group in tqdm(task_input.pred_df.groupby("sample_id")):
            predictions_dict.setdefault(sample, []).append(
                np.concatenate(group["pred"].to_numpy())
            )
            if sample not in genes_dict:
                genes_dict[sample] = group["target_genes"].to_numpy()

        for condition in cell_representation["gene"].unique():
            condition_de_df = task_input.de_res_wilcoxon_df[
                task_input.de_res_wilcoxon_df["target_gene"] == condition
            ]
            if len(condition_de_df) < self.min_de_genes:
                continue

            cell_condition = cell_representation[
                cell_representation.index.str.endswith(f"_{condition}")
            ]
            condition_cells = cell_condition[
                cell_condition["gene"] != "non-targeting"
            ].index
            control_cells = cell_condition[
                cell_condition["gene"] == "non-targeting"
            ].index

            if len(control_cells) != len(condition_cells):
                raise AssertionError(
                    f"Number of control cells ({len(control_cells)}) is not equal to number of condition cells ({len(condition_cells)})"
                )
            if len(control_cells) < 10:
                print(f"Less than 10 cells in condition {condition}. Skipping...")
                continue

            try:
                # Masked genes for this sample
                masked_genes = genes_dict[control_cells[0]]
                mask = masked_genes != "A"
                masked_genes = masked_genes[mask]
                n_masked_genes = len(masked_genes)
                # Get predictions for control and condition cells
                control_predictions = np.asarray(
                    [
                        predictions_dict[cell][-1][:n_masked_genes]
                        for cell in control_cells
                    ]
                )
                condition_predictions = np.asarray(
                    [
                        predictions_dict[cell][-1][:n_masked_genes]
                        for cell in condition_cells
                    ]
                )

                # Compute predicted log fold change
                pred_log_fc = condition_predictions.mean(
                    axis=0
                ) - control_predictions.mean(axis=0)

                # Select column for true log fold change
                col = "logfoldchanges"
                # Align true log fold change to masked genes
                true_log_fc = (
                    condition_de_df.set_index("names").reindex(masked_genes)[col].values
                )

                # Remove NaNs
                valid = ~np.isnan(true_log_fc)
                pred_log_fc = pred_log_fc[valid]
                true_log_fc = true_log_fc[valid]

                pred_log_fc_dict[condition] = pred_log_fc
                true_log_fc_dict[condition] = true_log_fc

            except Exception:
                raise

        return K562PerturbationOutput(
            pred_log_fc_dict=pred_log_fc_dict,
            true_log_fc_dict=true_log_fc_dict,
        )

    def _compute_metrics(
        self,
        task_input: K562PerturbationTaskInput,
        task_output: K562PerturbationOutput,
    ) -> List[List[MetricResult]]:
        """
        Computes perturbation prediction quality metrics for K562 cell line perturbation predictions.

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
            task_input (K562PerturbationTaskInput): Input object containing differential expression
                results and prediction data from the perturbation experiment.
            task_output (K562PerturbationOutput): Output object containing aligned predicted and
                true log fold changes for each perturbation condition.

        Returns:
            List[List[MetricResult]]: A list of 5 lists, where each inner list contains MetricResult
                objects for one metric type across all conditions:
                - [0]: Accuracy results for all conditions
                - [1]: Precision results for all conditions
                - [2]: Recall results for all conditions
                - [3]: F1 score results for all conditions
                - [4]: Spearman correlation results for all conditions

        Note:
            Each MetricResult includes the condition name in its params for identification.
        """
        accuracy_metric = MetricType.ACCURACY
        precision_metric = MetricType.PRECISION
        recall_metric = MetricType.RECALL
        f1_metric = MetricType.F1
        spearman_correlation_metric = MetricType.SPEARMAN_CORRELATION

        precision_results = []
        recall_results = []
        accuracy_results = []
        f1_results = []
        spearman_correlation_results = []

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
            precision_results.append(
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
            recall_results.append(
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
            f1_results.append(
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
            spearman_correlation_results.append(
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
            accuracy_results.append(
                MetricResult(
                    metric_type=accuracy_metric,
                    value=accuracy_value,
                    params={"condition": condition},
                )
            )

        return [
            accuracy_results,
            precision_results,
            recall_results,
            f1_results,
            spearman_correlation_results,
        ]
