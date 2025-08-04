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

            except Exception as e:
                print(f"Error processing condition {condition}: {e}")
                raise e
                continue

        return K562PerturbationOutput(
            pred_log_fc_dict=pred_log_fc_dict,
            true_log_fc_dict=true_log_fc_dict,
        )

    def _compute_metrics(
        self,
        task_input: K562PerturbationTaskInput,
        task_output: K562PerturbationOutput,
    ) -> List[MetricResult]:
        """
        Computes perturbation prediction quality metrics.

        For each perturbation, computes:
        - Mean Squared Error (MSE) between predicted and true log fold changes
        - Pearson correlation between predicted and true log fold changes
        - Jaccard similarity between binarized predicted and true log fold changes

        Args:
            task_input: K562PerturbationTaskInput with inputs for the task
            task_output: K562PerturbationOutput with task outputs from _run_task

        Returns:
            List[MetricResult]: MetricResult objects containing metric values and metadata
        """

        accuracy_results = []
        for condition in task_output.pred_log_fc_dict.keys():
            pred_log_fc = task_output.pred_log_fc_dict[condition]
            true_log_fc = task_output.true_log_fc_dict[condition]
            true_binary, pred_binary = binarize_values(true_log_fc, pred_log_fc)

            accuracy_metric = MetricType.ACCURACY
            accuracy_values = metrics_registry.compute(
                accuracy_metric,
                y_true=true_binary,
                y_pred=pred_binary,
            )
            accuracy_results.append(
                MetricResult(
                    metric_type=accuracy_metric,
                    value=accuracy_values,
                    params={"condition": condition},
                )
            )

        return accuracy_results
