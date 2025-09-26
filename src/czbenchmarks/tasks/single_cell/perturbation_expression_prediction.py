import logging
from pathlib import Path
from typing import Dict, List, Literal

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse as sp_sparse
import json

from ...constants import RANDOM_SEED
from ...metrics import metrics_registry
from ...metrics.types import MetricResult, MetricType
from ...tasks.types import CellRepresentation
from ..task import Task, TaskInput, TaskOutput
from ..utils import binarize_values, looks_like_lognorm

logger = logging.getLogger(__name__)


class PerturbationExpressionPredictionTaskInput(TaskInput):
    """Pydantic model for PerturbationTask inputs."""

    adata: ad.AnnData
    target_conditions_dict: Dict[str, List[str]]
    de_results: pd.DataFrame
    gene_index: pd.Index
    cell_index: pd.Index


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

    # Load DE results from parquet
    de_results_path = inputs_dir / "de_results.parquet"
    de_results = pd.read_parquet(de_results_path)

    # Load target conditions dict
    target_conditions_path = inputs_dir / "target_conditions_dict.json"
    with target_conditions_path.open("r") as f:
        target_conditions_dict = json.load(f)

    # Load the main AnnData object
    adata_path = inputs_dir / "control_matched_adata.h5ad"
    adata = ad.read_h5ad(adata_path)

    # Extract gene_index and cell_index
    gene_index = adata.var.index
    cell_index = pd.Index(adata.uns["cell_barcode_condition_index"])

    return PerturbationExpressionPredictionTaskInput(
        adata=adata,
        target_conditions_dict=target_conditions_dict,
        de_results=de_results,
        gene_index=gene_index,
        cell_index=cell_index,
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
        control_name: str = "non-targeting",
        condition_key: str = "condition",
        *,
        random_seed: int = RANDOM_SEED,
    ):
        """
        Args:
            control_prefix (str): Prefix for control conditions.
            random_seed (int): Random seed for reproducibility.
        """
        super().__init__(random_seed=random_seed)
        self.metric_column = "logfoldchange"  # TODO: logfoldchange only for now
        self.control_name = control_name
        self.condition_key = condition_key

    def _run_task(
        self,
        cell_representation: CellRepresentation,
        task_input: PerturbationExpressionPredictionTaskInput,
    ) -> PerturbationExpressionPredictionOutput:
        """
        Runs the perturbation evaluation task.

        Args:
            cell_representation: Cell expression matrix of shape (n_cells, n_genes)
            task_input: Task input containing AnnData with all necessary data

        Returns:
            PerturbationExpressionPredictionOutput: Predicted and true log fold changes
        """
        self._validate(task_input, cell_representation)
        pred_log_fc_dict = {}
        true_log_fc_dict = {}
        adata = task_input.adata

        # Extract data from AnnData
        obs = adata.obs
        de_results = task_input.de_results
        target_conditions_dict = task_input.target_conditions_dict

        # Get perturbation conditions (non-control)
        conditions = obs["condition"].astype(str)
        perturbation_conditions = np.unique(
            conditions[~conditions.str.startswith(self.control_name)]
        )

        # Extract base cell IDs for matching
        base_cell_ids = task_input.cell_index.str.split("_").str[0]

        for condition in perturbation_conditions:
            # Get target genes for this condition
            target_genes = target_conditions_dict.get(condition, [])
            valid_genes = [g for g in target_genes if g in task_input.gene_index]
            if not valid_genes:
                logger.warning(
                    "Skipping condition %s - no valid target genes", condition
                )
                continue
            # This is where the true and predicted log fold changes are computed for each condition
            # This outputs an array of true log fold changes for each cell in the condition
            # and a corresponding array of predicted log fold changes for each cell in the condition
            # Get the true DE results for this condition
            condition_de = de_results[de_results[self.condition_key] == condition]

            # Get true log fold changes from DE results
            true_lfc = (
                condition_de.set_index("gene_id")
                .reindex(valid_genes)[self.metric_column]
                .values
            )
            # Mask out genes with NaN true log fold change values
            valid_mask = ~np.isnan(true_lfc)
            n_filtered = (~valid_mask).sum()
            if n_filtered:
                logger.warning(
                    f"Filtered out {n_filtered} NaN true log fold changes for {condition}"
                )
            # Only keep genes with valid (non-NaN) true log fold change values
            final_genes = np.array(valid_genes)[valid_mask]
            true_lfc = true_lfc[valid_mask]
            # true_lfc could be float, so convert to string for join

            # If no valid genes remain for this condition, skip to next
            if len(final_genes) == 0:
                logger.warning(
                    f"Skipping condition {condition} - no valid genes remain after filtering"
                )
                continue

            # Get indices of the valid genes in task_input.gene_index for slicing the cell_representation matrix
            gene_indices = task_input.gene_index.get_indexer(final_genes)
            # Find cell barcodes for the current perturbation condition
            # This extracts the base cell IDs (before the underscore) for all cells in the current condition
            condition_cells = (
                obs[obs[self.condition_key] == condition].index.str.split("_").str[0]
            )
            # Find cell barcodes for the corresponding control cells
            # Control cells are expected to have a condition label like "controlPrefix_condition"
            control_cells = (
                obs[obs[self.condition_key] == f"{self.control_name}_{condition}"]
                .index.str.split("_")
                .str[0]
            )
            # Get indices of the condition and control cells in the cell_representation matrix
            condition_idx = np.where(base_cell_ids.isin(condition_cells))[0]
            control_idx = np.where(base_cell_ids.isin(control_cells))[0]
            # Compute predicted log fold change for each gene:
            #   - Take the mean expression of each gene across all condition cells
            #   - Subtract the mean expression of the same gene across all control cells
            pred_lfc = cell_representation[np.ix_(condition_idx, gene_indices)].mean(
                axis=0
            ) - cell_representation[np.ix_(control_idx, gene_indices)].mean(axis=0)
            # Store the predicted and true log fold changes for this condition
            pred_log_fc_dict[condition] = pred_lfc
            true_log_fc_dict[condition] = true_lfc

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

    def _validate(
        self,
        task_input: PerturbationExpressionPredictionTaskInput,
        cell_representation: CellRepresentation,
    ) -> None:
        if not looks_like_lognorm(cell_representation):
            raise ValueError(
                "Task input likelihood contains non-log-normalized data. Please provide a log-normalized cell representation."
            )

        if "cell_barcode_condition_index" not in task_input.adata.uns:
            raise ValueError("Task input contains no cell barcode index.")
        # Assert that the same values are in both gene and cell indices before re-assigning
        if not set(task_input.gene_index).issubset(set(task_input.adata.var.index)):
            raise ValueError(
                "Model data contains genes that are not in the task input."
            )
        if not set(task_input.cell_index).issubset(
            set(task_input.adata.uns["cell_barcode_condition_index"])
        ):
            raise ValueError(
                "Model data contains cells that are not in the task input."
            )

        if set(task_input.gene_index) != set(task_input.adata.var.index):
            logger.warning("Task input contains genes that are not in the model input.")

        if set(task_input.cell_index) != set(
            task_input.adata.uns["cell_barcode_condition_index"]
        ):
            logger.warning("Task input contains cells that are not in the model input.")
