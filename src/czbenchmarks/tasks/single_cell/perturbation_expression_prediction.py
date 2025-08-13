from typing import List, Dict
import pandas as pd
import numpy as np
import logging
from ..task import Task, TaskInput, TaskOutput
from ...tasks.types import CellRepresentation
from ...metrics import metrics_registry
from ...metrics.types import MetricResult, MetricType
from ..utils import binarize_values
from ...constants import RANDOM_SEED
from anndata import AnnData

logger = logging.getLogger(__name__)


class PerturbationExpressionPredictionTaskInput(TaskInput):
    """Pydantic model for PerturbationTask inputs."""

    de_results: pd.DataFrame
    dataset_adata: AnnData
    target_genes_to_save: Dict[str, List[str]]


class PerturbationExpressionPredictionOutput(TaskOutput):
    """Output for perturbation task."""

    pred_log_fc_dict: Dict[str, np.ndarray]
    true_log_fc_dict: Dict[str, np.ndarray]


class PerturbationExpressionPredictionTask(Task):
    display_name = "perturbation_expression_prediction"

    def __init__(
        self,
        min_de_genes: int = 10,
        control_gene: str = "non-targeting",
        metric_column: str = "logfoldchange",
        metric_type: str = "wilcoxon",
        pval_threshold: float = 1e-4,
        standardized_mean_diff: float = 0.5,
        min_logfoldchange: float = 1.0,
        *,
        random_seed: int = RANDOM_SEED,
    ):
        """
        Args:
            min_de_genes (int): Minimum number of DE genes for a perturbation condition.
            control_gene (str): Name of the control gene (default: "non-targeting").
            metric_column (str): Column name for metric (default: "logfoldchange").
            metric_type (str): Type of DE metric (default: "wilcoxon").
            pval_threshold (float): Adjusted p-value threshold for DE gene filtering.
            standardized_mean_diff (float): Minimum standardized mean difference for DE gene filtering.
            random_seed (int): Random seed for reproducibility.
        """
        super().__init__(random_seed=random_seed)
        self.min_de_genes = min_de_genes
        self.control_gene = control_gene
        self.metric_column = metric_column
        self.metric_type = metric_type
        self.pval_threshold = pval_threshold
        self.standardized_mean_diff = standardized_mean_diff
        self.min_logfoldchange = min_logfoldchange

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
        de_results = task_input.de_results[
            task_input.de_results["pval"] < self.pval_threshold
        ]

        if self.metric_type == "wilcoxon":
            de_results = de_results[
                np.abs(de_results[self.metric_column]) >= self.min_logfoldchange
            ]
        elif self.metric_type == "t-test":
            de_results = de_results[
                de_results["standardized_mean_diff"].abs()
                >= self.standardized_mean_diff
            ]

        control_prefix = self.control_gene
        condition_series = task_input.dataset_adata.obs["condition"].astype(str)
        condition_list = np.unique(
            condition_series[~condition_series.str.startswith(control_prefix)]
        )
        for condition in condition_list:
            condition_de_df = de_results[de_results["condition"] == condition]
            if len(condition_de_df) < self.min_de_genes:
                continue
            masked_genes = np.array(
                task_input.target_genes_to_save[
                    task_input.dataset_adata.obs.index[
                        task_input.dataset_adata.obs["condition"] == condition
                    ][0]
                ]
            )
            if len(masked_genes) == 0:
                continue
            true_log_fc = (
                condition_de_df.set_index("gene_id")
                .reindex(masked_genes)[self.metric_column]
                .values
            )
            valid = ~np.isnan(true_log_fc)
            masked_genes = masked_genes[valid]
            true_log_fc = true_log_fc[valid]
            col_indices = task_input.dataset_adata.var.index.get_indexer(masked_genes)
            condition_idx = np.where(
                task_input.dataset_adata.obs["condition"] == condition
            )[0]

            control_idx = np.where(
                task_input.dataset_adata.obs["condition"]
                == f"{control_prefix}_{condition}"
            )[0]

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
    ) -> List[List[MetricResult]]:
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
        metrics_dict = {
            "accuracy": accuracy_results,
            "precision": precision_results,
            "recall": recall_results,
            "f1": f1_results,
            "correlation": spearman_correlation_results,
        }

        return metrics_dict
