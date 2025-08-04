from typing import Literal, List, Dict
import pandas as pd
import scanpy as sc
import anndata as ad
import numpy as np
import scipy as sp
import logging
from ..task import Task, TaskInput, TaskOutput
from ...tasks.types import CellRepresentation
from ...types import ListLike
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

class PerturbationTaskInput(TaskInput):
    """Pydantic model for PerturbationTask inputs."""

    var_names: ListLike
    gene_pert: str
    perturbation_pred: pd.DataFrame
    perturbation_truth: Dict[str, pd.DataFrame]


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
        task_input: list,
    ) -> K562PerturbationOutput:
        """Runs the perturbation evaluation task.

        Gets predicted perturbation effects, ground truth effects, and control
        expression from the dataset for metric computation.

        Args:

        Returns:
            PerturbationOutput: Pydantic model with control data and averages
        """
        #TODO: reshape this thing. 
        de_res_wilcoxon_df, pred_df = task_input

        genes_dict = {}
        predictions_dict = {}
        pred_log_fc_dict = {}
        true_log_fc_dict = {}
        num_de_genes_dict = {}

        for sample, group in tqdm(pred_df.groupby('sample_id')):
            predictions_dict.setdefault(sample, []).append(np.concatenate(group['pred'].to_numpy()))
            if sample not in genes_dict:
                genes_dict[sample] = group['target_genes'].to_numpy()

        for condition in cell_representation["gene"].unique():
            condition_de_df = de_res_wilcoxon_df[de_res_wilcoxon_df["target_gene"] == condition]
            if len(condition_de_df) < self.min_de_genes:
                continue

            cell_condition = cell_representation[cell_representation.index.str.endswith(f"_{condition}")]
            condition_cells = cell_condition[cell_condition["gene"] != "non-targeting"].index
            control_cells = cell_condition[cell_condition["gene"] == "non-targeting"].index
            if len(control_cells) != len(condition_cells):
                raise AssertionError(f"Number of control cells ({len(control_cells)}) is not equal to number of condition cells ({len(condition_cells)})")
            if len(control_cells) < 10:
                print(f"Less than 10 cells in condition {condition}. Skipping...")
                continue
            
            try:
                # Masked genes for this sample
                masked_genes = genes_dict[control_cells[0]]
                mask = masked_genes != "A" #TODO: WHY?? 
                masked_genes = masked_genes[mask]
                n_masked_genes = len(masked_genes)
                # Get predictions for control and condition cells
                control_predictions = np.asarray([predictions_dict[cell][-1][:n_masked_genes] for cell in control_cells])
                condition_predictions = np.asarray([predictions_dict[cell][-1][:n_masked_genes] for cell in condition_cells])

               # Compute predicted log fold change
                pred_log_fc = condition_predictions.mean(axis=0) - control_predictions.mean(axis=0)

                # Select column for true log fold change
                col = "logfoldchanges"
                # Align true log fold change to masked genes
                true_log_fc = condition_de_df.set_index("names").reindex(masked_genes)[col].values

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
        task_input: list,
        task_output: K562PerturbationOutput,
    ) -> List[MetricResult]:
        """Computes perturbation prediction quality metrics.

        For each perturbation, computes:
        - MSE between predicted and true expression
        - Correlation between predicted and true expression changes from control

        Args:
            task_input: Pydantic model with inputs for the task
            task_output: Pydantic model with task outputs from _run_task

        Returns:
            List of MetricResult objects containing metric values and metadata
        """

    
        accuracy_results = []
        for condition in task_output.pred_log_fc_dict.keys():
            pred_log_fc = task_output.pred_log_fc_dict[condition]
            true_log_fc = task_output.true_log_fc_dict[condition]
            true_binary, pred_binary = binarize_values(true_log_fc, pred_log_fc)
            
            accuracy_metric = MetricType.ACCURACY
            accuracy_result = metrics_registry.compute(
                accuracy_metric,
                y_true=true_binary,
                y_pred=pred_binary,
            )

        """

        mean_squared_error_metric = MetricType.MEAN_SQUARED_ERROR
        pearson_correlation_metric = MetricType.PEARSON_CORRELATION
        jaccard_metric = MetricType.JACCARD

        gene_pert = task_input.gene_pert
        perturbation_pred = task_input.perturbation_pred
        perturbation_truth = task_input.perturbation_truth
        perturbation_ctrl = task_output.perturbation_ctrl
        avg_perturbation_ctrl = task_output.avg_perturbation_ctrl

        if gene_pert in perturbation_truth.keys():
            # Run differential expression analysis between control and
            # predicted/truth Create AnnData objects for control, prediction,
            # and truth
            adata_ctrl = ad.AnnData(X=perturbation_ctrl)
            adata_pred = ad.AnnData(X=perturbation_pred.values)
            adata_truth = ad.AnnData(X=perturbation_truth[gene_pert].values)

            # Ensure they have the same var_names
            genes = perturbation_pred.columns
            adata_ctrl.var_names = genes
            adata_pred.var_names = genes
            adata_truth.var_names = genes

            # Create combined AnnData for control vs prediction
            adata_ctrl_pred = ad.AnnData(
                X=np.vstack([adata_ctrl.X, adata_pred.X]),
                obs={
                    "condition": ["ctrl"] * adata_ctrl.n_obs
                    + ["pred"] * adata_pred.n_obs
                },
            )
            adata_ctrl_pred.var_names = genes

            # Create combined AnnData for control vs truth
            adata_ctrl_truth = ad.AnnData(
                X=np.vstack([adata_ctrl.X, adata_truth.X]),
                obs={
                    "condition": ["ctrl"] * adata_ctrl.n_obs
                    + ["truth"] * adata_truth.n_obs
                },
            )
            adata_ctrl_truth.var_names = genes

            # Run rank_genes_groups for control vs prediction
            sc.tl.rank_genes_groups(
                adata_ctrl_pred,
                groupby="condition",
                groups=["pred"],
                reference="ctrl",
                method="wilcoxon",
            )

            # Run rank_genes_groups for control vs truth
            sc.tl.rank_genes_groups(
                adata_ctrl_truth,
                groupby="condition",
                groups=["truth"],
                reference="ctrl",
                method="wilcoxon",
            )

            # Store the results for later use if needed
            de_results_pred = sc.get.rank_genes_groups_df(adata_ctrl_pred, group="pred")
            de_results_truth = sc.get.rank_genes_groups_df(
                adata_ctrl_truth, group="truth"
            )

            avg_perturbation_pred = perturbation_pred.mean(axis=0)
            avg_perturbation_truth = perturbation_truth[gene_pert].mean(axis=0)

            intersecting_genes = list(
                set(avg_perturbation_pred.index)
                & set(avg_perturbation_truth.index)
                & set(avg_perturbation_ctrl.index)
            )

            # 1. Calculate metrics for all genes
            mse_all = metrics_registry.compute(
                mean_squared_error_metric,
                y_true=avg_perturbation_truth[intersecting_genes],
                y_pred=avg_perturbation_pred[intersecting_genes],
            )
            delta_pearson_corr_all = metrics_registry.compute(
                pearson_correlation_metric,
                x=avg_perturbation_truth[intersecting_genes]
                - avg_perturbation_ctrl[intersecting_genes],
                y=avg_perturbation_pred[intersecting_genes]
                - avg_perturbation_ctrl[intersecting_genes],
            ).statistic

            # 2. Calculate metrics for top 20 DE genes
            top20_de_genes = (
                de_results_truth.sort_values("scores", ascending=False)
                .head(20)["names"]
                .tolist()
            )
            top20_de_genes = [
                gene for gene in top20_de_genes if gene in intersecting_genes
            ]

            mse_top20 = metrics_registry.compute(
                mean_squared_error_metric,
                y_true=avg_perturbation_truth[top20_de_genes],
                y_pred=avg_perturbation_pred[top20_de_genes],
            )
            delta_pearson_corr_top20 = metrics_registry.compute(
                pearson_correlation_metric,
                x=avg_perturbation_truth[top20_de_genes]
                - avg_perturbation_ctrl[top20_de_genes],
                y=avg_perturbation_pred[top20_de_genes]
                - avg_perturbation_ctrl[top20_de_genes],
            ).statistic

            # 3. Calculate metrics for top 100 DE genes
            top100_de_genes = (
                de_results_truth.sort_values("scores", ascending=False)
                .head(100)["names"]
                .tolist()
            )
            top100_de_genes = [
                gene for gene in top100_de_genes if gene in intersecting_genes
            ]

            mse_top100 = metrics_registry.compute(
                mean_squared_error_metric,
                y_true=avg_perturbation_truth[top100_de_genes],
                y_pred=avg_perturbation_pred[top100_de_genes],
            )
            delta_pearson_corr_top100 = metrics_registry.compute(
                pearson_correlation_metric,
                x=avg_perturbation_truth[top100_de_genes]
                - avg_perturbation_ctrl[top100_de_genes],
                y=avg_perturbation_pred[top100_de_genes]
                - avg_perturbation_ctrl[top100_de_genes],
            ).statistic

            # Calculate Jaccard similarity for top DE genes
            top20_pred_de_genes = set(
                de_results_pred.sort_values("scores", ascending=False)
                .head(20)["names"]
                .tolist()
            )
            top20_truth_de_genes = set(
                de_results_truth.sort_values("scores", ascending=False)
                .head(20)["names"]
                .tolist()
            )

            jaccard_top20 = metrics_registry.compute(
                jaccard_metric,
                y_true=top20_truth_de_genes,
                y_pred=top20_pred_de_genes,
            )

            top100_pred_de_genes = set(
                de_results_pred.sort_values("scores", ascending=False)
                .head(100)["names"]
                .tolist()
            )
            top100_truth_de_genes = set(
                de_results_truth.sort_values("scores", ascending=False)
                .head(100)["names"]
                .tolist()
            )

            jaccard_top100 = metrics_registry.compute(
                jaccard_metric,
                y_true=top100_truth_de_genes,
                y_pred=top100_pred_de_genes,
            )

            return [
                MetricResult(
                    metric_type=mean_squared_error_metric,
                    value=mse_all,
                    params={"subset": "all"},
                ),
                MetricResult(
                    metric_type=pearson_correlation_metric,
                    value=delta_pearson_corr_all,
                    params={"subset": "all"},
                ),
                MetricResult(
                    metric_type=mean_squared_error_metric,
                    value=mse_top20,
                    params={"subset": "top20"},
                ),
                MetricResult(
                    metric_type=pearson_correlation_metric,
                    value=delta_pearson_corr_top20,
                    params={"subset": "top20"},
                ),
                MetricResult(
                    metric_type=mean_squared_error_metric,
                    value=mse_top100,
                    params={"subset": "top100"},
                ),
                MetricResult(
                    metric_type=pearson_correlation_metric,
                    value=delta_pearson_corr_top100,
                    params={"subset": "top100"},
                ),
                MetricResult(
                    metric_type=jaccard_metric,
                    value=jaccard_top20,
                    params={"subset": "top20"},
                ),
                MetricResult(
                    metric_type=jaccard_metric,
                    value=jaccard_top100,
                    params={"subset": "top100"},
                ),
            ]
        else:
            raise ValueError(
                f"Perturbation {gene_pert} is not available in the ground truth "
                "test perturbations."
            )
        """
    @staticmethod
    def compute_baseline(
        cell_representation: CellRepresentation,
        var_names: ListLike,
        obs_names: ListLike,
        baseline_type: Literal["median", "mean"] = "median",
    ) -> CellRepresentation:
        """Set a baseline perturbation prediction using mean or median expression.

        This method creates a baseline prediction by either taking the mean or
        the median of the control cells' gene expression. This baseline
        represents a simple no-change prediction.

        Args:
            cell_representation: The gene expression matrix of control cells.
            var_names: The names of the genes.
            obs_names: The names of the observations (cells).
            baseline_type: The type of baseline to use, either "mean" or "median".

        Returns:
            A DataFrame representing the baseline perturbation prediction.
        """
        # Create baseline prediction by replicating the aggregated expression values
        # across all cells in the dataset.
        baseline_func = np.median if baseline_type == "median" else np.mean
        if baseline_type == "median" and sp.sparse.issparse(cell_representation):
            cell_representation = cell_representation.toarray()

        perturb_baseline_pred = pd.DataFrame(
            np.tile(
                baseline_func(cell_representation, axis=0),
                (cell_representation.shape[0], 1),
            ),
            columns=var_names,  # Use gene names from the dataset
            index=obs_names,  # Use cell names from the dataset
        )

        # Store the baseline prediction in the dataset for evaluation
        return perturb_baseline_pred
