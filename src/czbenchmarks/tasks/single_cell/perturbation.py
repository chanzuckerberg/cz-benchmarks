from typing import Literal, List, Dict
import pandas as pd
import scanpy as sc
import anndata as ad
import numpy as np
import scipy as sp
import logging
from ..base import BaseTask
from ...datasets.types import CellRepresentation, ListLike
from ...metrics import metrics_registry
from ...metrics.types import MetricResult, MetricType
from ...constants import RANDOM_SEED
from ..types import TaskInput, MetricInput

logger = logging.getLogger(__name__)


__all__ = [
    "PerturbationTaskInput",
    "PerturbationMetricInput",
    "PerturbationTask",
]


class PerturbationTaskInput(TaskInput):
    """Pydantic model for PerturbationTask inputs."""

    var_names: ListLike


class PerturbationMetricInput(MetricInput):
    """Pydantic model for PerturbationTask metric inputs."""

    gene_pert: str
    perturbation_pred: pd.DataFrame
    perturbation_truth: Dict[str, pd.DataFrame]


class PerturbationTask(BaseTask):
    """Task for evaluating perturbation prediction quality.

    This task computes metrics to assess how well a model predicts gene expression
    changes in response to perturbations. Compares predicted vs ground truth
    effects using MSE and correlation metrics.
    """

    def __init__(self, *, random_seed: int = RANDOM_SEED):
        super().__init__(random_seed=random_seed)
        self.display_name = "perturbation"

    def _run_task(
        self,
        cell_representation: CellRepresentation,
        task_input: PerturbationTaskInput,
    ) -> dict:
        """Runs the perturbation evaluation task.

        Gets predicted perturbation effects, ground truth effects, and control
        expression from the dataset for metric computation.

        Args:
            cell_representation: gene expression data or embedding for task
            task_input: Pydantic model with inputs for the task

        Returns:
            Dictionary of gene perturbation, perturbation predictions,
            perturbation truth, and perturbation control
        """

        if sp.sparse.issparse(cell_representation):
            cell_representation = cell_representation.toarray()

        perturbation_ctrl = cell_representation

        avg_perturbation_ctrl = pd.Series(
            data=perturbation_ctrl.mean(0),
            index=task_input.var_names,
            name="ctrl",
        )

        return {
            "perturbation_ctrl": perturbation_ctrl,
            "avg_perturbation_ctrl": avg_perturbation_ctrl,
        }

    def _compute_metrics(
        self,
        task_output: dict,
        metric_input: PerturbationMetricInput,
    ) -> List[MetricResult]:
        """Computes perturbation prediction quality metrics.

        For each perturbation, computes:
        - MSE between predicted and true expression
        - Correlation between predicted and true expression changes from control

        Args:
            task_output: Dictionary of outputs from _run_task
            metric_input: Pydantic model with inputs for the metrics

        Returns:
            List of MetricResult objects containing metric values and metadata
        """
        # FIXME BYOTASK: refactor so pandas objects are not required as inputs
        # FIXME BYOTASK: this is quite involved and should be broken into more
        # simple functions

        mean_squared_error_metric = MetricType.MEAN_SQUARED_ERROR
        pearson_correlation_metric = MetricType.PEARSON_CORRELATION
        jaccard_metric = MetricType.JACCARD

        gene_pert = metric_input.gene_pert
        perturbation_pred = metric_input.perturbation_pred
        perturbation_truth = metric_input.perturbation_truth
        perturbation_ctrl = task_output["perturbation_ctrl"]
        avg_perturbation_ctrl = task_output["avg_perturbation_ctrl"]

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
            de_results_pred = sc.get.rank_genes_groups_df(
                adata_ctrl_pred, group="pred"
            )
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
                    params={"subset": "top20_de"},
                ),
                MetricResult(
                    metric_type=pearson_correlation_metric,
                    value=delta_pearson_corr_top20,
                    params={"subset": "top20_de"},
                ),
                MetricResult(
                    metric_type=mean_squared_error_metric,
                    value=mse_top100,
                    params={"subset": "top100_de"},
                ),
                MetricResult(
                    metric_type=pearson_correlation_metric,
                    value=delta_pearson_corr_top100,
                    params={"subset": "top100_de"},
                ),
                MetricResult(
                    metric_type=jaccard_metric,
                    value=jaccard_top20,
                    params={"subset": "top20_de"},
                ),
                MetricResult(
                    metric_type=jaccard_metric,
                    value=jaccard_top100,
                    params={"subset": "top100_de"},
                ),
            ]
        return []

    @staticmethod
    def compute_baseline(
        cell_representation: CellRepresentation,
        var_names: ListLike,
        obs_names: ListLike,
        baseline_type: Literal["median", "mean"] = "median",
    ) -> pd.DataFrame:
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
        if baseline_type == "mean":
            baseline_embedding = np.mean(cell_representation, axis=0)
        elif baseline_type == "median":
            if sp.sparse.issparse(cell_representation):
                cell_representation = cell_representation.toarray()
            baseline_embedding = np.median(cell_representation, axis=0)
        else:
            raise ValueError("Invalid baseline type specified")

        return pd.DataFrame(
            data=np.tile(baseline_embedding, (len(obs_names), 1)),
            index=obs_names,
            columns=var_names,
        )
