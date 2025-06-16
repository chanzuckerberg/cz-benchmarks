from typing import Literal, List
import pandas as pd
import scanpy as sc
import anndata as ad
import numpy as np
import scipy as sp
import logging
from ..base import BaseTask
from ...datasets.types import Embedding, ListLike
from ...metrics import metrics_registry
from ...metrics.types import MetricResult, MetricType
from ...constants import RANDOM_SEED

logger = logging.getLogger(__name__)


class PerturbationTask(BaseTask):
    """Task for evaluating perturbation prediction quality.

    This task computes metrics to assess how well a model predicts gene expression
    changes in response to perturbations. Compares predicted vs ground truth
    perturbation effects using MSE and correlation metrics.
    """

    def __init__(self, *, random_seed: int = RANDOM_SEED):
        super().__init__(random_seed=random_seed)
        self.display_name = "perturbation"

    def _run_task(self, cell_representation: Embedding, var_names: ListLike) -> dict:
        """Runs the perturbation evaluation task.

        Gets predicted perturbation effects, ground truth effects, and control
        expression from the dataset for metric computation.

        Args:
            cell_representation: gene expression data or embedding for task
            var_names: list of gene names

        Returns:
            Dictionary of gene perturbation, perturbation predictions,
            perturbation truth, and perturbation control
        """

        if sp.sparse.issparse(cell_representation):
            cell_representation = cell_representation.toarray()

        perturbation_ctrl = cell_representation.mean(0)

        avg_perturbation_ctrl = pd.Series(
            data=perturbation_ctrl,
            index=var_names,
            name="ctrl",
        )

        return {
            "perturbation_ctrl": perturbation_ctrl,
            "avg_perturbation_ctrl": avg_perturbation_ctrl,
        }

    def _compute_metrics(
        self,
        gene_pert: str,
        perturbation_pred: Embedding,
        perturbation_truth: pd.DataFrame,
        perturbation_ctrl: Embedding,
        avg_perturbation_ctrl: pd.Series,
        **kwargs,
    ) -> List[MetricResult]:
        """Computes perturbation prediction quality metrics.

        For each perturbation, computes:
        - MSE between predicted and true expression
        - Correlation between predicted and true expression changes from control

        Args:
            gene_pert: The perturbation gene to evaluate
            perturbation_pred: The predicted perturbation effects with genes as column names
            perturbation_ctrl: The gene expression matrix for the control conditions
            perturbation_truth: The gene expression matrix for the ground truth perturbation effects
            avg_perturbation_ctrl: The average control expression

        Returns:
            List of MetricResult objects containing metric values and metadata
        """
        # FIXME BYOTASK: refactor so pandas objects are not required as inputs
        # FIXME BYOTASK: this is quite involved and should be broken into more simple functions

        mean_squared_error_metric = MetricType.MEAN_SQUARED_ERROR
        pearson_correlation_metric = MetricType.PEARSON_CORRELATION
        jaccard_metric = MetricType.JACCARD

        if gene_pert in perturbation_truth.keys():
            # Run differential expression analysis between control and predicted/truth
            # Create AnnData objects for control, prediction, and truth
            adata_ctrl = ad.AnnData(X=perturbation_ctrl)
            adata_pred = ad.AnnData(X=perturbation_pred)
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

    @staticmethod
    def set_baseline(
        cell_representation: Embedding,
        var_names: ListLike,
        obs_names: ListLike,
        baseline_type: Literal["median", "mean"] = "median",
    ) -> Embedding:
        """Set a baseline embedding for perturbation prediction.

        Creates baseline predictions using simple statistical methods (median and mean)
        applied to the control data, and evaluates these predictions against ground
        truth.

        Args:
            cell_representation: gene expression data or embedding for task
            gene_pert: The perturbation gene to evaluate
            baseline_type: The statistical method to use for baseline prediction
                (median or mean)
            **kwargs: Additional arguments passed to the evaluation

        Returns:
            Dataframe containing baseline performance metrics
            for different statistical methods (median, mean)
        """

        # Iterate through different statistical baseline functions (median and mean)

        # Create baseline prediction by replicating the aggregated expression values
        # across all cells in the dataset.
        baseline_func = np.median if baseline_type == "median" else np.mean
        perturb_baseline_pred = pd.DataFrame(
            np.tile(
                baseline_func(cell_representation, axis=0), (cell_representation.shape[0], 1)
            ),
            columns=var_names,  # Use gene names from the dataset
            index=obs_names,  # Use cell names from the dataset
        )

        # Store the baseline prediction in the dataset for evaluation
        return perturb_baseline_pred

    def _run_task_for_dataset(
        self,
        cell_representation: Embedding,
        var_names: ListLike,
        gene_pert: str,
        perturbation_pred: Embedding,
        perturbation_truth: pd.DataFrame,
    ) -> List[MetricResult]:
        """Run task for a dataset or list of datasets and compute metrics.

        This method runs the task implementation and computes the corresponding metrics.

        Args:
            cell_representation: gene expression data or embedding for task
            var_names: list of gene names
            gene_pert: perturbation gene to evaluate
            perturbation_pred: predicted perturbation effects
            perturbation_truth: ground truth perturbation effects

        Returns:
            List of MetricResult objects

        """
        task_output = self._run_task(
            cell_representation=cell_representation, var_names=var_names
        )
        metrics = self._compute_metrics(
            gene_pert=gene_pert,
            perturbation_pred=perturbation_pred,
            perturbation_truth=perturbation_truth,
            perturbation_ctrl=task_output["perturbation_ctrl"],
            avg_perturbation_ctrl=task_output["avg_perturbation_ctrl"],
        )
        return metrics
