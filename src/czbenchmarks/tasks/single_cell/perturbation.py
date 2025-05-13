from typing import Literal, Set, List
import pandas as pd
import scanpy as sc
import anndata as ad
import numpy as np
import logging
from ..base import BaseTask
from ...datasets import PerturbationSingleCellDataset, DataType
from ...metrics import metrics_registry
from ...metrics.types import MetricResult, MetricType
from ...models.types import ModelType

logger = logging.getLogger(__name__)


class PerturbationTask(BaseTask):
    """Task for evaluating perturbation prediction quality.

    This task computes metrics to assess how well a model predicts gene expression
    changes in response to perturbations. Compares predicted vs ground truth
    perturbation effects using MSE and correlation metrics.
    """

    @property
    def display_name(self) -> str:
        """A pretty name to use when displaying task results"""
        return "perturbation"

    @property
    def required_inputs(self) -> Set[DataType]:
        """Required input data types.

        Returns:
            Set of required input DataTypes (ground truth perturbation effects)
        """
        return {DataType.PERTURBATION_TRUTH}

    @property
    def required_outputs(self) -> Set[DataType]:
        """Required output data types.

        Returns:
            required output types from models this task to run
            (predicted perturbation effects)
        """
        return {DataType.PERTURBATION_PRED}

    def _run_task(self, data: PerturbationSingleCellDataset, model_type: ModelType):
        """Runs the perturbation evaluation task.

        Gets predicted perturbation effects, ground truth effects, and control
        expression from the dataset for metric computation.

        Args:
            data: Dataset containing perturbation predictions and ground truth
        """
        self.gene_pert, self.perturbation_pred = data.get_output(
            model_type, DataType.PERTURBATION_PRED
        )
        self.perturbation_truth = data.perturbation_truth
        self.perturbation_ctrl = data.adata.X.toarray()
        self.avg_perturbation_ctrl = pd.Series(
            data=self.perturbation_ctrl.mean(0),
            index=data.adata.var_names,
            name="ctrl",
        )

    def _compute_metrics(self) -> List[MetricResult]:
        """Computes perturbation prediction quality metrics.

        For each perturbation, computes:
        - MSE between predicted and true expression
        - Correlation between predicted and true expression changes from control

        Returns:
            List of MetricResult objects containing metric values and metadata
        """

        avg_perturbation_control = self.avg_perturbation_ctrl

        mean_squared_error_metric = MetricType.MEAN_SQUARED_ERROR
        pearson_correlation_metric = MetricType.PEARSON_CORRELATION
        jaccard_metric = MetricType.JACCARD

        if self.gene_pert in self.perturbation_truth.keys():
            # Run differential expression analysis between control and predicted/truth
            # Create AnnData objects for control, prediction, and truth
            adata_ctrl = ad.AnnData(X=self.perturbation_ctrl)
            adata_pred = ad.AnnData(X=self.perturbation_pred.values)
            adata_truth = ad.AnnData(X=self.perturbation_truth[self.gene_pert].values)

            # Ensure they have the same var_names
            genes = self.perturbation_pred.columns
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
            self.de_results_pred = sc.get.rank_genes_groups_df(
                adata_ctrl_pred, group="pred"
            )
            self.de_results_truth = sc.get.rank_genes_groups_df(
                adata_ctrl_truth, group="truth"
            )

            avg_perturbation_pred = self.perturbation_pred.mean(axis=0)
            avg_perturbation_truth = self.perturbation_truth[self.gene_pert].mean(
                axis=0
            )

            intersecting_genes = list(
                set(avg_perturbation_pred.index)
                & set(avg_perturbation_truth.index)
                & set(avg_perturbation_control.index)
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
                - avg_perturbation_control[intersecting_genes],
                y=avg_perturbation_pred[intersecting_genes]
                - avg_perturbation_control[intersecting_genes],
            ).statistic

            # 2. Calculate metrics for top 20 DE genes
            top20_de_genes = (
                self.de_results_truth.sort_values("scores", ascending=False)
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
                - avg_perturbation_control[top20_de_genes],
                y=avg_perturbation_pred[top20_de_genes]
                - avg_perturbation_control[top20_de_genes],
            ).statistic

            # 3. Calculate metrics for top 100 DE genes
            top100_de_genes = (
                self.de_results_truth.sort_values("scores", ascending=False)
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
                - avg_perturbation_control[top100_de_genes],
                y=avg_perturbation_pred[top100_de_genes]
                - avg_perturbation_control[top100_de_genes],
            ).statistic

            # Calculate Jaccard similarity for top DE genes
            top20_pred_de_genes = set(
                self.de_results_pred.sort_values("scores", ascending=False)
                .head(20)["names"]
                .tolist()
            )
            top20_truth_de_genes = set(
                self.de_results_truth.sort_values("scores", ascending=False)
                .head(20)["names"]
                .tolist()
            )

            jaccard_top20 = metrics_registry.compute(
                jaccard_metric,
                y_true=top20_truth_de_genes,
                y_pred=top20_pred_de_genes,
            )

            top100_pred_de_genes = set(
                self.de_results_pred.sort_values("scores", ascending=False)
                .head(100)["names"]
                .tolist()
            )
            top100_truth_de_genes = set(
                self.de_results_truth.sort_values("scores", ascending=False)
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
                f"Perturbation {self.gene_pert} is not available in the ground truth "
                "test perturbations."
            )

    def set_baseline(
        self,
        data: PerturbationSingleCellDataset,
        gene_pert: str,
        baseline_type: Literal["median", "mean"] = "median",
        **kwargs,
    ):
        """Set a baseline embedding for perturbation prediction.

        Creates baseline predictions using simple statistical methods (median and mean)
        applied to the control data, and evaluates these predictions against ground
        truth.

        Args:
            data: PerturbationSingleCellDataset containing control and perturbed data
            gene_pert: The perturbation gene to evaluate
            baseline_type: The statistical method to use for baseline prediction
                (median or mean)
            **kwargs: Additional arguments passed to the evaluation

        Returns:
            List of MetricResult objects containing baseline performance metrics
            for different statistical methods (median, mean)
        """

        # Iterate through different statistical baseline functions (median and mean)

        # Create baseline prediction by replicating the aggregated expression values
        # across all cells in the dataset.
        baseline_func = np.median if baseline_type == "median" else np.mean
        perturb_baseline_pred = pd.DataFrame(
            np.tile(
                baseline_func(data.adata.X.toarray(), axis=0), (data.adata.shape[0], 1)
            ),
            columns=data.adata.var_names,  # Use gene names from the dataset
            index=data.adata.obs_names,  # Use cell names from the dataset
        )

        # Store the baseline prediction in the dataset for evaluation
        data.set_output(
            ModelType.BASELINE,
            DataType.PERTURBATION_PRED,
            (gene_pert, perturb_baseline_pred),
        )
